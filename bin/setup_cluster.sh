#!/usr/bin/env bash
# bin/setup_cluster.sh — check (and with --install, build) everything a cluster needs to run lingua-clean,
# simplescale and OLMES evals. Venvs hold architecture-specific binaries (aarch64 on Dalton, x86_64 on DM1 / H200),
# so every cluster builds its own; never copy .venv* / wheelhouse / cuda-12.9 between clusters.
#
#   bash bin/setup_cluster.sh             # report only
#   bash bin/setup_cluster.sh --install   # also build missing or wrong-architecture venvs
#
# Checks, in order:
#   1. cluster profile (lingua/cluster.py) and ~/scottg/pretrain -> this cluster's storage root (PRETRAIN_ROOT)
#   2. uv, and uv-managed CPython 3.11 (lingua-clean .venv-kd) and 3.12 (its headers: simplescale's venv uses the
#      header-less system python3.12, and the Triton / SGLang JIT needs Python.h)
#   3. lingua-clean .venv-kd           (bin/install_kd_requirements.sh; aarch64 also builds flash-attn / xformers)
#   4. simplescale .venv               (uv sync --extra sglang, from the universal uv.lock)
#   5. OLMES .venv-lmeval              (x86_64: the standard install with the [gpu] extra; aarch64: by hand, see
#                                       README "Setting up on a new cluster")
set -uo pipefail

INSTALL=0
[ "${1:-}" = "--install" ] && INSTALL=1
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
ARCH="$(uname -m)"
STATUS=0
ok() { echo "  ok    $*"; }
bad() { echo "  TODO  $*"; STATUS=1; }

# A venv is usable here if its python runs and reports this machine's architecture.
venv_arch() { "$1/bin/python" -c 'import platform; print(platform.machine())' 2>/dev/null; }
venv_ok() { [ -x "$1/bin/python" ] && [ "$(venv_arch "$1")" = "${ARCH}" ]; }
venv_state() { if [ ! -e "$1" ]; then echo "missing"; elif venv_ok "$1"; then echo "ok"; else echo "wrong architecture or broken"; fi; }

echo "== 1. cluster and storage"
if ! PROFILE="$(python3 "${ROOT_DIR}/lingua/cluster.py" 2>&1)"; then
    bad "no cluster profile: ${PROFILE}"
else
    ok "profile: $(echo "${PROFILE}" | awk -F': ' '/^name:/ {print $2}') ($(echo "${PROFILE}" | sed -n 's/^notes: //p')), ${ARCH}"
fi
if [ -d "${HOME}/scottg/pretrain" ]; then
    ok "~/scottg/pretrain -> $(readlink -f "${HOME}/scottg/pretrain")"
    [ -d "${HOME}/scottg/pretrain/assets" ] || bad "~/scottg/pretrain/assets missing: transfer models/data (see the data-transfer route) before training"
else
    bad "~/scottg/pretrain missing: ln -s <this cluster's storage, e.g. /checkpoint/<team>/${USER}> ~/scottg, then mkdir -p ~/scottg/pretrain"
fi

echo "== 2. uv and Python"
if command -v uv >/dev/null; then
    ok "uv $(uv --version | awk '{print $2}')"
    for v in 3.11 3.12; do
        if uv python find "cpython-${v}" >/dev/null 2>&1 && ls -d "${HOME}"/.local/share/uv/python/cpython-"${v}".*-"${ARCH}"-*/include >/dev/null 2>&1; then
            ok "uv CPython ${v}"
        elif [ "${INSTALL}" = 1 ]; then
            uv python install "${v}" && ok "installed uv CPython ${v}" || bad "uv python install ${v} failed"
        else
            bad "uv CPython ${v} missing (uv python install ${v})"
        fi
    done
else
    bad "uv missing: curl -LsSf https://astral.sh/uv/install.sh | sh"
fi

echo "== 3. lingua-clean .venv-kd"
KD="${ROOT_DIR}/.venv-kd"
state="$(venv_state "${KD}")"
if [ "${state}" = ok ]; then
    ok "${KD}"
elif [ "${INSTALL}" = 1 ]; then
    [ -e "${KD}" ] && mv "${KD}" "${KD}.broken-$(date +%s)"
    echo "  building ${KD} (bin/install_kd_requirements.sh)"
    bash "${ROOT_DIR}/bin/install_kd_requirements.sh" && ok "${KD}" || bad "bin/install_kd_requirements.sh failed"
else
    bad "${KD}: ${state} (bash bin/install_kd_requirements.sh; on aarch64 run it on a GPU node)"
fi

echo "== 4. simplescale .venv"
SS="$(cd -- "${ROOT_DIR}/.." && pwd)/simplescale"
if [ ! -d "${SS}" ]; then
    bad "${SS} missing: git clone git@github.com:scottgeng00/simplescale.git ${SS}"
else
    state="$(venv_state "${SS}/.venv")"
    if [ "${state}" = ok ]; then
        ok "${SS}/.venv"
    elif [ "${INSTALL}" = 1 ]; then
        [ -e "${SS}/.venv" ] && mv "${SS}/.venv" "${SS}/.venv.broken-$(date +%s)"
        echo "  building ${SS}/.venv (uv sync --extra sglang)"
        (cd "${SS}" && uv sync --extra test --extra sglang --python /usr/bin/python3) && ok "${SS}/.venv" \
            || bad "uv sync in ${SS} failed"
    else
        bad "${SS}/.venv: ${state} (cd ${SS} && uv sync --extra test --extra sglang --python /usr/bin/python3)"
    fi
fi

echo "== 5. OLMES env"
OLMES="${OLMES_CONDA_ENV:-${HOME}/scottg/pretrain/olmes/.venv-lmeval}"
state="$(venv_state "${OLMES}")"
OLMES_REPO="$(dirname "${OLMES}")"
if [ "${state}" = ok ]; then
    ok "${OLMES}"
elif [ ! -d "${OLMES_REPO}/.git" ]; then
    bad "${OLMES_REPO}: no OLMES checkout (our fork, branch lm-eval-0.4.13)"
elif [ "${INSTALL}" = 1 ] && [ "${ARCH}" = x86_64 ]; then
    [ -e "${OLMES}" ] && mv "${OLMES}" "${OLMES}.broken-$(date +%s)"
    echo "  building ${OLMES} (uv pip install -e .[gpu])"
    (cd "${OLMES_REPO}" && UV_LINK_MODE=copy uv venv "${OLMES}" --python 3.11 \
        && UV_LINK_MODE=copy uv pip install --python "${OLMES}/bin/python" -e ".[gpu]") && ok "${OLMES}" \
        || bad "OLMES install failed"
else
    bad "${OLMES}: ${state} (x86_64: --install builds it; aarch64: by hand, README 'Setting up on a new cluster')"
fi

echo
if [ "${STATUS}" = 0 ]; then echo "All set: source scripts/env.sh"; else echo "Some items need attention (TODO above)."; fi
exit "${STATUS}"
