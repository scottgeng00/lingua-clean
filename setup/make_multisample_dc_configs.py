"""Configs for the data-constrained multi-sample study (w1152): a fixed real pool of 0.25x Chinchilla
(body + head) tokens, repeated, with or without synthetic views of the same documents.

Pool: the first pool_tokens_v2(1152, 0.25) = 2,148,532,224 tokens (2,049 bs256 steps per pass) of the filtered,
deduplicated sources the synthetic arms were generated from (setup/build_pool_stream.py), re-shuffled every pass.
Recipe otherwise as setup/make_multisample_configs.py (LR 2.1e-3, bs256, WSD 1-sqrt over the last 10%, warmup 1
token per body + head param, matrix-only weight decay).

    real_p16[_lrX_wdY]       real pool x 16 (main run, 4x Chinchilla); decay branches real_p{4,6,8,12}[_lrX_wdY]
                             (suffix only for (LR, wd) other than (2.1e-3, 0.1), e.g. 1.05e-3 / 0.2: same LR x wd)
    r4s_<arm>[_3x]           real pool x 4 + synthetic views of the pool documents up to 2x [3x] Chinchilla
                             (MixedTokenStream r = 1/2 [2/3]); written once assets/data/ms_<arm>_pool025[_3c]_qwen3_tokens
                             exists (setup/export_multisample.py --pool-mult 0.25 --order sample-major)
    fresh_4x                 unique-data reference (dclm_qwen3_tokens_32m: unfiltered DCLM, same kind of data as
                             multisample_w1152/dclm_2x), main run to the same 4x horizon; decay branches fresh_{1,2,3}x
                             at the real curve's 4 / 8 / 12-pass step counts; written once that stream exists
    real_p16_kd<t>_a<alpha>  KD on the real-only curve: (1 - alpha) CE + alpha KL(teacher || student), T = 1, Qwen3
                             <t> in {4b, 8b, 14b}-Base teacher on every token (FP8 linears, LINGUA_TEACHER_FP8=1), alpha
                             in KD_ALPHAS; decay branches real_p{4,8,12}_kd<t>_a<alpha>
    r4s_<arm>_<gen>[_3x]     generator-size ablation: r4s_<arm>[_3x] (direct_nothink, mix) with views from Qwen3.5-2B /
                             27B (gen q2b / q27b) instead of 9B, from assets/data/ms_<arm>_<gen>_pool025_qwen3_tokens
    r4s_<arm>_3x_kd8b_a<alpha>  KD on top of r4s_<arm>_3x (mix, direct_nothink), teacher 8B, alpha in {0.5, 1}
    r4s_basegen[_3x]         r4s with Qwen3.5-9B-Base "Document 2" samples; one export (ms_basegen_pool025) for both
Launched by scripts/dataconstrained_orchestrate.py --group multisample_dc_w1152 (branches wait for their checkpoint).

    .venv-kd/bin/python setup/make_multisample_dc_configs.py --hp 2.1e-3:0.1 1.05e-3:0.2
"""
import argparse
import json
import os
import sys

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "setup"))
import make_dataconstrained_configs as dc  # noqa: E402
from scripts.launch_olmes_suite import DOWNSTREAM  # noqa: E402

GROUP = "multisample_dc_w1152"
OUT = f"{ROOT}/apps/main/configs/experiments/{GROUP}"
DATA = "/home/sgeng/scottg/pretrain/assets/data"
POOL_STREAM = f"{DATA}/dclm_f200w_qwen3_tokens"
FRESH_STREAM = f"{DATA}/dclm_qwen3_tokens_32m"  # dclm_8m_0..3, ~40B unique tokens
ARMS = ["direct_nothink", "direct_think", "mix", "ideas", "gacha"]
HELDOUTS = [f"{DATA}/ms_{a}_qwen3_tokens/ms_{a}_heldout" for a in ARMS]
WIDTH, LR, BS, POOL_MULT = 1152, 2.1e-3, 256, 0.25
MAIN_PASSES, BRANCH_PASSES = 16, (4, 6, 8, 12)
QOS = "h100_mdf_explore"  # DM1 (switched from scale 2026-11-18: explore had room)
MODELS = "/home/sgeng/scottg/pretrain/assets/models"
TEACHERS = {t: f"{MODELS}/Qwen3-{t.upper()}-Base" for t in ("4b", "8b", "14b")}  # Qwen3 tokenizer = the student's
KD_ALPHAS, KD_BRANCH_PASSES = (0.5, 0.9, 1.0), (4, 8, 12)
KD_SYN = {"teacher": "8b", "alphas": (0.5, 1.0), "arms": ("mix", "direct_nothink"), "mult": 3}


def finish(cfg, entry, name, lr=LR):
    """Group, LR and eval suite of the multi-sample study on top of data_constrainedv2's config()."""
    cfg["name"] = f"{GROUP}-{name}".replace("_", "-")
    cfg["dump_dir"] = "${oc.env:MIDTRAIN_ROOT}/experiments/" + GROUP + "/" + name
    cfg["optim"]["lr"] = lr
    cfg["logging"]["wandb"]["project"] = f"lingua-clean-{GROUP}"
    val = cfg["eval"]["validation"]
    val["token_streams"] = val["token_streams"] + [h for h in HELDOUTS if os.path.exists(f"{h}/meta.json")]
    # fp32 rolling (Paloma) logits are batch_tokens x 151,936 x 4 bytes: the default 32k OOMs an 80 GB H100.
    val["token_stream_rolling_batch_tokens"] = 8192
    cfg["eval"]["olmes"].update(tasks=DOWNSTREAM, model_type="vllm", timeout_hours=7.0)
    entry.update(name=name, lr=lr, qos=QOS)
    return name, cfg, entry


def kd(cfg, entry, teacher, alpha):
    """Online KD from a frozen FP8 Qwen3 teacher (apps/main/train.py), same loss as setup/make_kd_configs.py."""
    cfg["teacher_model_path"] = TEACHERS[teacher]
    cfg["data"].update(use_kl_distillation=True, kl_alpha=alpha, kl_temperature=1.0, kd_skip_teacher_nll=True)
    entry.update(teacher=teacher, alpha=alpha, env={"LINGUA_TEACHER_FP8": "1"})
    return f"_kd{teacher}_a{dc.fmt(alpha)}"


def real_runs(lr, wd, pool, per_pass, warmup_params, teacher=None, alpha=None):
    tag = "" if (lr, wd) == (LR, 0.1) else f"_lr{dc.lr_fmt(lr)}_wd{dc.fmt(wd)}"
    if teacher:
        tag += f"_kd{teacher}_a{dc.fmt(alpha)}"
    main = f"real_p{MAIN_PASSES}{tag}"
    main_dir = "${oc.env:MIDTRAIN_ROOT}/experiments/" + GROUP + "/" + main
    branches = {k: k * per_pass for k in (KD_BRANCH_PASSES if teacher else BRANCH_PASSES)}
    kw = dict(warmup_params=warmup_params, stream=POOL_STREAM, group=dc.GROUP_V2)
    cfg, entry = dc.config(WIDTH, main, MAIN_PASSES * per_pass, lr, BS, wd, 0, "repeat", pool,
                           extra_saves={dc.decay_start(e) for e in branches.values()}, **kw)
    entry.update(passes=MAIN_PASSES, main=main, pool_tokens=pool)
    if teacher:
        kd(cfg, entry, teacher, alpha)
    out = [finish(cfg, entry, main, lr)]
    for k, e in branches.items():
        name = f"real_p{k}{tag}"
        cfg, entry = dc.config(WIDTH, name, e, lr, BS, wd, 0, "repeat", pool,
                               branch_from=f"{main_dir}/checkpoints/{dc.decay_start(e):010d}", **kw)
        entry.update(passes=k, main=main, pool_tokens=pool)
        if teacher:
            kd(cfg, entry, teacher, alpha)
        out.append(finish(cfg, entry, name, lr))
    return out


def fresh_runs(per_pass, warmup_params):
    """Unique-data reference: same steps as the real curve (pool passes' worth of tokens), never repeating a doc."""
    meta = f"{FRESH_STREAM}/meta.json"
    if not os.path.exists(meta):
        return []
    n_tokens = json.load(open(meta))["tokens"] - 1
    main, main_dir = "fresh_4x", "${oc.env:MIDTRAIN_ROOT}/experiments/" + GROUP + "/fresh_4x"
    branches = {f"fresh_{k // 4}x": k * per_pass for k in (4, 8, 12)}
    assert n_tokens >= MAIN_PASSES * per_pass * BS * dc.SEQ
    kw = dict(warmup_params=warmup_params, stream=FRESH_STREAM, group=dc.GROUP_V2)
    cfg, entry = dc.config(WIDTH, main, MAIN_PASSES * per_pass, LR, BS, 0.1, 0, "unique", n_tokens,
                           extra_saves={dc.decay_start(e) for e in branches.values()}, **kw)
    entry.update(main=main)
    out = [finish(cfg, entry, main)]
    for name, e in branches.items():
        cfg, entry = dc.config(WIDTH, name, e, LR, BS, 0.1, 0, "unique", n_tokens,
                               branch_from=f"{main_dir}/checkpoints/{dc.decay_start(e):010d}", **kw)
        entry.update(main=main)
        out.append(finish(cfg, entry, name))
    return out


# Synthetic-augmented runs: total tokens = mult x Chinchilla = 4 * mult pool passes' worth, real fixed at 4 passes.
R4S = {2: ("", "pool025"), 3: ("_3x", "pool025_3c")}  # mult -> (name suffix, synthetic export tag)


def r4s_runs(pool, per_pass, warmup_params):
    out = []
    specs = [(mult, arm, None, None) for mult in R4S for arm in ARMS] + \
        [(KD_SYN["mult"], arm, KD_SYN["teacher"], a) for arm in KD_SYN["arms"] for a in KD_SYN["alphas"]]
    for mult, arm, teacher, alpha in specs:
        suffix, tag = R4S[mult]
        syn = f"{DATA}/ms_{arm}_{tag}_qwen3_tokens"
        if not os.path.exists(f"{syn}/meta.json"):
            continue
        passes = 4 * mult  # steps of 4 * mult pool passes; real sequences are exactly 4 passes of them
        ratio = (passes - 4) / passes  # synthetic share: 1/2 at 2x, 2/3 at 3x
        cfg, entry = dc.config(WIDTH, f"r4s_{arm}{suffix}", passes * per_pass, LR, BS, 0.1, 0, "repeat", pool,
                               warmup_params=warmup_params, stream=POOL_STREAM, group=dc.GROUP_V2)
        n_syn = json.load(open(f"{syn}/meta.json"))["tokens"] - 1
        need = (passes - 4) * pool
        assert n_syn >= need, f"{syn}: {n_syn:,} synthetic tokens < {need:,} needed"
        cfg["data"].update(token_stream_mix_dir=syn, token_stream_mix_ratio=ratio, token_stream_mix_tokens=n_syn)
        entry.update(arm=arm, r=ratio, mult=mult, pool_tokens=pool)
        kd_tag = kd(cfg, entry, teacher, alpha) if teacher else ""
        out.append(finish(cfg, entry, f"r4s_{arm}{suffix}{kd_tag}"))
    return out


# Generator-size ablation: the same R4+S recipe on synthetic views from other Qwen3.5 instruct models (thinking off,
# same prompts / sampling as the 9B arms). One export per (arm, generator) serves both horizons.
GEN_ARMS, GENERATORS = ("direct_nothink", "mix"), ("q2b", "q27b")  # q2b = Qwen3.5-2B, q27b = Qwen3.5-27B


def r4s_gen_runs(pool, per_pass, warmup_params):
    """r4s_<arm>_<gen>[_3x] from assets/data/ms_<arm>_<gen>_pool025_qwen3_tokens (written once its meta.json exists)."""
    out = []
    for gen in GENERATORS:
        for arm in GEN_ARMS:
            syn = f"{DATA}/ms_{arm}_{gen}_pool025_qwen3_tokens"
            if not os.path.exists(f"{syn}/meta.json"):
                continue
            n_syn = json.load(open(f"{syn}/meta.json"))["tokens"] - 1
            for mult, (suffix, _) in R4S.items():
                passes = 4 * mult
                ratio = (passes - 4) / passes
                need = (passes - 4) * pool
                assert n_syn >= need, f"{syn}: {n_syn:,} synthetic tokens < {need:,} needed"
                name = f"r4s_{arm}_{gen}{suffix}"
                cfg, entry = dc.config(WIDTH, name, passes * per_pass, LR, BS, 0.1, 0, "repeat", pool,
                                       warmup_params=warmup_params, stream=POOL_STREAM, group=dc.GROUP_V2)
                cfg["data"].update(token_stream_mix_dir=syn, token_stream_mix_ratio=ratio, token_stream_mix_tokens=n_syn)
                entry.update(arm=arm, generator=gen, r=ratio, mult=mult, pool_tokens=pool)
                out.append(finish(cfg, entry, name))
    return out


# Arms with one export (sample-major, >= 3x worth of views) serving both horizons: r4s_<arm>[_3x] from
# assets/data/ms_<arm>_pool025_qwen3_tokens. basegen = Qwen3.5-9B-Base writing "Document 2" after the source
# (simplescale.harnesses.multisample:base_continue); arv2 = Active Reading v2 (active_reading_v2, 12-48 planned
# alternative web documents per source); mix16 / mix16plan = 8 of a 16-prompt pool per source (prompt_pool; uniform /
# planner-weighted draw; source-reference and code-execution filtered).
SINGLE_EXPORT_ARMS = ("basegen", "arv2", "mix16", "mix16plan")


def r4s_single_export_runs(pool, per_pass, warmup_params):
    out = []
    for arm in SINGLE_EXPORT_ARMS:
        syn = f"{DATA}/ms_{arm}_pool025_qwen3_tokens"
        if not os.path.exists(f"{syn}/meta.json"):
            continue
        n_syn = json.load(open(f"{syn}/meta.json"))["tokens"] - 1
        for mult, (suffix, _) in R4S.items():
            passes = 4 * mult
            ratio = (passes - 4) / passes
            need = (passes - 4) * pool
            assert n_syn >= need, f"{syn}: {n_syn:,} synthetic tokens < {need:,} needed"
            name = f"r4s_{arm}{suffix}"
            cfg, entry = dc.config(WIDTH, name, passes * per_pass, LR, BS, 0.1, 0, "repeat", pool,
                                   warmup_params=warmup_params, stream=POOL_STREAM, group=dc.GROUP_V2)
            cfg["data"].update(token_stream_mix_dir=syn, token_stream_mix_ratio=ratio, token_stream_mix_tokens=n_syn)
            entry.update(arm=arm, r=ratio, mult=mult, pool_tokens=pool)
            out.append(finish(cfg, entry, name))
    return out


# Anneal with real: the WSD decay phase (last 10%) on the real pool only, plain NTP (KD runs drop the teacher).
# 3x R4+S runs saved no decay-start checkpoint, so each gets a trunk: the same run (same seeds and schedule) stopped
# at the decay start (stop_step; manifest done_step = that step, no eval), with two branches off it: <base>_realdecay
# and <base>_redecay (the original decay again, a same-trunk control); KD trunks also get <base>_realdecay_kd
# (real-only decay, teacher kept). KD-only runs branch off the real_p16_kd* main
# runs' existing 12-pass decay-start checkpoints; their controls are the existing real_p12_kd* runs.
REALDECAY_R4S = {"r4s_mix_3x": ("ms_mix_pool025_3c", None, None), "r4s_mix_3x_kd8b_a1": ("ms_mix_pool025_3c", "8b", 1.0),
                 "r4s_arv2_3x": ("ms_arv2_pool025", None, None),
                 "r4s_direct_nothink_3x": ("ms_direct_nothink_pool025_3c", None, None),
                 "r4s_direct_nothink_3x_kd8b_a1": ("ms_direct_nothink_pool025_3c", "8b", 1.0)}
REALDECAY_KD = {"real_p12_kd8b_a1": ("8b", 1.0), "real_p12_kd4b_a0p9": ("4b", 0.9)}


def realdecay_runs(pool, per_pass, warmup_params):
    out, steps = [], 12 * per_pass
    start = dc.decay_start(steps)
    root = "${oc.env:MIDTRAIN_ROOT}/experiments/" + GROUP
    kw = dict(warmup_params=warmup_params, stream=POOL_STREAM, group=dc.GROUP_V2)

    def r4s_cfg(name, syn, teacher, alpha, **extra):
        cfg, entry = dc.config(WIDTH, name, steps, LR, BS, 0.1, 0, "repeat", pool, **kw, **extra)
        syn = f"{DATA}/{syn}_qwen3_tokens"
        n_syn = json.load(open(f"{syn}/meta.json"))["tokens"] - 1
        cfg["data"].update(token_stream_mix_dir=syn, token_stream_mix_ratio=2 / 3, token_stream_mix_tokens=n_syn)
        entry.update(r=2 / 3, mult=3, pool_tokens=pool)
        if teacher:
            kd(cfg, entry, teacher, alpha)
        return cfg, entry

    for base, (syn, teacher, alpha) in REALDECAY_R4S.items():
        trunk = f"{base}_trunk"
        cfg, entry = r4s_cfg(trunk, syn, teacher, alpha, extra_saves={start})
        cfg["stop_step"] = start
        entry.update(done_step=start, no_eval=True)  # steps stays the schedule length (the launch passes it on)
        out.append(finish(cfg, entry, trunk))
        src = f"{root}/{trunk}/checkpoints/{start:010d}"
        cfg, entry = r4s_cfg(f"{base}_redecay", syn, teacher, alpha, branch_from=src)
        out.append(finish(cfg, entry, f"{base}_redecay"))
        cfg, entry = dc.config(WIDTH, f"{base}_realdecay", steps, LR, BS, 0.1, 0, "repeat", pool, branch_from=src, **kw)
        entry.update(realdecay_of=base, pool_tokens=pool)
        out.append(finish(cfg, entry, f"{base}_realdecay"))
        if teacher:  # <base>_realdecay_kd: real-only decay, teacher kept (same KD loss as the trunk)
            cfg, entry = dc.config(WIDTH, f"{base}_realdecay_kd", steps, LR, BS, 0.1, 0, "repeat", pool,
                                   branch_from=src, **kw)
            entry.update(realdecay_of=base, pool_tokens=pool)
            kd(cfg, entry, teacher, alpha)
            out.append(finish(cfg, entry, f"{base}_realdecay_kd"))
    for base, (teacher, alpha) in REALDECAY_KD.items():
        main = base.replace("real_p12", f"real_p{MAIN_PASSES}")
        cfg, entry = dc.config(WIDTH, f"{base}_realdecay", steps, LR, BS, 0.1, 0, "repeat", pool,
                               branch_from=f"{root}/{main}/checkpoints/{start:010d}", **kw)
        entry.update(realdecay_of=base, main=main, pool_tokens=pool)
        out.append(finish(cfg, entry, f"{base}_realdecay"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hp", nargs="+", default=["2.1e-3:0.1"], help="LR:wd pairs for the real-only curve")
    a = ap.parse_args()
    pool = dc.pool_tokens_v2(WIDTH, POOL_MULT)
    per_pass = pool // (BS * dc.SEQ)
    meta = json.load(open(f"{POOL_STREAM}/meta.json"))
    assert meta["tokens"] > pool, (meta["tokens"], pool)
    s = dc.shape(WIDTH)
    warmup_params = s["body"] + s["embedding"]
    hps = [tuple(map(float, h.split(":"))) for h in a.hp]
    runs = [r for lr, wd in hps for r in real_runs(lr, wd, pool, per_pass, warmup_params)] + \
        [r for t in TEACHERS for al in KD_ALPHAS for r in real_runs(LR, 0.1, pool, per_pass, warmup_params, t, al)] + \
        r4s_runs(pool, per_pass, warmup_params) + fresh_runs(per_pass, warmup_params) + \
        r4s_gen_runs(pool, per_pass, warmup_params) + r4s_single_export_runs(pool, per_pass, warmup_params) + \
        realdecay_runs(pool, per_pass, warmup_params)
    os.makedirs(OUT, exist_ok=True)
    mpath = f"{OUT}/manifest.json"
    manifest = json.load(open(mpath)) if os.path.exists(mpath) else {}
    for name, cfg, entry in runs:
        with open(f"{OUT}/{name}.yaml", "w") as f:
            f.write(f"# Generated by setup/make_multisample_dc_configs.py ({entry['steps']} steps, {entry['gpus']} GPUs, "
                    f"pool {pool:,} tokens = {per_pass} steps per pass).\n")
            yaml.safe_dump(cfg, f, sort_keys=False)
        manifest[name] = entry
        print(name, entry["steps"], entry.get("branch_from", ""))
    with open(mpath, "w") as f:
        json.dump(manifest, f, indent=1)


if __name__ == "__main__":
    main()
