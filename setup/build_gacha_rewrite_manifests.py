"""Direct vs Gacha rewrite manifests over one of our DCLM subsets, replicating Jacqueline's rewrite_big2 setup
(gacha_pilot/rewrite_big2/make_manifests.py) with different seed docs.

Scans chunk files --chunks (default all 16) of each subset in --subsets (default dclm_8m_0 and dclm_8m_1), file by
file in that order, keeps docs with 200-1,200 words (stripped text), skips docs whose stripped text hashes to an
already-kept source, and stops at --sources (default: all). Each source
gets two rewrites per arm with her exact sampling settings (copied from her manifest); Gacha rows carry a
per-source dice seed (--seed-base + n; hers were 10M+ and 60M+) and thinking-off planning. Rows point at their
source line (_ref); the simplescale handlers gacha:rewrite_doc / gacha:rewrite_doc_direct cap the text at
6,000 chars and build her prompt. source_file / source_line locate each source for alignment.

    .venv-kd/bin/python setup/build_gacha_rewrite_manifests.py --out /home/sgeng/scottg/pretrain/simplescale_runs/gacha_dclm8m_01/manifest
"""

import argparse
import hashlib
import json
import os

DATA = "/home/sgeng/scottg/pretrain/assets/data/dclm_subsets"
HERS = "/checkpoint/comem-sage/jacquelinehe/pretrain_data/gacha_pilot/rewrite_big2/gacha_nothink.jsonl"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subsets", nargs="+", default=["dclm_8m_0", "dclm_8m_1"])
    ap.add_argument("--chunks", nargs="+", default=[f"{c:02d}" for c in range(16)])
    ap.add_argument("--sources", type=int, default=None)
    ap.add_argument("--seed-base", type=int, default=70_000_000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    first = json.loads(open(HERS).readline())
    sampling, planning = first["sampling_params"], first["planning_sampling_params"]
    assert planning["chat_template_kwargs"] == {"enable_thinking": False}
    files = [f"{DATA}/{s}/{s}.chunk.{c}.jsonl" for s in a.subsets for c in a.chunks]
    os.makedirs(a.out, exist_ok=True)
    key = lambda t: hashlib.blake2b(t.encode(), digest_size=8).digest()
    seen, n, short_long, dupes, rows = set(), 0, 0, 0, 0
    with open(f"{a.out}/direct.jsonl", "w") as direct, open(f"{a.out}/gacha.jsonl", "w") as gacha:
        for f in files:
            offset = 0
            for line, raw in enumerate(open(f, "rb")):
                if a.sources is not None and n == a.sources:
                    break
                start, offset, rows = offset, offset + len(raw), rows + 1
                text = json.loads(raw)["text"].strip()
                if not 200 <= len(text.split()) <= 1200:
                    short_long += 1
                    continue
                k = key(text)
                if k in seen:
                    dupes += 1
                    continue
                seen.add(k)
                base = {"source_id": f"d{n}", "source_file": os.path.basename(f), "source_line": line,
                        "_ref": {"path": f, "offset": start, "length": len(raw)},
                        "generations": 2, "sampling_params": sampling}
                direct.write(json.dumps(base | {"id": f"direct-d{n}"}) + "\n")
                gacha.write(json.dumps(base | {"id": f"gacha-d{n}", "seed": a.seed_base + n,
                                               "planning_sampling_params": planning}) + "\n")
                n += 1
    stats = {"files": files, "sources": n, "rows_scanned": rows, "skipped_length": short_long,
             "skipped_duplicates": dupes, "seed_base": a.seed_base, "sampling_params": sampling,
             "planning_sampling_params": planning, "copied_from": HERS}
    json.dump(stats, open(f"{a.out}/manifest_stats.json", "w"), indent=2)
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
