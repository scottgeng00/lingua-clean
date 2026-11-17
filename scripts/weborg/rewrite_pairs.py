"""Smoke test of how rewriting changes a document: sample rewrites from the first lines of rw_gacha / rw_direct chunk
00 (shuffled), keep sources rewritten by both arms (rewrite_big2: gacha-tN / direct-tN share source N), fetch the
source DCLM-baseline doc via the manifest (line N -> path, byte offset, length). Output: pairs.jsonl with
source / gacha / direct text per source id.
"""
import json, re, sys
G = "/checkpoint/comem-sage/jacquelinehe/pretrain_data/gacha_merged"
P = "/checkpoint/comem-sage/jacquelinehe/pretrain_data/gacha_pilot/rewrite_big2"
OUT = "/home/sgeng/scottg/pretrain/embeddings/rewrite_pairs/pairs.jsonl"
N_LINES = int(sys.argv[1]) if len(sys.argv) > 1 else 400_000
pat = re.compile(r"rewrite_big2/runs/(?:gacha_nothink|direct)/(?:gacha|direct)-t(\d+)/(\d)")
arms = {}
for arm in ("rw_gacha", "rw_direct"):
    d = {}
    with open(f"{G}/{arm}/{arm}.chunk.00.jsonl") as f:
        for i, line in enumerate(f):
            if i >= N_LINES:
                break
            r = json.loads(line)
            m = pat.fullmatch(r["id"])
            if m and m.group(1) not in d:
                d[int(m.group(1))] = r["text"]
    arms[arm] = d
    print(arm, len(d), "rewrite_big2 sources", flush=True)
both = sorted(set(arms["rw_gacha"]) & set(arms["rw_direct"]))
print("sources with both rewrites:", len(both), flush=True)
want, refs = set(both), {}
with open(f"{P}/direct.jsonl") as f:
    for i, line in enumerate(f):
        if i in want:
            refs[i] = json.loads(line)["_ref"]
        if i > both[-1]:
            break
n = 0
with open(OUT, "w") as out:
    fh = {}
    for s in both:
        ref = refs[s]
        h = fh.setdefault(ref["path"], open(ref["path"], "rb"))
        h.seek(ref["offset"])
        src = json.loads(h.read(ref["length"]))["text"]
        out.write(json.dumps({"source_id": s, "source": src, "gacha": arms["rw_gacha"][s], "direct": arms["rw_direct"][s]}) + "\n")
        n += 1
print("wrote", n, "triples to", OUT)
