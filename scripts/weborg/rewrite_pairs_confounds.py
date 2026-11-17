"""Confound checks for rewrite_pairs: (1) reclassify rewrites with markdown stripped, (2) label stability ceiling:
source first half vs full source, (3) source-rewrite cosine vs the source's nearest other source in the sample."""
import json, re
import numpy as np
import torch
from scripts.weborg.load import load
M = "/home/sgeng/scottg/pretrain/assets/models"; D = "/home/sgeng/scottg/pretrain/embeddings/rewrite_pairs"
rows = [json.loads(l) for l in open(f"{D}/pairs.jsonl")]
z = np.load(f"{D}/pairs_labels.npz")
def strip_md(t):
    t = re.sub(r"(?m)^\s{0,3}#{1,6}\s*", "", t)          # headers
    t = re.sub(r"\*\*|__", "", t)                         # bold
    t = re.sub(r"(?m)^\s*[\*\-•]\s+", "", t)              # bullets
    t = re.sub(r"(?m)^\s*\d+(\.\d+)*[\.\)]\s+", "", t)    # numbered items / section numbers
    t = re.sub(r"(?m)^\s*```\w*\s*$", "", t)              # code fences
    t = re.sub(r"(?<!\*)\*(?!\s)([^*\n]+)\*", r"\1", t)   # italics
    return t
tok, topic = load(f"{M}/TopicClassifier-NoURL"); _, fmt = load(f"{M}/FormatClassifier-NoURL")
def classify(texts):
    tp, fp = [], []
    with torch.inference_mode():
        for i in range(0, len(texts), 64):
            b = tok(texts[i:i + 64], truncation=True, max_length=2048, padding=True, return_tensors="pt").to("cuda")
            tp.append(topic(**b).logits.argmax(-1).cpu()); fp.append(fmt(**b).logits.argmax(-1).cpu())
    return torch.cat(tp).numpy(), torch.cat(fp).numpy()
out = {}
for a in ("gacha", "direct"):
    out[f"{a}_strip_topic"], out[f"{a}_strip_format"] = classify([strip_md(r[a]) for r in rows])
out["src_half_topic"], out["src_half_format"] = classify([r["source"][:max(200, len(r["source"]) // 2)] for r in rows])
np.savez(f"{D}/pairs_confounds.npz", **out)
st, sf = z["source_topic"].argmax(1), z["source_format"].argmax(1)
print(f"ceiling (source first half vs full source): topic {100*(out['src_half_topic']==st).mean():.1f}%, format {100*(out['src_half_format']==sf).mean():.1f}%")
for a in ("gacha", "direct"):
    print(f"{a}: topic kept {100*(z[a+'_topic'].argmax(1)==st).mean():.1f}% -> {100*(out[a+'_strip_topic']==st).mean():.1f}% after stripping markdown; "
          f"format kept {100*(z[a+'_format'].argmax(1)==sf).mean():.1f}% -> {100*(out[a+'_strip_format']==sf).mean():.1f}%")
g, d = z["gacha_topic"].argmax(1), z["direct_topic"].argmax(1)
print(f"gacha vs direct agree: topic {100*(g==d).mean():.1f}%, format {100*(z['gacha_format'].argmax(1)==z['direct_format'].argmax(1)).mean():.1f}%")
e = {k: torch.from_numpy(z[k + "_emb"].astype(np.float32)).cuda() for k in ("source", "gacha", "direct")}
S = e["source"] @ e["source"].T; S.fill_diagonal_(-1); nn = S.max(1).values.cpu().numpy()
for a in ("gacha", "direct"):
    c = (e["source"] * e[a]).sum(1).cpu().numpy()
    R = e[a] @ e["source"].T; rank = (R > R.diag()[:, None]).sum(1).cpu().numpy()
    print(f"{a}: cos(source, rewrite) median {np.median(c):.3f} vs cos(source, nearest other source) median {np.median(nn):.3f}; "
          f"rewrite closer to its own source than to any other of {len(rows)-1:,} sources: {100*(rank==0).mean():.1f}%")
