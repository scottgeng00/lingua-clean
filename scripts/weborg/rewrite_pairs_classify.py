"""WebOrganizer topic/format of each source / gacha / direct triple (pairs.jsonl) + Qwen3-Embedding cosine between
source and each rewrite. Local GPU. Output: pairs_labels.npz."""
import json
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from scripts.weborg.load import load
M = "/home/sgeng/scottg/pretrain/assets/models"
D = "/home/sgeng/scottg/pretrain/embeddings/rewrite_pairs"
rows = [json.loads(l) for l in open(f"{D}/pairs.jsonl")]
out = {"source_id": np.array([r["source_id"] for r in rows])}
tok, topic = load(f"{M}/TopicClassifier-NoURL")
_, fmt = load(f"{M}/FormatClassifier-NoURL")
etok = AutoTokenizer.from_pretrained(f"{M}/Qwen3-Embedding-0.6B")
emb = AutoModel.from_pretrained(f"{M}/Qwen3-Embedding-0.6B", torch_dtype=torch.bfloat16,
                                attn_implementation="flash_attention_2").cuda().eval()
for key in ("source", "gacha", "direct"):
    texts = [r[key] for r in rows]
    tp, fp, ev = [], [], []
    with torch.inference_mode():
        for i in range(0, len(texts), 64):
            b = tok(texts[i:i + 64], truncation=True, max_length=2048, padding=True, return_tensors="pt").to("cuda")
            tp.append(topic(**b).logits.float().softmax(-1).cpu()); fp.append(fmt(**b).logits.float().softmax(-1).cpu())
            e = etok(texts[i:i + 64], truncation=True, max_length=2048, padding=True, return_tensors="pt").to("cuda")
            h = emb(**e).last_hidden_state
            last = e["attention_mask"].sum(1) - 1
            ev.append(F.normalize(h[torch.arange(len(last)), last].float(), dim=-1).cpu())
    out[f"{key}_topic"], out[f"{key}_format"] = torch.cat(tp).numpy(), torch.cat(fp).numpy()
    out[f"{key}_emb"] = torch.cat(ev).numpy().astype(np.float16)
    print(key, "done", flush=True)
np.savez(f"{D}/pairs_labels.npz", **out)
