"""Load WebOrganizer gte-base classifiers (custom 'new' architecture, written for transformers 4.x) under our
transformers: build from config so non-persistent buffers (position_ids, rotary caches) are real, load weights from
safetensors, and restore the removed PreTrainedModel.get_extended_attention_mask (standard BERT additive mask)."""
import torch
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer


def _ext_mask(self, attention_mask, input_shape=None, device=None, dtype=None):
    dtype = dtype or next(self.parameters()).dtype
    m = attention_mask[:, None, None, :].to(dtype)
    return (1.0 - m) * torch.finfo(dtype).min


def load(path, device="cuda", dtype=torch.bfloat16, memory_efficient=False):
    cfg = AutoConfig.from_pretrained(path, trust_remote_code=True)
    cfg.use_memory_efficient_attention = memory_efficient
    cfg.unpad_inputs = memory_efficient
    model = AutoModelForSequenceClassification.from_config(cfg, trust_remote_code=True)
    res = model.load_state_dict(load_file(f"{path}/model.safetensors"), strict=False)
    assert not res.missing_keys and not res.unexpected_keys, res
    type(model.new).get_extended_attention_mask = _ext_mask
    return AutoTokenizer.from_pretrained(path), model.to(device=device, dtype=dtype).eval()
