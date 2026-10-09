"""Download an HF causal-LM checkpoint and convert it to a Lingua DCP checkpoint.

Output layout (rooted at ``--output``):

    output/
      .metadata                ← DCP shard metadata (consumed by load_from_checkpoint)
      __0_0.distcp             ← DCP shard
      consolidated/
        consolidated.pth       ← Lingua-format consolidated state_dict
        params.json            ← model args readable by `dataclass_from_dict(LMTransformerArgs, ...)`
      tokenizer/               ← AutoTokenizer.save_pretrained() target

Supported model families:
  - OLMo-2 (any `OLMo-2*` or `*olmo2*` repo id)
  - meta-llama/* (downloads `original/consolidated.00.pth`; assumes a single-shard original)
  - TinyLlama/*
  - Qwen2/Qwen2.5 and Qwen3 dense models (detected from config.architectures;
    --model may also be a local HF directory)

Sample usage:
    python setup/hf_to_lingua_dcp.py \\
        --model allenai/OLMo-2-0425-1B \\
        --revision stage1-step928646-tokens3897B \\
        --output ${STUDENT_INIT_PATH}

After this, point a recipe's `checkpoint.init_ckpt_path` at ${STUDENT_INIT_PATH}
(the directory that contains `.metadata` and `*.distcp`). The HF mirror at
${STUDENT_HF_PATH} (used for the tokenizer and HF-side analysis) is downloaded
separately by `scripts/fetch_models.sh`.
"""
import argparse
import json
import os
import re

import torch
from huggingface_hub import hf_hub_download
from torch.distributed.checkpoint.format_utils import torch_save_to_dcp
from transformers import AutoModelForCausalLM, AutoTokenizer


def permute(w, n_heads, dim1, dim2):
    """Lingua qk weight layout → HF (interleaved → split-half)."""
    return w.view(n_heads, dim1 // n_heads // 2, 2, dim2).transpose(1, 2).reshape(dim1, dim2)


def inverse_permute(w, n_heads, dim1, dim2):
    """HF qk weight layout → Lingua (split-half → interleaved)."""
    return w.view(n_heads, 2, dim1 // n_heads // 2, dim2).transpose(1, 2).reshape(dim1, dim2)


def inverse_permute_norm(w, n_heads):
    """1D QK-norm: HF split-half → Lingua interleaved."""
    head_dim = w.shape[0] // n_heads
    return w.view(n_heads, 2, head_dim // 2).transpose(1, 2).reshape(-1)


def _hf_architecture(name_or_path, revision=None):
    from transformers import AutoConfig
    kwargs = {"revision": revision} if revision is not None else {}
    archs = getattr(AutoConfig.from_pretrained(name_or_path, **kwargs), "architectures", None) or [None]
    return archs[0]


_DTYPE_MAP = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}


def _resolve_dtype(name: str) -> torch.dtype:
    if name not in _DTYPE_MAP:
        raise ValueError(f"Unsupported dtype {name!r}; choose from {list(_DTYPE_MAP)}")
    return _DTYPE_MAP[name]


def convert_olmo2(model, output_dir, tokenizer_path):
    state_dict = model.state_dict()
    config = model.config
    hidden_dim = config.hidden_size
    n_heads = config.num_attention_heads
    n_kv_heads = config.num_key_value_heads

    rename_map = {
        "model.embed_tokens.weight": "tok_embeddings.weight",
        "model.layers.{}.self_attn.q_proj.weight": "layers.{}.attention.wq.weight",
        "model.layers.{}.self_attn.k_proj.weight": "layers.{}.attention.wk.weight",
        "model.layers.{}.self_attn.v_proj.weight": "layers.{}.attention.wv.weight",
        "model.layers.{}.self_attn.o_proj.weight": "layers.{}.attention.wo.weight",
        "model.layers.{}.self_attn.q_norm.weight": "layers.{}.attention.q_norm.weight",
        "model.layers.{}.self_attn.k_norm.weight": "layers.{}.attention.k_norm.weight",
        "model.layers.{}.mlp.gate_proj.weight": "layers.{}.feed_forward.w1.weight",
        "model.layers.{}.mlp.up_proj.weight": "layers.{}.feed_forward.w3.weight",
        "model.layers.{}.mlp.down_proj.weight": "layers.{}.feed_forward.w2.weight",
        "model.layers.{}.post_attention_layernorm.weight": "layers.{}.attention_norm.weight",
        "model.layers.{}.post_feedforward_layernorm.weight": "layers.{}.ffn_norm.weight",
        "model.norm.weight": "norm.weight",
        "lm_head.weight": "output.weight",
    }

    out = {}
    for key, value in state_dict.items():
        if "layers" in key:
            abstract_key = re.sub(r"(\d+)", "{}", key)
            layer_num = re.search(r"\d+", key).group(0)
            new_key = rename_map.get(abstract_key)
            if new_key is None:
                print(f"  Skipping unmapped key: {key}")
                continue
            new_key = new_key.format(layer_num)
        else:
            new_key = rename_map.get(key)
            if new_key is None:
                print(f"  Skipping unmapped key: {key}")
                continue

        if "wq" in new_key:
            value = inverse_permute(value, n_heads=n_heads, dim1=hidden_dim, dim2=hidden_dim)
        elif "wk" in new_key:
            kv_dim = hidden_dim // n_heads * n_kv_heads
            value = inverse_permute(value, n_heads=n_kv_heads, dim1=kv_dim, dim2=hidden_dim)
        elif "q_norm" in new_key:
            value = inverse_permute_norm(value, n_heads=n_heads)
        elif "k_norm" in new_key:
            value = inverse_permute_norm(value, n_heads=n_kv_heads)

        out[new_key] = value

    params = {
        "model": {
            "dim": config.hidden_size,
            "n_layers": config.num_hidden_layers,
            "n_heads": config.num_attention_heads,
            "n_kv_heads": config.num_key_value_heads,
            "vocab_size": config.vocab_size,
            "ffn_dim_multiplier": 1.5,
            "multiple_of": 256,
            "norm_eps": config.rms_norm_eps,
            "rope_theta": config.rope_theta,
            "max_seqlen": config.max_position_embeddings,
            "hidden_dim": config.intermediate_size,
            "qk_norm": True,
            "post_norm": True,
        },
        "data": {"tokenizer": {"name": "hf", "path": os.path.join(output_dir, "tokenizer")}},
        "distributed": {"model_dtype": "bf16"},
    }
    params_path = os.path.join(output_dir, "consolidated", "params.json")
    with open(params_path, "w") as f:
        json.dump(params, f, indent=4)
    print(f"  Wrote params.json to {params_path}")
    return out


def convert_tinyllama(model, output_dir, tokenizer_path):
    state_dict = model.state_dict()
    config = model.config
    hidden_dim = config.hidden_size
    n_heads = config.num_attention_heads
    n_kv_heads = config.num_key_value_heads

    rename_map = {
        "model.embed_tokens.weight": "tok_embeddings.weight",
        "model.layers.{}.self_attn.q_proj.weight": "layers.{}.attention.wq.weight",
        "model.layers.{}.self_attn.k_proj.weight": "layers.{}.attention.wk.weight",
        "model.layers.{}.self_attn.v_proj.weight": "layers.{}.attention.wv.weight",
        "model.layers.{}.self_attn.o_proj.weight": "layers.{}.attention.wo.weight",
        "model.layers.{}.self_attn.rotary_emb.inv_freq": None,
        "model.layers.{}.mlp.gate_proj.weight": "layers.{}.feed_forward.w1.weight",
        "model.layers.{}.mlp.up_proj.weight": "layers.{}.feed_forward.w3.weight",
        "model.layers.{}.mlp.down_proj.weight": "layers.{}.feed_forward.w2.weight",
        "model.layers.{}.input_layernorm.weight": "layers.{}.attention_norm.weight",
        "model.layers.{}.post_attention_layernorm.weight": "layers.{}.ffn_norm.weight",
        "model.norm.weight": "norm.weight",
        "lm_head.weight": "output.weight",
    }

    out = {}
    for key, value in state_dict.items():
        if "layers" in key:
            abstract_key = re.sub(r"(\d+)", "{}", key)
            layer_num = re.search(r"\d+", key).group(0)
            new_key = rename_map.get(abstract_key)
            if new_key is None:
                continue
            new_key = new_key.format(layer_num)
        else:
            new_key = rename_map.get(key)
            if new_key is None:
                continue

        if "wq" in new_key:
            value = inverse_permute(value, n_heads=n_heads, dim1=hidden_dim, dim2=hidden_dim)
        elif "wk" in new_key:
            kv_dim = hidden_dim // n_heads * n_kv_heads
            value = inverse_permute(value, n_heads=n_kv_heads, dim1=kv_dim, dim2=hidden_dim)

        out[new_key] = value

    params = {
        "model": {
            "dim": config.hidden_size,
            "n_layers": config.num_hidden_layers,
            "n_heads": config.num_attention_heads,
            "n_kv_heads": config.num_key_value_heads,
            "vocab_size": config.vocab_size,
            "norm_eps": config.rms_norm_eps,
            "rope_theta": config.rope_theta,
            "max_seqlen": config.max_position_embeddings,
            "hidden_dim": config.intermediate_size,
        },
        "data": {"tokenizer": {"name": "hf", "path": os.path.join(output_dir, "tokenizer")}},
        "distributed": {"model_dtype": "bf16"},
    }
    with open(os.path.join(output_dir, "consolidated", "params.json"), "w") as f:
        json.dump(params, f, indent=4)
    return out


def convert_qwen(model, output_dir, tokenizer_path):
    """Qwen2 (q/k/v biases) and Qwen3 (per-head QK-norm) dense models -> Lingua."""
    config = model.config
    arch = config.architectures[0]
    assert arch in ("Qwen2ForCausalLM", "Qwen3ForCausalLM"), arch
    is_qwen3 = arch == "Qwen3ForCausalLM"
    dim = config.hidden_size
    n_heads = config.num_attention_heads
    n_kv_heads = config.num_key_value_heads
    head_dim = getattr(config, "head_dim", None) or dim // n_heads
    tied = bool(config.tie_word_embeddings)

    rename_map = {
        "model.embed_tokens.weight": "tok_embeddings.weight",
        "model.layers.{}.self_attn.q_proj.weight": "layers.{}.attention.wq.weight",
        "model.layers.{}.self_attn.k_proj.weight": "layers.{}.attention.wk.weight",
        "model.layers.{}.self_attn.v_proj.weight": "layers.{}.attention.wv.weight",
        "model.layers.{}.self_attn.q_proj.bias": "layers.{}.attention.wq.bias",
        "model.layers.{}.self_attn.k_proj.bias": "layers.{}.attention.wk.bias",
        "model.layers.{}.self_attn.v_proj.bias": "layers.{}.attention.wv.bias",
        "model.layers.{}.self_attn.o_proj.weight": "layers.{}.attention.wo.weight",
        "model.layers.{}.self_attn.q_norm.weight": "layers.{}.attention.q_norm.weight",
        "model.layers.{}.self_attn.k_norm.weight": "layers.{}.attention.k_norm.weight",
        "model.layers.{}.mlp.gate_proj.weight": "layers.{}.feed_forward.w1.weight",
        "model.layers.{}.mlp.up_proj.weight": "layers.{}.feed_forward.w3.weight",
        "model.layers.{}.mlp.down_proj.weight": "layers.{}.feed_forward.w2.weight",
        "model.layers.{}.input_layernorm.weight": "layers.{}.attention_norm.weight",
        "model.layers.{}.post_attention_layernorm.weight": "layers.{}.ffn_norm.weight",
        "model.norm.weight": "norm.weight",
        "lm_head.weight": "output.weight",
    }

    out = {}
    for key, value in model.state_dict().items():
        abstract_key = re.sub(r"\.(\d+)\.", ".{}.", key)
        new_key = rename_map.get(abstract_key)
        if new_key is None:
            raise ValueError(f"Unmapped Qwen key: {key}")
        if "{}" in new_key:
            new_key = new_key.format(re.search(r"\.(\d+)\.", key).group(1))

        # RoPE: HF split-half -> Lingua interleaved, for q/k rows (weights and biases)
        if new_key.endswith("wq.weight"):
            value = inverse_permute(value, n_heads=n_heads, dim1=n_heads * head_dim, dim2=dim)
        elif new_key.endswith("wk.weight"):
            value = inverse_permute(value, n_heads=n_kv_heads, dim1=n_kv_heads * head_dim, dim2=dim)
        elif new_key.endswith("wq.bias"):
            value = inverse_permute_norm(value, n_heads=n_heads)
        elif new_key.endswith("wk.bias"):
            value = inverse_permute_norm(value, n_heads=n_kv_heads)
        elif new_key.endswith(("q_norm.weight", "k_norm.weight")):
            value = inverse_permute_norm(value, n_heads=1)  # per-head norm of size head_dim
        out[new_key] = value.clone()
    if tied:
        # LMTransformer's tied head reaches the embedding as output.tied_module.weight too.
        out["output.tied_module.weight"] = out["tok_embeddings.weight"]

    # FFN width: lingua computes round_up(ffn_dim_multiplier * int(8*dim/3), multiple_of);
    # with multiple_of = intermediate_size this reproduces it exactly.
    base = int(2 * 4 * dim / 3)
    params = {
        "model": {
            "dim": dim,
            "n_layers": config.num_hidden_layers,
            "n_heads": n_heads,
            "n_kv_heads": n_kv_heads,
            "head_dim": head_dim,
            "vocab_size": config.vocab_size,
            "ffn_dim_multiplier": config.intermediate_size / base,
            "multiple_of": config.intermediate_size,
            "norm_eps": config.rms_norm_eps,
            "rope_theta": config.rope_theta,
            "max_seqlen": config.max_position_embeddings,
            "weight_tying": tied,
            "qk_norm": is_qwen3,
            "qk_norm_per_head": is_qwen3,
            "post_norm": False,
            "qkv_bias": not is_qwen3,
            "hf_arch": arch,
        },
        "data": {"tokenizer": {"name": "hf", "path": tokenizer_path}},
        "distributed": {"model_dtype": "bf16"},
    }
    params_path = os.path.join(output_dir, "consolidated", "params.json")
    with open(params_path, "w") as f:
        json.dump(params, f, indent=4)
    print(f"  Wrote params.json to {params_path}")
    return out


def download_and_convert(hf_model_name, output_dir, dtype="float32", revision=None):
    consolidated_dir = os.path.join(output_dir, "consolidated")
    consolidated_path = os.path.join(consolidated_dir, "consolidated.pth")
    os.makedirs(consolidated_dir, exist_ok=True)

    tokenizer_kwargs = {"revision": revision} if revision is not None else {}
    tokenizer = AutoTokenizer.from_pretrained(hf_model_name, **tokenizer_kwargs)
    tokenizer.save_pretrained(os.path.join(output_dir, "tokenizer"))

    torch_dtype = _resolve_dtype(dtype)
    model_kwargs = {"revision": revision} if revision is not None else {}

    if hf_model_name.startswith("meta-llama"):
        # The official Llama releases ship a single-shard `original/consolidated.00.pth`
        # in the Lingua-compatible layout already; just download it.
        hf_hub_download(
            repo_id=hf_model_name,
            filename="original/consolidated.00.pth",
            local_dir=output_dir,
            **model_kwargs,
        )
        os.rename(os.path.join(output_dir, "original/consolidated.00.pth"), consolidated_path)
        os.rmdir(os.path.join(output_dir, "original"))
    elif "OLMo-2" in hf_model_name or "olmo2" in hf_model_name.lower():
        print(f"Loading OLMo-2 model from HuggingFace (revision={revision})...")
        model = AutoModelForCausalLM.from_pretrained(
            hf_model_name, torch_dtype=torch_dtype, trust_remote_code=True, **model_kwargs
        )
        final_state_dict = convert_olmo2(model, output_dir, tokenizer_path=os.path.join(output_dir, "tokenizer"))
        torch.save(final_state_dict, consolidated_path)
    elif _hf_architecture(hf_model_name, revision) in ("Qwen2ForCausalLM", "Qwen3ForCausalLM"):
        print(f"Loading Qwen model from {hf_model_name} (revision={revision})...")
        model = AutoModelForCausalLM.from_pretrained(hf_model_name, torch_dtype=torch_dtype, **model_kwargs)
        final_state_dict = convert_qwen(model, output_dir, tokenizer_path=os.path.join(output_dir, "tokenizer"))
        torch.save(final_state_dict, consolidated_path)
    elif hf_model_name.startswith("TinyLlama"):
        print(f"Loading TinyLlama model from HuggingFace (revision={revision})...")
        model = AutoModelForCausalLM.from_pretrained(
            hf_model_name, torch_dtype=torch_dtype, trust_remote_code=True, **model_kwargs
        )
        final_state_dict = convert_tinyllama(model, output_dir, tokenizer_path=os.path.join(output_dir, "tokenizer"))
        torch.save(final_state_dict, consolidated_path)
    else:
        raise ValueError(
            f"Unsupported model family: {hf_model_name!r}. "
            "Add a converter branch for this family."
        )

    if os.path.exists(os.path.join(output_dir, ".metadata")):
        print(f"DCP format already exists at {output_dir}, skipping conversion.")
        return

    print("Converting consolidated.pth → DCP (may be slow on networked FS)...")
    torch_save_to_dcp(consolidated_path, output_dir)
    print(f"Done. Lingua DCP checkpoint at: {output_dir}")


def main():
    ap = argparse.ArgumentParser(description="HuggingFace causal-LM → Lingua DCP checkpoint.")
    ap.add_argument("--model", required=True, help="HF repo id (e.g. allenai/OLMo-2-0425-1B).")
    ap.add_argument(
        "--output",
        required=True,
        help="Output directory. After conversion contains .metadata + __0_0.distcp "
             "(consumed by load_from_checkpoint).",
    )
    ap.add_argument(
        "--dtype",
        default="float32",
        choices=list(_DTYPE_MAP.keys()),
        help="Load dtype for the HF model before conversion. Use float32 unless RAM-bound.",
    )
    ap.add_argument(
        "--revision",
        default=None,
        help="HF branch / tag / commit SHA (e.g. 'stage1-step928646-tokens3897B').",
    )
    args = ap.parse_args()
    download_and_convert(args.model, args.output, dtype=args.dtype, revision=args.revision)


if __name__ == "__main__":
    main()
