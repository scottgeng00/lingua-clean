"""Batch-size x LR sweep, 20B tokens, on DCLM and on FineWeb-Edu -> apps/main/configs/experiments/lrsweep/.

Model: Qwen3-0.6B shape with the Qwen3.5 tokenizer (~694M params incl. the tied 248k-vocab embedding),
context 4096, doc masking. AdamW (beta1 0.9, beta2 0.95, eps 1e-8), grad clip 1.0 (global norm),
weight decay 0.1 on 2D matrices only (attention Q/K/V/O, MLP gate/up/down); none on RMSNorm / QK-norm
gains or the tied embedding / LM head (optim.decay_matrices_only).

Schedule (optim.scheduler = wsd_sqrt), defined in tokens and converted per batch size here:
linear warmup 0 -> peak over the first 200M tokens (1%), constant, then 1 - sqrt over the last 20%
(from 16B tokens) down to exactly 0 at 20B.

Grid (tokens/step, seqs/step, steps, LRs):
    262,144    64   76,294   7.07e-4 1.00e-3 1.41e-3 2.00e-3 2.83e-3
    524,288   128   38,147   1.00e-3 1.41e-3 2.00e-3 2.83e-3 4.00e-3
  1,048,576   256   19,073   1.41e-3 2.00e-3 2.83e-3 4.00e-3 5.66e-3
plus 2 extra seeds (model init and data order both changed) at 524,288 / 2.00e-3, for each dataset.

Data (Qwen3.5 tokens, counted exactly incl. one EOS per doc; data/token_counts_qwen35.json):
  dclm:   dclm_8m_0 + dclm_8m_1, 1:1 per document.
  fwedu:  fwedu_8m_0 + fwedu_8m_1 + the first FWEDU2_DOCS docs of fwedu_8m_2 (they hold < 20B tokens
          on their own), sampled per document in proportion to each source's doc count.
Each run is 32 GPUs (8 nodes): data.batch_size = seqs/step / 32.

Evals at the decay start (16B tokens; its checkpoint is kept for later WSD branches) and at 20B: Paloma
BPB (16 OLMES sources, hf backend) and lingua validation loss on dclm_heldout and fwedu_heldout (10k
docs each, decontaminated against the training subsets).

    python3 setup/make_lrsweep_configs.py
"""

import os

import yaml

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "apps/main/configs/experiments/lrsweep")
ASSETS = "/home/sgeng/scottg/pretrain/assets"
TOKENIZER = f"{ASSETS}/models/Qwen3.5-2B-tokenizer-eot"
GPUS = 32
SEQ = 4096
TOKENS = 20_000_000_000
WARMUP_TOKENS = 200_000_000
DECAY_FRACTION = 0.2
DUMP_TOKENS = 500_000_000  # resumption checkpoint interval (preemptible QoS)
FWEDU2_DOCS = 3_968_000  # 16.27B (fwedu_8m_0 + _1) + 3.968M x 1017.7 = 20.31B tokens available

GRID = {
    262_144: (76_294, ["7.07e-4", "1.00e-3", "1.41e-3", "2.00e-3", "2.83e-3"]),
    524_288: (38_147, ["1.00e-3", "1.41e-3", "2.00e-3", "2.83e-3", "4.00e-3"]),
    1_048_576: (19_073, ["1.41e-3", "2.00e-3", "2.83e-3", "4.00e-3", "5.66e-3"]),
}
CENTER = (524_288, "2.00e-3")
EXTRA_SEEDS = [1, 2]

PALOMA = [
    "paloma_4chan_meta_sep::paloma", "paloma_c4_100_domains::paloma", "paloma_c4_en::paloma",
    "paloma_dolma_100_programing_languages::paloma", "paloma_dolma_100_subreddits::paloma",
    "paloma_dolma-v1_5::paloma", "paloma_falcon-refinedweb::paloma", "paloma_gab::paloma",
    "paloma_m2d2_s2orc_unsplit::paloma", "paloma_m2d2_wikipedia_unsplit::paloma",
    "paloma_manosphere_meta_sep::paloma", "paloma_mc4::paloma", "paloma_ptb::paloma",
    "paloma_redpajama::paloma", "paloma_twitterAAE_HELM_fixed::paloma", "paloma_wikitext_103::paloma",
]


def data_mix(dataset):
    if dataset == "dclm":
        return {"dclm_subsets/dclm_8m_0": 1.0, "dclm_subsets/dclm_8m_1": 1.0}, None
    assert FWEDU2_DOCS is not None
    w = FWEDU2_DOCS / 8_000_000
    return ({"fineweb_edu_subsets/fwedu_8m_0": 1.0, "fineweb_edu_subsets/fwedu_8m_1": 1.0,
             "fineweb_edu_subsets/fwedu_8m_2": round(w, 6)},
            {"fineweb_edu_subsets/fwedu_8m_2": FWEDU2_DOCS})


def lr_tag(lr):
    m, e = lr.split("e-")
    return f"{m.replace('.', 'p')}e{int(e)}"


def config(dataset, tokens_per_step, steps, lr, seed):
    seqs = tokens_per_step // SEQ
    assert seqs % GPUS == 0
    warmup = round(WARMUP_TOKENS / tokens_per_step)
    decay_start = steps - round(steps * DECAY_FRACTION)
    name = f"{dataset}_bs{seqs}_lr{lr_tag(lr)}" + (f"_seed{seed}" if seed else "")
    sources, max_docs = data_mix(dataset)
    data = {"root_dir": f"{ASSETS}/data", "seed": 42 + seed, "batch_size": seqs // GPUS, "prefetch_size": 64,
            "seq_len": SEQ, "n_views": 2, "load_async": True, "add_bos": False, "add_eos": True,
            "disable_cross_doc_attn": True, "sources": sources,
            "tokenizer": {"name": "hf", "path": TOKENIZER}}
    if max_docs:
        data["source_max_docs"] = max_docs
    cfg = {
        "name": f"qwen3-0p6b-lrsweep-{name}".replace("_", "-"),
        "dump_dir": "${oc.env:MIDTRAIN_ROOT}/experiments/lrsweep/" + name,
        "steps": steps,
        "probe_freq": None,
        "seed": 777 + seed,
        "grad_acc_steps": 1,
        "optim": {"lr": float(lr), "weight_decay": 0.1, "epsilon": 1e-8, "beta1": 0.9, "beta2": 0.95, "clip": 1.0,
                  "scheduler": "wsd_sqrt", "warmup": warmup, "decay_fraction": DECAY_FRACTION,
                  "decay_matrices_only": True},
        "distributed": {"fsdp_type": "full_shard", "compile": True, "model_dtype": "bf16",
                        "matmul_allow_tf32": True, "selective_activation_checkpointing": False, "tp_size": 1},
        "model": {"dim": 1024, "n_layers": 28, "n_heads": 16, "n_kv_heads": 8, "head_dim": 128,
                  "ffn_dim_multiplier": 1.125, "multiple_of": 256, "norm_eps": 1e-6, "rope_theta": 1000000,
                  "max_seqlen": SEQ, "weight_tying": True, "qk_norm": True, "qk_norm_per_head": True,
                  "post_norm": False, "hf_arch": "Qwen3ForCausalLM", "init_base_std": 0.02, "seed": 42 + seed},
        "checkpoint": {"init_ckpt_path": None,
                       "dump": {"every": round(DUMP_TOKENS / tokens_per_step), "keep": 2},
                       "eval": {"every": decay_start, "keep": -1}},
        "data": data,
        "profiling": {"run": False},
        "logging": {"freq": 10, "wandb": {"project": "lingua-clean-lrsweep", "entity": "${oc.env:WANDB_ENTITY,null}",
                                          "resume": "allow"}},
        "async_eval_gpus": 4,
        "eval_backend": "olmes",
        "eval": {
            "dump_dir": "${oc.env:EVAL_ROOT}/slurm_eval_logs",
            "generator": {"max_tokens": 4096, "temperature": 0.0, "dtype": "bf16"},
            "harness": None,
            "olmes": {"tasks": PALOMA, "model_type": "vllm", "tokenizer_path": TOKENIZER, "tokenizer_revision": "main",
                      "max_length": 4096, "gpu_memory_utilization": 0.8, "trust_remote_code": True,
                      "olmes_env": "${oc.env:OLMES_CONDA_ENV}", "timeout_hours": 7.0},
            "validation": {"max_steps": None, "use_val_from_train_src": False, "root_dir": f"{ASSETS}/data",
                           "sources": ["dclm_subsets/dclm_heldout", "fineweb_edu_subsets/fwedu_heldout"]},
        },
    }
    header = (f"# LR sweep: {dataset}, {tokens_per_step:,} tokens/step ({seqs} x {SEQ}), lr {lr}, seed offset {seed}.\n"
              f"# warmup {warmup} steps (200M tokens), decay from step {decay_start} (16B tokens) to 0 at {steps}.\n"
              f"# Generated by setup/make_lrsweep_configs.py.\n"
              f"#   CONFIG=experiments/lrsweep/{name} NNODES={GPUS // 4} STEPS_OVERRIDE={steps} bash scripts/launch_pretrain.sh\n")
    return name, header, cfg


def main():
    os.makedirs(OUT, exist_ok=True)
    for dataset in ("dclm", "fwedu"):
        for tps, (steps, lrs) in GRID.items():
            assert abs(steps * tps - TOKENS) < tps, (tps, steps)
            for lr in lrs:
                seeds = [0] + (EXTRA_SEEDS if (tps, lr) == CENTER else [])
                for seed in seeds:
                    name, header, cfg = config(dataset, tps, steps, lr, seed)
                    with open(f"{OUT}/{name}.yaml", "w") as f:
                        f.write(header)
                        yaml.safe_dump(cfg, f, sort_keys=False, width=120)
                    print(name, steps)


if __name__ == "__main__":
    main()
