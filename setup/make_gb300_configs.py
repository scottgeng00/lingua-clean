"""Write the GB300-cluster sweep configs to apps/main/configs/experiments/gb300/.

  pt_dclm8m0_lr*: from-scratch CE pretraining on dclm_8m_0 (one epoch, 10.0B tokens), LR sweep over the
                  genscale grid (1.3e-3 / 2.65e-3 / 5.3e-3 / 1.06e-2) at wd 0.1.
  kd9bbase_a*:    KD from Qwen3.5-9B-Base on dclm_8m_0 + dclm_8m_1 (1:1 per doc, 20.0B tokens),
                  loss = a*KL(T=1) + (1-a)*CE, a in {0.1, 0.3, 0.5, 0.7, 0.9}, lr 2.65e-3, wd 0.1.

Everything else is genscale/r_lr2p65e3_wd01.yaml: Qwen3-0.6B shape, Qwen3.5 tokenizer (~694M params),
256 x 4096 tokens per step, doc masking. Only the final checkpoint is evaluated, with the OLMES suite
lingua:v2 (listed task by task so eval_olmes spreads them over the eval job's GPUs), no lingua validation.

GPUs per run must be a multiple of the 16 data chunks: data.batch_size = 256 / (GPUs * grad_acc_steps).
KD uses 4 seqs/GPU x 2 accumulation: at 8 seqs/GPU the cached activations left NCCL no room for the
P2P buffers of the first checkpoint save (CUDA OOM in NCCL at step 500).

    python3 setup/make_gb300_configs.py
"""

import os

import yaml

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "apps/main/configs/experiments/gb300")
ASSETS = "/home/sgeng/scottg/pretrain/assets"
TOKENIZER = f"{ASSETS}/models/Qwen3.5-2B-tokenizer-eot"
TEACHER = f"{ASSETS}/models/Qwen3.5-9B-Base"
GLOBAL_BATCH = 256
SEQ_LEN = 4096

# OLMES suite lingua:v2 (olmes fork, branch lm-eval-0.4.13) with paloma::paloma expanded.
LINGUA_V2 = [
    "arc:rc::olmes:full", "basic_skills:rc::olmes", "coqa:rc::gen2mc:xlarge", "csqa:rc::olmes:full",
    "drop:rc::gen2mc:xlarge", "hellaswag:rc::olmes:full", "jeopardy:rc::gen2mc:xlarge", "lab_bench_dbqa",
    "lab_bench_protocolqa", "lambada", "medmcqa:rc::none", "medqa_en:rc::none", "mmlu:rc::olmes",
    "naturalqs:rc::gen2mc:xlarge", "piqa:rc::olmes:full", "qasper_yesno:rc::olmes", "sciq:rc::olmo3",
    "sciriff_yesno:rc::olmes", "squad:rc::gen2mc:xlarge", "winogrande:rc::olmes:full",
    "arc:rc:bpb::olmes:full", "basic_skills:rc:bpb::olmes", "coqa:rc:bpb::gen2mc:xlarge",
    "csqa:rc:bpb::olmes:full", "drop:rc:bpb::gen2mc:xlarge", "hellaswag:rc:bpb::olmes:full",
    "jeopardy:rc:bpb::gen2mc:xlarge", "lab_bench_dbqa:bpb", "lab_bench_protocolqa:bpb", "lambada:bpb",
    "medmcqa:rc:bpb::none", "medqa_en:rc:bpb::none", "mmlu:rc:bpb::olmes", "naturalqs:rc:bpb::gen2mc:xlarge",
    "piqa:rc:bpb::olmes:full", "qasper_yesno:rc:bpb::olmes", "sciq:rc:bpb::olmo3", "sciriff_yesno:rc:bpb::olmes",
    "squad:rc:bpb::gen2mc:xlarge", "winogrande:rc:bpb::olmes:full",
    "minerva_math_algebra:bpb::olmes", "minerva_math_counting_and_probability:bpb::olmes",
    "minerva_math_geometry:bpb::olmes", "minerva_math_intermediate_algebra:bpb::olmes",
    "minerva_math_number_theory:bpb::olmes", "minerva_math_prealgebra:bpb::olmes",
    "minerva_math_precalculus:bpb::olmes",
    "codex_humaneval:3shot:bpb::none", "mbpp:3shot:bpb::none", "mt_mbpp_v2fix",
    "paloma_4chan_meta_sep::paloma", "paloma_c4_100_domains::paloma", "paloma_c4_en::paloma",
    "paloma_dolma_100_programing_languages::paloma", "paloma_dolma_100_subreddits::paloma",
    "paloma_dolma-v1_5::paloma", "paloma_falcon-refinedweb::paloma", "paloma_gab::paloma",
    "paloma_m2d2_s2orc_unsplit::paloma", "paloma_m2d2_wikipedia_unsplit::paloma",
    "paloma_manosphere_meta_sep::paloma", "paloma_mc4::paloma", "paloma_ptb::paloma",
    "paloma_redpajama::paloma", "paloma_twitterAAE_HELM_fixed::paloma", "paloma_wikitext_103::paloma",
    "gsm8k::olmo3:n32:pass32",
]


def base(name, steps, lr, gpus, sources, grad_acc=1):
    return {
        "name": f"qwen3-0p6b-gb300-{name}".replace("_", "-"),
        "dump_dir": "${oc.env:MIDTRAIN_ROOT}/experiments/gb300/" + name,
        "steps": steps,
        "probe_freq": None,
        "seed": 777,
        "grad_acc_steps": grad_acc,
        "optim": {"lr": lr, "weight_decay": 0.1, "warmup": 954, "lr_min_ratio": 0.01, "clip": 1.0,
                  "beta1": 0.9, "beta2": 0.95, "scheduler": "cosine"},
        "distributed": {"fsdp_type": "full_shard", "compile": True, "model_dtype": "bf16",
                        "matmul_allow_tf32": True, "selective_activation_checkpointing": False, "tp_size": 1},
        "model": {"dim": 1024, "n_layers": 28, "n_heads": 16, "n_kv_heads": 8, "head_dim": 128,
                  "ffn_dim_multiplier": 1.125, "multiple_of": 256, "norm_eps": 1e-6, "rope_theta": 1000000,
                  "max_seqlen": SEQ_LEN, "weight_tying": True, "qk_norm": True, "qk_norm_per_head": True,
                  "post_norm": False, "hf_arch": "Qwen3ForCausalLM", "init_base_std": 0.02, "seed": 42},
        # Resumption checkpoints every 500 steps (g3_comm_shared can be preempted); the final step is
        # always saved and evaluated, and eval.every never fires before it.
        "checkpoint": {"init_ckpt_path": None, "dump": {"every": 500, "keep": 2},
                       "eval": {"every": 1_000_000, "keep": -1}},
        "data": {"root_dir": f"{ASSETS}/data", "seed": 42, "batch_size": GLOBAL_BATCH // (gpus * grad_acc),
                 "prefetch_size": 64, "seq_len": SEQ_LEN, "n_views": 2, "load_async": True,
                 "add_bos": False, "add_eos": True, "disable_cross_doc_attn": True,
                 "sources": sources, "tokenizer": {"name": "hf", "path": TOKENIZER}},
        "profiling": {"run": False},
        "logging": {"freq": 10, "wandb": {"project": "lingua-clean-pretrain",
                                          "entity": "${oc.env:WANDB_ENTITY,null}", "resume": "allow"}},
        "async_eval_gpus": 4,
        "eval_backend": "olmes",
        "eval": {
            "dump_dir": "${oc.env:EVAL_ROOT}/slurm_eval_logs",
            "generator": {"max_tokens": 4096, "temperature": 0.0, "dtype": "bf16"},
            "harness": None,
            "olmes": {"tasks": LINGUA_V2, "model_type": "vllm", "tokenizer_path": TOKENIZER,
                      "tokenizer_revision": "main", "max_length": 4096, "gpu_memory_utilization": 0.8,
                      "trust_remote_code": True, "olmes_env": "${oc.env:OLMES_CONDA_ENV}", "timeout_hours": 7.0},
            "validation": None,
        },
    }


def write(name, cfg, header):
    os.makedirs(OUT, exist_ok=True)
    with open(f"{OUT}/{name}.yaml", "w") as f:
        f.write(header)
        yaml.safe_dump(cfg, f, sort_keys=False, width=120)
    print(f"{OUT}/{name}.yaml")


def main():
    # 9,537 steps x 256 x 4096 = 10.0B tokens, just under one epoch of dclm_8m_0 (~10.10B Qwen3.5 tokens).
    pt_gpus = 16
    for tag, lr in [("1p3e3", 1.3e-3), ("2p65e3", 2.65e-3), ("5p3e3", 5.3e-3), ("1p06e2", 1.06e-2)]:
        name = f"pt_dclm8m0_lr{tag}"
        cfg = base(name, 9537, lr, pt_gpus, {"dclm_subsets/dclm_8m_0": 1.0})
        write(name, cfg, f"# CE pretraining on dclm_8m_0, 10.0B tokens, lr {lr}. Generated by setup/make_gb300_configs.py.\n"
                         f"#   CONFIG=experiments/gb300/{name} NNODES={pt_gpus // 4} STEPS_OVERRIDE=9537 bash scripts/launch_pretrain.sh\n")

    # 19,074 steps x 256 x 4096 = 20.0B tokens over dclm_8m_0 + dclm_8m_1 (20.3B tokens, ~0.99 epoch).
    kd_gpus = 32
    for tag, alpha in [("0p1", 0.1), ("0p3", 0.3), ("0p5", 0.5), ("0p7", 0.7), ("0p9", 0.9)]:
        name = f"kd9bbase_a{tag}"
        cfg = base(name, 19074, 2.65e-3, kd_gpus, {"dclm_subsets/dclm_8m_0": 1.0, "dclm_subsets/dclm_8m_1": 1.0},
                   grad_acc=2)
        cfg["teacher_model_path"] = TEACHER
        cfg["data"].update({"use_kl_distillation": True, "kl_alpha": alpha, "kl_temperature": 1.0})
        write(name, cfg, f"# KD from Qwen3.5-9B-Base, loss = {alpha}*KL(T=1) + {1 - alpha:.1f}*CE, dclm_8m_0 + dclm_8m_1, 20.0B tokens.\n"
                         f"# Generated by setup/make_gb300_configs.py.\n"
                         f"#   CONFIG=experiments/gb300/{name} NNODES={kd_gpus // 4} STEPS_OVERRIDE=19074 bash scripts/launch_pretrain.sh\n")


if __name__ == "__main__":
    main()
