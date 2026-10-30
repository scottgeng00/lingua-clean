"""Configs for the data-constrained Qwen3 ladder (doc: "Data-constrained pretraining: first sweep instructions").

Writes apps/main/configs/experiments/qwen3_ladder_dataconstrained/<run>.yaml plus manifest.json (every run with its
step count, GPU count and, for decay branches, the checkpoint it starts from). Run names start with the model width:

    w{width}_s1_bs{B}_lr{lr}_wd{wd}            stage 1 main run: unique data to 2x Chinchilla, decays over the last 10%
    w{width}_s1_bs{B}_lr{lr}_wd{wd}_d1x        its decay branch evaluated at 1x Chinchilla
    w{width}_s2_bs{B}_lr{lr}_wd{wd}            stage 2 main run: 16 passes over the pool, decays over the last 10%
    w{width}_s2_bs{B}_lr{lr}_wd{wd}_p{k}       its decay branch evaluated after pass k
    ..._seed{n}                                 another seed (model init)

Model: Qwen3 dense block (pre-norm RMSNorm, per-head QK-norm, SwiGLU 4x width, RoPE 1e6, no biases), head size 128,
MHA, untied embeddings, Qwen3 tokenizer (vocab padded to 151,936), context 4096, document masking. Depth from the
Delphi rule: layers = round(width / (55 + 4 log2 width)).

Data: lingua/token_stream.py over the Qwen3-tokenized global DCLM order (setup/pretokenize_dclm_qwen3.py).
Stage 1 reads the stream once; stage 2 repeats a pool of the first documents, cut to a whole number of sequences
divisible by 256 (125.83M tokens for width 512, 1.274B for 1152), re-shuffled and repacked every pass.

Schedule: WSD, linear warmup over 1 token per body parameter, then constant, then 1 - sqrt decay to 0 over the last
10% of the tokens up to each evaluation point. Earlier evaluation points are separate branch runs that resume the
main run's checkpoint at E - round(0.1 E) and decay to E (a run with steps = E and decay_fraction 0.1 decays from
exactly that step).

Examples (one call per sweep step; re-running is idempotent):
    python3 setup/make_dataconstrained_configs.py --width 512 --stage 1 --lr 2e-3 3e-3 4e-3 6e-3 8e-3 --bs 8 16 32 64 --wd 0.1
    python3 setup/make_dataconstrained_configs.py --width 512 --stage 2 --lr 2e-3 3e-3 4e-3 6e-3 --bs 8 16 32 64 --wd 0.1 0.3 1.0
    python3 setup/make_dataconstrained_configs.py --width 512 --stage 2 --final --lr 3e-3 --bs 16 --wd 0.3 --seed 0 1
    python3 setup/make_dataconstrained_configs.py --width 512 --stage 2 --lr 6e-3 --bs 64 --wd 0.3 --seed 0 1 \
        --extend 32 --extend-evals 20 24 28      # w512_s2final_..._x32[_pK]: continue the final runs to 32 passes
"""

import argparse
import json
import math
import os

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "apps/main/configs/experiments/qwen3_ladder_dataconstrained")
ASSETS = "/home/sgeng/scottg/pretrain/assets"
TOKENIZER = f"{ASSETS}/models/Qwen3-0.6B-Base-tokenizer"
STREAM = f"{ASSETS}/data/dclm_qwen3_tokens"
HELDOUT = f"{ASSETS}/data/dclm_qwen3_tokens/dclm_heldout"
GROUP = "qwen3_ladder_dataconstrained"
SEQ, VOCAB, HEAD = 4096, 151_936, 128
SWEEP_PASSES = [1, 2, 3, 4, 6, 8, 12]  # stage 2 branches during the sweep; the main run evaluates at 16
N_PASSES = 16
SEQS_PER_GPU = {512: 16, 1152: 8}
EVAL_GPUS = {512: 1, 1152: 4}
PALOMA = [
    "paloma_4chan_meta_sep::paloma", "paloma_c4_100_domains::paloma", "paloma_c4_en::paloma",
    "paloma_dolma_100_programing_languages::paloma", "paloma_dolma_100_subreddits::paloma",
    "paloma_dolma-v1_5::paloma", "paloma_falcon-refinedweb::paloma", "paloma_gab::paloma",
    "paloma_m2d2_s2orc_unsplit::paloma", "paloma_m2d2_wikipedia_unsplit::paloma",
    "paloma_manosphere_meta_sep::paloma", "paloma_mc4::paloma", "paloma_ptb::paloma",
    "paloma_redpajama::paloma", "paloma_twitterAAE_HELM_fixed::paloma", "paloma_wikitext_103::paloma",
]


def shape(width):
    layers = round(width / (55 + 4 * math.log2(width)))
    heads = width // HEAD
    assert heads * HEAD == width
    ffn = 4 * width
    per_layer = 4 * width * width + 3 * width * ffn + 2 * width + 2 * HEAD  # attn, SwiGLU, 2 RMSNorm, q/k-norm
    body = layers * per_layer + width  # + final norm
    return {"layers": layers, "heads": heads, "ffn": ffn, "body": body, "embedding": VOCAB * width}


def chinchilla(width):
    return 20 * shape(width)["body"]


def pool_tokens(width):
    """0.25x Chinchilla, rounded down to a whole number of sequences divisible by 256."""
    seqs = chinchilla(width) // 4 // SEQ
    return (seqs - seqs % 256) * SEQ


def fmt(x):
    """0.1 -> 0p1, 1.0 -> 1, 0.03 -> 0p03."""
    return f"{x:g}".replace(".", "p")


def lr_fmt(lr):
    """2e-3 -> 2e-3, 1.5e-3 -> 1p5e-3, 1e-2 -> 1e-2."""
    m, e = f"{lr:e}".split("e")
    return f"{float(m):g}".replace(".", "p") + f"e{int(e)}"


def decay_start(steps):
    return steps - round(steps * 0.1)


def config(width, name, steps, lr, bs, wd, seed, mode, n_tokens, branch_from=None, extra_saves=()):
    s = shape(width)
    gpus = max(1, bs // SEQS_PER_GPU[width])
    tps = bs * SEQ
    warmup = round(s["body"] / tps)
    dump_every = max(100, round(steps / 20))  # resumption checkpoints (preemptible QoS)
    cfg = {
        "name": f"{GROUP}-{name}".replace("_", "-"),
        "dump_dir": "${oc.env:MIDTRAIN_ROOT}/experiments/" + GROUP + "/" + name,
        "steps": steps,
        "probe_freq": None,
        "seed": 777 + seed,
        "grad_acc_steps": 1,
        "expected_params": {"body": s["body"], "embedding": s["embedding"], "output": s["embedding"]},
        "optim": {"lr": lr, "weight_decay": wd, "epsilon": 1e-8, "beta1": 0.9, "beta2": 0.95, "clip": 1.0,
                  "scheduler": "wsd_sqrt", "warmup": warmup, "decay_fraction": 0.1, "decay_matrices_only": True},
        "distributed": {"fsdp_type": "full_shard", "compile": True, "model_dtype": "bf16", "matmul_allow_tf32": True,
                        "selective_activation_checkpointing": False, "tp_size": 1},
        "model": {"dim": width, "n_layers": s["layers"], "n_heads": s["heads"], "n_kv_heads": s["heads"],
                  "head_dim": HEAD, "ffn_dim_multiplier": 1.5, "multiple_of": 256, "norm_eps": 1e-6,
                  "rope_theta": 1_000_000, "max_seqlen": SEQ, "vocab_size": VOCAB, "weight_tying": False,
                  "qk_norm": True, "qk_norm_per_head": True, "post_norm": False, "hf_arch": "Qwen3ForCausalLM",
                  "init_base_std": 0.02, "seed": 42 + seed},
        "checkpoint": {"init_ckpt_path": None, "dump": {"every": dump_every, "keep": 1},
                       "eval": {"every": 10**9, "keep": -1},
                       "extra_save_steps": sorted(extra_saves), "branch_from": branch_from},
        "data": {"root_dir": f"{ASSETS}/data", "sources": {}, "seed": 42, "batch_size": bs // gpus, "seq_len": SEQ,
                 "n_views": 2, "add_bos": False, "add_eos": True, "disable_cross_doc_attn": True,
                 "tokenizer": {"name": "hf", "path": TOKENIZER},
                 "token_stream_dir": STREAM, "token_stream_mode": mode, "token_stream_tokens": n_tokens,
                 "token_stream_seed": 0},
        "profiling": {"run": False},
        "logging": {"freq": 10, "wandb": {"project": "lingua-clean-dataconstrained",
                                          "entity": "${oc.env:WANDB_ENTITY,null}", "resume": "allow"}},
        "async_eval_gpus": EVAL_GPUS[width],
        "eval_backend": "olmes",
        "eval": {
            "dump_dir": "${oc.env:EVAL_ROOT}/slurm_eval_logs",
            "generator": {"max_tokens": SEQ, "temperature": 0.0, "dtype": "bf16"},
            "harness": None,
            "olmes": {"tasks": PALOMA, "model_type": "hf", "tokenizer_path": TOKENIZER, "tokenizer_revision": "main",
                      "max_length": SEQ, "trust_remote_code": True, "olmes_env": "${oc.env:OLMES_CONDA_ENV}",
                      "timeout_hours": 10.0, "rolling_no_prefix": True},
            "validation": {"max_steps": None, "use_val_from_train_src": False, "root_dir": f"{ASSETS}/data",
                           "sources": [], "token_streams": [HELDOUT], "token_stream_seq_len": SEQ},
        },
    }
    entry = {"name": name, "width": width, "steps": steps, "gpus": gpus, "lr": lr, "bs": bs, "wd": wd, "seed": seed,
             "tokens": steps * tps, "warmup": warmup, "decay_start": decay_start(steps)}
    if branch_from:
        entry["branch_from"] = branch_from
    return cfg, entry


def runs_for(width, stage, lr, bs, wd, seed, final=False):
    """(name, config, manifest entry) for one configuration: the main run and its decay branches."""
    tps = bs * SEQ
    base = f"w{width}_s{stage}{'final' if final else ''}_bs{bs}_lr{lr_fmt(lr)}_wd{fmt(wd)}" + (f"_seed{seed}" if seed else "")
    main_dir = "${oc.env:MIDTRAIN_ROOT}/experiments/" + GROUP + "/" + base
    if stage == 1:
        C = chinchilla(width)
        main_steps, mode, n_tokens = round(2 * C / tps), "unique", round(2 * C / tps) * tps
        branches = {"d1x": round(C / tps)}
    else:
        assert pool_tokens(width) % tps == 0, (width, bs)
        per_pass = pool_tokens(width) // tps
        main_steps, mode, n_tokens = N_PASSES * per_pass, "repeat", pool_tokens(width)
        passes = range(1, N_PASSES) if final else SWEEP_PASSES
        branches = {f"p{k}": k * per_pass for k in passes}
    out = []
    starts = {decay_start(e) for e in branches.values()}
    cfg, entry = config(width, base, main_steps, lr, bs, wd, seed, mode, n_tokens, extra_saves=starts)
    entry.update(stage=stage, eval_point="2x" if stage == 1 else f"p{N_PASSES}", main=base)
    out.append((base, cfg, entry))
    for tag, e in branches.items():
        src = f"{main_dir}/checkpoints/{decay_start(e):010d}"
        cfg, entry = config(width, f"{base}_{tag}", e, lr, bs, wd, seed, mode, n_tokens, branch_from=src)
        entry.update(stage=stage, eval_point=tag, main=base)
        if decay_start(e) > e - 20:
            entry["warning"] = f"decay only {e - decay_start(e)} steps"
        out.append((f"{base}_{tag}", cfg, entry))
    return out


def extension_runs(width, lr, bs, wd, seed, n_passes, eval_passes):
    """Extend a --final stage 2 run to n_passes: resume its stable-phase checkpoint at 13.5 passes (the start of its
    pass-15 decay branch; the last checkpoint before its own decay) with steps = n_passes passes, so it holds the peak
    learning rate to 0.9 * n_passes and decays to 0 at n_passes, exactly as an n_passes run from scratch would (same
    warmup, same constant LR, same position-determined data). Decay branches at eval_passes resume the extension."""
    tps = bs * SEQ
    per_pass = pool_tokens(width) // tps
    final = f"w{width}_s2final_bs{bs}_lr{lr_fmt(lr)}_wd{fmt(wd)}" + (f"_seed{seed}" if seed else "")
    src = "${oc.env:MIDTRAIN_ROOT}/experiments/" + GROUP + f"/{final}/checkpoints/{decay_start(15 * per_pass):010d}"
    base = f"{final}_x{n_passes}"
    main_dir = "${oc.env:MIDTRAIN_ROOT}/experiments/" + GROUP + "/" + base
    branches = {f"p{k}": k * per_pass for k in eval_passes}
    starts = {decay_start(e) for e in branches.values()}
    assert min(starts) > decay_start(15 * per_pass), "extension branches must start after the resumed checkpoint"
    out = []
    cfg, entry = config(width, base, n_passes * per_pass, lr, bs, wd, seed, "repeat", pool_tokens(width),
                        branch_from=src, extra_saves=starts)
    entry.update(stage=2, eval_point=f"p{n_passes}", main=final)
    out.append((base, cfg, entry))
    for tag, e in branches.items():
        cfg, entry = config(width, f"{base}_{tag}", e, lr, bs, wd, seed, "repeat", pool_tokens(width),
                            branch_from=f"{main_dir}/checkpoints/{decay_start(e):010d}")
        entry.update(stage=2, eval_point=tag, main=base)
        out.append((f"{base}_{tag}", cfg, entry))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, required=True, choices=[512, 1152])
    ap.add_argument("--stage", type=int, required=True, choices=[1, 2])
    ap.add_argument("--lr", type=float, nargs="+", required=True)
    ap.add_argument("--bs", type=int, nargs="+", required=True)
    ap.add_argument("--wd", type=float, nargs="+", required=True)
    ap.add_argument("--seed", type=int, nargs="+", default=[0])
    ap.add_argument("--final", action="store_true", help="stage 2 final run: a decay branch after every pass")
    ap.add_argument("--extend", type=int, help="extend the --final run(s) to this many passes (from 13.5 passes)")
    ap.add_argument("--extend-evals", type=int, nargs="*", default=[], help="decay branches of the extension")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    man_path = f"{OUT}/manifest.json"
    manifest = json.load(open(man_path)) if os.path.exists(man_path) else {}
    new = 0
    for lr in a.lr:
        for bs in a.bs:
            for wd in a.wd:
                for seed in a.seed:
                    todo = (extension_runs(a.width, lr, bs, wd, seed, a.extend, a.extend_evals) if a.extend
                            else runs_for(a.width, a.stage, lr, bs, wd, seed, a.final))
                    for name, cfg, entry in todo:
                        with open(f"{OUT}/{name}.yaml", "w") as f:
                            f.write(f"# Generated by setup/make_dataconstrained_configs.py ({entry['eval_point']}, "
                                    f"{entry['steps']} steps, {entry['gpus']} GPU(s)).\n")
                            yaml.safe_dump(cfg, f, sort_keys=False, width=120)
                        new += name not in manifest
                        manifest[name] = entry
    json.dump(manifest, open(man_path, "w"), indent=1, sort_keys=True)
    s = shape(a.width)
    print(f"width {a.width}: {s['layers']} layers, {s['heads']} heads, body {s['body']:,}, embedding {s['embedding']:,}; "
          f"chinchilla {chinchilla(a.width):,}, pool {pool_tokens(a.width):,} tokens. {new} new runs, manifest {len(manifest)}")


if __name__ == "__main__":
    main()
