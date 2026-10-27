# Multi-rewrite study: repeating real data vs multiple rewrites per doc

**Status:** complete (all 12 arms at step 19074). Numbers pulled 2026-10-27 07:01 UTC; first partial
snapshot 2026-10-26 22:50 UTC.

## Question

S1 (10B real tokens, seen once) gets the model to a reasonable spot. The rest of the budget is a
~10B-token slot made from a smaller real pool S2. We ask whether repeating S2 is worse than training on
k distinct synthetic rewrites of S2 ("augmentation across epochs"), and whether more diverse generation
methods give better rewrites.

## Setup

- Model and training:
  - Qwen3-0.6B architecture with the Qwen3.5 tokenizer (~694M params), doc masking.
  - 19,074 steps x 256 x 4096 = 20.0B tokens.
  - AdamW lr 5.3e-3, wd 0.1, beta2 0.95, warmup 954, cosine to 1%. These are the best real-data hparams from the generator-scaling sweep.
- S1: `dclm_subsets/dclm_8m_0` (8M docs, ~10.1B tokens) at weight 1, i.e. one epoch.
- S2 pool: a prefix of `dclm_subsets/dclm_8m_1`. k=2 uses the first 4M docs (~5.1B tokens); k=4 uses the first 2M docs (~2.5B tokens).
- **Repeat arms:** S1 + kxS2. Sources sampled 1:1 per document, so the slot is ~10B tokens of S2 repeated ~k times.
- **Rewrite arms:** S1 + k distinct Qwen3.5-9B rewrites of every S2 doc. Each view is its own source with the same weight w = 1262.1 / sum(view tokens/doc):
  - The slot is token-matched to S1 (~10.1B tokens).
  - Every rewrite is seen the same ~w * 8M / N times; the "seen" column in the method list below.
- All checkpoints are evaluated at the final step.

### Generation methods (all Qwen3.5-9B, TP1, simplescale; source truncated to 11k tokens, max 4096 new tokens)

View 0 of iid / nemo / wofmt is the existing full-8M wiki rephrase `dclm_rewrites/dclm_8m_1_qwen35_9b_wiki_g1_max4096`.

| method | views | prompt / decoding | tok/doc | seen |
|---|---|---|---|---|
| repeat | real S2 docs | n/a | 1273 | k |
| iid | wiki x k | Megadocs/WRAP wiki rephrase, T=1, independent samples | 984-989 | 1.28x |
| t12 | wiki x k | same prompt, T=1.2, no top_p / min_p | 1104-1113 | 1.14x |
| nemo | wiki, diverse QA (+ distill, knowledge list for k=4) | Nemotron-CC prompts with their system prompts; answer headers stripped | 315 / 519 / 428 | 1.94x (k=2), 2.25x (k=4) |
| wofmt | wiki + (k-1) WebOrganizer formats | wiki prompt retargeted to "write a full {format}" | 1114-1131 | 1.16-1.19x |
| gacha | gacha x k | Gacha decoding (arXiv:2610.01382) on the wiki prompt, JSON-schema-constrained planning turns, dice seeded by doc id | 1109-1115 | 1.13x |

More on the methods:
- **wofmt format sampling:** k-1 distinct formats per doc, drawn without replacement and seeded by the doc id.
  - Draw probabilities are S2's WebOrganizer (NoURL) format distribution.
  - Excluded formats: Truncated, Spam / Ads, Legal Notices, and Knowledge Article (which the wiki view already covers); the remaining rates are renormalized.
- **Data:** row-aligned exports under `dclm_rewrites/dclm_8m_1_rewrite_first{4m,2m}_qwen35_9b_<method>_g<view>_max4096` (views 0-1 cover the first 4M rows, views 2-3 the first 2M).
- **T=1.2 quality:** about 2.4% of outputs degenerate into multilingual token soup (more than 10% non-ASCII). They are NOT filtered.
- **Gacha quality:** about 4% of samples lost the idea step and about 7% kept only 1-2 of 4 decisions (malformed planning output).
  - A first attempt without JSON constraints lost the idea step in 21% of samples; it was discarded.

## Runs

- Configs: `apps/main/configs/experiments/genscale/mr_{repeat,iid,t12,nemo,wofmt,gacha}_k{2,4}_lr5p3e3_wd01.yaml`.
- References: `r_lr5p3e3_wd01` (R = dclm_8m_0 + dclm_8m_1, 20B fresh real tokens), `s2x2_lr5p3e3_wd01`, `s1_rw9bwiki_lr5p3e3_wd01` (S1 + one 9B wiki rewrite of all 8M S2 docs).
- Pipeline:
  1. `setup/build_multirewrite_manifests.py` builds the manifests.
  2. `setup/submit_dclm_multirewrites.sh` runs generation.
  3. `setup/prepare_dclm_multirewrites.py` exports the views.
  4. `setup/make_multirewrite_configs.py` writes the configs.
  - The harnesses `rewrite:wiki|nemotron|wofmt` and `gacha:rewrite_wiki` are in the simplescale repo.

## Metrics

- PPL = exp(mean per-token NLL).
- **heldout PPL:** `dclm_heldout`, 1000 DCLM docs with no URL / exact / normalized-text overlap with any dclm_8m_k. This is the primary DCLM number.
- **OOD-5 PPL:** exp of the mean per-token NLL over the math / flan / wiki / pes2o / stackexchange Dolmino val sets.
- **Accuracy (OLMES):** SciQ (rc::olmo3), ARC-Easy, PIQA (rc::olmes:full), MMLU (rc::olmes), TriviaQA (F1). avg acc is the mean of these 5.
- **Gold BPB(5):** mean bits_per_byte_corr over SciQ / ARC-E / PIQA / MMLU plus the TriviaQA :bpb task.
- GSM8K and MBPP BPB are reported separately (GSM8K accuracy is at the floor).
- **Noise:** about +-0.1 avg acc across seeds; about +-0.3-0.4 across data splits.

## Results (step 19074)

| run | step | heldout PPL | OOD-5 PPL | SciQ | ARC-E | PIQA | MMLU | TQA | avg acc | gold BPB(5) | GSM8K BPB | MBPP BPB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| r | 19074 | 13.00 | 13.98 | 81.3 | 70.5 | 68.6 | 34.0 | 27.3 | 56.3 | 0.929 | 0.623 | 0.859 |
| s2x2 | 19074 | 13.05 | 14.05 | 81.3 | 70.5 | 68.6 | 33.7 | 26.9 | 56.2 | 0.929 | 0.619 | 0.857 |
| s1_rw9bwiki | 19074 | 13.44 | 14.17 | 83.6 | 71.7 | 70.0 | 34.6 | 28.4 | 57.7 | 0.945 | 0.610 | 0.858 |
| mr_repeat_k2 | 19074 | 13.03 | 14.05 | 81.8 | 71.2 | 70.6 | 34.2 | 27.4 | 57.0 | 0.926 | 0.620 | 0.856 |
| mr_repeat_k4 | 19074 | 13.07 | 14.10 | 81.7 | 71.2 | 69.0 | 34.5 | 26.7 | 56.6 | 0.930 | 0.626 | 0.863 |
| mr_iid_k2 | 19074 | 13.62 | 14.36 | 82.9 | 73.1 | 69.4 | 34.7 | 27.5 | 57.5 | 0.952 | 0.618 | 0.867 |
| mr_iid_k4 | 19074 | 13.75 | 14.53 | 81.7 | 71.9 | 69.1 | 34.9 | 27.2 | 57.0 | 0.962 | 0.623 | 0.883 |
| mr_t12_k2 | 19074 | 13.77 | 14.17 | 82.1 | 70.1 | 68.7 | 34.3 | 25.5 | 56.1 | 0.957 | 0.632 | 0.837 |
| mr_t12_k4 | 19074 | 13.79 | 14.17 | 79.9 | 71.5 | 69.3 | 34.4 | 25.5 | 56.1 | 0.959 | 0.633 | 0.851 |
| mr_nemo_k2 | 19074 | 13.60 | 14.29 | 85.2 | 73.7 | 69.6 | 35.5 | 30.1 | 58.8 | 0.937 | 0.621 | 0.859 |
| mr_nemo_k4 | 19074 | 13.62 | 14.39 | 83.7 | 75.3 | 70.0 | 35.6 | 30.0 | 58.9 | 0.930 | 0.623 | 0.871 |
| mr_wofmt_k2 | 19074 | 13.60 | 14.16 | 80.0 | 72.1 | 68.9 | 35.1 | 26.8 | 56.6 | 0.946 | 0.617 | 0.878 |
| mr_wofmt_k4 | 19074 | 13.67 | 14.21 | 81.3 | 72.5 | 69.0 | 34.8 | 26.7 | 56.9 | 0.949 | 0.615 | 0.878 |
| mr_gacha_k2 | 19074 | 13.74 | 14.46 | 81.2 | 71.2 | 68.7 | 35.0 | 26.0 | 56.4 | 0.955 | 0.618 | 0.881 |
| mr_gacha_k4 | 19074 | 13.77 | 14.54 | 81.2 | 71.8 | 69.1 | 34.5 | 26.2 | 56.6 | 0.959 | 0.622 | 0.878 |

Row names: `mr_repeat_k2` = S1 + 2xS2, `mr_iid_k4` = S1 + 4 iid rewrites per S2 doc, etc.

Average accuracy, rewrite arms vs the matching repeat arm (repeat: 57.0 at k=2, 56.6 at k=4):

| method | k=2 | k=4 |
|---|---|---|
| repeat (real) | 57.0 | 56.6 |
| iid | 57.5 (+0.5) | 57.0 (+0.4) |
| t12 | 56.1 (-0.9) | 56.1 (-0.5) |
| **nemo** | **58.8 (+1.8)** | **58.9 (+2.3)** |
| wofmt | 56.6 (-0.4) | 56.9 (+0.3) |
| gacha | 56.4 (-0.6) | 56.6 (+0.0) |

## Takeaways

1. **Repeating real data barely hurts at 2-4 epochs.**
   - 2xS2 scores 57.0 and 4xS2 56.6, vs 56.3 for R with fresh data.
   - heldout PPL is 13.03 / 13.07 vs 13.00.
   - At this scale repetition hasn't started to bite, so the repeat arms are a strong baseline that most rewrite methods fail to beat.
2. **Only the Nemotron-CC prompt mixture clearly beats repetition.**
   - 58.8 / 58.9, i.e. +1.8 / +2.3 over repeat.
   - Gains are broad: TQA +2.7-3.3, ARC-E +2.5-4.1, SciQ +2.0-3.4, MMLU +1.1-1.3.
   - It holds up at k=4, and its gold BPB stays at the repeat level (0.937 / 0.930).
   - Caveats:
     - Its short rewrites are repeated ~2x vs ~1.1-1.3x for the other methods, which works against it.
     - The gain plausibly comes from the QA / knowledge-list *tasks* (new formats that elicit knowledge) rather than from diversity per se.
3. **The other "diverse" generation methods do not help.**
   - Within noise of repeat, or worse: iid +0.5 / +0.4, WebOrganizer formats -0.4 / +0.3, Gacha -0.6 / +0.0, T=1.2 -0.9 / -0.5.
   - More views (k=4) do not help any method relative to k=2.
   - Gacha and WebOrganizer formats change surface form (genre, register, structure) but the content is still a paraphrase of the same document. Nemotron's QA / extraction formats are the only ones that change *what is asked of the content*.
4. **Higher temperature hurts** (56.1 at both k). TQA falls to 25.5, consistent with the ~2.4% unfiltered degenerate outputs.
5. **Every rewrite arm loses on in-distribution likelihood.**
   - heldout DCLM PPL 13.6-13.8 vs 13.0.
   - Gold BPB is worse (0.946-0.962) for all methods except nemo, mostly from TriviaQA BPB.
   - Downstream accuracy and likelihood metrics disagree.

So, for "augmentation across epochs": what you ask the generator to *do* with a document matters more than
how diverse the samples are. Surface-form diversity (temperature, genre / format, Gacha's decision axes)
doesn't beat simply repeating the real data at 2-4 epochs; task diversity (QA, distillation, knowledge
lists) does, by ~2 points.

## Reproduce

```
.venv/bin/python analysis/results_viewer/genscale_table.py r_lr5p3e3_wd01 s2x2_lr5p3e3_wd01 s1_rw9bwiki_lr5p3e3_wd01 \
    mr_repeat_k2_lr5p3e3_wd01 mr_repeat_k4_lr5p3e3_wd01 mr_iid_k2_lr5p3e3_wd01 mr_iid_k4_lr5p3e3_wd01 \
    mr_t12_k2_lr5p3e3_wd01 mr_t12_k4_lr5p3e3_wd01 mr_nemo_k2_lr5p3e3_wd01 mr_nemo_k4_lr5p3e3_wd01 \
    mr_wofmt_k2_lr5p3e3_wd01 mr_wofmt_k4_lr5p3e3_wd01 mr_gacha_k2_lr5p3e3_wd01 mr_gacha_k4_lr5p3e3_wd01
```
