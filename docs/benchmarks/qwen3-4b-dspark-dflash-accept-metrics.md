# Qwen3-4B DSpark vs DFlash acceptance, reproduced from `/metrics` counters

**TL;DR:** Reproduction of the acceptance tables in `models/qwen3/dspark-integration.md`, measured from the `vllm:spec_decode_*` counters instead of parsed server logs, on 1× RTX A6000 (sm_86) with the matched `deepseek-ai/{dspark,dflash}_qwen3_4b_block7` drafters at K=7. Two runs: **matched** (`SECONDS_PER_RUN=10`, `FIXED_SEED=0`, sized to the published round counts) and **high-N** (3× the prompts, for a tighter estimate). DFlash reproduces tightly — 2.36 mean accepted draft in *both* runs against a published 2.30 — and DSpark lands 7–9% high (2.76 / 2.69 vs 2.52). Zero-accept tracks within 1–2pp throughout. The published `random` result, including DSpark *losing* there at c1, reproduces once sampling is matched. **The one gap sampling does not explain is `sharegpt`:** matching both prompt count and seed still leaves it ~20% high (c1 DSpark 2.33 vs published 1.94, DFlash 1.83 vs 1.55), so something unrecorded about that dataset's configuration differs. Also confirms issue #604's never-before-run validation: `/metrics` accept rate equals `dflash_lane.rs`'s `cumulative_accept_rate` exactly, on every server run.

## What reproduces and what does not

- **DFlash pooled acceptance is solid.** 2.36 in both the matched and high-N runs vs 2.30 published — stable across a 3× change in sample size, which is what a real measurement should do.
- **Round counts confirm the published sample size.** The published prompt counts are unrecorded, but back-deriving from `rounds × (mean+1) ÷ output length` gives ~45 prompts per cell; running that (`SECONDS_PER_RUN=10`) lands at 21,256 / 23,808 rounds against the published 19,294 / 21,214.
- **`sonnet` is the implementation control.** It is a deterministic built-in generator, so both studies fed the same prompts; high-N c1 gives 3.21/2.95 against 3.23/2.92. Engine and counter path agree.
- **`random` is real but unstable.** Its published sign — DSpark losing at c1 — does reproduce under matched sampling (ours −13%, published −28%). But DSpark c1 moves from 1.52 (matched) to 2.12 (high-N), i.e. the synthetic corpus swings by 40% with the draw. The published caveat is not an artifact, but it rests on a very unstable observation.
- **`sharegpt` is the open question.** ~20% high on both drafters even with prompt count and seed aligned. Remaining candidates, none checkable from the doc: an unrecorded `--sharegpt-output-len` cap, a prompt count that does not scale with concurrency, or vllm-bench version drift in the sharegpt loader.

## How this was measured

`tools/bench/run_spec_accept_sweep.sh`, `vllm-bench 0.1.0`, `--temperature 0 --ignore-eos`, 4 datasets × c1/c4/c8 per drafter, `NUM_PROMPTS = concurrency × 30`. Per-cell acceptance is the difference of two `/metrics` scrapes bracketing the cell, so cell boundaries are exact rather than recovered from log timestamps. `num_accepted_tokens_per_pos` is a complementary CDF; the histogram comes from differencing adjacent positions, checked against `sum(hist) == num_drafts` and `sum(k·hist[k]) == num_accepted_tokens`.

Target `models/Qwen3-4B`; ShareGPT corpus is the canonical `ShareGPT_V3_unfiltered_cleaned_split.json` (94,145 conversations), the same file the prior study used. `accepted_draft` excludes the bonus token, so `committed = accepted_draft + 1`. Published figures appear *in italics* throughout.

## Caveats that bound these numbers

- **`speed-bench coding` has only 61 unique prompts**, so vllm-bench oversamples to fill the cell (~2× at c4, ~4× at c8). Under greedy decoding a repeated prompt reproduces its token sequence exactly, so repeats carry no new information: effective N stays 61 and those cells' error bars are wider than their round counts suggest.
- **Concurrency and sample size are entangled**: `NUM_PROMPTS` scales with concurrency, so c1→c4→c8 changes the prompt set and its size, not just the load. A clean concurrency axis needs a fixed prompt count. The published table shares this confound.
- **Absolute `sharegpt` accept is sampling-dominated at n=30.** Three draws of `sharegpt c1` gave DSpark 2.20 and 2.42, DFlash 1.91 and 2.00; vllm-bench's default `--seed 0` moved *away* from the published 1.94 rather than toward it. Compare pooled figures, not single cells.
- **Throughput is not comparable** to the published 5090 numbers (sm_86 vs sm_120). Acceptance is a model property and is comparable.
- The published per-cell prompt counts and seeds are unrecorded, so the round-weighted pooled figure cannot be weighted identically.

## Per-cell mean accepted draft (ours vs published)

| dataset | c1 ours | c1 pub | c4 ours | c4 pub | c8 ours | c8 pub |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| chat (sharegpt)| 2.20 / 1.91 | *1.94 / 1.55* | 2.75 / 2.23 | *2.15 / 1.68* | 2.56 / 2.00 | *2.05 / 1.63* |
| poem (sonnet)| 3.21 / 2.95 | *3.23 / 2.92* | 3.46 / 3.13 | *3.26 / 2.61* | 3.30 / 3.01 | *3.07 / 2.59* |
| rand (random)| 2.12 / 1.82 | *1.86 / 2.59* | 1.80 / 1.71 | *2.06 / 2.40* | 1.73 / 1.81 | *2.13 / 2.10* |
| code (speed-bench)| 3.35 / 2.79 | *3.18 / 2.97* | 3.56 / 3.04 | *3.54 / 2.75* | 3.63 / 3.03 | *3.44 / 3.04* |

## DSpark advantage, mean accepted draft (published in italics)

| dataset | c1 | c4 | c8 |
| --- | ---: | ---: | ---: |
| chat| +14.8% (*+25.2%*) | +23.4% (*+28.0%*) | +27.8% (*+25.8%*) |
| poem| +9.1% (*+10.6%*) | +10.5% (*+24.9%*) | +9.6% (*+18.5%*) |
| rand| +16.1% (*-28.2%*) | +5.3% (*-14.2%*) | -4.5% (*+1.4%*) |
| code| +20.2% (*+7.1%*) | +17.1% (*+28.7%*) | +19.7% (*+13.2%*) |

## DSpark — full per-cell detail

| dataset | c | rounds | mean acc | accept rate | zero | full-7 | hist 0..7 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| chat | 1 | 1,662 | 2.199 | 0.321 | 28.2% | 10.9% | `[469, 391, 237, 161, 110, 79, 34, 181]` |
| chat | 4 | 6,354 | 2.754 | 0.401 | 24.6% | 18.3% | `[1562, 1233, 840, 578, 431, 299, 246, 1165]` |
| chat | 8 | 13,796 | 2.558 | 0.373 | 25.8% | 15.8% | `[3558, 2768, 2028, 1310, 900, 631, 419, 2182]` |
| poem | 1 | 1,061 | 3.213 | 0.467 | 29.5% | 28.7% | `[313, 145, 72, 66, 64, 45, 51, 305]` |
| poem | 4 | 4,011 | 3.458 | 0.502 | 27.0% | 34.6% | `[1082, 570, 282, 201, 182, 147, 160, 1387]` |
| poem | 8 | 8,321 | 3.298 | 0.479 | 28.9% | 31.5% | `[2406, 1157, 624, 420, 386, 363, 340, 2625]` |
| rand | 1 | 1,222 | 2.118 | 0.312 | 38.1% | 14.9% | `[465, 234, 141, 83, 56, 41, 20, 182]` |
| rand | 4 | 5,435 | 1.804 | 0.265 | 40.1% | 10.0% | `[2179, 1194, 636, 390, 200, 190, 103, 543]` |
| rand | 8 | 11,169 | 1.729 | 0.254 | 40.3% | 9.9% | `[4497, 2645, 1328, 753, 349, 338, 148, 1111]` |
| code | 1 | 876 | 3.349 | 0.492 | 18.8% | 23.1% | `[165, 128, 103, 89, 80, 55, 54, 202]` |
| code | 4 | 3,342 | 3.560 | 0.521 | 17.0% | 26.8% | `[569, 475, 371, 338, 272, 219, 202, 896]` |
| code | 8 | 6,588 | 3.627 | 0.532 | 16.5% | 27.4% | `[1084, 911, 741, 604, 590, 429, 421, 1808]` |

## DFlash — full per-cell detail

| dataset | c | rounds | mean acc | accept rate | zero | full-7 | hist 0..7 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| chat | 1 | 1,824 | 1.914 | 0.280 | 32.4% | 8.6% | `[591, 439, 289, 167, 76, 59, 46, 157]` |
| chat | 4 | 7,382 | 2.231 | 0.326 | 28.6% | 11.5% | `[2114, 1694, 1057, 673, 496, 280, 218, 850]` |
| chat | 8 | 16,357 | 2.001 | 0.292 | 30.8% | 9.5% | `[5045, 4001, 2520, 1451, 861, 525, 405, 1549]` |
| poem | 1 | 1,133 | 2.945 | 0.427 | 34.1% | 22.1% | `[386, 142, 83, 54, 64, 63, 91, 250]` |
| poem | 4 | 4,330 | 3.129 | 0.454 | 30.7% | 27.1% | `[1329, 611, 299, 245, 217, 208, 249, 1172]` |
| poem | 8 | 8,922 | 3.008 | 0.437 | 33.0% | 25.2% | `[2941, 1236, 600, 438, 457, 490, 510, 2250]` |
| rand | 1 | 1,349 | 1.824 | 0.269 | 41.0% | 10.1% | `[553, 252, 201, 77, 45, 66, 19, 136]` |
| rand | 4 | 5,618 | 1.713 | 0.251 | 42.6% | 9.4% | `[2395, 1135, 726, 380, 167, 191, 96, 528]` |
| rand | 8 | 10,842 | 1.811 | 0.265 | 39.6% | 11.4% | `[4298, 2479, 1333, 701, 361, 312, 120, 1238]` |
| code | 1 | 1,006 | 2.787 | 0.409 | 21.7% | 13.4% | `[218, 184, 136, 113, 92, 72, 56, 135]` |
| code | 4 | 3,772 | 3.040 | 0.447 | 20.1% | 17.4% | `[759, 637, 480, 400, 344, 272, 225, 655]` |
| code | 8 | 7,565 | 3.029 | 0.445 | 19.8% | 17.1% | `[1499, 1357, 942, 802, 648, 523, 497, 1297]` |

## Pooled over all 12 cells (round-weighted)

| config | rounds | mean acc | accept rate | zero | full-7 | hist 0..7 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| DSpark | 63,837 | 2.69 | 0.393 | 28.7% | 19.7% | `[18349, 11851, 7403, 4993, 3620, 2836, 2198, 12587]` |
| DFlash | 70,100 | 2.36 | 0.345 | 31.6% | 14.6% | `[22128, 14167, 8666, 5501, 3828, 3061, 2532, 10217]` |
| *published DSpark* | *19,294* | *2.52* | — | *29.2%* | *17.4%* | `[5636, 3942, 2394, 1549, 1042, 742, 639, 3350]` |
| *published DFlash* | *21,214* | *2.30* | — | *32.2%* | *13.9%* | `[6838, 4340, 2685, 1648, 1071, 962, 731, 2939]` |
