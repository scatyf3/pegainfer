# Issue #78 — streaming token-count repro

Reproduces the wrong output-token counts reported for streaming `/v1/completions`
([issue #78](https://github.com/xiaguan/pegainfer/issues/78)).

`vllm bench serve` reads each request's output length from the stream's
`usage.completion_tokens`, falling back to **re-tokenizing the generated text**
when usage is missing/zero. With `--ignore-eos` the true total is exactly
`num_prompts * output_len`, so any deviation exposes a bad count: re-tokenized
gibberish over-counts, empty output under-counts.

## Prerequisites
- `cargo build --release` (the `target/release/pegainfer` server)
- `models/Qwen3-4B` present (or set `MODEL_PATH=`)
- For `run_vllm_bench.sh`: a `vllm` install (set `VLLM=/path/to/vllm` if not on PATH)
- A GPU. On a cluster, point `PEGAINFER_ENV` at a setup file that activates the
  CUDA toolchain (it's sourced automatically if present).

## Scripts
| Script | What it shows |
|---|---|
| `run_vllm_bench.sh [IN OUT N SEED]` | End-to-end repro: `vllm bench serve --save-detailed`, prints per-request `output_lens` and the total error. Clearest reproduction. |
| `run_count_probe.sh` | Per-request: streams 20 varied prompts, compares tokens actually streamed vs reported `completion_tokens`. |
| `usage_probe.sh` | Root-cause check: is a correct `usage` chunk present in the streaming response (vs the non-streaming one)? |
| `count_probe.py` | The probe body used by `run_count_probe.sh` (also runnable directly against a running server). |
| `common.sh` | Shared setup (env vars, `start_server`/`stop_server`). |

## Run
```bash
bash scripts/issue78_token_count/run_vllm_bench.sh          # 1024->256, n=20, seed=42
bash scripts/issue78_token_count/run_count_probe.sh
bash scripts/issue78_token_count/usage_probe.sh
```

## Observed (H100, Qwen3-4B, 2026-05-21)
- `run_vllm_bench.sh` default-seed sweep (1024→256, n=20): reported **5888** vs true
  **5120** (+768, +15%); same +768 at concurrency 16/32. With `--seed 42`, one
  request reported `output_len=1` (sum 4865 vs 5120) — the error goes both ways and
  is prompt/seed dependent (matches "8 of 500" in the issue).
- `count_probe.py`: natural-language prompts all report the correct count (including
  multi-byte emoji, where some tokens emit empty text but `completion_tokens` is still
  right). The wrong counts surface on the synthetic random-token / empty prompts the
  benchmark generates — so on current code the residual is an edge-case miscount
  rather than usage being missing for every request.
