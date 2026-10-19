//! EAGLE-3 single-stream decode A/B on real GSM8K prompts, built to be compared
//! one-to-one against vLLM's EAGLE-3 (`tools/bench/eagle3_vllm_ab.py`).
//!
//! Both engines run the same chain length γ = [`EAGLE3_CHAIN_LENGTH`] (the vLLM
//! script reads it from this harness's summary as `num_speculative_tokens`), the
//! same drafter, and the **same prompt token ids**: this harness tokenizes the
//! lm-eval `gsm8k` 8-shot prompts once and dumps them, and vLLM consumes the dump
//! as `prompt_token_ids`. Decode is greedy with `ignore_eos` and a fixed budget,
//! bs=1, requests issued back to back.
//!
//! Timing separates prefill from decode: TTFT is submit → first token, decode
//! tok/s is `(n - 1) / (finish - first token)`. Per-round acceptance comes from
//! the executor's `accepted_draft=` debug lines (`PEGAINFER_TEST_LOG=1`).
//!
//! Env:
//! - `PEGAINFER_TEST_MODEL_PATH`, `PEGAINFER_EAGLE3_TEST_MODEL_PATH` (required)
//! - `PEGAINFER_GSM8K_DIR` (default `<crate>/../models/gsm8k`, `train.jsonl` + `test.jsonl`)
//! - `PEGAINFER_AB_MODE` = `both` (default) | `off` | `on`
//! - `PEGAINFER_AB_NUM_PROMPTS` (default 20)
//! - `PEGAINFER_AB_OUT` directory: writes `prompts.jsonl`, `ab_config.json` and
//!   `pegainfer_<off|on>[_profiled].json`
//! - `PEGAINFER_AB_PROFILE_RANGE=1`: wrap the measured loop (after warm-up) in
//!   `cudaProfilerStart/Stop`, for `nsys --capture-range=cudaProfilerApi`
//!
//! Requires a CUDA GPU; skips cleanly when the model, drafter or data are absent.

use std::io::Write as _;
use std::path::Path;
use std::path::PathBuf;
use std::time::Duration;
use std::time::Instant;

use pegainfer_frontend::engine::Terminal;
use pegainfer_frontend::sampler::SamplingParams;
use pegainfer_qwen3::DEFAULT_KV_CACHE_MEMORY_MARGIN_BYTES;
use pegainfer_qwen3::DEFAULT_KV_PAGE_SIZE;
use pegainfer_qwen3::DecodeOverlap;
use pegainfer_qwen3::EAGLE3_CHAIN_LENGTH;
use pegainfer_qwen3::Qwen3LaunchOptions;
use pegainfer_qwen3::Qwen3MemoryOptions;
use pegainfer_qwen3::Qwen3OffloadOptions;

mod common;

use common::harness::EngineHarness;
use common::harness::request;

const MODEL_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../models/Qwen3-4B");
const DRAFT_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../models/Qwen3-4B_eagle3");
const GSM8K_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../models/gsm8k");
/// Fixed decode budget per prompt (EOS ignored) so both engines do identical work.
const GENERATED_TOKENS: usize = 256;
const NUM_FEWSHOT: usize = 8;
const DEFAULT_NUM_PROMPTS: usize = 20;
/// The 8-shot prompt is ~1.3k tokens; the v1 drafter only speculates prompts that
/// fit one prefill chunk, so the cap must sit above the longest prompt.
const MAX_PREFILL_TOKENS: usize = 4096;

#[link(name = "cudart")]
unsafe extern "C" {
    #[link_name = "cudaProfilerStart"]
    fn cuda_profiler_start() -> i32;
    #[link_name = "cudaProfilerStop"]
    fn cuda_profiler_stop() -> i32;
}

fn env_path_or_default(var: &str, default: &str, probe: &str) -> Option<String> {
    match std::env::var(var) {
        Ok(path) => Some(path),
        Err(_) if Path::new(default).join(probe).exists() => Some(default.to_string()),
        Err(_) => None,
    }
}

fn read_gsm8k(path: &Path) -> Vec<(String, String)> {
    let text = std::fs::read_to_string(path).expect("read gsm8k jsonl");
    text.lines()
        .filter(|l| !l.trim().is_empty())
        .map(|line| {
            let v: serde_json::Value = serde_json::from_str(line).expect("parse gsm8k line");
            (
                v["question"].as_str().expect("question").to_string(),
                v["answer"].as_str().expect("answer").to_string(),
            )
        })
        .collect()
}

/// lm-eval `gsm8k` 8-shot prefix: `Question: ..\nAnswer: ..` exemplars joined by
/// blank lines, then the held-out question with a trailing `Answer:`.
fn build_prompt(fewshot: &[(String, String)], question: &str) -> String {
    let mut s = String::new();
    for (q, a) in fewshot {
        s.push_str(&format!("Question: {q}\nAnswer: {a}\n\n"));
    }
    s.push_str(&format!("Question: {question}\nAnswer:"));
    s
}

fn launch_options(draft: Option<PathBuf>) -> Qwen3LaunchOptions {
    Qwen3LaunchOptions {
        device_ordinal: 0,
        tp_size: 1,
        cuda_graph: true,
        dump_graph_png: None,
        offload: Qwen3OffloadOptions::disabled(),
        no_prefix_cache: true,
        max_prefill_tokens: MAX_PREFILL_TOKENS,
        memory: Qwen3MemoryOptions::new(
            0.85,
            DEFAULT_KV_CACHE_MEMORY_MARGIN_BYTES,
            DEFAULT_KV_PAGE_SIZE,
        )
        .validate()
        .expect("valid memory options"),
        lora: None,
        decode_overlap: DecodeOverlap::Off,
        batch_invariant: false,
        dflash_draft_model_path: None,
        eagle3_draft_model_path: draft,
    }
}

struct Timed {
    tokens: usize,
    ttft: Duration,
    total: Duration,
}

impl Timed {
    fn decode(&self) -> Duration {
        self.total.saturating_sub(self.ttft)
    }
}

/// One greedy request, folded by hand so the first-token instant is visible.
fn timed_generate(engine: &EngineHarness, prompt: &[u32]) -> Timed {
    let start = Instant::now();
    let mut stream = engine.submit(request(
        prompt.to_vec(),
        SamplingParams {
            ignore_eos: true,
            ..SamplingParams::default()
        },
        GENERATED_TOKENS,
    ));
    let mut tokens = 0usize;
    let mut ttft = None;
    loop {
        let update = stream
            .recv()
            .expect("engine closed the stream without a terminal");
        if !update.tokens.is_empty() && ttft.is_none() {
            ttft = Some(start.elapsed());
        }
        tokens += update.tokens.len();
        match update.terminal {
            None => {}
            Some(Terminal::Finished { .. }) => break,
            Some(Terminal::Rejected { reason, .. }) => panic!("generation rejected: {reason}"),
            Some(Terminal::Failed { message, .. }) => panic!("generation failed: {message}"),
        }
    }
    let total = start.elapsed();
    Timed {
        tokens,
        ttft: ttft.unwrap_or(total),
        total,
    }
}

fn run_mode(
    model_path: &str,
    draft: Option<PathBuf>,
    prompts: &[Vec<u32>],
    profile_range: bool,
) -> serde_json::Value {
    let spec = draft.is_some();
    let engine = EngineHarness::new(
        pegainfer_qwen3::launch(Path::new(model_path), launch_options(draft)).expect("engine"),
    );
    // Two warm-ups: graph capture for the plain path and the verify span.
    for p in prompts.iter().take(2) {
        let _ = timed_generate(&engine, p);
    }
    if profile_range {
        assert_eq!(unsafe { cuda_profiler_start() }, 0, "cudaProfilerStart");
    }
    let runs: Vec<Timed> = prompts.iter().map(|p| timed_generate(&engine, p)).collect();
    if profile_range {
        assert_eq!(unsafe { cuda_profiler_stop() }, 0, "cudaProfilerStop");
    }
    drop(engine);
    std::thread::sleep(Duration::from_secs(2));

    let tokens: usize = runs.iter().map(|r| r.tokens).sum();
    let total: f64 = runs.iter().map(|r| r.total.as_secs_f64()).sum();
    let decode_tokens: usize = runs.iter().map(|r| r.tokens.saturating_sub(1)).sum();
    let decode: f64 = runs.iter().map(|r| r.decode().as_secs_f64()).sum();
    let ttft_ms: f64 =
        runs.iter().map(|r| r.ttft.as_secs_f64() * 1e3).sum::<f64>() / runs.len() as f64;
    serde_json::json!({
        "engine": "pegainfer",
        "spec": spec,
        "gamma": if spec { EAGLE3_CHAIN_LENGTH } else { 0 },
        "num_prompts": runs.len(),
        "generated_tokens": tokens,
        "e2e_tok_s": tokens as f64 / total,
        "decode_tok_s": decode_tokens as f64 / decode,
        "tpot_ms": decode * 1e3 / decode_tokens as f64,
        "mean_ttft_ms": ttft_ms,
        "profiled": profile_range,
    })
}

#[test]
fn eagle3_gsm8k_single_stream_ab() {
    common::harness::init_capture_logging();
    let (Some(model_path), Some(draft_path), Some(gsm8k_dir)) = (
        env_path_or_default("PEGAINFER_TEST_MODEL_PATH", MODEL_PATH, "config.json"),
        env_path_or_default(
            "PEGAINFER_EAGLE3_TEST_MODEL_PATH",
            DRAFT_PATH,
            "config.json",
        ),
        env_path_or_default("PEGAINFER_GSM8K_DIR", GSM8K_DIR, "test.jsonl"),
    ) else {
        eprintln!(
            "skipping eagle3 GSM8K A/B: set PEGAINFER_TEST_MODEL_PATH + PEGAINFER_EAGLE3_TEST_MODEL_PATH \
             and provide models/gsm8k/{{train,test}}.jsonl (or PEGAINFER_GSM8K_DIR)"
        );
        return;
    };
    let mode = std::env::var("PEGAINFER_AB_MODE").unwrap_or_else(|_| "both".into());
    let num_prompts = std::env::var("PEGAINFER_AB_NUM_PROMPTS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(DEFAULT_NUM_PROMPTS);
    let profile_range = std::env::var("PEGAINFER_AB_PROFILE_RANGE").is_ok_and(|v| v == "1");
    let out_dir = std::env::var("PEGAINFER_AB_OUT").ok().map(PathBuf::from);

    let tokenizer = common::load_tokenizer(&model_path);
    let gsm8k = Path::new(&gsm8k_dir);
    let train = read_gsm8k(&gsm8k.join("train.jsonl"));
    let test = read_gsm8k(&gsm8k.join("test.jsonl"));
    let fewshot = &train[..NUM_FEWSHOT];
    let prompts: Vec<Vec<u32>> = test
        .iter()
        .take(num_prompts)
        .map(|(q, _)| {
            tokenizer
                .encode(&build_prompt(fewshot, q), false)
                .expect("encode failed")
        })
        .collect();
    let max_len = prompts.iter().map(Vec::len).max().unwrap_or(0);
    assert!(
        max_len <= MAX_PREFILL_TOKENS,
        "prompt of {max_len} tokens would be chunked and decode without EAGLE"
    );
    eprintln!(
        "GSM8K A/B: {} prompts, {NUM_FEWSHOT}-shot, max prompt {max_len} tokens, γ={EAGLE3_CHAIN_LENGTH}, \
         {GENERATED_TOKENS} tokens/request, bs=1",
        prompts.len()
    );

    if let Some(dir) = &out_dir {
        std::fs::create_dir_all(dir).expect("create PEGAINFER_AB_OUT");
        let mut f = std::fs::File::create(dir.join("prompts.jsonl")).expect("prompts.jsonl");
        for p in &prompts {
            writeln!(f, "{}", serde_json::json!({ "prompt_token_ids": p })).unwrap();
        }
        let meta = serde_json::json!({
            "gamma": EAGLE3_CHAIN_LENGTH,
            "generated_tokens": GENERATED_TOKENS,
            "max_model_len": MAX_PREFILL_TOKENS,
            "num_fewshot": NUM_FEWSHOT,
        });
        std::fs::write(dir.join("ab_config.json"), meta.to_string()).unwrap();
    }

    let mut results = Vec::new();
    if mode == "both" || mode == "off" {
        results.push(run_mode(
            &model_path,
            None,
            &prompts,
            profile_range && mode == "off",
        ));
    }
    if mode == "both" || mode == "on" {
        results.push(run_mode(
            &model_path,
            Some(PathBuf::from(&draft_path)),
            &prompts,
            profile_range && mode == "on",
        ));
    }

    eprintln!(
        "────────── EAGLE-3 GSM8K single-stream A/B (pegainfer, γ={EAGLE3_CHAIN_LENGTH}) ──────────"
    );
    for r in &results {
        eprintln!(
            "  spec {:<3}  decode {:7.1} tok/s  TPOT {:6.2} ms  TTFT {:6.1} ms  e2e {:7.1} tok/s",
            if r["spec"].as_bool().unwrap() {
                "ON"
            } else {
                "OFF"
            },
            r["decode_tok_s"].as_f64().unwrap(),
            r["tpot_ms"].as_f64().unwrap(),
            r["mean_ttft_ms"].as_f64().unwrap(),
            r["e2e_tok_s"].as_f64().unwrap(),
        );
    }
    if let [off, on] = results.as_slice() {
        let speedup = on["decode_tok_s"].as_f64().unwrap() / off["decode_tok_s"].as_f64().unwrap();
        eprintln!("  decode speedup: {speedup:.2}×");
        assert!(
            speedup > 0.5,
            "speculative decode catastrophically slower ({speedup:.2}×)"
        );
    }
    if let Some(dir) = &out_dir {
        for r in &results {
            let name = if r["spec"].as_bool().unwrap() {
                "on"
            } else {
                "off"
            };
            std::fs::write(
                dir.join(format!(
                    "pegainfer_{name}{}.json",
                    if r["profiled"].as_bool().unwrap() {
                        "_profiled"
                    } else {
                        ""
                    }
                )),
                serde_json::to_string_pretty(r).unwrap(),
            )
            .unwrap();
        }
    }
}
