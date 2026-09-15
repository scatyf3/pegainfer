//! EAGLE-3 speculative-decoding throughput A/B (same head, speculative off vs on).
//!
//! Measures end-to-end decode tok/s for a fixed token budget at concurrency 1 and
//! 4 on the same prompts and hardware. A measurement harness, not a pass/fail
//! gate: it only guards against a catastrophic slowdown. Read the printed numbers
//! with `--nocapture`. Per-position acceptance lives in the spec-decode counters
//! (`num_accepted_tokens_per_pos`) the server exports; a serving-load A/B belongs
//! to `vllm bench serve`.
//!
//! Requires a CUDA GPU, Qwen3-4B weights, and the EAGLE-3 drafter. Set
//! `PEGAINFER_TEST_MODEL_PATH` + `PEGAINFER_EAGLE3_TEST_MODEL_PATH`; skips when
//! either is absent.

use std::path::Path;
use std::path::PathBuf;
use std::time::Duration;
use std::time::Instant;

use pegainfer_frontend::sampler::SamplingParams;
use pegainfer_qwen3::DEFAULT_KV_CACHE_MEMORY_MARGIN_BYTES;
use pegainfer_qwen3::DEFAULT_KV_PAGE_SIZE;
use pegainfer_qwen3::DEFAULT_MAX_PREFILL_TOKENS;
use pegainfer_qwen3::DecodeOverlap;
use pegainfer_qwen3::Qwen3LaunchOptions;
use pegainfer_qwen3::Qwen3MemoryOptions;
use pegainfer_qwen3::Qwen3OffloadOptions;

mod common;

use common::harness::EngineHarness;
use common::harness::request;

const MODEL_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../models/Qwen3-4B");
const DRAFT_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../models/Qwen3-4B_eagle3");
const GENERATED_TOKENS: usize = 256;
const CONCURRENCY: [usize; 2] = [1, 4];

fn target_path_or_skip() -> Option<String> {
    match std::env::var("PEGAINFER_TEST_MODEL_PATH") {
        Ok(path) => Some(path),
        Err(_) if Path::new(MODEL_PATH).join("config.json").exists() => {
            Some(MODEL_PATH.to_string())
        }
        Err(_) => None,
    }
}

fn draft_path_or_skip() -> Option<String> {
    match std::env::var("PEGAINFER_EAGLE3_TEST_MODEL_PATH") {
        Ok(path) => Some(path),
        Err(_) if Path::new(DRAFT_PATH).join("config.json").exists() => {
            Some(DRAFT_PATH.to_string())
        }
        Err(_) => None,
    }
}

fn launch_options(draft: Option<PathBuf>) -> Qwen3LaunchOptions {
    Qwen3LaunchOptions {
        device_ordinal: 0,
        tp_size: 1,
        cuda_graph: true,
        dump_graph_png: None,
        offload: Qwen3OffloadOptions::disabled(),
        no_prefix_cache: true,
        max_prefill_tokens: DEFAULT_MAX_PREFILL_TOKENS,
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

/// Run `concurrency` greedy requests at once (cycling the prompts) and return
/// the aggregate generated tok/s.
fn run_batch(engine: &EngineHarness, prompts: &[Vec<u32>], concurrency: usize) -> f64 {
    let start = Instant::now();
    let streams: Vec<_> = prompts
        .iter()
        .cycle()
        .take(concurrency)
        .map(|prompt| {
            engine.submit(request(
                prompt.clone(),
                SamplingParams {
                    ignore_eos: true,
                    ..SamplingParams::default()
                },
                GENERATED_TOKENS,
            ))
        })
        .collect();
    let tokens: usize = streams
        .into_iter()
        .map(|stream| stream.expect_finished().tokens.len())
        .sum();
    tokens as f64 / start.elapsed().as_secs_f64()
}

/// tok/s per concurrency level, after one discarded warm-up batch each.
fn measure(engine: &EngineHarness, prompts: &[Vec<u32>]) -> Vec<f64> {
    CONCURRENCY
        .iter()
        .map(|&concurrency| {
            let _ = run_batch(engine, prompts, concurrency);
            let rounds = 3;
            (0..rounds)
                .map(|_| run_batch(engine, prompts, concurrency))
                .sum::<f64>()
                / f64::from(rounds)
        })
        .collect()
}

#[test]
fn eagle3_speculative_throughput_ab() {
    let (Some(model_path), Some(draft_path)) = (target_path_or_skip(), draft_path_or_skip()) else {
        eprintln!(
            "skipping eagle3 perf A/B: set PEGAINFER_TEST_MODEL_PATH + PEGAINFER_EAGLE3_TEST_MODEL_PATH"
        );
        return;
    };

    let tokenizer = common::load_tokenizer(&model_path);
    let prompts: Vec<Vec<u32>> = [
        "Write a short essay about the history of the Roman Empire.",
        "Explain how a transformer neural network works, step by step.",
        "List ten facts about the planet Mars and describe each one.",
        "Q: A train travels 60 km in 45 minutes. What is its average speed in km/h? A:",
    ]
    .iter()
    .map(|p| tokenizer.encode(p, false).expect("encode failed"))
    .collect();

    let baseline = {
        let engine = EngineHarness::new(
            pegainfer_qwen3::launch(Path::new(&model_path), launch_options(None))
                .expect("baseline engine"),
        );
        let tps = measure(&engine, &prompts);
        drop(engine);
        std::thread::sleep(Duration::from_secs(2));
        tps
    };
    let spec = {
        let engine = EngineHarness::new(
            pegainfer_qwen3::launch(
                Path::new(&model_path),
                launch_options(Some(PathBuf::from(&draft_path))),
            )
            .expect("speculative engine"),
        );
        measure(&engine, &prompts)
    };

    eprintln!("───────────── EAGLE-3 decode A/B (same head) ─────────────");
    eprintln!("  concurrency   spec OFF tok/s   spec ON tok/s   speedup");
    for (i, concurrency) in CONCURRENCY.iter().enumerate() {
        eprintln!(
            "  c{concurrency:<10} {:>14.1} {:>15.1} {:>8.2}×",
            baseline[i],
            spec[i],
            spec[i] / baseline[i]
        );
    }
    eprintln!("──────────────────────────────────────────────────────────");

    for (i, concurrency) in CONCURRENCY.iter().enumerate() {
        let speedup = spec[i] / baseline[i];
        assert!(
            speedup > 0.5,
            "c{concurrency}: speculative decode is catastrophically slower ({speedup:.2}×)"
        );
    }
}
