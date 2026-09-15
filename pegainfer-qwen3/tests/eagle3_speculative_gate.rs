//! EAGLE-3 speculative-decoding losslessness gate.
//!
//! The draft/verify transaction is shared with DFlash: every committed token is
//! the verify forward's own pick, so the EAGLE-3 chain can only change *how many*
//! tokens ride one step, never *which*. This gate checks that end to end with the
//! same regret test as `dflash_speculative_gate`: an exact `spec == baseline`
//! match is the wrong oracle (the verify path runs the prefill attention kernel,
//! plain decode the decode kernel, and a bf16 near-tie can flip), so at the first
//! divergence — where both runs still share a context — the speculative pick must
//! sit within `MARGIN_TOL` of the argmax in the *prefill* kernel's distribution.
//!
//! Beyond the single-request gate this covers:
//! - concurrent requests with staggered budgets: bs>1 drafting and verify spans
//!   truncated to each request's remaining budget;
//! - a sampled request batched with a greedy one: the batch falls back to plain
//!   decode and the greedy request's draft state is dropped mid-flight;
//! - a request aborted mid-speculation, then a fresh request on the same engine.
//!
//! Requires a CUDA GPU, Qwen3-4B weights, and the EAGLE-3 drafter. Set
//! `PEGAINFER_TEST_MODEL_PATH` (target) and `PEGAINFER_EAGLE3_TEST_MODEL_PATH`
//! (drafter); skips cleanly when either is absent. Baseline and speculative
//! engines run sequentially so only one Qwen3-4B is resident at a time.

use std::path::Path;
use std::path::PathBuf;
use std::time::Duration;

use pegainfer_frontend::sampler::SamplingParams;
use pegainfer_qwen3::DEFAULT_KV_CACHE_MEMORY_MARGIN_BYTES;
use pegainfer_qwen3::DEFAULT_KV_PAGE_SIZE;
use pegainfer_qwen3::DEFAULT_MAX_PREFILL_TOKENS;
use pegainfer_qwen3::DecodeOverlap;
use pegainfer_qwen3::Qwen3LaunchOptions;
use pegainfer_qwen3::Qwen3MemoryOptions;
use pegainfer_qwen3::Qwen3OffloadOptions;
use vllm_text::tokenizer::DynTokenizer;

mod common;

use common::harness::EngineHarness;
use common::harness::Outcome;
use common::harness::request;

const MODEL_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../models/Qwen3-4B");
const DRAFT_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../models/Qwen3-4B_eagle3");
const GENERATED_TOKENS: usize = 64;
/// Top-K logprobs requested from the baseline; wide enough that the speculative
/// pick is in the set on any real tie.
const LOGPROBS: usize = 20;
/// Max acceptable regret at a divergence, in the prefill distribution. Same band
/// as the DFlash gate: EAGLE-3 reuses the verify path unchanged, so a wider band
/// would need its own evidence.
const MARGIN_TOL: f32 = 0.20;

/// Every test launches a Qwen3-4B engine and two at once overflow a 16 GB card;
/// serialize the engine-holding bodies.
static GPU: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn target_path_or_skip() -> Option<String> {
    match std::env::var("PEGAINFER_TEST_MODEL_PATH") {
        Ok(path) => Some(path),
        Err(_) if Path::new(MODEL_PATH).join("config.json").exists() => {
            Some(MODEL_PATH.to_string())
        }
        Err(_) => {
            eprintln!(
                "skipping eagle3 gate: {MODEL_PATH}/config.json missing; set PEGAINFER_TEST_MODEL_PATH"
            );
            None
        }
    }
}

fn draft_path_or_skip() -> Option<String> {
    match std::env::var("PEGAINFER_EAGLE3_TEST_MODEL_PATH") {
        Ok(path) => Some(path),
        Err(_) if Path::new(DRAFT_PATH).join("config.json").exists() => {
            Some(DRAFT_PATH.to_string())
        }
        Err(_) => {
            eprintln!(
                "skipping eagle3 gate: {DRAFT_PATH}/config.json missing; set PEGAINFER_EAGLE3_TEST_MODEL_PATH"
            );
            None
        }
    }
}

fn launch_options(draft: Option<PathBuf>) -> Qwen3LaunchOptions {
    Qwen3LaunchOptions {
        device_ordinal: 0,
        tp_size: 1,
        cuda_graph: true,
        dump_graph_png: None,
        offload: Qwen3OffloadOptions::disabled(),
        // The speculative engine forces the prefix cache off; match it on the
        // baseline so both take the same cold prefill path.
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

fn baseline_engine(model_path: &str) -> EngineHarness {
    EngineHarness::new(
        pegainfer_qwen3::launch(Path::new(model_path), launch_options(None))
            .expect("failed to start baseline engine"),
    )
}

fn speculative_engine(model_path: &str, draft_path: &str) -> EngineHarness {
    EngineHarness::new(
        pegainfer_qwen3::launch(
            Path::new(model_path),
            launch_options(Some(PathBuf::from(draft_path))),
        )
        .expect("failed to start speculative engine"),
    )
}

/// Drop the baseline engine and let its scheduler thread free GPU memory before
/// the speculative engine loads the same target.
fn release(engine: EngineHarness) {
    drop(engine);
    std::thread::sleep(Duration::from_secs(2));
}

/// One decoded position: the chosen token and (when requested) the top-K
/// `(token, logprob)` distribution that produced it.
struct Step {
    id: u32,
    top_logprobs: Vec<(u32, f32)>,
}

fn to_steps(outcome: Outcome) -> Vec<Step> {
    let Outcome {
        tokens, logprobs, ..
    } = outcome;
    tokens
        .into_iter()
        .enumerate()
        .map(|(i, id)| Step {
            id,
            top_logprobs: logprobs
                .get(i)
                .and_then(Option::as_ref)
                .map(|lp| lp.top_logprobs.clone())
                .unwrap_or_default(),
        })
        .collect()
}

/// Submit one greedy request and collect its decoded steps.
fn generate(
    engine: &EngineHarness,
    prompt_tokens: Vec<u32>,
    logprobs: usize,
    max_tokens: usize,
) -> Vec<Step> {
    let mut req = request(prompt_tokens, SamplingParams::default(), max_tokens);
    req.logprobs = (logprobs > 0).then_some(logprobs);
    to_steps(engine.submit(req).expect_finished())
}

/// Prefill `context` with prompt scoring and return the first generated token's
/// distribution — the prefill kernel's reference for the verify path.
fn prefill_next(engine: &EngineHarness, context: Vec<u32>, logprobs: usize) -> Step {
    let mut req = request(context, SamplingParams::default(), 1);
    req.logprobs = (logprobs > 0).then_some(logprobs);
    req.prompt_logprobs = Some(logprobs);
    let mut generated = to_steps(engine.submit(req).expect_finished());
    assert!(
        !generated.is_empty(),
        "prompt-scoring prefill finished without a token"
    );
    generated.swap_remove(0)
}

/// Compare speculative `spec` steps against plain-greedy `base`, tolerating only
/// a benign tie flip. `engine` must be the live speculative engine: at a
/// divergence it re-prefills the shared context for the prefill-kernel reference.
fn check_lossless(
    engine: &EngineHarness,
    tokenizer: &DynTokenizer,
    label: &str,
    prompt_tokens: &[u32],
    base: &[Step],
    spec: &[Step],
) -> Result<(), String> {
    let matched = base
        .iter()
        .zip(spec)
        .take_while(|(b, s)| b.id == s.id)
        .count();
    if matched == base.len().min(spec.len()) {
        eprintln!(
            "{label}: {matched}/{} tokens identical (100% lossless)",
            base.len()
        );
        return Ok(());
    }

    let spec_id = spec[matched].id;
    let decode_argmax = base[matched].top_logprobs[0].0;
    let lo = matched.saturating_sub(2);
    let hi = (matched + 3).min(base.len()).min(spec.len());
    let decode = |steps: &[Step]| -> Vec<(u32, String)> {
        steps
            .iter()
            .map(|s| (s.id, tokenizer.decode(&[s.id], false).unwrap_or_default()))
            .collect()
    };
    eprintln!("  [diag] {label} matched={matched}");
    eprintln!("  [diag] base[{lo}..{hi}] = {:?}", decode(&base[lo..hi]));
    eprintln!("  [diag] spec[{lo}..{hi}] = {:?}", decode(&spec[lo..hi]));

    let mut context = prompt_tokens.to_vec();
    context.extend(base[..matched].iter().map(|s| s.id));
    let prefill_ref = prefill_next(engine, context, LOGPROBS);
    if prefill_ref.id == spec_id {
        eprintln!(
            "{label}: kernel-gap flip at token {matched} — verify(prefill)→{spec_id}, \
             decode→{decode_argmax}; spec matches prefill greedy. Not a spec bug."
        );
        return Ok(());
    }

    let prefill_regret = prefill_ref
        .top_logprobs
        .iter()
        .find(|(t, _)| *t == spec_id)
        .map(|(_, lp)| prefill_ref.top_logprobs[0].1 - lp);
    if let Some(regret) = prefill_regret
        && regret <= MARGIN_TOL
    {
        eprintln!(
            "{label}: tie flip at token {matched} — verify(prefill)→{}, spec→{spec_id}, \
             decode→{decode_argmax}; regret {regret:.3} ≤ {MARGIN_TOL}. Not a spec bug.",
            prefill_ref.id,
        );
        return Ok(());
    }
    Err(format!(
        "{label}: at token {matched} spec chose {spec_id} but prefill greedy says {} and \
         decode greedy says {decode_argmax} (spec regret in prefill dist: {prefill_regret:?} > \
         {MARGIN_TOL}) — real spec bug",
        prefill_ref.id,
    ))
}

fn assert_no_failures(failures: &[String], what: &str) {
    assert!(
        failures.is_empty(),
        "{what} is not lossless:\n{}",
        failures.join("\n")
    );
}

#[test]
fn eagle3_speculative_greedy_matches_plain_greedy() {
    common::harness::init_capture_logging();
    let (Some(model_path), Some(draft_path)) = (target_path_or_skip(), draft_path_or_skip()) else {
        return;
    };
    let _gpu = GPU
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);

    let prompts = [
        "The capital of France is",
        "Here is a short story about a dragon. Once upon a time",
        "def fibonacci(n):",
        "Q: What is 17 multiplied by 23? A: Let's think step by step.",
        "The three primary colors are",
    ];
    let tokenizer = common::load_tokenizer(&model_path);
    let encoded: Vec<Vec<u32>> = prompts
        .iter()
        .map(|p| tokenizer.encode(p, false).expect("encode failed"))
        .collect();

    let baseline: Vec<Vec<Step>> = {
        let engine = baseline_engine(&model_path);
        let out = encoded
            .iter()
            .map(|t| generate(&engine, t.clone(), LOGPROBS, GENERATED_TOKENS))
            .collect();
        release(engine);
        out
    };

    let engine = speculative_engine(&model_path, &draft_path);
    let mut failures = Vec::new();
    for (i, prompt) in prompts.iter().enumerate() {
        let spec = generate(&engine, encoded[i].clone(), 0, GENERATED_TOKENS);
        if let Err(failure) = check_lossless(
            &engine,
            &tokenizer,
            &format!("prompt {i} ({prompt:?})"),
            &encoded[i],
            &baseline[i],
            &spec,
        ) {
            failures.push(failure);
        }
    }
    drop(engine);
    assert_no_failures(&failures, "EAGLE-3 speculative greedy decode");
}

/// Concurrent greedy requests with staggered budgets: at any step the batch
/// mixes full and budget-truncated verify spans, and the lane drafts bs>1.
#[test]
fn eagle3_concurrent_heterogeneous_is_lossless() {
    common::harness::init_capture_logging();
    let (Some(model_path), Some(draft_path)) = (target_path_or_skip(), draft_path_or_skip()) else {
        return;
    };
    let _gpu = GPU
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);

    let cases: [(&str, usize); 4] = [
        ("def fibonacci(n):", 64),
        ("The three primary colors are", 24),
        (
            "Q: What is 17 multiplied by 23? A: Let's think step by step.",
            48,
        ),
        ("Here is a short story about a dragon. Once upon a time", 7),
    ];
    let tokenizer = common::load_tokenizer(&model_path);
    let encoded: Vec<Vec<u32>> = cases
        .iter()
        .map(|(p, _)| tokenizer.encode(p, false).expect("encode failed"))
        .collect();

    let baselines: Vec<Vec<Step>> = {
        let engine = baseline_engine(&model_path);
        let out = encoded
            .iter()
            .zip(&cases)
            .map(|(t, (_, max_tokens))| generate(&engine, t.clone(), LOGPROBS, *max_tokens))
            .collect();
        release(engine);
        out
    };

    let engine = speculative_engine(&model_path, &draft_path);
    let streams: Vec<_> = encoded
        .iter()
        .zip(&cases)
        .map(|(t, (_, max_tokens))| {
            engine.submit(request(t.clone(), SamplingParams::default(), *max_tokens))
        })
        .collect();
    let specs: Vec<Vec<Step>> = streams
        .into_iter()
        .map(|stream| to_steps(stream.expect_finished()))
        .collect();

    let mut failures = Vec::new();
    for (i, (prompt, _)) in cases.iter().enumerate() {
        if let Err(failure) = check_lossless(
            &engine,
            &tokenizer,
            &format!("concurrent {i} ({prompt:?})"),
            &encoded[i],
            &baselines[i],
            &specs[i],
        ) {
            failures.push(failure);
        }
    }
    drop(engine);
    assert_no_failures(&failures, "concurrent EAGLE-3 speculative decode");
}

/// Request-lifecycle edges on one speculative engine: an abort mid-speculation
/// must release the draft state cleanly, and a sampled request batched with a
/// greedy one must push the batch onto plain decode without corrupting the greedy
/// output.
#[test]
fn eagle3_abort_and_mixed_batch_stay_lossless() {
    common::harness::init_capture_logging();
    let (Some(model_path), Some(draft_path)) = (target_path_or_skip(), draft_path_or_skip()) else {
        return;
    };
    let _gpu = GPU
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);

    let victim_prompt = "Q: What is 17 multiplied by 23? A: Let's think step by step.";
    let sampled_prompt = "Write a haiku about the ocean.";
    let tokenizer = common::load_tokenizer(&model_path);
    let victim_tokens = tokenizer
        .encode(victim_prompt, false)
        .expect("encode failed");
    let sampled_tokens = tokenizer
        .encode(sampled_prompt, false)
        .expect("encode failed");

    let baseline = {
        let engine = baseline_engine(&model_path);
        let out = generate(&engine, victim_tokens.clone(), LOGPROBS, GENERATED_TOKENS);
        release(engine);
        out
    };

    let engine = speculative_engine(&model_path, &draft_path);
    let mut failures = Vec::new();

    // 1. Abort once the request has emitted tokens, so it holds live draft state.
    let mut aborted = engine.submit(request(
        victim_tokens.clone(),
        SamplingParams::default(),
        GENERATED_TOKENS,
    ));
    loop {
        let update = aborted
            .recv()
            .expect("engine closed the stream before any token");
        assert!(
            update.terminal.is_none(),
            "request finished before it could be aborted"
        );
        if !update.tokens.is_empty() {
            break;
        }
    }
    aborted.control.abort();
    let aborted_outcome = aborted.outcome();
    eprintln!(
        "aborted request stopped after {} token(s)",
        aborted_outcome.tokens.len()
    );

    let after_abort = generate(&engine, victim_tokens.clone(), 0, GENERATED_TOKENS);
    if let Err(failure) = check_lossless(
        &engine,
        &tokenizer,
        "after abort",
        &victim_tokens,
        &baseline,
        &after_abort,
    ) {
        failures.push(failure);
    }

    // 2. A sampled request shares the batch with the greedy victim.
    let victim = engine.submit(request(
        victim_tokens.clone(),
        SamplingParams::default(),
        GENERATED_TOKENS,
    ));
    let sampled = engine.submit(request(
        sampled_tokens,
        SamplingParams {
            temperature: 0.8,
            seed: Some(7),
            ..SamplingParams::default()
        },
        GENERATED_TOKENS,
    ));
    let mixed = to_steps(victim.expect_finished());
    let sampled_outcome = sampled.expect_finished();
    assert!(
        !sampled_outcome.tokens.is_empty(),
        "sampled request produced no tokens"
    );
    if let Err(failure) = check_lossless(
        &engine,
        &tokenizer,
        "greedy batched with sampled",
        &victim_tokens,
        &baseline,
        &mixed,
    ) {
        failures.push(failure);
    }

    drop(engine);
    assert_no_failures(&failures, "EAGLE-3 lifecycle edges");
}
