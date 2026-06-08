//! In-process inference benchmark CLI.
//!
//! Usage:
//!   cargo run -r --bin bench_serving -- [GLOBAL_OPTIONS] <SUBCOMMAND> [OPTIONS]
//!
//! Examples:
//!   cargo run -r --bin bench_serving -- request --prompt "Tell me a story" --output-len 128
//!   cargo run -r --bin bench_serving -- request --prompt-len 512 --output-len 64
//!   cargo run -r --bin bench_serving -- matrix --prompt-lens 32,128,512 --output-lens 32,128
//!   cargo run -r --bin bench_serving -- curve --prompt-len 1024 --output-len 256 --window 32

use std::fmt::Write as _;
use std::fs;
use std::io::{IsTerminal, stdout};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::thread;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, ensure};
use clap::{Args as ClapArgs, Parser, Subcommand, ValueEnum};
use comfy_table::modifiers::UTF8_ROUND_CORNERS;
use comfy_table::presets::{ASCII_FULL_CONDENSED, UTF8_FULL_CONDENSED};
use comfy_table::{Cell, CellAlignment, Table};
use cudarc::driver::Profiler;
use cudarc::runtime::result::device as cuda_device;
use log::{debug, info};
use pegainfer::logging;
use pegainfer::sampler::SamplingParams;
use pegainfer::scheduler::{SchedulerHandle, SchedulerRequest, TokenEvent};
use pegainfer::server_engine::{ModelType, detect_model_type};
use pegainfer_core::{
    engine::{EngineLoadOptions, EpBackend},
    parallel::ParallelConfig,
};
use pegainfer_vllm_support::load_tokenizer as load_vllm_tokenizer;
use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;
use vllm_text::tokenizer::DynTokenizer;

const SNAPSHOT_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../bench_snapshots");
const SNAPSHOT_PREFILL_OUTPUT_LEN: usize = 1;
const SNAPSHOT_DECODE_PROMPT_LEN: usize = 1024;
const SNAPSHOT_DECODE_OUTPUT_LEN: usize = 256;

fn snapshot_prefill_prompt_len(model_type: ModelType) -> usize {
    match model_type {
        // Kimi serves TP1/DP8, where the PPLX fabric buffers cap prompts at
        // 2048 tokens (full-lifetime KV cap is 8192) — probe the largest
        // prompt the serving shape admits.
        #[cfg(feature = "kimi-k2")]
        ModelType::KimiK2 => 2_048,
        _ => 10_000,
    }
}
const REGRESSION_TPOT_PCT: f64 = 2.0;
const REGRESSION_TTFT_PCT: f64 = 3.0;

const DEFAULT_MODEL_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../models/Qwen3-4B");
const DEFAULT_REQUEST_PROMPT: &str = "Tell me a story";
const DEFAULT_CURVE_PROMPT_LEN: usize = 512;
const SYNTHETIC_PATTERN: &str = "token_id = 100 + (idx % 1000)";
const TOP_LEVEL_EXAMPLES: &str = "\
Examples:
  cargo run -r --bin bench_serving -- request
  cargo run -r --bin bench_serving -- request --prompt \"Tell me a story about Rust\" --output-len 128
  cargo run -r --bin bench_serving -- request --prompt-len 512 --output-len 64
  cargo run -r --bin bench_serving -- matrix --prompt-lens 32,128,512,2048 --output-lens 32,128,256
  cargo run -r --bin bench_serving -- curve --prompt-len 1024 --output-len 256 --window 32
  cargo run -r --bin bench_serving -- --format json --out bench.json request --prompt-len 512 --output-len 64
  cargo run -r --bin bench_serving -- snapshot
  cargo run -r --bin bench_serving -- compare bench_snapshots/rtx-5070-ti/qwen3-4b.json";
const REQUEST_EXAMPLES: &str = "\
Examples:
  cargo run -r --bin bench_serving -- request
  cargo run -r --bin bench_serving -- request --prompt \"Tell me a story about Rust\" --output-len 128
  cargo run -r --bin bench_serving -- request --prompt-file prompts/story.txt --output-len 128
  cargo run -r --bin bench_serving -- request --prompt-len 512 --output-len 64 --warmup 3 --iters 10";
const MATRIX_EXAMPLES: &str = "\
Examples:
  cargo run -r --bin bench_serving -- matrix
  cargo run -r --bin bench_serving -- matrix --prompt-lens 32,128,512,2048 --output-lens 32,128,256
  cargo run -r --bin bench_serving -- --format json --out matrix.json matrix --prompt-lens 128,512 --output-lens 64,256";
const CURVE_EXAMPLES: &str = "\
Examples:
  cargo run -r --bin bench_serving -- curve
  cargo run -r --bin bench_serving -- curve --prompt-len 1024 --output-len 256 --window 32
  cargo run -r --bin bench_serving -- curve --prompt \"Summarize KV cache behavior\" --output-len 128 --window 16";
const SNAPSHOT_EXAMPLES: &str = "\
Examples:
  cargo run -r --bin bench_serving -- snapshot
  cargo run -r --bin bench_serving -- snapshot --warmup 3 --iters 10";
const COMPARE_EXAMPLES: &str = "\
Examples:
  cargo run -r --bin bench_serving -- compare bench_snapshots/rtx-5070-ti/qwen3-4b.json
  cargo run -r --bin bench_serving -- compare bench_snapshots/rtx-5070-ti/qwen3-4b.json --baseline HEAD~3";
const MIXED_EXAMPLES: &str = "\
Examples:
  cargo run -r --bin bench_serving -- mixed
  cargo run -r --bin bench_serving -- mixed --bg-concurrency 8 --qps 0.5 --num-injections 10
  cargo run -r --bin bench_serving -- mixed --bg-concurrency 2 --bg-output-len 512 \\
    --inj-prompt-len 4000 --qps 1.0 --num-injections 3 --warmup 2
  cargo run -r --bin bench_serving -- --format json --out mixed.json mixed --skip-baseline";

#[derive(Debug, Clone, Copy, ValueEnum)]
enum OutputFormat {
    Text,
    Json,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum CliEpBackend {
    Nccl,
    #[value(name = "deepep")]
    DeepEp,
}

impl From<CliEpBackend> for EpBackend {
    fn from(value: CliEpBackend) -> Self {
        match value {
            CliEpBackend::Nccl => Self::Nccl,
            CliEpBackend::DeepEp => Self::DeepEp,
        }
    }
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Measure one request shape end-to-end.
    #[command(after_help = REQUEST_EXAMPLES)]
    Request(RequestArgs),
    /// Sweep prompt_len x output_len and summarize each cell.
    #[command(after_help = MATRIX_EXAMPLES)]
    Matrix(MatrixArgs),
    /// Measure TPOT as context grows during decode.
    #[command(after_help = CURVE_EXAMPLES)]
    Curve(CurveArgs),
    /// Run standard profiles and write a regression-trackable snapshot.
    #[command(after_help = SNAPSHOT_EXAMPLES)]
    Snapshot(SnapshotArgs),
    /// Compare a snapshot against its git baseline.
    #[command(after_help = COMPARE_EXAMPLES)]
    Compare(CompareArgs),
    /// Measure decode ITL while long prompts arrive at low QPS (mixed load).
    #[command(after_help = MIXED_EXAMPLES)]
    Mixed(MixedArgs),
}

#[derive(Parser, Debug)]
#[command(
    name = "bench_serving",
    about = "pegainfer in-process inference benchmark",
    after_help = TOP_LEVEL_EXAMPLES
)]
struct Cli {
    /// Model directory (contains config.json, tokenizer, safetensors)
    #[arg(long, default_value = DEFAULT_MODEL_PATH)]
    model_path: String,

    /// Enable CUDA graph on decode path
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    cuda_graph: bool,

    /// Render result to terminal as text or structured JSON
    #[arg(long, default_value = "text")]
    format: OutputFormat,

    /// Optional label to tag this benchmark run
    #[arg(long)]
    label: Option<String>,

    /// Optional output path for the rendered report
    #[arg(long)]
    out: Option<String>,

    /// Capture only measured iterations for nsys `-c cudaProfilerApi`
    #[arg(long, default_value_t = false)]
    cuda_profiler_capture: bool,

    /// Tensor-parallel world size for Kimi-K2
    #[arg(long, default_value_t = 1)]
    tp_size: usize,

    /// Data-parallel world size for Kimi-K2
    #[arg(long, default_value_t = 8)]
    dp_size: usize,

    /// Expert-parallel backend for Kimi-K2 (TP1/DP8 requires deepep; TP8/DP1 requires nccl)
    #[arg(long, default_value = "deepep")]
    ep_backend: CliEpBackend,

    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Clone, ClapArgs)]
struct PromptInputArgs {
    /// Inline prompt text
    #[arg(long, conflicts_with_all = ["prompt_file", "prompt_len"])]
    prompt: Option<String>,

    /// Read prompt text from file
    #[arg(long, conflicts_with_all = ["prompt", "prompt_len"])]
    prompt_file: Option<String>,

    /// Use a synthetic prompt with exactly this many token ids
    #[arg(long, conflicts_with_all = ["prompt", "prompt_file"])]
    prompt_len: Option<usize>,
}

#[derive(Debug, Clone, ClapArgs)]
struct RunArgs {
    /// Warmup iterations
    #[arg(long, default_value_t = 5)]
    warmup: usize,

    /// Measured iterations
    #[arg(long, default_value_t = 20)]
    iters: usize,

    /// RNG seed (matters once sampling becomes non-greedy)
    #[arg(long, default_value_t = 42)]
    seed: u64,
}

#[derive(Debug, ClapArgs)]
struct RequestArgs {
    #[command(flatten)]
    prompt_input: PromptInputArgs,

    /// Max generated tokens
    #[arg(long, default_value_t = 64)]
    output_len: usize,

    /// Number of concurrent requests per measured iteration
    #[arg(long, default_value_t = 1)]
    concurrency: usize,

    /// Number of *distinct* synthetic prompts to tile across the concurrent
    /// batch (0 = one per request, fully diverse). `1` makes every concurrent
    /// request identical, which collapses MoE routing onto a narrow expert set
    /// and under-measures decode TPOT — sweep this to quantify the
    /// routing-diversity → TPOT curve (see the MoE bench-diversity lesson).
    #[arg(long, default_value_t = 0)]
    distinct_prompts: usize,

    #[command(flatten)]
    run: RunArgs,
}

#[derive(Debug, ClapArgs)]
struct MatrixArgs {
    /// Synthetic prompt lengths to sweep
    #[arg(long, value_delimiter = ',', default_value = "32,128,512,2048")]
    prompt_lens: Vec<usize>,

    /// Output lengths to sweep
    #[arg(long, value_delimiter = ',', default_value = "32,128,256")]
    output_lens: Vec<usize>,

    #[command(flatten)]
    run: RunArgs,
}

#[derive(Debug, ClapArgs)]
struct CurveArgs {
    #[command(flatten)]
    prompt_input: PromptInputArgs,

    /// Max generated tokens
    #[arg(long, default_value_t = 256)]
    output_len: usize,

    /// Group decode positions into windows of this size
    #[arg(long, default_value_t = 32)]
    window: usize,

    #[command(flatten)]
    run: RunArgs,
}

#[derive(Debug, ClapArgs)]
struct SnapshotArgs {
    #[command(flatten)]
    run: RunArgs,
}

#[derive(Debug, ClapArgs)]
struct CompareArgs {
    /// Path to snapshot JSON file
    path: String,

    /// Git ref to compare against
    #[arg(long, default_value = "HEAD")]
    baseline: String,
}

#[derive(Debug, ClapArgs)]
struct MixedArgs {
    /// Prompt length of each background decode stream (decode-heavy steady state)
    #[arg(long, default_value_t = 1024)]
    bg_prompt_len: usize,

    /// Number of long-lived background decode streams kept active for the run
    #[arg(long, default_value_t = 8)]
    bg_concurrency: usize,

    /// Max generated tokens per background stream (size to outlast the whole run)
    #[arg(long, default_value_t = 8192)]
    bg_output_len: usize,

    /// Prompt length of each injected long prompt (the prefill that stalls decode)
    #[arg(long, default_value_t = 10_000)]
    inj_prompt_len: usize,

    /// Max generated tokens per injected prompt (1 = prefill-dominated)
    #[arg(long, default_value_t = 1)]
    inj_output_len: usize,

    /// Arrival rate of injected long prompts, in requests per second
    #[arg(long, default_value_t = 0.5)]
    qps: f64,

    /// Number of long prompts to inject; bounds the run length
    #[arg(long, default_value_t = 10)]
    num_injections: usize,

    /// Skip the decode-only baseline control (only measure the mixed run)
    #[arg(long, default_value_t = false)]
    skip_baseline: bool,

    /// Fraction of injections that reuse a shared prompt and so hit the prefix
    /// cache (warm prefill, ~no stall); the rest get distinct prompts (cold,
    /// worst-case stall). 0.0 = all cold (default), 1.0 = all warm, 0.5 = half.
    /// Warm/cold are interleaved evenly across the run.
    #[arg(long, default_value_t = 0.0)]
    inj_warm_frac: f64,

    /// Background tokens each stream must emit before injection starts (head-start)
    #[arg(long, default_value_t = 8)]
    head_start_tokens: usize,

    /// `--iters` is ignored by `mixed`; `--warmup`/`--seed` apply.
    #[command(flatten)]
    run: RunArgs,
}

#[derive(Debug, Clone, Serialize)]
struct RunInfo {
    command: &'static str,
    model_path: String,
    model_type: String,
    cuda_graph: bool,
    load_ms: f64,
    label: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
struct PromptDescriptor {
    source: String,
    prompt_tokens: usize,
    prompt_preview: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct DurationStats {
    avg_ms: f64,
    p50_ms: f64,
    p95_ms: f64,
    p99_ms: f64,
    max_ms: f64,
    samples: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CountStats {
    min: usize,
    max: usize,
    avg: f64,
    samples: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct GeneratedTokenTrace {
    hash: String,
    prefix: Vec<u32>,
    len: usize,
}

#[derive(Debug, Clone, Serialize)]
struct RequestWorkload {
    prompt: PromptDescriptor,
    output_len: usize,
    concurrency: usize,
    warmup: usize,
    iters: usize,
    seed: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct RequestMetrics {
    ttft_ms: DurationStats,
    first_decode_step_ms: Option<DurationStats>,
    steady_tpot_ms: Option<DurationStats>,
    e2e_ms: DurationStats,
    generated_tokens: CountStats,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    generated_token_traces: Vec<GeneratedTokenTrace>,
    request_tok_s: Option<f64>,
    decode_tok_s: Option<f64>,
}

#[derive(Debug, Clone, Serialize)]
struct RequestIterationTiming {
    index: usize,
    ttft_ms: f64,
    first_decode_step_ms: Option<f64>,
    steady_tpot_ms: Option<DurationStats>,
    e2e_ms: f64,
    generated_tokens: usize,
    generated_token_trace: GeneratedTokenTrace,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SnapshotProfile {
    prompt_len: usize,
    output_len: usize,
    metrics: RequestMetrics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SnapshotReport {
    commit: String,
    date: String,
    model: String,
    gpu: String,
    /// Parallel layout the snapshot was measured under (e.g. "tp1-dp8-deepep").
    /// Absent in snapshots that predate multi-GPU model lines.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    parallel: Option<String>,
    /// Cold prefill: a distinct prompt per iteration so the prefix cache never
    /// hits — measures real prefill compute (the TTFT regression gate).
    prefill_heavy: SnapshotProfile,
    /// Warm prefill: the same prompt every iteration, so iterations after the
    /// first hit the default-on prefix cache (#216) — what a repeated prompt
    /// actually costs. `Option` so pre-existing baselines (without this field)
    /// still deserialize for `compare`.
    #[serde(default)]
    prefill_cached: Option<SnapshotProfile>,
    decode_heavy: SnapshotProfile,
}

#[derive(Debug, Clone, Serialize)]
struct MixedLoadConfig {
    bg_prompt_len: usize,
    bg_concurrency: usize,
    bg_output_len: usize,
    inj_prompt_len: usize,
    inj_output_len: usize,
    qps: f64,
    num_injections: usize,
    inj_warm_frac: f64,
    warmup: usize,
    seed: u64,
}

/// Inter-token-latency of the background decode streams
#[derive(Debug, Clone, Serialize)]
struct MixedLoadItl {
    /// Every background decode gap.
    all: DurationStats,
    /// Gaps with no overlapping injection window (decode unaffected by prefill).
    steady: Option<DurationStats>,
    /// Gaps overlapping an in-flight prefill (the unified-step stall tail).
    stall: Option<DurationStats>,
    stall_gap_count: usize,
    total_gap_count: usize,
}

#[derive(Debug, Clone, Serialize)]
struct InjectionRecord {
    index: usize,
    /// Whether this injection reused the shared prompt (intended prefix-cache hit).
    warm: bool,
    /// Wall time from submit to last token of the injected prompt (≈ prefill time).
    prefill_ms: f64,
    /// Offset of this injection's submit from the first injection's submit.
    arrival_offset_ms: f64,
}

#[derive(Debug, Clone, Serialize)]
struct MixedDecisionInputs {
    baseline_p50_ms: Option<f64>,
    baseline_p99_ms: Option<f64>,
    mixed_p50_ms: f64,
    mixed_p99_ms: f64,
    p99_delta_ms: Option<f64>,
    p99_delta_pct: Option<f64>,
}

#[derive(Debug, Clone, Serialize)]
struct MixedLoadReport {
    commit: String,
    date: String,
    gpu: String,
    run: RunInfo,
    config: MixedLoadConfig,
    /// Decode-only control (None when --skip-baseline).
    baseline_itl: Option<DurationStats>,
    mixed_itl: MixedLoadItl,
    injections: Vec<InjectionRecord>,
    decision_inputs: MixedDecisionInputs,
    /// Non-fatal measurement caveats (e.g. a background stream finished early).
    warnings: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
struct RequestReport {
    run: RunInfo,
    workload: RequestWorkload,
    metrics: RequestMetrics,
    iterations: Vec<RequestIterationTiming>,
}

#[derive(Debug, Clone, Serialize)]
struct MatrixWorkload {
    prompt_lens: Vec<usize>,
    output_lens: Vec<usize>,
    warmup: usize,
    iters: usize,
    seed: u64,
    synthetic_pattern: &'static str,
}

#[derive(Debug, Clone, Serialize)]
struct MatrixCell {
    prompt_len: usize,
    output_len: usize,
    ttft_ms: DurationStats,
    e2e_ms: DurationStats,
    first_decode_step_ms: Option<DurationStats>,
    steady_tpot_ms: Option<DurationStats>,
    generated_tokens: CountStats,
    request_tok_s: Option<f64>,
    decode_tok_s: Option<f64>,
}

#[derive(Debug, Clone, Serialize)]
struct MatrixReport {
    run: RunInfo,
    workload: MatrixWorkload,
    cells: Vec<MatrixCell>,
}

#[derive(Debug, Clone, Serialize)]
struct CurveWorkload {
    prompt: PromptDescriptor,
    output_len: usize,
    window: usize,
    warmup: usize,
    iters: usize,
    seed: u64,
}

#[derive(Debug, Clone, Serialize)]
struct CurveWindow {
    ctx_start: usize,
    ctx_end: usize,
    tpot_ms: DurationStats,
    decode_tok_s: Option<f64>,
}

#[derive(Debug, Clone, Serialize)]
struct CurveReport {
    run: RunInfo,
    workload: CurveWorkload,
    windows: Vec<CurveWindow>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum BenchReport {
    Request(Box<RequestReport>),
    Matrix(MatrixReport),
    Curve(CurveReport),
    Mixed(Box<MixedLoadReport>),
}

fn dur_ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1000.0
}

fn percentiles(sorted: &[Duration]) -> (Duration, Duration, Duration, Duration, Duration) {
    assert!(!sorted.is_empty());
    let n = sorted.len();
    let sum: Duration = sorted.iter().sum();
    let avg = sum / n as u32;
    let p = |pct: f64| sorted[((pct / 100.0) * (n - 1) as f64).round() as usize];
    (avg, p(50.0), p(95.0), p(99.0), sorted[n - 1])
}

fn summarize_durations(samples: &[Duration]) -> DurationStats {
    let mut sorted = samples.to_vec();
    sorted.sort();
    let (avg, p50, p95, p99, max) = percentiles(&sorted);
    DurationStats {
        avg_ms: dur_ms(avg),
        p50_ms: dur_ms(p50),
        p95_ms: dur_ms(p95),
        p99_ms: dur_ms(p99),
        max_ms: dur_ms(max),
        samples: sorted.len(),
    }
}

fn summarize_counts(samples: &[usize]) -> CountStats {
    assert!(!samples.is_empty());
    let min = *samples.iter().min().unwrap();
    let max = *samples.iter().max().unwrap();
    let sum: usize = samples.iter().sum();
    CountStats {
        min,
        max,
        avg: sum as f64 / samples.len() as f64,
        samples: samples.len(),
    }
}

fn aggregate_tok_s(tokens: usize, total: Duration) -> Option<f64> {
    if tokens == 0 || total.is_zero() {
        None
    } else {
        Some(tokens as f64 / total.as_secs_f64())
    }
}

fn generated_token_hash(tokens: &[u32]) -> String {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for token in tokens {
        for byte in token.to_le_bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x0100_0000_01b3);
        }
    }
    format!("{hash:016x}")
}

fn generated_token_trace(tokens: &[u32]) -> GeneratedTokenTrace {
    GeneratedTokenTrace {
        hash: generated_token_hash(tokens),
        prefix: tokens.iter().copied().take(16).collect(),
        len: tokens.len(),
    }
}

fn new_table() -> Table {
    let mut table = Table::new();
    if stdout().is_terminal() {
        table.load_preset(UTF8_FULL_CONDENSED);
        table.apply_modifier(UTF8_ROUND_CORNERS);
    } else {
        table.load_preset(ASCII_FULL_CONDENSED);
    }
    table
}

fn key_cell(label: impl Into<String>) -> Cell {
    Cell::new(label.into())
}

fn value_cell(value: impl Into<String>) -> Cell {
    Cell::new(value.into())
}

fn numeric_cell(value: impl Into<String>) -> Cell {
    Cell::new(value.into()).set_alignment(CellAlignment::Right)
}

fn format_rate(value: Option<f64>) -> String {
    value.map_or_else(|| "-".to_string(), |v| format!("{v:.2}"))
}

fn format_duration_ms(value: f64) -> String {
    format!("{value:.2}")
}

fn format_count_avg(value: f64) -> String {
    format!("{value:.2}")
}

fn push_table(out: &mut String, table: &Table) {
    out.push_str(&table.to_string());
    out.push('\n');
}

fn render_run_summary(report: &RunInfo) -> Table {
    let mut table = new_table();
    table.add_row(vec![
        key_cell("model"),
        value_cell(format!("{} ({})", report.model_path, report.model_type)),
    ]);
    table.add_row(vec![
        key_cell("cuda_graph"),
        value_cell(report.cuda_graph.to_string()),
    ]);
    table.add_row(vec![
        key_cell("load_ms"),
        numeric_cell(format_duration_ms(report.load_ms)),
    ]);
    if let Some(label) = &report.label {
        table.add_row(vec![key_cell("label"), value_cell(label.clone())]);
    }
    table
}

fn render_request_meta(report: &RequestReport) -> Table {
    let mut table = render_run_summary(&report.run);
    table.add_row(vec![
        key_cell("prompt_source"),
        value_cell(report.workload.prompt.source.clone()),
    ]);
    table.add_row(vec![
        key_cell("prompt_tokens"),
        numeric_cell(report.workload.prompt.prompt_tokens.to_string()),
    ]);
    if let Some(preview) = &report.workload.prompt.prompt_preview {
        table.add_row(vec![
            key_cell("prompt"),
            value_cell(format!("\"{preview}\"")),
        ]);
    }
    table.add_row(vec![
        key_cell("output_len"),
        numeric_cell(report.workload.output_len.to_string()),
    ]);
    table.add_row(vec![
        key_cell("warmup / iters"),
        value_cell(format!(
            "{} / {}",
            report.workload.warmup, report.workload.iters
        )),
    ]);
    table.add_row(vec![
        key_cell("seed"),
        numeric_cell(report.workload.seed.to_string()),
    ]);
    table
}

fn render_duration_table(rows: Vec<(String, DurationStats)>) -> Table {
    let mut table = new_table();
    table.set_header(vec![
        Cell::new("metric"),
        Cell::new("avg_ms").set_alignment(CellAlignment::Right),
        Cell::new("p50_ms").set_alignment(CellAlignment::Right),
        Cell::new("p95_ms").set_alignment(CellAlignment::Right),
        Cell::new("p99_ms").set_alignment(CellAlignment::Right),
        Cell::new("max_ms").set_alignment(CellAlignment::Right),
        Cell::new("samples").set_alignment(CellAlignment::Right),
    ]);
    for (label, stats) in rows {
        table.add_row(vec![
            key_cell(label),
            numeric_cell(format_duration_ms(stats.avg_ms)),
            numeric_cell(format_duration_ms(stats.p50_ms)),
            numeric_cell(format_duration_ms(stats.p95_ms)),
            numeric_cell(format_duration_ms(stats.p99_ms)),
            numeric_cell(format_duration_ms(stats.max_ms)),
            numeric_cell(stats.samples.to_string()),
        ]);
    }
    table
}

fn render_request_summary(report: &RequestReport) -> Table {
    let mut table = new_table();
    table.set_header(vec![
        Cell::new("metric"),
        Cell::new("value").set_alignment(CellAlignment::Right),
    ]);
    table.add_row(vec![
        key_cell("generated_tokens_avg"),
        numeric_cell(format_count_avg(report.metrics.generated_tokens.avg)),
    ]);
    table.add_row(vec![
        key_cell("generated_tokens_min"),
        numeric_cell(report.metrics.generated_tokens.min.to_string()),
    ]);
    table.add_row(vec![
        key_cell("generated_tokens_max"),
        numeric_cell(report.metrics.generated_tokens.max.to_string()),
    ]);
    table.add_row(vec![
        key_cell("generated_token_runs"),
        numeric_cell(report.metrics.generated_tokens.samples.to_string()),
    ]);
    table.add_row(vec![
        key_cell("request_tok_s"),
        numeric_cell(format_rate(report.metrics.request_tok_s)),
    ]);
    table.add_row(vec![
        key_cell("decode_tok_s"),
        numeric_cell(format_rate(report.metrics.decode_tok_s)),
    ]);
    table
}

fn render_matrix_meta(report: &MatrixReport) -> Table {
    let mut table = render_run_summary(&report.run);
    table.add_row(vec![
        key_cell("prompt_lens"),
        value_cell(
            report
                .workload
                .prompt_lens
                .iter()
                .map(std::string::ToString::to_string)
                .collect::<Vec<_>>()
                .join(","),
        ),
    ]);
    table.add_row(vec![
        key_cell("output_lens"),
        value_cell(
            report
                .workload
                .output_lens
                .iter()
                .map(std::string::ToString::to_string)
                .collect::<Vec<_>>()
                .join(","),
        ),
    ]);
    table.add_row(vec![
        key_cell("synthetic_pattern"),
        value_cell(report.workload.synthetic_pattern),
    ]);
    table.add_row(vec![
        key_cell("warmup / iters"),
        value_cell(format!(
            "{} / {}",
            report.workload.warmup, report.workload.iters
        )),
    ]);
    table.add_row(vec![
        key_cell("seed"),
        numeric_cell(report.workload.seed.to_string()),
    ]);
    table
}

fn render_matrix_table(report: &MatrixReport) -> Table {
    let mut table = new_table();
    table.set_header(vec![
        Cell::new("prompt_tok").set_alignment(CellAlignment::Right),
        Cell::new("output_tok").set_alignment(CellAlignment::Right),
        Cell::new("ttft_avg").set_alignment(CellAlignment::Right),
        Cell::new("ttft_p95").set_alignment(CellAlignment::Right),
        Cell::new("e2e_avg").set_alignment(CellAlignment::Right),
        Cell::new("req_tok/s").set_alignment(CellAlignment::Right),
        Cell::new("decode_tok/s").set_alignment(CellAlignment::Right),
        Cell::new("gen_avg").set_alignment(CellAlignment::Right),
    ]);
    for cell in &report.cells {
        table.add_row(vec![
            numeric_cell(cell.prompt_len.to_string()),
            numeric_cell(cell.output_len.to_string()),
            numeric_cell(format_duration_ms(cell.ttft_ms.avg_ms)),
            numeric_cell(format_duration_ms(cell.ttft_ms.p95_ms)),
            numeric_cell(format_duration_ms(cell.e2e_ms.avg_ms)),
            numeric_cell(format_rate(cell.request_tok_s)),
            numeric_cell(format_rate(cell.decode_tok_s)),
            numeric_cell(format_count_avg(cell.generated_tokens.avg)),
        ]);
    }
    table
}

fn render_curve_meta(report: &CurveReport) -> Table {
    let mut table = render_run_summary(&report.run);
    table.add_row(vec![
        key_cell("prompt_source"),
        value_cell(report.workload.prompt.source.clone()),
    ]);
    table.add_row(vec![
        key_cell("prompt_tokens"),
        numeric_cell(report.workload.prompt.prompt_tokens.to_string()),
    ]);
    if let Some(preview) = &report.workload.prompt.prompt_preview {
        table.add_row(vec![
            key_cell("prompt"),
            value_cell(format!("\"{preview}\"")),
        ]);
    }
    table.add_row(vec![
        key_cell("output_len"),
        numeric_cell(report.workload.output_len.to_string()),
    ]);
    table.add_row(vec![
        key_cell("window"),
        numeric_cell(report.workload.window.to_string()),
    ]);
    table.add_row(vec![
        key_cell("warmup / iters"),
        value_cell(format!(
            "{} / {}",
            report.workload.warmup, report.workload.iters
        )),
    ]);
    table.add_row(vec![
        key_cell("seed"),
        numeric_cell(report.workload.seed.to_string()),
    ]);
    table
}

fn render_curve_table(report: &CurveReport) -> Table {
    let mut table = new_table();
    table.set_header(vec![
        Cell::new("ctx_range"),
        Cell::new("avg_ms").set_alignment(CellAlignment::Right),
        Cell::new("p50_ms").set_alignment(CellAlignment::Right),
        Cell::new("p95_ms").set_alignment(CellAlignment::Right),
        Cell::new("p99_ms").set_alignment(CellAlignment::Right),
        Cell::new("tok/s").set_alignment(CellAlignment::Right),
        Cell::new("samples").set_alignment(CellAlignment::Right),
    ]);
    for window in &report.windows {
        table.add_row(vec![
            value_cell(format!("{}-{}", window.ctx_start, window.ctx_end)),
            numeric_cell(format_duration_ms(window.tpot_ms.avg_ms)),
            numeric_cell(format_duration_ms(window.tpot_ms.p50_ms)),
            numeric_cell(format_duration_ms(window.tpot_ms.p95_ms)),
            numeric_cell(format_duration_ms(window.tpot_ms.p99_ms)),
            numeric_cell(format_rate(window.decode_tok_s)),
            numeric_cell(window.tpot_ms.samples.to_string()),
        ]);
    }
    table
}

fn truncate_preview(text: &str, limit: usize) -> String {
    let one_line = text.replace('\n', "\\n");
    if one_line.chars().count() <= limit {
        return one_line;
    }
    let mut truncated = String::new();
    for ch in one_line.chars().take(limit) {
        truncated.push(ch);
    }
    truncated.push_str("...");
    truncated
}

fn synthetic_prompt_tokens(len: usize) -> Vec<u32> {
    (0..len).map(|i| ((i % 1000) + 100) as u32).collect()
}

/// Token-id bounds for synthetic concurrent prompts: above the low special
/// tokens and well under the smallest supported vocab (DeepSeek-V2-Lite ≈
/// 102 400), so every drawn id is an ordinary token on any model line.
const SYNTHETIC_TOKEN_LO: u32 = 100;
const SYNTHETIC_TOKEN_HI: u32 = 100_000;

/// One synthetic prompt of `len` random tokens, seeded per request so the
/// concurrent decode streams diverge. Identical concurrent prompts route a MoE
/// batch onto a narrow expert set, packing the Marlin expert GEMM into fat
/// tiles and under-measuring decode TPOT by ~7–15% (measured on Kimi-K2 via a
/// `--distinct-prompts` sweep; the bench trap behind the misread #225 "+51%
/// HTTP" gap). Distinct prompts exercise realistic broad expert routing. See
/// docs/lessons/moe-bench-prompt-diversity.md.
fn synthetic_random_prompt(len: usize, seed: u64, request_idx: usize) -> Vec<u32> {
    let mut rng =
        StdRng::seed_from_u64(seed ^ (request_idx as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
    (0..len)
        .map(|_| rng.random_range(SYNTHETIC_TOKEN_LO..SYNTHETIC_TOKEN_HI))
        .collect()
}

/// To control prefix cache hit/miss in mixed-load benchmarking, we salt the
/// synthetic prompt tokens by a large constant per injection, so that even the
/// same prompt length gets a different token sequence and so misses the prefix
/// cache when intended (e.g. `--inj-warm-frac 0.0`).
fn synthetic_prompt_tokens_salted(len: usize, salt: usize) -> Vec<u32> {
    let shift = salt.wrapping_mul(7919);
    (0..len)
        .map(|i| (((i + shift) % 1000) + 100) as u32)
        .collect()
}

#[derive(Debug, Clone)]
struct PromptSpec {
    descriptor: PromptDescriptor,
    tokens: Vec<u32>,
}

fn resolve_prompt_input(
    args: &PromptInputArgs,
    tokenizer: &DynTokenizer,
    default_text: Option<&str>,
    default_prompt_len: Option<usize>,
) -> Result<PromptSpec> {
    match (&args.prompt, &args.prompt_file, args.prompt_len) {
        (Some(prompt), None, None) => Ok(PromptSpec {
            descriptor: PromptDescriptor {
                source: "text".to_string(),
                prompt_tokens: tokenizer.encode(prompt, false)?.len(),
                prompt_preview: Some(truncate_preview(prompt, 96)),
            },
            tokens: tokenizer.encode(prompt, false)?,
        }),
        (None, Some(path), None) => {
            let prompt = fs::read_to_string(path)
                .with_context(|| format!("failed to read prompt file: {path}"))?;
            let tokens = tokenizer.encode(&prompt, false)?;
            Ok(PromptSpec {
                descriptor: PromptDescriptor {
                    source: format!("file:{path}"),
                    prompt_tokens: tokens.len(),
                    prompt_preview: Some(truncate_preview(&prompt, 96)),
                },
                tokens,
            })
        }
        (None, None, Some(prompt_len)) => {
            ensure!(prompt_len > 0, "--prompt-len must be > 0");
            Ok(PromptSpec {
                descriptor: PromptDescriptor {
                    source: format!("synthetic:{SYNTHETIC_PATTERN}"),
                    prompt_tokens: prompt_len,
                    prompt_preview: None,
                },
                tokens: synthetic_prompt_tokens(prompt_len),
            })
        }
        (None, None, None) => {
            if let Some(prompt) = default_text {
                let tokens = tokenizer.encode(prompt, false)?;
                Ok(PromptSpec {
                    descriptor: PromptDescriptor {
                        source: "text".to_string(),
                        prompt_tokens: tokens.len(),
                        prompt_preview: Some(truncate_preview(prompt, 96)),
                    },
                    tokens,
                })
            } else if let Some(prompt_len) = default_prompt_len {
                Ok(PromptSpec {
                    descriptor: PromptDescriptor {
                        source: format!("synthetic:{SYNTHETIC_PATTERN}"),
                        prompt_tokens: prompt_len,
                        prompt_preview: None,
                    },
                    tokens: synthetic_prompt_tokens(prompt_len),
                })
            } else {
                unreachable!("default prompt source must be provided");
            }
        }
        _ => unreachable!("clap enforces prompt input conflicts"),
    }
}

struct GenTimings {
    ttft: Duration,
    tbt: Vec<Duration>,
    total: Duration,
    emitted_tokens: usize,
    generated_tokens: Vec<u32>,
    decode_tokens_for_rate: usize,
    decode_time_for_rate: Duration,
}

trait BenchModel {
    fn validate_concurrency(&self, concurrency: usize) -> Result<()> {
        ensure!(concurrency > 0, "--concurrency must be > 0");
        Ok(())
    }

    /// Scheduler handle for open-loop mixed-load benchmarking.
    fn scheduler_handle(&self) -> Option<SchedulerHandle> {
        None
    }

    fn timed_generation(
        &mut self,
        prompt_tokens: &[u32],
        max_new_tokens: usize,
        sampling: &SamplingParams,
        rng: &mut StdRng,
    ) -> GenTimings;

    /// Run one request per prompt; the slice length is the concurrency. Each
    /// prompt is independent, so MoE models must be handed *distinct* prompts
    /// to exercise realistic expert routing (see `synthetic_random_prompt`).
    fn timed_generation_batch(
        &mut self,
        prompts: &[Vec<u32>],
        max_new_tokens: usize,
        sampling: &SamplingParams,
        rng: &mut StdRng,
    ) -> Vec<GenTimings> {
        prompts
            .iter()
            .map(|prompt| self.timed_generation(prompt, max_new_tokens, sampling, rng))
            .collect()
    }
}

fn run_timed<F>(prompt_tokens: &[u32], max_new_tokens: usize, mut generate: F) -> GenTimings
where
    F: FnMut(&[u32], usize, &mut dyn FnMut(u32) -> bool) -> Result<()>,
{
    let start = Instant::now();
    let mut first_at: Option<Instant> = None;
    let mut prev_at: Option<Instant> = None;
    let mut emitted_tokens = 0usize;
    let mut tbt = Vec::with_capacity(max_new_tokens.saturating_sub(1));
    let mut generated_tokens = Vec::with_capacity(max_new_tokens);

    generate(prompt_tokens, max_new_tokens, &mut |tok| {
        let now = Instant::now();
        emitted_tokens += 1;
        generated_tokens.push(tok);
        if first_at.is_none() {
            first_at = Some(now);
        } else if let Some(prev) = prev_at {
            tbt.push(now - prev);
        }
        prev_at = Some(now);
        true
    })
    .expect("generation failed");

    let total = start.elapsed();
    let ttft = first_at.map_or(total, |t| t - start);
    let decode_tokens_for_rate = emitted_tokens.saturating_sub(1);
    let decode_time_for_rate = tbt.iter().copied().sum();
    GenTimings {
        ttft,
        tbt,
        total,
        emitted_tokens,
        generated_tokens,
        decode_tokens_for_rate,
        decode_time_for_rate,
    }
}

struct SchedulerBenchModel {
    handle: SchedulerHandle,
}

impl BenchModel for SchedulerBenchModel {
    fn scheduler_handle(&self) -> Option<SchedulerHandle> {
        Some(self.handle.clone())
    }

    fn timed_generation(
        &mut self,
        prompt_tokens: &[u32],
        max_new_tokens: usize,
        sampling: &SamplingParams,
        _rng: &mut StdRng,
    ) -> GenTimings {
        run_timed(prompt_tokens, max_new_tokens, |toks, n, cb| {
            let (token_tx, mut token_rx) = mpsc::unbounded_channel();
            self.handle
                .submit(SchedulerRequest {
                    request_id: None,
                    queued_at_unix_s: None,
                    prompt_tokens: toks.to_vec(),
                    params: SamplingParams {
                        temperature: sampling.temperature,
                        top_k: sampling.top_k,
                        top_p: sampling.top_p,
                        ignore_eos: sampling.ignore_eos,
                    },
                    max_tokens: n,
                    lora_adapter: None,
                    token_tx,
                    logprobs: 0,
                    echo: false,
                })
                .map_err(|e| anyhow::anyhow!("scheduler submit failed: {e}"))?;

            loop {
                match token_rx.blocking_recv() {
                    Some(TokenEvent::Token { id, .. }) => {
                        if !cb(id) {
                            break;
                        }
                    }
                    Some(TokenEvent::PromptTokens { .. } | TokenEvent::Scheduled { .. }) => {}
                    Some(TokenEvent::Finished { .. }) => break,
                    Some(TokenEvent::Error { message, .. }) => {
                        anyhow::bail!("scheduler request failed: {message}");
                    }
                    Some(TokenEvent::Rejected { message, .. }) => {
                        anyhow::bail!("scheduler request rejected: {message}");
                    }
                    None => anyhow::bail!("scheduler channel closed"),
                }
            }

            Ok(())
        })
    }

    fn timed_generation_batch(
        &mut self,
        prompts: &[Vec<u32>],
        max_new_tokens: usize,
        sampling: &SamplingParams,
        _rng: &mut StdRng,
    ) -> Vec<GenTimings> {
        let mut workers = Vec::with_capacity(prompts.len());
        for (idx, prompt) in prompts.iter().enumerate() {
            let handle = self.handle.clone();
            let prompt_tokens = prompt.clone();
            let sampling = *sampling;
            workers.push(thread::spawn(move || {
                run_timed(&prompt_tokens, max_new_tokens, |toks, n, cb| {
                    let (token_tx, mut token_rx) = mpsc::unbounded_channel();
                    handle
                        .submit(SchedulerRequest {
                            request_id: Some(format!("bench-serving-{idx}")),
                            queued_at_unix_s: None,
                            prompt_tokens: toks.to_vec(),
                            params: SamplingParams {
                                temperature: sampling.temperature,
                                top_k: sampling.top_k,
                                top_p: sampling.top_p,
                                ignore_eos: sampling.ignore_eos,
                            },
                            max_tokens: n,
                            lora_adapter: None,
                            token_tx,
                            logprobs: 0,
                            echo: false,
                        })
                        .map_err(|e| anyhow::anyhow!("scheduler submit failed: {e}"))?;

                    loop {
                        match token_rx.blocking_recv() {
                            Some(TokenEvent::Token { id, .. }) => {
                                if !cb(id) {
                                    break;
                                }
                            }
                            Some(
                                TokenEvent::PromptTokens { .. } | TokenEvent::Scheduled { .. },
                            ) => {}
                            Some(TokenEvent::Finished { .. }) => break,
                            Some(TokenEvent::Error { message, .. }) => {
                                anyhow::bail!("scheduler request failed: {message}");
                            }
                            Some(TokenEvent::Rejected { message, .. }) => {
                                anyhow::bail!("scheduler request rejected: {message}");
                            }
                            None => anyhow::bail!("scheduler channel closed"),
                        }
                    }

                    Ok(())
                })
            }));
        }

        workers
            .into_iter()
            .map(|worker| worker.join().expect("bench request worker panicked"))
            .collect()
    }
}

#[cfg(feature = "deepseek-v2-lite")]
struct DeepSeekV2LiteBenchModel {
    generator: pegainfer_deepseek_v2_lite::DeepSeekV2LiteEp2Generator,
}

#[cfg(feature = "deepseek-v2-lite")]
impl BenchModel for DeepSeekV2LiteBenchModel {
    fn validate_concurrency(&self, concurrency: usize) -> Result<()> {
        ensure!(
            concurrency > 0 && concurrency <= 8,
            "DeepSeek-V2-Lite direct benchmark supports --concurrency 1..=8; concurrency=1 is the single-row control and >1 uses the narrow same-prompt batched decode path, got {concurrency}"
        );
        Ok(())
    }

    fn timed_generation(
        &mut self,
        prompt_tokens: &[u32],
        max_new_tokens: usize,
        sampling: &SamplingParams,
        _rng: &mut StdRng,
    ) -> GenTimings {
        assert_dsv2_lite_sampling_contract(sampling);
        let (result, attribution) = self
            .generator
            .generate_greedy_with_attribution(prompt_tokens, max_new_tokens, sampling.ignore_eos)
            .expect("DeepSeek-V2-Lite generation failed");
        timings_from_dsv2_lite_attribution(
            result.tokens,
            max_new_tokens,
            attribution.total_generation_us(),
            attribution.prefill_next_token_us(),
            attribution.per_token_decode_us(),
        )
    }

    fn timed_generation_batch(
        &mut self,
        prompts: &[Vec<u32>],
        max_new_tokens: usize,
        sampling: &SamplingParams,
        _rng: &mut StdRng,
    ) -> Vec<GenTimings> {
        assert_dsv2_lite_sampling_contract(sampling);
        if prompts.len() == 1 {
            return vec![self.timed_generation(&prompts[0], max_new_tokens, sampling, _rng)];
        }

        // This generator drives a narrow same-prompt batched decode kernel:
        // every row shares `prompts[0]`. Distinct per-request prompts are a
        // scheduler-path concern; this microbench takes one prompt by design.
        let result = self
            .generator
            .generate_greedy_batch_same_prompt_with_timings(
                &prompts[0],
                prompts.len(),
                max_new_tokens,
                sampling.ignore_eos,
            )
            .expect("DeepSeek-V2-Lite batched generation failed");
        timings_from_dsv2_lite_batched_generation(result, max_new_tokens)
    }
}

#[cfg(feature = "deepseek-v2-lite")]
fn assert_dsv2_lite_sampling_contract(sampling: &SamplingParams) {
    assert!(
        sampling.ignore_eos,
        "DeepSeek-V2-Lite direct attribution benchmark requires ignore_eos=true so output_len maps to an exact generated-token count"
    );
    assert!(
        (sampling.temperature <= 0.0 || sampling.top_k == 1) && sampling.top_p >= 1.0,
        "DeepSeek-V2-Lite direct attribution benchmark supports greedy decoding only; requested temperature={}, top_k={}, top_p={}",
        sampling.temperature,
        sampling.top_k,
        sampling.top_p
    );
}

#[cfg(feature = "deepseek-v2-lite")]
fn timings_from_dsv2_lite_attribution(
    generated_token_ids: Vec<u32>,
    expected_generated_tokens: usize,
    total_generation_us: u64,
    prefill_next_token_us: Option<u64>,
    per_token_decode_us: &[u64],
) -> GenTimings {
    // This bench helper intentionally panics on corrupted attribution data rather
    // than synthesizing a result. The surrounding trait does not carry errors,
    // and emitting bogus TPOT would be worse than aborting the benchmark.
    let emitted_tokens = generated_token_ids.len();
    assert_eq!(
        emitted_tokens, expected_generated_tokens,
        "DeepSeek-V2-Lite generated token count mismatch: got {} tokens for requested output_len={}",
        emitted_tokens, expected_generated_tokens
    );
    let expected_decode_steps = expected_generated_tokens.saturating_sub(1);
    assert_eq!(
        per_token_decode_us.len(),
        expected_decode_steps,
        "DeepSeek-V2-Lite timing count mismatch: got {} decode samples for {} generated tokens",
        per_token_decode_us.len(),
        emitted_tokens
    );
    assert!(
        total_generation_us > 0,
        "DeepSeek-V2-Lite total generation timing is zero; refusing to report TPOT"
    );
    if emitted_tokens > 0 {
        assert!(
            prefill_next_token_us.is_some_and(|us| us > 0),
            "DeepSeek-V2-Lite TTFT timing is missing or zero; refusing to report TPOT"
        );
    }
    if expected_decode_steps > 0 {
        assert!(
            per_token_decode_us.iter().all(|us| *us > 0),
            "DeepSeek-V2-Lite decode timing contains a zero-duration sample; refusing to report TPOT"
        );
    }
    let tbt: Vec<_> = per_token_decode_us
        .iter()
        .map(|us| Duration::from_micros(*us))
        .collect();
    let decode_time_for_rate = tbt.iter().copied().sum();
    GenTimings {
        ttft: Duration::from_micros(prefill_next_token_us.unwrap_or(total_generation_us)),
        tbt,
        total: Duration::from_micros(total_generation_us),
        emitted_tokens,
        generated_tokens: generated_token_ids,
        decode_tokens_for_rate: emitted_tokens.saturating_sub(1),
        decode_time_for_rate,
    }
}

#[cfg(feature = "deepseek-v2-lite")]
fn timings_from_dsv2_lite_batched_generation(
    result: pegainfer_deepseek_v2_lite::BatchedGenerationResult,
    expected_generated_tokens: usize,
) -> Vec<GenTimings> {
    let batch_size = result.tokens.len();
    assert!(
        batch_size > 0,
        "DeepSeek-V2-Lite batch result must contain at least one row"
    );
    assert_eq!(
        result.prefill_next_token_us.len(),
        batch_size,
        "DeepSeek-V2-Lite batch result TTFT count mismatch"
    );
    assert!(
        result.total_generation_us > 0,
        "DeepSeek-V2-Lite batch total generation timing is zero; refusing to report TPOT"
    );
    assert!(
        result.prefill_next_token_us.iter().all(|us| *us > 0),
        "DeepSeek-V2-Lite batch TTFT timing contains a zero-duration sample; refusing to report TPOT"
    );
    let expected_decode_steps = expected_generated_tokens.saturating_sub(1);
    assert_eq!(
        result.per_token_decode_us.len(),
        expected_decode_steps,
        "DeepSeek-V2-Lite batch timing count mismatch: got {} decode samples for {} generated tokens",
        result.per_token_decode_us.len(),
        expected_generated_tokens
    );
    if expected_decode_steps > 0 {
        assert!(
            result.per_token_decode_us.iter().all(|us| *us > 0),
            "DeepSeek-V2-Lite batch decode timing contains a zero-duration sample; refusing to report TPOT"
        );
    }

    let tbt: Vec<_> = result
        .per_token_decode_us
        .iter()
        .map(|us| Duration::from_micros(*us))
        .collect();
    let decode_time_for_rate: Duration = tbt.iter().copied().sum();
    let decode_tokens_for_rate = batch_size * expected_decode_steps;

    result
        .tokens
        .into_iter()
        .zip(result.prefill_next_token_us)
        .enumerate()
        .map(|(idx, (generated_token_ids, prefill_us))| {
            let emitted_tokens = generated_token_ids.len();
            assert_eq!(
                emitted_tokens, expected_generated_tokens,
                "DeepSeek-V2-Lite batch row {idx} generated token count mismatch: got {} tokens for requested output_len={}",
                emitted_tokens, expected_generated_tokens
            );
            GenTimings {
                ttft: Duration::from_micros(prefill_us),
                tbt: tbt.clone(),
                total: Duration::from_micros(result.total_generation_us),
                emitted_tokens,
                generated_tokens: generated_token_ids,
                decode_tokens_for_rate: if idx == 0 { decode_tokens_for_rate } else { 0 },
                decode_time_for_rate: if idx == 0 {
                    decode_time_for_rate
                } else {
                    Duration::ZERO
                },
            }
        })
        .collect()
}

fn command_seed(cli: &Cli) -> u64 {
    match &cli.command {
        Command::Request(args) => args.run.seed,
        Command::Matrix(args) => args.run.seed,
        Command::Curve(args) => args.run.seed,
        Command::Snapshot(args) => args.run.seed,
        Command::Mixed(args) => args.run.seed,
        Command::Compare(_) => 42,
    }
}

#[cfg(feature = "kimi-k2")]
fn kimi_parallel_config(tp_size: usize, dp_size: usize) -> Result<ParallelConfig> {
    ensure!(tp_size > 0, "--tp-size must be positive");
    ensure!(dp_size > 0, "--dp-size must be positive");
    Ok(ParallelConfig::new(tp_size, dp_size))
}

fn normalize_sizes(values: &[usize], flag: &str) -> Result<Vec<usize>> {
    ensure!(!values.is_empty(), "{flag} must not be empty");
    ensure!(values.iter().all(|v| *v > 0), "{flag} values must be > 0");
    let mut normalized = values.to_vec();
    normalized.sort_unstable();
    normalized.dedup();
    Ok(normalized)
}

fn validate_run_args(args: &RunArgs) -> Result<()> {
    ensure!(args.iters > 0, "--iters must be > 0");
    Ok(())
}

fn measure_timings(
    model: &mut dyn BenchModel,
    prompts: &[Vec<u32>],
    output_len: usize,
    run: &RunArgs,
    cuda_profiler_capture: bool,
) -> Result<Vec<GenTimings>> {
    ensure!(output_len > 0, "--output-len must be > 0");
    ensure!(!prompts.is_empty(), "concurrency must be > 0");
    model.validate_concurrency(prompts.len())?;
    validate_run_args(run)?;

    let sampling = SamplingParams {
        ignore_eos: true,
        ..SamplingParams::default()
    };
    let mut rng = StdRng::seed_from_u64(run.seed);

    for _ in 0..run.warmup {
        let _ = model.timed_generation_batch(prompts, output_len, &sampling, &mut rng);
    }

    let profiler = if cuda_profiler_capture {
        info!(
            "Starting CUDA profiler capture around {} measured iterations",
            run.iters
        );
        cuda_device::set(0).context("failed to set CUDA device before profiler capture")?;
        Some(Profiler::new().context("failed to start CUDA profiler capture")?)
    } else {
        None
    };

    let mut timings = Vec::with_capacity(run.iters * prompts.len());
    for _ in 0..run.iters {
        timings.extend(model.timed_generation_batch(prompts, output_len, &sampling, &mut rng));
    }
    drop(profiler);
    Ok(timings)
}

// ---------------------------------------------------------------------------
// Mixed-load ITL (open-loop: decode-heavy background + low-QPS long prefills)
// ---------------------------------------------------------------------------

/// One background decode stream's record over a mixed-load (or baseline) phase.
struct BgStream {
    /// Wall-clock instant of each emitted decode token.
    token_times: Vec<Instant>,
    /// True if the stream hit its `output_len` (Finished) before being stopped —
    /// signals that steady-state concurrency dropped mid-run.
    finished_early: bool,
}

struct InjectorOutcome {
    /// `[submit, last-token]` window of each injected prefill.
    windows: Vec<(Instant, Instant)>,
    records: Vec<InjectionRecord>,
    /// Injections whose prefill outlasted the `1/qps` slot (QPS not sustained).
    overruns: usize,
}

fn greedy_sampling() -> SamplingParams {
    SamplingParams {
        ignore_eos: true,
        ..SamplingParams::default()
    }
}

fn opt_summarize(samples: &[Duration]) -> Option<DurationStats> {
    (!samples.is_empty()).then(|| summarize_durations(samples))
}

/// Spawn `bg_concurrency` long-lived decode streams. Each records the instant of
/// every emitted token and stops when `stop` is set (or its `output_len` runs
/// out). `counters[idx]` tracks tokens emitted, for head-start coordination.
fn spawn_background_streams(
    handle: &SchedulerHandle,
    bg_prompt_len: usize,
    bg_output_len: usize,
    bg_concurrency: usize,
    stop: &Arc<AtomicBool>,
    counters: &Arc<[AtomicUsize]>,
) -> Vec<thread::JoinHandle<Result<BgStream>>> {
    (0..bg_concurrency)
        .map(|idx| {
            let handle = handle.clone();
            let stop = Arc::clone(stop);
            let counters = Arc::clone(counters);
            thread::spawn(move || -> Result<BgStream> {
                let prompt = synthetic_prompt_tokens(bg_prompt_len);
                let (token_tx, mut token_rx) = mpsc::unbounded_channel();
                handle
                    .submit(SchedulerRequest {
                        request_id: Some(format!("mixed-bg-{idx}")),
                        queued_at_unix_s: None,
                        prompt_tokens: prompt,
                        params: greedy_sampling(),
                        max_tokens: bg_output_len,
                        lora_adapter: None,
                        token_tx,
                        logprobs: 0,
                        echo: false,
                    })
                    .map_err(|e| anyhow::anyhow!("background submit failed: {e}"))?;

                let mut token_times = Vec::with_capacity(bg_output_len);
                let mut finished_early = false;
                loop {
                    match token_rx.blocking_recv() {
                        Some(TokenEvent::Token { .. }) => {
                            token_times.push(Instant::now());
                            counters[idx].fetch_add(1, Ordering::Relaxed);
                            if stop.load(Ordering::Acquire) {
                                break;
                            }
                        }
                        Some(TokenEvent::PromptTokens { .. } | TokenEvent::Scheduled { .. }) => {}
                        Some(TokenEvent::Finished { .. }) => {
                            finished_early = true;
                            break;
                        }
                        Some(TokenEvent::Error { message, .. }) => {
                            anyhow::bail!("background request failed: {message}");
                        }
                        Some(TokenEvent::Rejected { message, .. }) => {
                            anyhow::bail!("background request rejected: {message}");
                        }
                        None => anyhow::bail!("background channel closed"),
                    }
                }
                // Dropping `token_rx` cancels the request if it is still active.
                Ok(BgStream {
                    token_times,
                    finished_early,
                })
            })
        })
        .collect()
}

/// Run a few closed-loop decode batches at the target concurrency to JIT the
/// decode CUDA graph and warm the allocator before measurement begins.
fn mixed_warmup(
    handle: &SchedulerHandle,
    bg_prompt_len: usize,
    bg_concurrency: usize,
    rounds: usize,
) -> Result<()> {
    for _ in 0..rounds {
        let workers: Vec<_> = (0..bg_concurrency)
            .map(|idx| {
                let handle = handle.clone();
                thread::spawn(move || -> Result<()> {
                    let prompt = synthetic_prompt_tokens(bg_prompt_len);
                    let (token_tx, mut token_rx) = mpsc::unbounded_channel();
                    handle
                        .submit(SchedulerRequest {
                            request_id: Some(format!("mixed-warmup-{idx}")),
                            queued_at_unix_s: None,
                            prompt_tokens: prompt,
                            params: greedy_sampling(),
                            max_tokens: 16,
                            lora_adapter: None,
                            token_tx,
                            logprobs: 0,
                            echo: false,
                        })
                        .map_err(|e| anyhow::anyhow!("warmup submit failed: {e}"))?;
                    loop {
                        match token_rx.blocking_recv() {
                            Some(TokenEvent::Finished { .. }) => break,
                            Some(TokenEvent::Error { message, .. }) => {
                                anyhow::bail!("warmup request failed: {message}")
                            }
                            Some(TokenEvent::Rejected { message, .. }) => {
                                anyhow::bail!("warmup request rejected: {message}")
                            }
                            Some(_) => {}
                            None => anyhow::bail!("warmup channel closed"),
                        }
                    }
                    Ok(())
                })
            })
            .collect();
        for worker in workers {
            worker.join().expect("warmup worker panicked")?;
        }
    }
    Ok(())
}

/// Block until every background stream has emitted `target` tokens, so injection
/// starts only after the background is in steady-state decode (past its own
/// prefill / first-decode-step). Returns false on timeout.
fn wait_for_head_start(counters: &Arc<[AtomicUsize]>, target: usize, timeout: Duration) -> bool {
    let deadline = Instant::now() + timeout;
    loop {
        if counters.iter().all(|c| c.load(Ordering::Relaxed) >= target) {
            return true;
        }
        if Instant::now() >= deadline {
            return false;
        }
        thread::sleep(Duration::from_millis(5));
    }
}

/// Decide whether injection `index` is warm, spreading `warm_frac` of injections evenly across the run
fn injection_is_warm(index: usize, warm_frac: f64) -> bool {
    let count = |k: usize| (k as f64 * warm_frac).floor() as usize;
    count(index + 1) > count(index)
}

/// Submit `num_injections` long prompts paced by arrival at `qps`, draining each
/// to completion. Each `[submit, last-token]` window marks an in-flight prefill.
fn run_injector(
    handle: &SchedulerHandle,
    inj_prompt_len: usize,
    inj_output_len: usize,
    qps: f64,
    num_injections: usize,
    warm_frac: f64,
) -> Result<InjectorOutcome> {
    let period = Duration::from_secs_f64(1.0 / qps);
    let mut windows = Vec::with_capacity(num_injections);
    let mut records = Vec::with_capacity(num_injections);
    let mut overruns = 0usize;
    let warm_salt = num_injections + 100;
    let t0 = Instant::now();
    for index in 0..num_injections {
        // Evenly interleave round(warm_frac * num_injections) warm injections.
        let warm = injection_is_warm(index, warm_frac);
        // Warm → shared prompt (injection after the first hits the prefix cache).
        // Cold → distinct prompt per injection → real prefill every time.
        let salt = if warm { warm_salt } else { index + 1 };
        let prompt = synthetic_prompt_tokens_salted(inj_prompt_len, salt);
        let slot_start = Instant::now();
        let (token_tx, mut token_rx) = mpsc::unbounded_channel();
        handle
            .submit(SchedulerRequest {
                request_id: Some(format!("mixed-inj-{index}")),
                queued_at_unix_s: None,
                prompt_tokens: prompt,
                params: greedy_sampling(),
                max_tokens: inj_output_len,
                lora_adapter: None,
                token_tx,
                logprobs: 0,
                echo: false,
            })
            .map_err(|e| anyhow::anyhow!("injection submit failed: {e}"))?;
        let mut last = slot_start;
        loop {
            match token_rx.blocking_recv() {
                Some(TokenEvent::Token { .. }) => last = Instant::now(),
                Some(TokenEvent::PromptTokens { .. } | TokenEvent::Scheduled { .. }) => {}
                Some(TokenEvent::Finished { .. }) => break,
                Some(TokenEvent::Error { message, .. }) => {
                    anyhow::bail!("injection request failed: {message}")
                }
                Some(TokenEvent::Rejected { message, .. }) => {
                    anyhow::bail!("injection request rejected: {message}")
                }
                None => anyhow::bail!("injection channel closed"),
            }
        }
        windows.push((slot_start, last));
        records.push(InjectionRecord {
            index,
            warm,
            prefill_ms: dur_ms(last - slot_start),
            arrival_offset_ms: dur_ms(slot_start - t0),
        });
        let elapsed = slot_start.elapsed();
        if elapsed < period {
            thread::sleep(period.saturating_sub(elapsed));
        } else if index + 1 < num_injections {
            overruns += 1;
        }
    }
    Ok(InjectorOutcome {
        windows,
        records,
        overruns,
    })
}

/// A background decode gap `[a, b)` is a stall if it overlaps any in-flight
/// prefill window `[s, e)`.
fn gap_overlaps_any(a: Instant, b: Instant, windows: &[(Instant, Instant)]) -> bool {
    windows.iter().any(|&(s, e)| a < e && s < b)
}

fn collect_gaps(streams: &[BgStream]) -> Vec<Duration> {
    let mut gaps = Vec::new();
    for stream in streams {
        for pair in stream.token_times.windows(2) {
            gaps.push(pair[1] - pair[0]);
        }
    }
    gaps
}

fn build_mixed_itl(streams: &[BgStream], windows: &[(Instant, Instant)]) -> Option<MixedLoadItl> {
    let mut all = Vec::new();
    let mut steady = Vec::new();
    let mut stall = Vec::new();
    for stream in streams {
        for pair in stream.token_times.windows(2) {
            let (a, b) = (pair[0], pair[1]);
            all.push(b - a);
            if gap_overlaps_any(a, b, windows) {
                stall.push(b - a);
            } else {
                steady.push(b - a);
            }
        }
    }
    let total_gap_count = all.len();
    let stall_gap_count = stall.len();
    Some(MixedLoadItl {
        all: opt_summarize(&all)?,
        steady: opt_summarize(&steady),
        stall: opt_summarize(&stall),
        stall_gap_count,
        total_gap_count,
    })
}

/// Decode-only control: same background streams, no injector, run for `duration`.
fn run_baseline(
    handle: &SchedulerHandle,
    args: &MixedArgs,
    duration: Duration,
    warnings: &mut Vec<String>,
) -> Result<Option<DurationStats>> {
    let stop = Arc::new(AtomicBool::new(false));
    let counters: Arc<[AtomicUsize]> = (0..args.bg_concurrency)
        .map(|_| AtomicUsize::new(0))
        .collect();
    let bg_handles = spawn_background_streams(
        handle,
        args.bg_prompt_len,
        args.bg_output_len,
        args.bg_concurrency,
        &stop,
        &counters,
    );
    if !wait_for_head_start(&counters, args.head_start_tokens, Duration::from_secs(120)) {
        warnings.push("baseline: head-start not reached within 120s".to_string());
    }
    thread::sleep(duration);
    stop.store(true, Ordering::Release);

    let mut streams = Vec::with_capacity(args.bg_concurrency);
    for worker in bg_handles {
        streams.push(worker.join().expect("baseline worker panicked")?);
    }
    if streams.iter().any(|s| s.finished_early) {
        warnings.push(
            "baseline: a background stream hit --bg-output-len before the window closed"
                .to_string(),
        );
    }
    Ok(opt_summarize(&collect_gaps(&streams)))
}

fn run_mixed_load(
    model: &mut dyn BenchModel,
    cli: &Cli,
    model_type: ModelType,
    load_ms: f64,
    cuda_graph: bool,
    args: &MixedArgs,
) -> Result<BenchReport> {
    ensure!(args.bg_concurrency > 0, "--bg-concurrency must be > 0");
    ensure!(args.bg_prompt_len > 0, "--bg-prompt-len must be > 0");
    ensure!(args.bg_output_len > 0, "--bg-output-len must be > 0");
    ensure!(args.inj_prompt_len > 0, "--inj-prompt-len must be > 0");
    ensure!(args.inj_output_len > 0, "--inj-output-len must be > 0");
    ensure!(args.num_injections > 0, "--num-injections must be > 0");
    ensure!(args.qps > 0.0, "--qps must be > 0");
    ensure!(
        (0.0..=1.0).contains(&args.inj_warm_frac),
        "--inj-warm-frac must be in [0.0, 1.0]"
    );

    let handle = model.scheduler_handle().context(
        "mixed-load requires a scheduler-backed continuous-batching model; \
         this model exposes no scheduler handle",
    )?;

    let mut warnings = Vec::new();

    info!(
        "mixed-load warmup: {} round(s) at bg_concurrency={}",
        args.run.warmup, args.bg_concurrency
    );
    mixed_warmup(
        &handle,
        args.bg_prompt_len,
        args.bg_concurrency,
        args.run.warmup,
    )?;

    // ---- Mixed phase ----
    info!(
        "mixed-load: {} background decode streams (prompt={}, output={}); injecting {} prompt(s) of {} tokens at {} QPS",
        args.bg_concurrency,
        args.bg_prompt_len,
        args.bg_output_len,
        args.num_injections,
        args.inj_prompt_len,
        args.qps
    );
    let stop = Arc::new(AtomicBool::new(false));
    let counters: Arc<[AtomicUsize]> = (0..args.bg_concurrency)
        .map(|_| AtomicUsize::new(0))
        .collect();
    let bg_handles = spawn_background_streams(
        &handle,
        args.bg_prompt_len,
        args.bg_output_len,
        args.bg_concurrency,
        &stop,
        &counters,
    );
    if !wait_for_head_start(&counters, args.head_start_tokens, Duration::from_secs(120)) {
        warnings.push(format!(
            "head-start of {} tokens not reached within 120s; injection started anyway",
            args.head_start_tokens
        ));
    }

    let mixed_window_start = Instant::now();
    let inj = run_injector(
        &handle,
        args.inj_prompt_len,
        args.inj_output_len,
        args.qps,
        args.num_injections,
        args.inj_warm_frac,
    )?;
    stop.store(true, Ordering::Release);
    let mixed_window = mixed_window_start.elapsed();

    let mut streams = Vec::with_capacity(args.bg_concurrency);
    for worker in bg_handles {
        streams.push(worker.join().expect("background worker panicked")?);
    }

    if inj.overruns > 0 {
        warnings.push(format!(
            "{} injection(s) overran the {:.0}ms QPS slot (prefill longer than 1/qps); arrivals were not evenly paced",
            inj.overruns,
            1000.0 / args.qps
        ));
    }
    let early = streams.iter().filter(|s| s.finished_early).count();
    if early > 0 {
        warnings.push(format!(
            "{early} background stream(s) hit --bg-output-len before the run ended; raise --bg-output-len to keep steady-state concurrency constant"
        ));
    }

    let mixed_itl = build_mixed_itl(&streams, &inj.windows).context(
        "no background decode gaps recorded; increase --bg-output-len or --num-injections",
    )?;

    // ---- Baseline phase (decode-only control over the same wall-clock) ----
    let baseline_itl = if args.skip_baseline {
        None
    } else {
        info!(
            "mixed-load baseline: decode-only for {:.1}s",
            mixed_window.as_secs_f64()
        );
        run_baseline(&handle, args, mixed_window, &mut warnings)?
    };

    let mixed_p50_ms = mixed_itl.all.p50_ms;
    let mixed_p99_ms = mixed_itl.all.p99_ms;
    let decision_inputs = MixedDecisionInputs {
        baseline_p50_ms: baseline_itl.as_ref().map(|b| b.p50_ms),
        baseline_p99_ms: baseline_itl.as_ref().map(|b| b.p99_ms),
        mixed_p50_ms,
        mixed_p99_ms,
        p99_delta_ms: baseline_itl.as_ref().map(|b| mixed_p99_ms - b.p99_ms),
        p99_delta_pct: baseline_itl
            .as_ref()
            .map(|b| delta_pct(mixed_p99_ms, b.p99_ms)),
    };

    Ok(BenchReport::Mixed(Box::new(MixedLoadReport {
        commit: git_short_commit(),
        date: today_date(),
        gpu: gpu_name(),
        run: run_info(cli, "mixed", model_type, load_ms, cuda_graph),
        config: MixedLoadConfig {
            bg_prompt_len: args.bg_prompt_len,
            bg_concurrency: args.bg_concurrency,
            bg_output_len: args.bg_output_len,
            inj_prompt_len: args.inj_prompt_len,
            inj_output_len: args.inj_output_len,
            qps: args.qps,
            num_injections: args.num_injections,
            inj_warm_frac: args.inj_warm_frac,
            warmup: args.run.warmup,
            seed: args.run.seed,
        },
        baseline_itl,
        mixed_itl,
        injections: inj.records,
        decision_inputs,
        warnings,
    })))
}

fn build_request_metrics(timings: &[GenTimings]) -> RequestMetrics {
    let ttfts: Vec<Duration> = timings.iter().map(|t| t.ttft).collect();
    let e2e: Vec<Duration> = timings.iter().map(|t| t.total).collect();
    let first_steps: Vec<Duration> = timings
        .iter()
        .filter_map(|t| t.tbt.first().copied())
        .collect();
    let steady: Vec<Duration> = timings
        .iter()
        .flat_map(|t| t.tbt.iter().skip(1).copied())
        .collect();
    let generated: Vec<usize> = timings.iter().map(|t| t.emitted_tokens).collect();
    let generated_token_traces: Vec<GeneratedTokenTrace> = timings
        .iter()
        .map(|timing| generated_token_trace(&timing.generated_tokens))
        .collect();

    let total_emitted: usize = timings.iter().map(|t| t.emitted_tokens).sum();
    let total_request_time: Duration = timings.iter().map(|t| t.total).sum();
    let total_decode_steps: usize = timings.iter().map(|t| t.decode_tokens_for_rate).sum();
    let total_decode_time: Duration = timings.iter().map(|t| t.decode_time_for_rate).sum();

    RequestMetrics {
        ttft_ms: summarize_durations(&ttfts),
        first_decode_step_ms: (!first_steps.is_empty()).then(|| summarize_durations(&first_steps)),
        steady_tpot_ms: (!steady.is_empty()).then(|| summarize_durations(&steady)),
        e2e_ms: summarize_durations(&e2e),
        generated_tokens: summarize_counts(&generated),
        generated_token_traces,
        request_tok_s: aggregate_tok_s(total_emitted, total_request_time),
        decode_tok_s: aggregate_tok_s(total_decode_steps, total_decode_time),
    }
}

fn build_request_iterations(timings: &[GenTimings]) -> Vec<RequestIterationTiming> {
    timings
        .iter()
        .enumerate()
        .map(|(index, timing)| {
            let steady: Vec<Duration> = timing.tbt.iter().skip(1).copied().collect();
            RequestIterationTiming {
                index,
                ttft_ms: dur_ms(timing.ttft),
                first_decode_step_ms: timing.tbt.first().copied().map(dur_ms),
                steady_tpot_ms: (!steady.is_empty()).then(|| summarize_durations(&steady)),
                e2e_ms: dur_ms(timing.total),
                generated_tokens: timing.emitted_tokens,
                generated_token_trace: generated_token_trace(&timing.generated_tokens),
            }
        })
        .collect()
}

fn run_info(
    cli: &Cli,
    command: &'static str,
    model_type: ModelType,
    load_ms: f64,
    cuda_graph: bool,
) -> RunInfo {
    RunInfo {
        command,
        model_path: cli.model_path.clone(),
        model_type: format!("{model_type:?}"),
        cuda_graph,
        load_ms,
        label: cli.label.clone(),
    }
}

fn bench_request(
    model: &mut dyn BenchModel,
    tokenizer: &DynTokenizer,
    cli: &Cli,
    model_type: ModelType,
    load_ms: f64,
    cuda_graph: bool,
    args: &RequestArgs,
) -> Result<BenchReport> {
    let mut prompt = resolve_prompt_input(
        &args.prompt_input,
        tokenizer,
        Some(DEFAULT_REQUEST_PROMPT),
        None,
    )?;
    // A `--prompt-len` workload is synthetic: give every concurrent request a
    // distinct seeded-random prompt so the decode streams diverge and MoE
    // routing is realistic. An explicit `--prompt`/`--prompt-file` (or the
    // default text) is the caller's chosen prompt and is replicated as-is.
    let synthetic = args.prompt_input.prompt_len.is_some();
    let prompts: Vec<Vec<u32>> = if synthetic {
        // 0 = one distinct prompt per request (fully diverse). Otherwise tile
        // `distinct` unique prompts across the batch: idx → idx % distinct.
        let distinct = if args.distinct_prompts == 0 {
            args.concurrency
        } else {
            args.distinct_prompts.min(args.concurrency)
        };
        prompt.descriptor.source = format!(
            "synthetic-random[{SYNTHETIC_TOKEN_LO}..{SYNTHETIC_TOKEN_HI}) seed={} distinct={distinct}/{}",
            args.run.seed, args.concurrency
        );
        (0..args.concurrency)
            .map(|idx| synthetic_random_prompt(prompt.tokens.len(), args.run.seed, idx % distinct))
            .collect()
    } else {
        vec![prompt.tokens.clone(); args.concurrency]
    };
    info!(
        "Starting request benchmark: prompt_tokens={} output_len={} concurrency={} warmup={} iters={} seed={} source={}",
        prompt.descriptor.prompt_tokens,
        args.output_len,
        args.concurrency,
        args.run.warmup,
        args.run.iters,
        args.run.seed,
        prompt.descriptor.source,
    );
    let timings = measure_timings(
        model,
        &prompts,
        args.output_len,
        &args.run,
        cli.cuda_profiler_capture,
    )?;
    Ok(BenchReport::Request(Box::new(RequestReport {
        run: run_info(cli, "request", model_type, load_ms, cuda_graph),
        workload: RequestWorkload {
            prompt: prompt.descriptor,
            output_len: args.output_len,
            concurrency: args.concurrency,
            warmup: args.run.warmup,
            iters: args.run.iters,
            seed: args.run.seed,
        },
        metrics: build_request_metrics(&timings),
        iterations: build_request_iterations(&timings),
    })))
}

fn bench_matrix(
    model: &mut dyn BenchModel,
    cli: &Cli,
    model_type: ModelType,
    load_ms: f64,
    cuda_graph: bool,
    args: &MatrixArgs,
) -> Result<BenchReport> {
    validate_run_args(&args.run)?;
    let prompt_lens = normalize_sizes(&args.prompt_lens, "--prompt-lens")?;
    let output_lens = normalize_sizes(&args.output_lens, "--output-lens")?;
    info!(
        "Starting matrix benchmark: prompt_lens={:?} output_lens={:?} warmup={} iters={} seed={}",
        prompt_lens, output_lens, args.run.warmup, args.run.iters, args.run.seed
    );

    let mut cells = Vec::with_capacity(prompt_lens.len() * output_lens.len());
    for &prompt_len in &prompt_lens {
        let prompt_tokens = synthetic_prompt_tokens(prompt_len);
        for &output_len in &output_lens {
            debug!(
                "Running matrix cell: prompt_len={} output_len={}",
                prompt_len, output_len
            );
            let timings = measure_timings(
                model,
                std::slice::from_ref(&prompt_tokens),
                output_len,
                &args.run,
                cli.cuda_profiler_capture,
            )?;
            let metrics = build_request_metrics(&timings);
            cells.push(MatrixCell {
                prompt_len,
                output_len,
                ttft_ms: metrics.ttft_ms,
                e2e_ms: metrics.e2e_ms,
                first_decode_step_ms: metrics.first_decode_step_ms,
                steady_tpot_ms: metrics.steady_tpot_ms,
                generated_tokens: metrics.generated_tokens,
                request_tok_s: metrics.request_tok_s,
                decode_tok_s: metrics.decode_tok_s,
            });
        }
    }

    Ok(BenchReport::Matrix(MatrixReport {
        run: run_info(cli, "matrix", model_type, load_ms, cuda_graph),
        workload: MatrixWorkload {
            prompt_lens,
            output_lens,
            warmup: args.run.warmup,
            iters: args.run.iters,
            seed: args.run.seed,
            synthetic_pattern: SYNTHETIC_PATTERN,
        },
        cells,
    }))
}

fn bench_curve(
    model: &mut dyn BenchModel,
    tokenizer: &DynTokenizer,
    cli: &Cli,
    model_type: ModelType,
    load_ms: f64,
    cuda_graph: bool,
    args: &CurveArgs,
) -> Result<BenchReport> {
    ensure!(args.window > 0, "--window must be > 0");
    ensure!(args.output_len >= 2, "--output-len must be >= 2 for curve");

    let prompt = resolve_prompt_input(
        &args.prompt_input,
        tokenizer,
        None,
        Some(DEFAULT_CURVE_PROMPT_LEN),
    )?;
    info!(
        "Starting curve benchmark: prompt_tokens={} output_len={} window={} warmup={} iters={} seed={}",
        prompt.descriptor.prompt_tokens,
        args.output_len,
        args.window,
        args.run.warmup,
        args.run.iters,
        args.run.seed
    );
    let timings = measure_timings(
        model,
        std::slice::from_ref(&prompt.tokens),
        args.output_len,
        &args.run,
        cli.cuda_profiler_capture,
    )?;

    let mut tbt_by_pos: Vec<Vec<Duration>> = Vec::new();
    for timing in &timings {
        for (idx, &duration) in timing.tbt.iter().enumerate() {
            if idx >= tbt_by_pos.len() {
                tbt_by_pos.push(Vec::with_capacity(args.run.iters));
            }
            tbt_by_pos[idx].push(duration);
        }
    }

    let mut windows = Vec::new();
    let mut pos = 0usize;
    while pos < tbt_by_pos.len() {
        let end = (pos + args.window).min(tbt_by_pos.len());
        let mut samples = Vec::new();
        for bucket in &tbt_by_pos[pos..end] {
            samples.extend_from_slice(bucket);
        }
        if !samples.is_empty() {
            let stats = summarize_durations(&samples);
            windows.push(CurveWindow {
                ctx_start: prompt.descriptor.prompt_tokens + pos + 1,
                ctx_end: prompt.descriptor.prompt_tokens + end,
                decode_tok_s: (stats.avg_ms > 0.0).then(|| 1000.0 / stats.avg_ms),
                tpot_ms: stats,
            });
        }
        pos = end;
    }

    Ok(BenchReport::Curve(CurveReport {
        run: run_info(cli, "curve", model_type, load_ms, cuda_graph),
        workload: CurveWorkload {
            prompt: prompt.descriptor,
            output_len: args.output_len,
            window: args.window,
            warmup: args.run.warmup,
            iters: args.run.iters,
            seed: args.run.seed,
        },
        windows,
    }))
}

fn render_text(report: &BenchReport) -> String {
    let mut out = String::new();
    match report {
        BenchReport::Request(report) => {
            let _ = writeln!(out, "bench_serving request\n");
            push_table(&mut out, &render_request_meta(report));
            out.push('\n');
            push_table(
                &mut out,
                &render_duration_table(
                    std::iter::once(("ttft_ms".to_string(), report.metrics.ttft_ms.clone()))
                        .chain(
                            report
                                .metrics
                                .first_decode_step_ms
                                .clone()
                                .into_iter()
                                .map(|stats| ("first_decode_step_ms".to_string(), stats)),
                        )
                        .chain(
                            report
                                .metrics
                                .steady_tpot_ms
                                .clone()
                                .into_iter()
                                .map(|stats| ("steady_tpot_ms".to_string(), stats)),
                        )
                        .chain(std::iter::once((
                            "e2e_ms".to_string(),
                            report.metrics.e2e_ms.clone(),
                        )))
                        .collect(),
                ),
            );
            out.push('\n');
            push_table(&mut out, &render_request_summary(report));
        }
        BenchReport::Matrix(report) => {
            let _ = writeln!(out, "bench_serving matrix\n");
            push_table(&mut out, &render_matrix_meta(report));
            out.push('\n');
            push_table(&mut out, &render_matrix_table(report));
        }
        BenchReport::Curve(report) => {
            let _ = writeln!(out, "bench_serving curve\n");
            push_table(&mut out, &render_curve_meta(report));
            out.push('\n');
            push_table(&mut out, &render_curve_table(report));
        }
        BenchReport::Mixed(report) => out.push_str(&render_mixed_text(report)),
    }
    out
}

fn render_mixed_text(report: &MixedLoadReport) -> String {
    let mut out = String::new();
    let _ = writeln!(out, "bench_serving mixed-load ITL\n");

    let cfg = &report.config;
    let mut meta = render_run_summary(&report.run);
    meta.add_row(vec![
        key_cell("commit / gpu"),
        value_cell(format!("{} / {}", report.commit, report.gpu)),
    ]);
    meta.add_row(vec![
        key_cell("bg (prompt,conc,out)"),
        value_cell(format!(
            "({},{},{})",
            cfg.bg_prompt_len, cfg.bg_concurrency, cfg.bg_output_len
        )),
    ]);
    meta.add_row(vec![
        key_cell("injection (prompt,out)"),
        value_cell(format!(
            "({},{})  warm_frac={}",
            cfg.inj_prompt_len, cfg.inj_output_len, cfg.inj_warm_frac
        )),
    ]);
    meta.add_row(vec![
        key_cell("qps / num_injections"),
        value_cell(format!("{} / {}", cfg.qps, cfg.num_injections)),
    ]);
    meta.add_row(vec![
        key_cell("warmup / seed"),
        value_cell(format!("{} / {}", cfg.warmup, cfg.seed)),
    ]);
    push_table(&mut out, &meta);
    out.push('\n');

    let mut rows = Vec::new();
    if let Some(baseline) = &report.baseline_itl {
        rows.push(("baseline_itl".to_string(), baseline.clone()));
    }
    rows.push(("mixed_itl_all".to_string(), report.mixed_itl.all.clone()));
    if let Some(steady) = &report.mixed_itl.steady {
        rows.push(("mixed_itl_steady".to_string(), steady.clone()));
    }
    if let Some(stall) = &report.mixed_itl.stall {
        rows.push(("mixed_itl_stall".to_string(), stall.clone()));
    }
    push_table(&mut out, &render_duration_table(rows));
    out.push('\n');

    let total = report.mixed_itl.total_gap_count;
    let stalled = report.mixed_itl.stall_gap_count;
    let stall_pct = if total > 0 {
        100.0 * stalled as f64 / total as f64
    } else {
        0.0
    };
    let _ = writeln!(out, "stall gaps: {stalled}/{total} ({stall_pct:.1}%)");

    let dur = |ms: f64| Duration::from_secs_f64(ms / 1000.0);
    let prefill_line = |label: &str, ms: &[Duration]| {
        if ms.is_empty() {
            return String::new();
        }
        let s = summarize_durations(ms);
        format!(
            "{label}: p50={:.2}ms  p99={:.2}ms  max={:.2}ms (n={})\n",
            s.p50_ms,
            s.p99_ms,
            s.max_ms,
            ms.len()
        )
    };
    if !report.injections.is_empty() {
        let cold: Vec<Duration> = report
            .injections
            .iter()
            .filter(|r| !r.warm)
            .map(|r| dur(r.prefill_ms))
            .collect();
        let warm: Vec<Duration> = report
            .injections
            .iter()
            .filter(|r| r.warm)
            .map(|r| dur(r.prefill_ms))
            .collect();
        out.push_str(&prefill_line("injected prefill (cold)", &cold));
        out.push_str(&prefill_line("injected prefill (warm)", &warm));
    }

    let d = &report.decision_inputs;
    match (
        d.baseline_p50_ms,
        d.baseline_p99_ms,
        d.p99_delta_pct,
        d.p99_delta_ms,
    ) {
        (Some(bp50), Some(bp99), Some(dpct), Some(dms)) => {
            let _ = writeln!(
                out,
                "\nITL p50: baseline {:.2}ms → mixed {:.2}ms",
                bp50, d.mixed_p50_ms
            );
            let _ = writeln!(
                out,
                "ITL p99: baseline {:.2}ms → mixed {:.2}ms ({}, {:+.2}ms)",
                bp99,
                d.mixed_p99_ms,
                format_delta(dpct),
                dms
            );
        }
        _ => {
            let _ = writeln!(
                out,
                "\nITL (mixed, no baseline): p50={:.2}ms  p99={:.2}ms",
                d.mixed_p50_ms, d.mixed_p99_ms
            );
        }
    }

    for warning in &report.warnings {
        let _ = writeln!(out, "warning: {warning}");
    }
    out
}

fn emit_report(cli: &Cli, report: &BenchReport) -> Result<()> {
    let rendered = match cli.format {
        OutputFormat::Text => render_text(report),
        OutputFormat::Json => serde_json::to_string_pretty(report)?,
    };

    if let Some(path) = &cli.out {
        fs::write(path, &rendered).with_context(|| format!("failed to write report to {path}"))?;
        info!("Wrote benchmark report to {}", path);
    }

    println!("{rendered}");
    Ok(())
}

fn run_command(
    cli: &Cli,
    model_type: ModelType,
    load_ms: f64,
    cuda_graph: bool,
    model: &mut dyn BenchModel,
    tokenizer: &DynTokenizer,
) -> Result<BenchReport> {
    match &cli.command {
        Command::Request(args) => {
            bench_request(model, tokenizer, cli, model_type, load_ms, cuda_graph, args)
        }
        Command::Matrix(args) => bench_matrix(model, cli, model_type, load_ms, cuda_graph, args),
        Command::Curve(args) => {
            bench_curve(model, tokenizer, cli, model_type, load_ms, cuda_graph, args)
        }
        Command::Mixed(args) => run_mixed_load(model, cli, model_type, load_ms, cuda_graph, args),
        Command::Snapshot(_) | Command::Compare(_) => unreachable!(),
    }
}

// ---------------------------------------------------------------------------
// Snapshot / Compare
// ---------------------------------------------------------------------------

fn shell_output(program: &str, args: &[&str]) -> Option<String> {
    std::process::Command::new(program)
        .args(args)
        .output()
        .ok()
        .filter(|o| o.status.success())
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
}

fn git_short_commit() -> String {
    shell_output("git", &["rev-parse", "--short", "HEAD"]).unwrap_or_else(|| "unknown".into())
}

fn gpu_name() -> String {
    shell_output(
        "nvidia-smi",
        &["--query-gpu=name", "--format=csv,noheader", "--id=0"],
    )
    .unwrap_or_else(|| "unknown".into())
}

/// Produce a filesystem-safe slug from a GPU name string.
///
/// `"NVIDIA GeForce RTX 5070 Ti"` → `"rtx-5070-ti"`
fn gpu_slug_from(name: &str) -> String {
    let stripped = name
        .strip_prefix("NVIDIA GeForce ")
        .or_else(|| name.strip_prefix("NVIDIA "))
        .unwrap_or(name);
    stripped
        .to_lowercase()
        .chars()
        .map(|c| {
            if c.is_alphanumeric() || c == '-' {
                c
            } else {
                '-'
            }
        })
        .collect::<String>()
        .split('-')
        .filter(|s| !s.is_empty())
        .collect::<Vec<_>>()
        .join("-")
}

fn today_date() -> String {
    shell_output("date", &["+%Y-%m-%d"]).unwrap_or_else(|| "unknown".into())
}

fn model_display_name(model_path: &str) -> String {
    Path::new(model_path)
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("unknown")
        .to_string()
}

fn delta_pct(current: f64, baseline: f64) -> f64 {
    if baseline == 0.0 {
        return 0.0;
    }
    (current - baseline) / baseline * 100.0
}

fn format_delta(pct: f64) -> String {
    if pct >= 0.0 {
        format!("+{pct:.1}%")
    } else {
        format!("{pct:.1}%")
    }
}

/// Measure cold prefill TTFT: each iteration uses a distinct (salted) prompt so
/// the default-on prefix cache never hits, isolating real prefill compute.
/// Mirrors `measure_timings` but varies the prompt per iteration.
fn measure_cold_prefill_timings(
    model: &mut dyn BenchModel,
    prompt_len: usize,
    run: &RunArgs,
) -> Result<Vec<GenTimings>> {
    validate_run_args(run)?;
    let sampling = greedy_sampling();
    let mut rng = StdRng::seed_from_u64(run.seed);
    // Warmup with distinct prompts (content only matters for JIT/allocator warmup).
    for i in 0..run.warmup {
        let toks = synthetic_prompt_tokens_salted(prompt_len, 100_000 + i);
        let _ = model.timed_generation_batch(
            std::slice::from_ref(&toks),
            SNAPSHOT_PREFILL_OUTPUT_LEN,
            &sampling,
            &mut rng,
        );
    }
    let mut timings = Vec::with_capacity(run.iters);
    for i in 0..run.iters {
        // salt 1.. keeps the first token distinct from the cached prompt (salt 0)
        // and from every other iteration → guaranteed cache miss each time.
        let toks = synthetic_prompt_tokens_salted(prompt_len, i + 1);
        timings.extend(model.timed_generation_batch(
            std::slice::from_ref(&toks),
            SNAPSHOT_PREFILL_OUTPUT_LEN,
            &sampling,
            &mut rng,
        ));
    }
    Ok(timings)
}

/// Trim a profile's per-iteration token traces down to a single determinism
/// fingerprint. `compare` never reads the traces and (under greedy decode)
/// repeated-prompt profiles emit identical copies, so keeping one hash keeps the
/// committed snapshot small without losing the "did the output change" signal.
fn snapshot_metrics(mut metrics: RequestMetrics) -> RequestMetrics {
    metrics.generated_token_traces.truncate(1);
    metrics
}

fn run_snapshot(
    model: &mut dyn BenchModel,
    cli: &Cli,
    model_type: ModelType,
    args: &SnapshotArgs,
) -> Result<()> {
    let prefill_prompt_len = snapshot_prefill_prompt_len(model_type);

    info!("Running prefill-heavy cold ({prefill_prompt_len},{SNAPSHOT_PREFILL_OUTPUT_LEN})");
    let prefill_timings = measure_cold_prefill_timings(model, prefill_prompt_len, &args.run)?;
    let prefill_metrics = snapshot_metrics(build_request_metrics(&prefill_timings));

    info!(
        "Running prefill-heavy cached ({prefill_prompt_len},{SNAPSHOT_PREFILL_OUTPUT_LEN}) — repeated prompt, prefix-cache warm"
    );
    let cached_tokens = synthetic_prompt_tokens(prefill_prompt_len);
    let cached_timings = measure_timings(
        model,
        std::slice::from_ref(&cached_tokens),
        SNAPSHOT_PREFILL_OUTPUT_LEN,
        &args.run,
        cli.cuda_profiler_capture,
    )?;
    let cached_metrics = snapshot_metrics(build_request_metrics(&cached_timings));

    info!("Running decode-heavy ({SNAPSHOT_DECODE_PROMPT_LEN},{SNAPSHOT_DECODE_OUTPUT_LEN})");
    let decode_tokens = synthetic_prompt_tokens(SNAPSHOT_DECODE_PROMPT_LEN);
    let decode_timings = measure_timings(
        model,
        std::slice::from_ref(&decode_tokens),
        SNAPSHOT_DECODE_OUTPUT_LEN,
        &args.run,
        cli.cuda_profiler_capture,
    )?;
    let decode_metrics = snapshot_metrics(build_request_metrics(&decode_timings));

    let model_name = model_display_name(&cli.model_path);
    let gpu = gpu_name();
    let parallel = match model_type {
        #[cfg(feature = "kimi-k2")]
        ModelType::KimiK2 => Some(format!(
            "tp{}-dp{}-{}",
            cli.tp_size,
            cli.dp_size,
            format!("{:?}", cli.ep_backend).to_lowercase()
        )),
        _ => None,
    };
    let report = SnapshotReport {
        commit: git_short_commit(),
        date: today_date(),
        model: model_name.clone(),
        gpu: gpu.clone(),
        parallel,
        prefill_heavy: SnapshotProfile {
            prompt_len: prefill_prompt_len,
            output_len: SNAPSHOT_PREFILL_OUTPUT_LEN,
            metrics: prefill_metrics,
        },
        prefill_cached: Some(SnapshotProfile {
            prompt_len: prefill_prompt_len,
            output_len: SNAPSHOT_PREFILL_OUTPUT_LEN,
            metrics: cached_metrics,
        }),
        decode_heavy: SnapshotProfile {
            prompt_len: SNAPSHOT_DECODE_PROMPT_LEN,
            output_len: SNAPSHOT_DECODE_OUTPUT_LEN,
            metrics: decode_metrics,
        },
    };

    let dir = Path::new(SNAPSHOT_DIR).join(gpu_slug_from(&gpu));
    fs::create_dir_all(&dir)?;
    let filename = model_name.to_lowercase();
    let path = dir.join(format!("{filename}.json"));
    let snapshot_json = serde_json::to_string_pretty(&report)?;
    fs::write(&path, format!("{snapshot_json}\n"))?;

    println!("{}", render_snapshot_text(&report, &path));
    Ok(())
}

fn render_snapshot_text(report: &SnapshotReport, path: &Path) -> String {
    let mut out = String::new();
    let _ = writeln!(out, "bench_serving snapshot\n");
    let _ = writeln!(out, "model:  {}", report.model);
    let _ = writeln!(out, "gpu:    {}", report.gpu);
    if let Some(parallel) = &report.parallel {
        let _ = writeln!(out, "shape:  {parallel}");
    }
    let _ = writeln!(out, "commit: {}\n", report.commit);
    let _ = writeln!(
        out,
        "prefill_heavy ({},{}) [cold — distinct prompt/iter]:",
        report.prefill_heavy.prompt_len, report.prefill_heavy.output_len
    );
    let _ = writeln!(
        out,
        "  TTFT  p50={:.2}ms  p99={:.2}ms",
        report.prefill_heavy.metrics.ttft_ms.p50_ms, report.prefill_heavy.metrics.ttft_ms.p99_ms
    );
    if let Some(cached) = &report.prefill_cached {
        let _ = writeln!(
            out,
            "\nprefill_cached ({},{}) [warm — repeated prompt, prefix-cache hit]:",
            cached.prompt_len, cached.output_len
        );
        let _ = writeln!(
            out,
            "  TTFT  p50={:.2}ms  p99={:.2}ms",
            cached.metrics.ttft_ms.p50_ms, cached.metrics.ttft_ms.p99_ms
        );
    }
    let _ = writeln!(
        out,
        "\ndecode_heavy ({},{}):",
        report.decode_heavy.prompt_len, report.decode_heavy.output_len
    );
    if let Some(tpot) = &report.decode_heavy.metrics.steady_tpot_ms {
        let _ = writeln!(
            out,
            "  TPOT  p50={:.2}ms  p99={:.2}ms",
            tpot.p50_ms, tpot.p99_ms
        );
    }
    let _ = writeln!(out, "\nwritten to {}", path.display());
    out
}

fn run_compare(args: &CompareArgs) -> Result<()> {
    let current_content = fs::read_to_string(&args.path).with_context(|| {
        format!(
            "snapshot not found: {}\nrun `bench_serving snapshot` first",
            args.path
        )
    })?;
    let current: SnapshotReport =
        serde_json::from_str(&current_content).context("failed to parse current snapshot")?;

    // Resolve repo-root-relative path for git show
    let abs_path = fs::canonicalize(&args.path)?;
    let toplevel =
        shell_output("git", &["rev-parse", "--show-toplevel"]).context("not a git repository")?;
    let root = PathBuf::from(&toplevel);
    let rel_path = abs_path
        .strip_prefix(&root)
        .context("snapshot file is outside the git repository")?;

    let git_output = std::process::Command::new("git")
        .args(["show", &format!("{}:{}", args.baseline, rel_path.display())])
        .output()
        .context("failed to run git show")?;

    if !git_output.status.success() {
        anyhow::bail!(
            "no baseline at {}:{}\ncommit the current snapshot to establish a baseline",
            args.baseline,
            rel_path.display()
        );
    }

    let baseline: SnapshotReport =
        serde_json::from_slice(&git_output.stdout).context("failed to parse baseline snapshot")?;

    // Guard against comparing snapshots with different profile shapes
    ensure!(
        current.prefill_heavy.prompt_len == baseline.prefill_heavy.prompt_len
            && current.prefill_heavy.output_len == baseline.prefill_heavy.output_len
            && current.decode_heavy.prompt_len == baseline.decode_heavy.prompt_len
            && current.decode_heavy.output_len == baseline.decode_heavy.output_len,
        "profile shape mismatch: current ({},{}) + ({},{}) vs baseline ({},{}) + ({},{})\n\
         the snapshot profiles were changed — re-baseline by committing a fresh snapshot",
        current.prefill_heavy.prompt_len,
        current.prefill_heavy.output_len,
        current.decode_heavy.prompt_len,
        current.decode_heavy.output_len,
        baseline.prefill_heavy.prompt_len,
        baseline.prefill_heavy.output_len,
        baseline.decode_heavy.prompt_len,
        baseline.decode_heavy.output_len,
    );
    println!("{}", render_comparison(&current, &baseline, &args.baseline));
    Ok(())
}

fn render_comparison(
    current: &SnapshotReport,
    baseline: &SnapshotReport,
    ref_name: &str,
) -> String {
    let mut out = String::new();
    let _ = writeln!(out, "bench_serving compare\n");
    let _ = writeln!(
        out,
        "comparing {} (working tree) vs {} ({ref_name})\n",
        current.commit, baseline.commit
    );

    let mut table = new_table();
    table.set_header(vec![
        Cell::new("metric"),
        Cell::new("current").set_alignment(CellAlignment::Right),
        Cell::new("baseline").set_alignment(CellAlignment::Right),
        Cell::new("delta").set_alignment(CellAlignment::Right),
    ]);

    let pf = &current.prefill_heavy;
    let pf_b = &baseline.prefill_heavy;
    let pf_label = format!("({},{})", pf.prompt_len, pf.output_len);

    for (stat, cur, base) in [
        (
            "p50",
            pf.metrics.ttft_ms.p50_ms,
            pf_b.metrics.ttft_ms.p50_ms,
        ),
        (
            "p99",
            pf.metrics.ttft_ms.p99_ms,
            pf_b.metrics.ttft_ms.p99_ms,
        ),
    ] {
        table.add_row(vec![
            key_cell(format!("TTFT {stat} {pf_label}")),
            numeric_cell(format!("{cur:.2}ms")),
            numeric_cell(format!("{base:.2}ms")),
            numeric_cell(format_delta(delta_pct(cur, base))),
        ]);
    }

    let dc_label = format!(
        "({},{})",
        current.decode_heavy.prompt_len, current.decode_heavy.output_len
    );
    if let (Some(cur_tpot), Some(base_tpot)) = (
        &current.decode_heavy.metrics.steady_tpot_ms,
        &baseline.decode_heavy.metrics.steady_tpot_ms,
    ) {
        for (stat, cur, base) in [
            ("p50", cur_tpot.p50_ms, base_tpot.p50_ms),
            ("p99", cur_tpot.p99_ms, base_tpot.p99_ms),
        ] {
            table.add_row(vec![
                key_cell(format!("TPOT {stat} {dc_label}")),
                numeric_cell(format!("{cur:.2}ms")),
                numeric_cell(format!("{base:.2}ms")),
                numeric_cell(format_delta(delta_pct(cur, base))),
            ]);
        }
    }

    push_table(&mut out, &table);

    // Regression check
    let mut regressions = Vec::new();
    let ttft_d = delta_pct(
        current.prefill_heavy.metrics.ttft_ms.p50_ms,
        baseline.prefill_heavy.metrics.ttft_ms.p50_ms,
    );
    if ttft_d > REGRESSION_TTFT_PCT {
        regressions.push(format!(
            "TTFT p50 {ttft_d:+.1}% > {REGRESSION_TTFT_PCT}% threshold"
        ));
    }
    if let (Some(cur), Some(base)) = (
        &current.decode_heavy.metrics.steady_tpot_ms,
        &baseline.decode_heavy.metrics.steady_tpot_ms,
    ) {
        let tpot_d = delta_pct(cur.p50_ms, base.p50_ms);
        if tpot_d > REGRESSION_TPOT_PCT {
            regressions.push(format!(
                "TPOT p50 {tpot_d:+.1}% > {REGRESSION_TPOT_PCT}% threshold"
            ));
        }
    }

    out.push('\n');
    if regressions.is_empty() {
        let _ = writeln!(
            out,
            "no regression detected (threshold: TPOT >{REGRESSION_TPOT_PCT}%, TTFT >{REGRESSION_TTFT_PCT}%)"
        );
    } else {
        let _ = writeln!(out, "REGRESSION DETECTED:");
        for r in &regressions {
            let _ = writeln!(out, "  {r}");
        }
    }

    out
}

fn dispatch(
    cli: &Cli,
    model_type: ModelType,
    load_ms: f64,
    cuda_graph: bool,
    model: &mut dyn BenchModel,
    tokenizer: &DynTokenizer,
) -> Result<()> {
    if let Command::Snapshot(args) = &cli.command {
        run_snapshot(model, cli, model_type, args)
    } else {
        let report = run_command(cli, model_type, load_ms, cuda_graph, model, tokenizer)?;
        emit_report(cli, &report)
    }
}

fn main() -> Result<()> {
    logging::init_default();

    let cli = Cli::parse();

    // Compare needs no model loading
    if let Command::Compare(ref args) = cli.command {
        return run_compare(args);
    }

    debug!(
        "bench_serving starting: command={} model_path={} cuda_graph={} format={:?}",
        match &cli.command {
            Command::Request(_) => "request",
            Command::Matrix(_) => "matrix",
            Command::Curve(_) => "curve",
            Command::Snapshot(_) => "snapshot",
            Command::Compare(_) => "compare",
            Command::Mixed(_) => "mixed",
        },
        cli.model_path,
        cli.cuda_graph,
        cli.format
    );
    let model_type = detect_model_type(&cli.model_path)
        .with_context(|| format!("failed to detect model type from {}", cli.model_path))?;
    debug!("Detected model type: {:?}", model_type);
    let load_start = Instant::now();

    match model_type {
        #[cfg(feature = "deepseek-v2-lite")]
        ModelType::DeepSeekV2Lite => {
            let generator = pegainfer_deepseek_v2_lite::DeepSeekV2LiteEp2Generator::load(
                Path::new(&cli.model_path),
                EngineLoadOptions {
                    enable_cuda_graph: false,
                    enable_prefill_profile: false,
                    device_ordinals: vec![0, 1],
                    parallel_config: None,
                    ep_backend: EpBackend::Nccl,
                    seed: command_seed(&cli),
                },
            )?;
            let tokenizer = load_vllm_tokenizer(&cli.model_path)?;
            let load_ms = dur_ms(load_start.elapsed());
            let mut bench = DeepSeekV2LiteBenchModel { generator };
            dispatch(&cli, model_type, load_ms, false, &mut bench, &tokenizer)
        }
        #[cfg(feature = "deepseek-v4")]
        ModelType::DeepSeekV4 => {
            let handle = pegainfer_deepseek_v4::start_engine(
                Path::new(&cli.model_path),
                EngineLoadOptions {
                    enable_cuda_graph: false,
                    enable_prefill_profile: false,
                    device_ordinals: (0..8).collect(),
                    parallel_config: None,
                    ep_backend: EpBackend::Nccl,
                    seed: command_seed(&cli),
                },
            )?;
            let tokenizer = load_vllm_tokenizer(&cli.model_path)?;
            let load_ms = dur_ms(load_start.elapsed());
            let mut bench = SchedulerBenchModel { handle };
            dispatch(&cli, model_type, load_ms, false, &mut bench, &tokenizer)
        }
        #[cfg(feature = "kimi-k2")]
        ModelType::KimiK2 => {
            let parallel = kimi_parallel_config(cli.tp_size, cli.dp_size)?;
            let handle = pegainfer_kimi_k2::start_engine(
                Path::new(&cli.model_path),
                EngineLoadOptions {
                    enable_cuda_graph: cli.cuda_graph,
                    enable_prefill_profile: false,
                    device_ordinals: (0..parallel.ep_world()).collect(),
                    parallel_config: Some(parallel),
                    ep_backend: cli.ep_backend.into(),
                    seed: command_seed(&cli),
                },
            )?;
            let tokenizer = load_vllm_tokenizer(&cli.model_path)?;
            let load_ms = dur_ms(load_start.elapsed());
            let mut bench = SchedulerBenchModel { handle };
            dispatch(
                &cli,
                model_type,
                load_ms,
                cli.cuda_graph,
                &mut bench,
                &tokenizer,
            )
        }
        ModelType::Qwen3 => {
            let handle = pegainfer_qwen3_4b::start_engine(
                Path::new(&cli.model_path),
                EngineLoadOptions {
                    enable_cuda_graph: cli.cuda_graph,
                    enable_prefill_profile: false,
                    device_ordinals: vec![0],
                    parallel_config: None,
                    ep_backend: EpBackend::Nccl,
                    seed: command_seed(&cli),
                },
            )?;
            let tokenizer = load_vllm_tokenizer(&cli.model_path)?;
            let load_ms = dur_ms(load_start.elapsed());
            let mut bench = SchedulerBenchModel { handle };
            dispatch(
                &cli,
                model_type,
                load_ms,
                cli.cuda_graph,
                &mut bench,
                &tokenizer,
            )
        }
        ModelType::Qwen35 => {
            let handle = pegainfer_qwen35_4b::start_engine_with_capacity(
                Path::new(&cli.model_path),
                EngineLoadOptions {
                    enable_cuda_graph: cli.cuda_graph,
                    enable_prefill_profile: false,
                    device_ordinals: vec![0],
                    parallel_config: None,
                    ep_backend: EpBackend::Nccl,
                    seed: command_seed(&cli),
                },
                4,
            )?;
            let tokenizer = load_vllm_tokenizer(&cli.model_path)?;
            let load_ms = dur_ms(load_start.elapsed());
            let mut bench = SchedulerBenchModel { handle };
            dispatch(
                &cli,
                model_type,
                load_ms,
                cli.cuda_graph,
                &mut bench,
                &tokenizer,
            )
        }
    }
}

#[cfg(all(test, feature = "deepseek-v2-lite"))]
mod tests {
    use super::*;

    #[test]
    fn dsv2_lite_sampling_contract_accepts_bench_params() {
        let sampling = SamplingParams {
            ignore_eos: true,
            ..SamplingParams::default()
        };

        assert_dsv2_lite_sampling_contract(&sampling);
    }

    #[test]
    #[should_panic(expected = "supports greedy decoding only")]
    fn dsv2_lite_sampling_contract_rejects_non_greedy_params() {
        let sampling = SamplingParams {
            temperature: 0.8,
            top_k: -1,
            top_p: 0.95,
            ignore_eos: true,
        };

        assert_dsv2_lite_sampling_contract(&sampling);
    }

    #[test]
    #[should_panic(expected = "requires ignore_eos=true")]
    fn dsv2_lite_sampling_contract_rejects_eos_enabled_params() {
        let sampling = SamplingParams {
            ignore_eos: false,
            ..SamplingParams::default()
        };

        assert_dsv2_lite_sampling_contract(&sampling);
    }

    #[test]
    fn dsv2_lite_attribution_timings_preserve_decode_steps() {
        let timings = timings_from_dsv2_lite_attribution(
            vec![11, 304, 608],
            3,
            60_000,
            Some(20_000),
            &[19_000, 18_000],
        );

        assert_eq!(timings.ttft, Duration::from_micros(20_000));
        assert_eq!(
            timings.tbt,
            vec![Duration::from_micros(19_000), Duration::from_micros(18_000)]
        );
        assert_eq!(timings.total, Duration::from_micros(60_000));
        assert_eq!(timings.emitted_tokens, 3);
        assert_eq!(timings.generated_tokens, vec![11, 304, 608]);
        assert_eq!(timings.decode_tokens_for_rate, 2);
        assert_eq!(timings.decode_time_for_rate, Duration::from_micros(37_000));
    }

    #[test]
    fn dsv2_lite_batched_timings_use_shared_decode_time_for_rate() {
        let timings = timings_from_dsv2_lite_batched_generation(
            pegainfer_deepseek_v2_lite::BatchedGenerationResult {
                tokens: vec![vec![11, 304, 608], vec![11, 304, 608]],
                prefill_next_token_us: vec![20_000, 21_000],
                per_token_decode_us: vec![19_000, 18_000],
                total_generation_us: 80_000,
                stats: pegainfer_deepseek_v2_lite::GenerationStats::default(),
            },
            3,
        );

        assert_eq!(timings.len(), 2);
        assert_eq!(timings[0].decode_tokens_for_rate, 4);
        assert_eq!(
            timings[0].decode_time_for_rate,
            Duration::from_micros(37_000)
        );
        assert_eq!(timings[1].decode_tokens_for_rate, 0);
        assert_eq!(timings[1].decode_time_for_rate, Duration::ZERO);

        let metrics = build_request_metrics(&timings);
        assert_eq!(metrics.steady_tpot_ms.unwrap().p50_ms, 18.0);
        assert!(
            metrics.decode_tok_s.unwrap() > 100.0,
            "batched decode tok/s should use one shared step duration instead of duplicating it per row"
        );
    }

    #[test]
    #[should_panic(expected = "timing count mismatch")]
    fn dsv2_lite_attribution_timings_fail_on_missing_decode_samples() {
        let _ = timings_from_dsv2_lite_attribution(
            vec![11, 304, 608],
            3,
            60_000,
            Some(20_000),
            &[19_000],
        );
    }

    #[test]
    #[should_panic(expected = "generated token count mismatch")]
    fn dsv2_lite_attribution_timings_fail_on_short_generation() {
        let _ =
            timings_from_dsv2_lite_attribution(vec![11, 304], 3, 60_000, Some(20_000), &[19_000]);
    }

    #[test]
    #[should_panic(expected = "zero-duration")]
    fn dsv2_lite_attribution_timings_fail_on_zero_decode_samples() {
        let _ = timings_from_dsv2_lite_attribution(vec![11, 304], 2, 60_000, Some(20_000), &[0]);
    }

    #[test]
    #[should_panic(expected = "total generation timing is zero")]
    fn dsv2_lite_attribution_timings_fail_on_zero_total_generation() {
        let _ = timings_from_dsv2_lite_attribution(vec![11, 304], 2, 0, Some(20_000), &[19_000]);
    }

    #[test]
    #[should_panic(expected = "TTFT timing is missing or zero")]
    fn dsv2_lite_attribution_timings_fail_on_missing_ttft() {
        let _ = timings_from_dsv2_lite_attribution(vec![11, 304], 2, 60_000, None, &[19_000]);
    }

    #[test]
    #[should_panic(expected = "TTFT timing is missing or zero")]
    fn dsv2_lite_attribution_timings_fail_on_zero_ttft() {
        let _ = timings_from_dsv2_lite_attribution(vec![11, 304], 2, 60_000, Some(0), &[19_000]);
    }
}
