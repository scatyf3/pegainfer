//! Issue #80 e2e: `--served-model-name` makes the OpenAI API report a clean id
//! while the tokenizer still loads from the local path. `#[ignore]` — needs GPU.

use std::env;
use std::path::PathBuf;
use std::time::Duration;

use pegainfer_core::engine::EngineLoadOptions;
use reqwest::Client;
use serde_json::Value;
use tokio_util::sync::CancellationToken;

const SERVER_PORT: u16 = 18080;
const SERVED_NAME: &str = "test-served-name";
const SERVER_READY_TIMEOUT: Duration = Duration::from_secs(120);

fn model_path() -> PathBuf {
    let path: PathBuf = env::var("PEGAINFER_TEST_MODEL_PATH")
        .unwrap_or_else(|_| concat!(env!("CARGO_MANIFEST_DIR"), "/../models/Qwen3-4B").to_string())
        .into();
    assert!(
        path.join("config.json").exists(),
        "model path missing: {}",
        path.display(),
    );
    path
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "needs GPU + Qwen3-4B model + free TCP port 18080"]
async fn served_model_name_is_reported_in_api() {
    pegainfer_core::logging::init_stderr("warn");
    let model_path = model_path();

    // Served name is deliberately not the dir basename, proving real decoupling.
    eprintln!(
        "[setup] loading engine from {} (served as '{SERVED_NAME}')",
        model_path.display()
    );
    let handle = pegainfer_qwen3_4b::start_engine(
        &model_path,
        EngineLoadOptions {
            enable_cuda_graph: true,
            device_ordinals: vec![0],
            seed: 42,
        },
    )
    .expect("engine start failed");

    let shutdown = CancellationToken::new();
    let server_task = {
        let shutdown = shutdown.clone();
        let model_path = model_path.clone();
        tokio::spawn(async move {
            pegainfer::vllm_frontend::serve(
                handle,
                &model_path,
                Some(SERVED_NAME),
                SERVER_PORT,
                shutdown,
            )
            .await
            .expect("server exited with error");
        })
    };

    wait_for_health(SERVER_PORT).await;
    eprintln!("[setup] server up on port {SERVER_PORT}");

    let client = Client::builder()
        .timeout(Duration::from_secs(60))
        .build()
        .expect("client");

    let models: Value = client
        .get(format!("http://127.0.0.1:{SERVER_PORT}/v1/models"))
        .send()
        .await
        .expect("models request")
        .json()
        .await
        .expect("models json");
    let id = models["data"][0]["id"]
        .as_str()
        .expect("missing /v1/models data[0].id");
    assert_eq!(
        id, SERVED_NAME,
        "/v1/models should report the served name, got {id:?}"
    );
    assert!(
        !id.contains('/'),
        "served model id must not leak a filesystem path: {id:?}"
    );

    let completion: Value = client
        .post(format!("http://127.0.0.1:{SERVER_PORT}/v1/completions"))
        .json(&serde_json::json!({
            "model": SERVED_NAME,
            "prompt": "Hello",
            "max_tokens": 4,
            "temperature": 0.0,
        }))
        .send()
        .await
        .expect("completion request")
        .json()
        .await
        .expect("completion json");
    assert_eq!(
        completion["model"].as_str(),
        Some(SERVED_NAME),
        "completion response should echo the served name, got {:?}",
        completion["model"]
    );

    shutdown.cancel();
    let _ = tokio::time::timeout(Duration::from_secs(15), server_task).await;
}

async fn wait_for_health(port: u16) {
    let url = format!("http://127.0.0.1:{port}/health");
    let deadline = tokio::time::Instant::now() + SERVER_READY_TIMEOUT;
    loop {
        if let Ok(resp) = reqwest::get(&url).await
            && resp.status().is_success()
        {
            return;
        }
        if tokio::time::Instant::now() >= deadline {
            panic!("server health endpoint did not respond within {SERVER_READY_TIMEOUT:?}");
        }
        tokio::time::sleep(Duration::from_millis(500)).await;
    }
}
