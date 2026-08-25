//! Model-fixture lookup for the Qwen3.5 gates.
//!
//! Two flavours, and the difference is the whole point of this module:
//!
//! * `*_or_skip` — for plain `#[test]` gates that ride along on ordinary test
//!   runs. A machine without Qwen3.5 weights must still get a green workspace
//!   run, so a missing fixture prints `SKIP` and the caller returns early.
//! * `require_*` — for `#[ignore]` gates. `#[ignore]` already keeps them out of
//!   ordinary runs, so the only way one executes is a maintainer naming it.
//!   Once named, a missing or invalid fixture is a failure: skipping there
//!   reports success for a checkpoint that never loaded.
//!
//! Keep that invariant when adding a gate: `#[ignore]` takes `require_*`.

use std::path::Path;

const MODEL_PATH_ENV: &str = "PEGAINFER_TEST_MODEL_PATH";
#[allow(dead_code)]
const FRONTEND_MODEL_PATH_ENV: &str = "PEGAINFER_TEST_FRONTEND_MODEL_PATH";

pub(crate) fn model_path_or_skip(test_name: &str) -> Option<String> {
    resolve_model_path()
        .map_err(|reason| skip(test_name, &reason))
        .ok()
}

/// `model_path_or_skip` for explicitly selected gates: panics instead of skipping.
#[allow(dead_code)]
pub(crate) fn require_model_path(test_name: &str) -> String {
    resolve_model_path().unwrap_or_else(|reason| fail(test_name, &reason))
}

#[allow(dead_code)]
pub(crate) fn frontend_model_path_or_skip(
    engine_model_path: &Path,
    test_name: &str,
) -> Option<String> {
    resolve_frontend_model_path(engine_model_path)
        .map_err(|reason| skip(test_name, &reason))
        .ok()
}

/// `frontend_model_path_or_skip` for explicitly selected gates.
#[allow(dead_code)]
pub(crate) fn require_frontend_model_path(engine_model_path: &Path, test_name: &str) -> String {
    resolve_frontend_model_path(engine_model_path).unwrap_or_else(|reason| fail(test_name, &reason))
}

fn resolve_model_path() -> Result<String, String> {
    fixture_path_from_env(MODEL_PATH_ENV)
}

fn resolve_frontend_model_path(engine_model_path: &Path) -> Result<String, String> {
    match std::env::var(FRONTEND_MODEL_PATH_ENV) {
        Ok(path) => validated_fixture_path(FRONTEND_MODEL_PATH_ENV, path),
        Err(std::env::VarError::NotPresent) => Ok(engine_model_path.to_string_lossy().into_owned()),
        Err(std::env::VarError::NotUnicode(_)) => {
            Err(format!("{FRONTEND_MODEL_PATH_ENV} is not valid UTF-8"))
        }
    }
}

fn fixture_path_from_env(env: &str) -> Result<String, String> {
    match std::env::var(env) {
        Ok(path) => validated_fixture_path(env, path),
        Err(std::env::VarError::NotPresent) => Err(format!(
            "{env} is not set; point it at a public Qwen3.5 model fixture"
        )),
        Err(std::env::VarError::NotUnicode(_)) => Err(format!("{env} is not valid UTF-8")),
    }
}

fn validated_fixture_path(env: &str, path: String) -> Result<String, String> {
    if path.trim().is_empty() {
        return Err(format!("{env} is empty"));
    }

    let config_path = Path::new(&path).join("config.json");
    let raw = std::fs::read(&config_path)
        .map_err(|err| format!("cannot read {} from {env}: {err}", config_path.display()))?;
    let config: serde_json::Value = serde_json::from_slice(&raw).map_err(|err| {
        format!(
            "{} from {env} is not valid JSON: {err}",
            config_path.display()
        )
    })?;
    let root_model_type = config.get("model_type").and_then(serde_json::Value::as_str);
    let text_model_type = config
        .pointer("/text_config/model_type")
        .and_then(serde_json::Value::as_str);
    if root_model_type != Some("qwen3_5") && text_model_type != Some("qwen3_5_text") {
        return Err(format!(
            "{} from {env} is not a Qwen3.5 config",
            config_path.display()
        ));
    }

    Ok(path)
}

fn skip(test_name: &str, reason: &str) {
    eprintln!("SKIP {test_name}: {reason}");
}

fn fail(test_name: &str, reason: &str) -> ! {
    panic!(
        "{test_name} was selected explicitly but its model fixture is unusable: {reason}\n\
         This gate is #[ignore]d, so it only runs when a maintainer names it; a missing \
         fixture is a failure, not a skip."
    )
}
