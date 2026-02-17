//! Ollama API client for offloading inference (e.g. to AMD GPU via ROCm on Windows).

use serde::Deserialize;
use std::sync::mpsc::Sender;

#[derive(serde::Serialize)]
struct GenerateRequest {
    model: String,
    prompt: String,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    options: Option<GenerateOptions>,
}

#[derive(serde::Serialize)]
struct GenerateOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    num_predict: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f64>,
}

#[derive(Deserialize)]
struct GenerateChunk {
    response: Option<String>,
    done: Option<bool>,
}

/// Call Ollama /api/generate with streaming; send each "response" chunk via `tx` as Ok(chunk).
/// Runs synchronously (blocking) so it can be called from a sync Tauri command.
pub fn stream_generate(
    client: &reqwest::blocking::Client,
    base_url: &str,
    model: &str,
    prompt: &str,
    num_predict: Option<u32>,
    temperature: Option<f64>,
    tx: Sender<Result<String, String>>,
) -> Result<(), String> {
    let url = format!("{}/api/generate", base_url.trim_end_matches('/'));
    let body = GenerateRequest {
        model: model.to_string(),
        prompt: prompt.to_string(),
        stream: true,
        options: Some(GenerateOptions {
            num_predict,
            temperature,
        }),
    };

    let response = client
        .post(&url)
        .json(&body)
        .send()
        .map_err(|e| format!("Ollama request failed: {}", e))?;

    if !response.status().is_success() {
        let status = response.status();
        let text = response.text().unwrap_or_default();
        return Err(format!("Ollama error {}: {}", status, text));
    }

    let bytes = response
        .bytes()
        .map_err(|e| format!("Ollama response read failed: {}", e))?;

    for line in bytes.split(|&b| b == b'\n') {
        if line.is_empty() {
            continue;
        }
        let line = match std::str::from_utf8(line) {
            Ok(s) => s,
            Err(_) => continue,
        };
        let chunk: GenerateChunk = match serde_json::from_str(line) {
            Ok(c) => c,
            Err(_) => continue,
        };
        if let Some(ref s) = chunk.response {
            if !s.is_empty() {
                let _ = tx.send(Ok(s.clone()));
            }
        }
        if chunk.done == Some(true) {
            break;
        }
    }

    Ok(())
}
