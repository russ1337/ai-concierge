mod llm;
mod ollama;
mod rag;
use std::sync::mpsc;
use std::sync::Mutex;
use std::path::PathBuf;
use tauri::Emitter;

struct AppState {
    llm: Mutex<Option<llm::LlmEngine>>,
}

/// TinyLlama chat format so the model only generates the assistant reply.
/// If current_date is Some, inject it so the model knows today's date.
fn build_prompt_with_rag(
    prompt: &str,
    events_path: Option<&str>,
    current_date: Option<&str>,
) -> String {
    let date_line = current_date
        .map(|d| format!("Today's date: {}.\n", d))
        .unwrap_or_default();

    if let Some(path) = events_path {
        let path = std::path::Path::new(path);
        if path.exists() {
            match rag::retrieve_context(path, prompt, 5) {
                Ok(context) => {
                    return format!(
                        "<|system|>\n{}Relevant events:\n{}\nOnly output the assistant reply. Do not generate any user message or \"User:\" line.</s>\n<|user|>\n{}</s>\n<|assistant|>\n",
                        date_line, context, prompt
                    );
                }
                Err(e) => {
                    log::warn!("RAG retrieval failed: {}; using raw prompt", e);
                }
            }
        } else {
            log::warn!("Events file not found: {}; using raw prompt", path.display());
        }
    }
    if date_line.is_empty() {
        format!("<|user|>\n{}</s>\n<|assistant|>\n", prompt)
    } else {
        format!(
            "<|system|>\n{}Only output the assistant reply. Do not generate any user message or \"User:\" line.</s>\n<|user|>\n{}</s>\n<|assistant|>\n",
            date_line, prompt
        )
    }
}

/// Strip any model-generated "User:" or "<|user|>" so we never show fake user prompts.
fn strip_fake_user_prompts(response: &str) -> String {
    let markers = ["\nUser:", "\n<|user|>", "\n\nUser:"];
    let truncate_at = markers
        .iter()
        .filter_map(|m| response.find(m))
        .min()
        .unwrap_or(response.len());
    response[..truncate_at].trim_end().to_string()
}

#[tauri::command]
fn generate(
    prompt: String,
    model_dir: String,
    events_path: Option<String>,
    current_date: Option<String>,
    max_tokens: Option<u32>,
    temperature: Option<f64>,
    state: tauri::State<AppState>,
) -> Result<String, String> {
    let path = PathBuf::from(&model_dir);

    let mut guard = state.llm.lock().map_err(|e| e.to_string())?;

    if guard.is_none() {
        log::info!("Loading model from {}", model_dir);
        let engine = llm::load(&path).map_err(|e| e.to_string())?;
        *guard = Some(engine);
    }

    let engine = guard.as_ref().ok_or("Model not loaded")?;
    let max_tokens = max_tokens.unwrap_or(128) as usize;
    let temperature = temperature.unwrap_or(0.0);
    let seed = 299792458u64;

    let prompt_to_use = build_prompt_with_rag(&prompt, events_path.as_deref(), current_date.as_deref());

    let raw = engine
        .generate(&prompt_to_use, max_tokens, temperature, seed)
        .map_err(|e| e.to_string())?;
    Ok(strip_fake_user_prompts(&raw))
}

#[tauri::command]
fn generate_stream(
    prompt: String,
    model_dir: String,
    events_path: Option<String>,
    current_date: Option<String>,
    max_tokens: Option<u32>,
    temperature: Option<f64>,
    ollama_url: Option<String>,
    ollama_model: Option<String>,
    window: tauri::Window,
    state: tauri::State<AppState>,
) -> Result<(), String> {
    let prompt_to_use = build_prompt_with_rag(&prompt, events_path.as_deref(), current_date.as_deref());
    let max_tokens_val = max_tokens.unwrap_or(128);
    let temperature_val = temperature.unwrap_or(0.0);

    if let (Some(ref url), Some(ref model)) = (ollama_url, ollama_model) {
        let (tx, rx) = mpsc::channel::<Result<String, String>>();
        let url = url.clone();
        let model = model.clone();
        let prompt = prompt_to_use.clone();
        std::thread::spawn(move || {
            let client = reqwest::blocking::Client::new();
            if let Err(e) = ollama::stream_generate(
                &client,
                &url,
                &model,
                &prompt,
                Some(max_tokens_val),
                Some(temperature_val as f64),
                tx.clone(),
            ) {
                let _ = tx.send(Err(e));
            }
        });
        while let Ok(msg) = rx.recv() {
            match msg {
                Ok(chunk) => {
                    let _ = window.emit("chat-token", chunk);
                }
                Err(e) => return Err(e),
            }
        }
        return Ok(());
    }

    let path = PathBuf::from(&model_dir);
    let mut guard = state.llm.lock().map_err(|e| e.to_string())?;

    if guard.is_none() {
        log::info!("Loading model from {}", model_dir);
        let engine = llm::load(&path).map_err(|e| e.to_string())?;
        *guard = Some(engine);
    }

    let engine = guard.as_ref().ok_or("Model not loaded")?;
    let max_tokens = max_tokens_val as usize;
    let seed = 299792458u64;

    engine
        .generate_stream(&prompt_to_use, max_tokens, temperature_val, seed, |chunk| {
            let _ = window.emit("chat-token", chunk);
        })
        .map_err(|e| e.to_string())
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
  let state = AppState {
    llm: Mutex::new(None),
  };
  tauri::Builder::default()
    .setup(|app| {
      if cfg!(debug_assertions) {
        app.handle().plugin(
          tauri_plugin_log::Builder::default()
            .level(log::LevelFilter::Info)
            .build(),
        )?;
      }
      Ok(())
    }).manage(state)
    .invoke_handler(tauri::generate_handler![generate, generate_stream])
    .run(tauri::generate_context!())
    .expect("error while running tauri application");
}
