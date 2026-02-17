mod llm;
mod rag;
use std::sync::Mutex;
use std::path::PathBuf;
use tauri::Emitter;

struct AppState {
    llm: Mutex<Option<llm::LlmEngine>>,
}

fn build_prompt_with_rag(prompt: &str, events_path: Option<&str>) -> String {
    if let Some(path) = events_path {
        let path = std::path::Path::new(path);
        if path.exists() {
            match rag::retrieve_context(path, prompt, 5) {
                Ok(context) => {
                    return format!(
                        "Relevant events:\n{}\n\nUser: {}\n\nAssistant:",
                        context, prompt
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
    prompt.to_string()
}

#[tauri::command]
fn generate(
    prompt: String,
    model_dir: String,
    events_path: Option<String>,
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
    let max_tokens = 256usize;
    let temperature = 0.8f64;
    let seed = 299792458u64;

    let prompt_to_use = build_prompt_with_rag(&prompt, events_path.as_deref());

    engine
        .generate(&prompt_to_use, max_tokens, temperature, seed)
        .map_err(|e| e.to_string())
}

#[tauri::command]
fn generate_stream(
    prompt: String,
    model_dir: String,
    events_path: Option<String>,
    window: tauri::Window,
    state: tauri::State<AppState>,
) -> Result<(), String> {
    let path = PathBuf::from(&model_dir);

    let mut guard = state.llm.lock().map_err(|e| e.to_string())?;

    if guard.is_none() {
        log::info!("Loading model from {}", model_dir);
        let engine = llm::load(&path).map_err(|e| e.to_string())?;
        *guard = Some(engine);
    }

    let engine = guard.as_ref().ok_or("Model not loaded")?;

    let max_tokens = 256usize;
    let temperature = 0.8f64;
    let seed = 299792458u64;

    let prompt_to_use = build_prompt_with_rag(&prompt, events_path.as_deref());

    engine
        .generate_stream(&prompt_to_use, max_tokens, temperature, seed, |chunk| {
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
