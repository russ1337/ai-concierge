use std::path::Path;
use tokenizers::Tokenizer;
use candle_core::{Device, DType, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::generation::{LogitsProcessor, Sampling};
use candle_transformers::models::llama::{Llama, LlamaConfig, Cache, LlamaEosToks};
use candle_transformers::utils::apply_repeat_penalty;

#[derive(Debug)]
pub struct LlmError(String);

impl std::fmt::Display for LlmError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for LlmError {}

const EOS_TOKEN: &str = "</s>";
const DEFAULT_REPEAT_PENALTY: f32 = 1.1;
const DEFAULT_REPEAT_LAST_N: usize = 64;

pub struct LlmEngine {
    pub model: Llama,
    pub tokenizer: Tokenizer,
    pub device: Device,
    pub config: candle_transformers::models::llama::Config,
}

fn safetensors_paths(model_dir: &Path) -> Result<Vec<std::path::PathBuf>, LlmError> {
    let mut paths: Vec<_> = std::fs::read_dir(model_dir)
        .map_err(|e| LlmError(format!("Failed to read model dir: {}", e)))?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("safetensors"))
        .collect();
    paths.sort();
    Ok(paths)
}

fn load_config(model_dir: &Path) -> Result<candle_transformers::models::llama::Config, LlmError> {
    let config_path = model_dir.join("config.json");
    let config_bytes = std::fs::read(&config_path)
        .map_err(|e| LlmError(format!("Failed to read config.json: {}", e)))?;
    let llama_config: LlamaConfig = serde_json::from_slice(&config_bytes)
        .map_err(|e| LlmError(format!("Invalid config.json: {}", e)))?;
    let config = llama_config.into_config(false);
    Ok(config)
}

pub fn load(model_dir: &Path) -> Result<LlmEngine, LlmError> {
    let device = Device::Cpu;
    let dtype = DType::F16;

    let config = load_config(model_dir)?;
    let tokenizer_path = model_dir.join("tokenizer.json");
    let tokenizer = Tokenizer::from_file(tokenizer_path)
        .map_err(|e| LlmError(format!("Failed to load tokenizer: {}", e)))?;

    let paths = safetensors_paths(model_dir)?;
    if paths.is_empty() {
        return Err(LlmError("No .safetensors files found in model dir".into()));
    }

    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&paths, dtype, &device) }
        .map_err(|e| LlmError(format!("Failed to load weights: {}", e)))?;

    let model = Llama::load(vb, &config)
        .map_err(|e| LlmError(format!("Failed to load model: {}", e)))?;

    Ok(LlmEngine {
        model,
        tokenizer,
        device,
        config,
    })
}

impl LlmEngine {
    pub fn generate(
        &self,
        prompt: &str,
        max_tokens: usize,
        temperature: f64,
        seed: u64,
    ) -> Result<String, LlmError> {
        let mut tokens = self
            .tokenizer
            .encode(prompt, true)
            .map_err(|e| LlmError(format!("Encode error: {}", e)))?
            .get_ids()
            .to_vec();

        let prompt_len = tokens.len();

        let dtype = DType::F16;
        let mut cache = Cache::new(true, dtype, &self.config, &self.device)
            .map_err(|e| LlmError(format!("Cache creation failed: {}", e)))?;

        let sampling = if temperature <= 0.0 {
            Sampling::ArgMax
        } else {
            Sampling::All { temperature }
        };
        let mut logits_processor = LogitsProcessor::from_sampling(seed, sampling);

        let eos_token_id = self.config.eos_token_id.clone().or_else(|| {
            self.tokenizer
                .token_to_id(EOS_TOKEN)
                .map(LlamaEosToks::Single)
        });

        let mut index_pos = 0usize;

        for _ in 0..max_tokens {
            let (context_size, context_index) = if cache.use_kv_cache && tokens.len() > prompt_len {
                (1, index_pos)
            } else {
                (tokens.len(), 0)
            };

            let ctxt = &tokens[tokens.len().saturating_sub(context_size)..];
            let input = Tensor::new(ctxt, &self.device)
                .map_err(|e| LlmError(format!("Tensor creation failed: {}", e)))?
                .unsqueeze(0)
                .map_err(|e| LlmError(format!("Unsqueeze failed: {}", e)))?;

            let logits = self
                .model
                .forward(&input, context_index, &mut cache)
                .map_err(|e| LlmError(format!("Forward failed: {}", e)))?
                .squeeze(0)
                .map_err(|e| LlmError(format!("Squeeze failed: {}", e)))?;

            let logits = if (DEFAULT_REPEAT_PENALTY - 1.0).abs() < 1e-6 {
                logits
            } else {
                let start_at = tokens.len().saturating_sub(DEFAULT_REPEAT_LAST_N);
                apply_repeat_penalty(&logits, DEFAULT_REPEAT_PENALTY, &tokens[start_at..])
                    .map_err(|e| LlmError(format!("Repeat penalty failed: {}", e)))?
            };

            let next_token = logits_processor
                .sample(&logits)
                .map_err(|e| LlmError(format!("Sample failed: {}", e)))?;

            index_pos += ctxt.len();
            tokens.push(next_token);

            match &eos_token_id {
                Some(LlamaEosToks::Single(id)) if next_token == *id => break,
                Some(LlamaEosToks::Multiple(ids)) if ids.contains(&next_token) => break,
                _ => {}
            }
        }

        let generated_ids: Vec<u32> = tokens[prompt_len..].to_vec();
        let text = self
            .tokenizer
            .decode(&generated_ids, true)
            .map_err(|e| LlmError(format!("Decode error: {}", e)))?;

        Ok(text)
    }

    pub fn generate_stream<E>(
        &self,
        prompt: &str,
        max_tokens: usize,
        temperature: f64,
        seed: u64,
        mut emit: E,
    ) -> Result<(), LlmError>
    where
        E: FnMut(&str),
    {
        let mut tokens = self
            .tokenizer
            .encode(prompt, true)
            .map_err(|e| LlmError(format!("Encode error: {}", e)))?
            .get_ids()
            .to_vec();

        let prompt_len = tokens.len();

        let dtype = DType::F16;
        let mut cache = Cache::new(true, dtype, &self.config, &self.device)
            .map_err(|e| LlmError(format!("Cache creation failed: {}", e)))?;

        let sampling = if temperature <= 0.0 {
            Sampling::ArgMax
        } else {
            Sampling::All { temperature }
        };
        let mut logits_processor = LogitsProcessor::from_sampling(seed, sampling);

        let eos_token_id = self.config.eos_token_id.clone().or_else(|| {
            self.tokenizer
                .token_to_id(EOS_TOKEN)
                .map(LlamaEosToks::Single)
        });

        let mut index_pos = 0usize;
        let mut last_emitted_len = 0usize;

        for _ in 0..max_tokens {
            let (context_size, context_index) = if cache.use_kv_cache && tokens.len() > prompt_len {
                (1, index_pos)
            } else {
                (tokens.len(), 0)
            };

            let ctxt = &tokens[tokens.len().saturating_sub(context_size)..];
            let input = Tensor::new(ctxt, &self.device)
                .map_err(|e| LlmError(format!("Tensor creation failed: {}", e)))?
                .unsqueeze(0)
                .map_err(|e| LlmError(format!("Unsqueeze failed: {}", e)))?;

            let logits = self
                .model
                .forward(&input, context_index, &mut cache)
                .map_err(|e| LlmError(format!("Forward failed: {}", e)))?
                .squeeze(0)
                .map_err(|e| LlmError(format!("Squeeze failed: {}", e)))?;

            let logits = if (DEFAULT_REPEAT_PENALTY - 1.0).abs() < 1e-6 {
                logits
            } else {
                let start_at = tokens.len().saturating_sub(DEFAULT_REPEAT_LAST_N);
                apply_repeat_penalty(&logits, DEFAULT_REPEAT_PENALTY, &tokens[start_at..])
                    .map_err(|e| LlmError(format!("Repeat penalty failed: {}", e)))?
            };

            let next_token = logits_processor
                .sample(&logits)
                .map_err(|e| LlmError(format!("Sample failed: {}", e)))?;

            index_pos += ctxt.len();
            tokens.push(next_token);

            let generated_ids: Vec<u32> = tokens[prompt_len..].to_vec();
            let full_text = self
                .tokenizer
                .decode(&generated_ids, true)
                .map_err(|e| LlmError(format!("Decode error: {}", e)))?;
            let current_len = full_text.len();
            if current_len > last_emitted_len {
                let chunk = &full_text[last_emitted_len..];
                if !chunk.is_empty() {
                    emit(chunk);
                }
                last_emitted_len = current_len;
            }

            match &eos_token_id {
                Some(LlamaEosToks::Single(id)) if next_token == *id => break,
                Some(LlamaEosToks::Multiple(ids)) if ids.contains(&next_token) => break,
                _ => {}
            }
        }

        Ok(())
    }
}