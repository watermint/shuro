use std::path::{Path, PathBuf};
use std::fs;
use tokio::fs as async_fs;
use reqwest::Client;
use tracing::{info, warn};
use indicatif::{ProgressBar, ProgressStyle};

use crate::error::{Result, ShuroError};
use crate::config::Config;

pub struct SetupManager {
    client: Client,
    shuro_dir: PathBuf,
}

#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub name: String,
    pub filename: String,
    pub url: String,
    pub size_mb: f64,
}

impl SetupManager {
    pub fn new() -> Result<Self> {
        let shuro_dir = PathBuf::from(".shuro");
        
        // Create .shuro directory structure if it doesn't exist
        fs::create_dir_all(shuro_dir.join("models"))?;
        fs::create_dir_all(shuro_dir.join("cache"))?;
        
        let client = Client::builder()
            .user_agent("shuro/0.1.0")
            .build()
            .map_err(|e| ShuroError::Http(e))?;

        Ok(Self { client, shuro_dir })
    }

    /// Initialize the application, downloading necessary files if needed
    pub async fn initialize(&self, config: &mut Config) -> Result<()> {
        info!("Initializing Shuro application...");
        info!("Current explore model: {}", config.whisper.explore_model);
        info!("Current transcribe model: {}", config.whisper.transcribe_model);

        // Check and download required whisper models
        self.ensure_whisper_models(config).await?;

        info!("Updated explore model: {}", config.whisper.explore_model);
        info!("Updated transcribe model: {}", config.whisper.transcribe_model);
        info!("Initialization completed successfully");
        Ok(())
    }

    /// Ensure required whisper models are available
    async fn ensure_whisper_models(&self, config: &mut Config) -> Result<()> {
        let _models_dir = self.shuro_dir.join("models");
        
        // Define available models
        let available_models = self.get_available_models();
        
        // Check and download exploration model
        if !self.model_exists(&config.whisper.explore_model) {
            info!("Exploration model not found: {}", config.whisper.explore_model);
            let model = self.select_appropriate_model(&available_models, &config.whisper.explore_model)?;
            let local_path = self.download_model(&model).await?;
            config.whisper.explore_model = local_path;
        } else {
            // Model exists, but if it's just a name, resolve to full path
            config.whisper.explore_model = self.resolve_model_path(&config.whisper.explore_model);
        }

        // Check and download transcription model
        if !self.model_exists(&config.whisper.transcribe_model) {
            info!("Transcription model not found: {}", config.whisper.transcribe_model);
            let model = self.select_appropriate_model(&available_models, &config.whisper.transcribe_model)?;
            let local_path = self.download_model(&model).await?;
            config.whisper.transcribe_model = local_path;
        } else {
            // Model exists, but if it's just a name, resolve to full path
            config.whisper.transcribe_model = self.resolve_model_path(&config.whisper.transcribe_model);
        }

        Ok(())
    }

    pub fn get_available_models(&self) -> Vec<ModelInfo> {
        vec![
            ModelInfo {
                name: "tiny".to_string(),
                filename: "ggml-tiny.bin".to_string(),
                url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.bin".to_string(),
                size_mb: 39.0,
            },
            ModelInfo {
                name: "tiny.en".to_string(),
                filename: "ggml-tiny.en.bin".to_string(),
                url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.en.bin".to_string(),
                size_mb: 39.0,
            },
            ModelInfo {
                name: "base".to_string(),
                filename: "ggml-base.bin".to_string(),
                url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin".to_string(),
                size_mb: 142.0,
            },
            ModelInfo {
                name: "base.en".to_string(),
                filename: "ggml-base.en.bin".to_string(),
                url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin".to_string(),
                size_mb: 142.0,
            },
            ModelInfo {
                name: "small".to_string(),
                filename: "ggml-small.bin".to_string(),
                url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.bin".to_string(),
                size_mb: 244.0,
            },
            ModelInfo {
                name: "small.en".to_string(),
                filename: "ggml-small.en.bin".to_string(),
                url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.en.bin".to_string(),
                size_mb: 244.0,
            },
            ModelInfo {
                name: "medium".to_string(),
                filename: "ggml-medium.bin".to_string(),
                url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-medium.bin".to_string(),
                size_mb: 769.0,
            },
            ModelInfo {
                name: "medium.en".to_string(),
                filename: "ggml-medium.en.bin".to_string(),
                url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-medium.en.bin".to_string(),
                size_mb: 769.0,
            },
            ModelInfo {
                name: "large-v1".to_string(),
                filename: "ggml-large-v1.bin".to_string(),
                url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v1.bin".to_string(),
                size_mb: 1550.0,
            },
            ModelInfo {
                name: "large-v2".to_string(),
                filename: "ggml-large-v2.bin".to_string(),
                url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v2.bin".to_string(),
                size_mb: 1550.0,
            },
            ModelInfo {
                name: "large-v3".to_string(),
                filename: "ggml-large-v3.bin".to_string(),
                url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3.bin".to_string(),
                size_mb: 1550.0,
            },
        ]
    }

    fn select_appropriate_model(&self, models: &[ModelInfo], preferred: &str) -> Result<ModelInfo> {
        // Try to find the exact preferred model
        if let Some(model) = models.iter().find(|m| m.name == preferred) {
            return Ok(model.clone());
        }

        // Fall back to base model if preferred not found
        if let Some(model) = models.iter().find(|m| m.name == "base") {
            warn!("Preferred model '{}' not found, using 'base' instead", preferred);
            return Ok(model.clone());
        }

        // Fall back to tiny model as last resort
        if let Some(model) = models.iter().find(|m| m.name == "tiny") {
            warn!("Base model not found, using 'tiny' instead");
            return Ok(model.clone());
        }

        Err(ShuroError::Config("No suitable whisper model found".to_string()))
    }

    fn model_exists(&self, model_path: &str) -> bool {
        // Check if it's an absolute path that exists
        if Path::new(model_path).is_absolute() && Path::new(model_path).exists() {
            return true;
        }

        // Check if it's a relative path in .shuro/models
        if model_path.starts_with(".shuro/models/") && Path::new(model_path).exists() {
            return true;
        }

        // If it's just a model name (like "base" or "tiny"), find the corresponding file
        if !model_path.contains('/') && !model_path.ends_with(".bin") {
            let models = self.get_available_models();
            if let Some(model) = models.iter().find(|m| m.name == model_path) {
                let local_path = self.shuro_dir.join("models").join(&model.filename);
                return local_path.exists();
            }
        }

        // Check if the filename exists in .shuro/models
        let filename = Path::new(model_path).file_name()
            .and_then(|name| name.to_str())
            .unwrap_or(model_path);
        
        let local_path = self.shuro_dir.join("models").join(filename);
        local_path.exists()
    }

    pub async fn download_model(&self, model: &ModelInfo) -> Result<String> {
        let local_path = self.shuro_dir.join("models").join(&model.filename);
        
        // Check if already exists
        if local_path.exists() {
            info!("Model {} already exists at {}", model.name, local_path.display());
            return Ok(local_path.to_string_lossy().to_string());
        }

        info!("Downloading {} model ({:.1} MB)...", model.name, model.size_mb);
        
        // Create progress bar
        let pb = ProgressBar::new((model.size_mb * 1_000_000.0) as u64);
        pb.set_style(ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({eta})")
            .unwrap()
            .progress_chars("#>-"));

        // Download the file
        let response = self.client.get(&model.url).send().await
            .map_err(|e| ShuroError::Http(e))?;

        if !response.status().is_success() {
            return Err(ShuroError::Config(format!(
                "Failed to download model {}: HTTP {}",
                model.name, response.status()
            )));
        }

        // Create temporary file
        let temp_path = local_path.with_extension("tmp");
        let mut file = async_fs::File::create(&temp_path).await?;
        
        // Download with progress
        use tokio::io::AsyncWriteExt;
        let content_length = response.content_length().unwrap_or(0);
        let bytes = response.bytes().await.map_err(|e| ShuroError::Http(e))?;
        
        file.write_all(&bytes).await?;
        let downloaded = bytes.len() as u64;
        pb.set_position(downloaded);
        
        // If we have content length, use it for final progress update
        if content_length > 0 {
            pb.set_length(content_length);
        }

        file.flush().await?;
        drop(file);

        // Move temp file to final location
        async_fs::rename(&temp_path, &local_path).await?;
        
        pb.finish_with_message(format!("Downloaded {}", model.name));
        info!("Successfully downloaded {} to {}", model.name, local_path.display());

        Ok(local_path.to_string_lossy().to_string())
    }

    /// Get the path to the .shuro directory
    pub fn shuro_dir(&self) -> &PathBuf {
        &self.shuro_dir
    }

    /// Resolve a model name to the actual file path
    fn resolve_model_path(&self, model_name: &str) -> String {
        // If it's already a path, return as-is
        if model_name.contains('/') || model_name.ends_with(".bin") {
            return model_name.to_string();
        }

        // Map model names to filenames
        let filename = format!("ggml-{}.bin", model_name);
        let local_path = self.shuro_dir.join("models").join(filename);
        
        local_path.to_string_lossy().to_string()
    }
} 