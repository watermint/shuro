use std::path::{Path, PathBuf};
use std::process::Command;
use serde_json;
use tokio::fs;
use tracing::{info, warn, debug};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use serde::{Serialize, Deserialize};

use crate::config::WhisperConfig;
use crate::error::{Result, ShuroError};
use crate::quality::{Transcription, QualityValidator, WhisperCppOutput};

pub struct WhisperTranscriber {
    config: WhisperConfig,
    validator: QualityValidator,
    cache_dir: PathBuf,
    audio_cache_dir: PathBuf,
}

#[derive(Debug)]
pub struct TuneParams {
    pub tempo: i32,
    pub model: String,
    pub temperature: f32,
}

#[derive(Debug)]
pub struct TuneResult {
    pub best_transcription: Transcription,
    pub best_tempo: i32,
    pub quality_score: f64,
    pub all_attempts: Vec<(i32, f64)>, // (tempo, quality_score)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscriptionCache {
    pub transcription: Transcription,
    pub model: String,
    pub temperature: f32,
    pub language: Option<String>,
    pub audio_path: String,
    pub audio_modified: Option<u64>, // File modification time
    pub cached_at: u64, // Unix timestamp
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioCache {
    pub audio_path: String,
    pub video_path: String,
    pub video_modified: Option<u64>,
    pub cached_at: u64,
}

#[derive(Debug)]
pub struct CacheInfo {
    pub total_files: u64,
    pub total_size: u64,
    pub oldest_entry: Option<u64>,
    pub newest_entry: Option<u64>,
    pub models_used: Vec<String>,
    pub audio_files: u64,
    pub audio_size: u64,
}

impl WhisperTranscriber {
    pub fn new(config: WhisperConfig, validator: QualityValidator) -> Self {
        let cache_base = std::env::current_dir()
            .unwrap_or_default()
            .join(".shuro")
            .join("cache");
        
        let cache_dir = cache_base.join("transcriptions");
        let audio_cache_dir = cache_base.join("audio");
        
        Self { config, validator, cache_dir, audio_cache_dir }
    }

    /// Generate cache key based on audio file, model, and parameters
    fn generate_cache_key<P: AsRef<Path>>(
        &self,
        audio_path: P,
        model: &str,
        temperature: f32,
        language: Option<&str>
    ) -> Result<String> {
        let audio_path = audio_path.as_ref();
        
        // Get file metadata
        let metadata = std::fs::metadata(audio_path)
            .map_err(|e| ShuroError::Whisper(format!("Failed to read audio metadata: {}", e)))?;
        
        let modified = metadata.modified()
            .map_err(|e| ShuroError::Whisper(format!("Failed to get modification time: {}", e)))?
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        
        // Create hash input
        let mut hasher = DefaultHasher::new();
        audio_path.to_string_lossy().hash(&mut hasher);
        model.hash(&mut hasher);
        temperature.to_bits().hash(&mut hasher);
        language.unwrap_or("").hash(&mut hasher);
        modified.hash(&mut hasher);
        
        let hash = hasher.finish();
        Ok(format!("{:016x}", hash))
    }

    /// Ensure cache directory exists
    async fn ensure_cache_dir(&self) -> Result<()> {
        tokio::fs::create_dir_all(&self.cache_dir).await
            .map_err(|e| ShuroError::Whisper(format!("Failed to create cache directory: {}", e)))?;
        Ok(())
    }

    /// Load transcription from cache
    async fn load_from_cache(&self, cache_key: &str) -> Result<Option<Transcription>> {
        let cache_file = self.cache_dir.join(format!("{}.json", cache_key));
        
        if !cache_file.exists() {
            debug!("Cache miss: {}", cache_key);
            return Ok(None);
        }

        let content = tokio::fs::read_to_string(&cache_file).await
            .map_err(|e| ShuroError::Whisper(format!("Failed to read cache file: {}", e)))?;
        
        let cached: TranscriptionCache = serde_json::from_str(&content)
            .map_err(|e| ShuroError::Whisper(format!("Failed to parse cache: {}", e)))?;
        
        info!("Cache hit: {} (cached {} ago)", 
              cache_key,
              format_duration(std::time::SystemTime::now()
                  .duration_since(std::time::UNIX_EPOCH)
                  .unwrap_or_default()
                  .as_secs()
                  .saturating_sub(cached.cached_at)));
        
        Ok(Some(cached.transcription))
    }

    /// Save transcription to cache
    async fn save_to_cache<P: AsRef<Path>>(
        &self,
        cache_key: &str,
        transcription: &Transcription,
        audio_path: P,
        model: &str,
        temperature: f32,
        language: Option<&str>
    ) -> Result<()> {
        self.ensure_cache_dir().await?;
        
        let audio_path = audio_path.as_ref();
        let audio_modified = std::fs::metadata(audio_path)
            .ok()
            .and_then(|m| m.modified().ok())
            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
            .map(|d| d.as_secs());
        
        let cached = TranscriptionCache {
            transcription: transcription.clone(),
            model: model.to_string(),
            temperature,
            language: language.map(|s| s.to_string()),
            audio_path: audio_path.to_string_lossy().to_string(),
            audio_modified,
            cached_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        };

        let cache_file = self.cache_dir.join(format!("{}.json", cache_key));
        let content = serde_json::to_string_pretty(&cached)
            .map_err(|e| ShuroError::Whisper(format!("Failed to serialize cache: {}", e)))?;
        
        tokio::fs::write(&cache_file, content).await
            .map_err(|e| ShuroError::Whisper(format!("Failed to write cache file: {}", e)))?;
        
        debug!("Cached transcription: {}", cache_key);
        Ok(())
    }

    /// List all cached transcriptions
    pub async fn list_cache(&self) -> Result<Vec<TranscriptionCache>> {
        self.ensure_cache_dir().await?;
        
        let mut entries = tokio::fs::read_dir(&self.cache_dir).await
            .map_err(|e| ShuroError::Whisper(format!("Failed to read cache directory: {}", e)))?;
        
        let mut cached_items = Vec::new();
        
        while let Some(entry) = entries.next_entry().await
            .map_err(|e| ShuroError::Whisper(format!("Failed to read directory entry: {}", e)))? {
            
            if let Some(extension) = entry.path().extension() {
                if extension == "json" {
                    match tokio::fs::read_to_string(entry.path()).await {
                        Ok(content) => {
                            if let Ok(cached) = serde_json::from_str::<TranscriptionCache>(&content) {
                                cached_items.push(cached);
                            }
                        }
                        Err(_) => continue, // Skip unreadable files
                    }
                }
            }
        }
        
        // Sort by cache time (newest first)
        cached_items.sort_by(|a, b| b.cached_at.cmp(&a.cached_at));
        
        Ok(cached_items)
    }

    /// Clear all cached transcriptions
    pub async fn clear_cache(&self) -> Result<u64> {
        self.ensure_cache_dir().await?;
        
        let mut entries = tokio::fs::read_dir(&self.cache_dir).await
            .map_err(|e| ShuroError::Whisper(format!("Failed to read cache directory: {}", e)))?;
        
        let mut deleted_count = 0;
        
        while let Some(entry) = entries.next_entry().await
            .map_err(|e| ShuroError::Whisper(format!("Failed to read directory entry: {}", e)))? {
            
            if let Some(extension) = entry.path().extension() {
                if extension == "json" {
                    if tokio::fs::remove_file(entry.path()).await.is_ok() {
                        deleted_count += 1;
                    }
                }
            }
        }
        
        info!("Cleared {} cached transcriptions", deleted_count);
        Ok(deleted_count)
    }

    /// Clean old cache entries
    pub async fn clean_cache(&self, max_age_days: u64) -> Result<u64> {
        self.ensure_cache_dir().await?;
        
        let cutoff_time = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs()
            .saturating_sub(max_age_days * 24 * 60 * 60);
        
        let mut entries = tokio::fs::read_dir(&self.cache_dir).await
            .map_err(|e| ShuroError::Whisper(format!("Failed to read cache directory: {}", e)))?;
        
        let mut deleted_count = 0;
        
        while let Some(entry) = entries.next_entry().await
            .map_err(|e| ShuroError::Whisper(format!("Failed to read directory entry: {}", e)))? {
            
            if let Some(extension) = entry.path().extension() {
                if extension == "json" {
                    // Check if file is old enough to delete
                    if let Ok(content) = tokio::fs::read_to_string(entry.path()).await {
                        if let Ok(cached) = serde_json::from_str::<TranscriptionCache>(&content) {
                            if cached.cached_at < cutoff_time {
                                if tokio::fs::remove_file(entry.path()).await.is_ok() {
                                    deleted_count += 1;
                                }
                            }
                        }
                    }
                }
            }
        }
        
        info!("Cleaned {} old cached transcriptions (older than {} days)", deleted_count, max_age_days);
        Ok(deleted_count)
    }

    /// Get cache statistics
    pub async fn cache_info(&self) -> Result<CacheInfo> {
        self.ensure_cache_dir().await?;
        self.ensure_audio_cache_dir().await?;
        
        // Transcription cache stats
        let mut transcription_entries = tokio::fs::read_dir(&self.cache_dir).await
            .map_err(|e| ShuroError::Whisper(format!("Failed to read cache directory: {}", e)))?;
        
        let mut total_files = 0;
        let mut total_size = 0;
        let mut oldest = u64::MAX;
        let mut newest = 0;
        let mut models = std::collections::HashSet::new();
        
        while let Some(entry) = transcription_entries.next_entry().await
            .map_err(|e| ShuroError::Whisper(format!("Failed to read directory entry: {}", e)))? {
            
            if let Some(extension) = entry.path().extension() {
                if extension == "json" {
                    if let Ok(metadata) = entry.metadata().await {
                        total_size += metadata.len();
                    }
                    
                    if let Ok(content) = tokio::fs::read_to_string(entry.path()).await {
                        if let Ok(cached) = serde_json::from_str::<TranscriptionCache>(&content) {
                            total_files += 1;
                            models.insert(cached.model);
                            oldest = oldest.min(cached.cached_at);
                            newest = newest.max(cached.cached_at);
                        }
                    }
                }
            }
        }
        
        // Audio cache stats
        let mut audio_entries = tokio::fs::read_dir(&self.audio_cache_dir).await
            .map_err(|e| ShuroError::Whisper(format!("Failed to read audio cache directory: {}", e)))?;
        
        let mut audio_files = 0;
        let mut audio_size = 0;
        
        while let Some(entry) = audio_entries.next_entry().await
            .map_err(|e| ShuroError::Whisper(format!("Failed to read directory entry: {}", e)))? {
            
            if let Some(extension) = entry.path().extension() {
                if extension == "wav" {
                    if let Ok(metadata) = entry.metadata().await {
                        audio_size += metadata.len();
                        audio_files += 1;
                    }
                } else if extension == "json" {
                    if let Ok(content) = tokio::fs::read_to_string(entry.path()).await {
                        if let Ok(cached) = serde_json::from_str::<AudioCache>(&content) {
                            oldest = oldest.min(cached.cached_at);
                            newest = newest.max(cached.cached_at);
                        }
                    }
                }
            }
        }
        
        Ok(CacheInfo {
            total_files,
            total_size,
            oldest_entry: if oldest == u64::MAX { None } else { Some(oldest) },
            newest_entry: if newest == 0 { None } else { Some(newest) },
            models_used: models.into_iter().collect(),
            audio_files,
            audio_size,
        })
    }

    /// Convert a model name to the actual file path
    fn resolve_model_path(&self, model_name: &str) -> String {
        // If it's already a path, return as-is
        if model_name.contains('/') || model_name.ends_with(".bin") {
            debug!("Model path already resolved: {}", model_name);
            return model_name.to_string();
        }

        // Map model names to filenames
        let filename = format!("ggml-{}.bin", model_name);
        let local_path = std::env::current_dir()
            .unwrap_or_default()
            .join(".shuro")
            .join("models")
            .join(filename);
        
        let resolved_path = local_path.to_string_lossy().to_string();
        debug!("Resolved model '{}' to path: {}", model_name, resolved_path);
        resolved_path
    }

    /// Main tuning function - simplified for whisper.cpp compatibility
    pub async fn tune_transcription<P: AsRef<Path>>(&self, audio_path: P) -> Result<TuneResult> {
        let audio_path = audio_path.as_ref();
        info!("Starting transcription tuning for: {}", audio_path.display());

        // Check cache for final transcription model first (most refined result)
        let transcribe_cache_key = self.generate_cache_key(
            audio_path,
            &self.config.transcribe_model,
            self.config.temperature,
            None
        )?;
        
        if let Some(cached_transcription) = self.load_from_cache(&transcribe_cache_key).await? {
            let quality = cached_transcription.quality();
            let quality_score = quality.score();
            
            info!("Using cached transcription (quality: {})", quality_score);
            
            return Ok(TuneResult {
                best_transcription: cached_transcription,
                best_tempo: 100,
                quality_score,
                all_attempts: vec![(100, quality_score)],
            });
        }

        // If transcription model cache miss, check exploration model cache
        let explore_cache_key = self.generate_cache_key(
            audio_path,
            &self.config.explore_model,
            self.config.temperature,
            None
        )?;
        
        if let Some(cached_transcription) = self.load_from_cache(&explore_cache_key).await? {
            let quality = cached_transcription.quality();
            let quality_score = quality.score();
            
            info!("Using cached exploration transcription (quality: {})", quality_score);
            
            return Ok(TuneResult {
                best_transcription: cached_transcription,  
                best_tempo: 100,
                quality_score,
                all_attempts: vec![(100, quality_score)],
            });
        }

        // Since whisper.cpp doesn't support speed adjustment, we'll do a simpler approach:
        // 1. Try with exploration model first
        // 2. If quality is good, refine with transcription model
        
        let explore_model_path = self.resolve_model_path(&self.config.explore_model);
        info!("Exploring with model: {}", explore_model_path);
        let params = TuneParams {
            tempo: 100, // Normal speed (not used by whisper.cpp but kept for compatibility)
            model: explore_model_path,
            temperature: self.config.temperature,
        };

        let mut all_attempts = Vec::new();

        match self.transcribe_with_params(audio_path, &params).await {
            Ok(transcription) => {
                let quality = transcription.quality();
                let quality_score = quality.score();
                
                debug!("Exploration quality score: {}", quality_score);
                all_attempts.push((100, quality_score));

                // Skip if hallucinations detected
                if quality.has_hallucinations() {
                    return Err(ShuroError::Hallucination);
                }

                // Validate quality
                if let Err(e) = self.validator.validate_transcription(&transcription) {
                    return Err(ShuroError::Quality(format!("Quality validation failed: {}", e)));
                }

                // If exploration model is different from transcription model, refine
                if self.config.transcribe_model != self.config.explore_model {
                    let transcribe_model_path = self.resolve_model_path(&self.config.transcribe_model);
                    info!("Refining with transcription model: {}", transcribe_model_path);
                    
                    let refine_params = TuneParams {
                        tempo: 100,
                        model: transcribe_model_path,
                        temperature: self.config.temperature,
                    };

                    match self.transcribe_with_params(audio_path, &refine_params).await {
                        Ok(refined_transcription) => {
                            // Validate the refined transcription
                            if self.validator.validate_transcription(&refined_transcription).is_ok() {
                                let refined_quality = refined_transcription.quality();
                                let refined_score = refined_quality.score();
                                info!("Refined transcription quality score: {}", refined_score);
                                
                                // Cache the refined result
                                let refine_cache_key = self.generate_cache_key(
                                    audio_path,
                                    &self.config.transcribe_model,
                                    self.config.temperature,
                                    None
                                )?;
                                
                                if let Err(e) = self.save_to_cache(
                                    &refine_cache_key,
                                    &refined_transcription,
                                    audio_path,
                                    &self.config.transcribe_model,
                                    self.config.temperature,
                                    None
                                ).await {
                                    warn!("Failed to cache refined transcription: {}", e);
                                }
                                
                                return Ok(TuneResult {
                                    best_transcription: refined_transcription,
                                    best_tempo: 100,
                                    quality_score: refined_score,
                                    all_attempts,
                                });
                            }
                        }
                        Err(e) => {
                            warn!("Refinement failed, using exploration result: {}", e);
                        }
                    }
                }

                // Cache the exploration result
                if let Err(e) = self.save_to_cache(
                    &explore_cache_key,
                    &transcription,
                    audio_path,
                    &self.config.explore_model,
                    self.config.temperature,
                    None
                ).await {
                    warn!("Failed to cache exploration transcription: {}", e);
                }

                // Return the exploration result
                info!("Tuning completed with quality score: {}", quality_score);
                Ok(TuneResult {
                    best_transcription: transcription,
                    best_tempo: 100,
                    quality_score,
                    all_attempts,
                })
            }
            Err(e) => {
                Err(ShuroError::Whisper(format!("Transcription failed: {}", e)))
            }
        }
    }

    /// Simple transcription without tuning
    pub async fn transcribe<P: AsRef<Path>>(&self, audio_path: P, language: Option<&str>) -> Result<Transcription> {
        let transcribe_model_path = self.resolve_model_path(&self.config.transcribe_model);
        
        // Generate cache key
        let cache_key = self.generate_cache_key(
            &audio_path,
            &self.config.transcribe_model,
            self.config.temperature,
            language
        )?;
        
        // Check cache first
        if let Some(cached_transcription) = self.load_from_cache(&cache_key).await? {
            return Ok(cached_transcription);
        }
        
        let params = TuneParams {
            tempo: 100, // Normal speed
            model: transcribe_model_path,
            temperature: self.config.temperature,
        };

        let transcription = self.transcribe_with_params_and_language(&audio_path, &params, language).await?;
        
        // Save to cache
        if let Err(e) = self.save_to_cache(
            &cache_key,
            &transcription,
            &audio_path,
            &self.config.transcribe_model,
            self.config.temperature,
            language
        ).await {
            warn!("Failed to cache transcription: {}", e);
        }
        
        Ok(transcription)
    }

    async fn transcribe_with_params<P: AsRef<Path>>(&self, audio_path: P, params: &TuneParams) -> Result<Transcription> {
        self.transcribe_with_params_and_language(audio_path, params, None).await
    }

    async fn transcribe_with_params_and_language<P: AsRef<Path>>(
        &self, 
        audio_path: P, 
        params: &TuneParams,
        language: Option<&str>
    ) -> Result<Transcription> {
        let audio_path = audio_path.as_ref();
        let temp_dir = tempfile::tempdir()
            .map_err(|e| ShuroError::Whisper(format!("Failed to create temp dir: {}", e)))?;
        
        debug!("Using temp directory: {}", temp_dir.path().display());
        
        let output_base = temp_dir.path().join("transcript");
        let output_path = temp_dir.path().join("transcript.json");

        // Prepare whisper.cpp command
        let mut cmd = Command::new(&self.config.binary_path);
        cmd.arg("-oj")  // Output JSON format
           .arg("-of").arg(&output_base)  // Output file base name (without extension)
           .arg("-m").arg(&params.model)
           .arg("-f").arg(audio_path);  // Use -f flag for input file

        // Add language if specified
        if let Some(lang) = language {
            cmd.arg("-l").arg(lang);
        }

        // Add temperature 
        if params.temperature > 0.0 {
            cmd.arg("-tp").arg(format!("{:.2}", params.temperature));
        }

        // Note: whisper.cpp doesn't seem to support speed/tempo adjustment
        // The tempo tuning will be handled differently - we may need to 
        // preprocess audio with ffmpeg to adjust speed

        debug!("Executing whisper command: {:?}", cmd);

        // Execute whisper
        let output = cmd.output()
            .map_err(|e| ShuroError::Whisper(format!("Failed to execute whisper: {}", e)))?;

        debug!("Whisper exit status: {}", output.status);
        debug!("Whisper stdout: {}", String::from_utf8_lossy(&output.stdout));
        debug!("Whisper stderr: {}", String::from_utf8_lossy(&output.stderr));
        debug!("Expected output file: {}", output_path.display());
        debug!("Output file exists: {}", output_path.exists());

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(ShuroError::Whisper(format!(
                "Whisper failed: {}",
                stderr
            )));
        }

        // Read the transcription result
        let json_content = fs::read_to_string(&output_path).await
            .map_err(|e| ShuroError::Whisper(format!("Failed to read transcription: {}", e)))?;

        debug!("JSON content first 500 chars: {}", 
            json_content.chars().take(500).collect::<String>());

        let whisper_output: WhisperCppOutput = serde_json::from_str(&json_content)
            .map_err(|e| {
                // Save failed JSON to a file for inspection
                let failed_json_path = std::env::current_dir()
                    .unwrap_or_default()
                    .join("failed_whisper_output.json");
                let _ = std::fs::write(&failed_json_path, &json_content);
                eprintln!("Saved failed JSON to: {}", failed_json_path.display());
                
                ShuroError::Whisper(format!("Failed to parse transcription JSON: {}", e))
            })?;

        let transcription = Transcription::from(whisper_output);

        Ok(transcription)
    }

    // Note: Tempo range generation removed since whisper.cpp doesn't support speed adjustment
    // This functionality may be re-implemented using ffmpeg preprocessing in the future

    /// Generate cache key for audio based on video file
    fn generate_audio_cache_key<P: AsRef<Path>>(&self, video_path: P) -> Result<String> {
        let video_path = video_path.as_ref();
        
        // Get file metadata
        let metadata = std::fs::metadata(video_path)
            .map_err(|e| ShuroError::Whisper(format!("Failed to read video metadata: {}", e)))?;
        
        let modified = metadata.modified()
            .map_err(|e| ShuroError::Whisper(format!("Failed to get modification time: {}", e)))?
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        
        let size = metadata.len();
        
        // Create hash input
        let mut hasher = DefaultHasher::new();
        video_path.to_string_lossy().hash(&mut hasher);
        modified.hash(&mut hasher);
        size.hash(&mut hasher);
        
        let hash = hasher.finish();
        Ok(format!("audio_{:016x}", hash))
    }

    /// Ensure audio cache directory exists
    async fn ensure_audio_cache_dir(&self) -> Result<()> {
        tokio::fs::create_dir_all(&self.audio_cache_dir).await
            .map_err(|e| ShuroError::Whisper(format!("Failed to create audio cache directory: {}", e)))?;
        Ok(())
    }

    /// Check if cached audio exists and return path
    pub async fn get_cached_audio<P: AsRef<Path>>(&self, video_path: P) -> Result<Option<PathBuf>> {
        let cache_key = self.generate_audio_cache_key(&video_path)?;
        let audio_file = self.audio_cache_dir.join(format!("{}.wav", cache_key));
        let metadata_file = self.audio_cache_dir.join(format!("{}.json", cache_key));
        
        if !audio_file.exists() || !metadata_file.exists() {
            debug!("Audio cache miss: {}", cache_key);
            return Ok(None);
        }

        // Verify metadata
        let content = tokio::fs::read_to_string(&metadata_file).await
            .map_err(|e| ShuroError::Whisper(format!("Failed to read audio cache metadata: {}", e)))?;
        
        let cached: AudioCache = serde_json::from_str(&content)
            .map_err(|e| ShuroError::Whisper(format!("Failed to parse audio cache: {}", e)))?;
        
        info!("Audio cache hit: {} (cached {} ago)", 
              cache_key,
              format_duration(std::time::SystemTime::now()
                  .duration_since(std::time::UNIX_EPOCH)
                  .unwrap_or_default()
                  .as_secs()
                  .saturating_sub(cached.cached_at)));
        
        Ok(Some(audio_file))
    }

    /// Cache extracted audio file
    pub async fn cache_audio<P: AsRef<Path>>(
        &self,
        video_path: P,
        audio_path: P,
    ) -> Result<PathBuf> {
        self.ensure_audio_cache_dir().await?;
        
        let video_path = video_path.as_ref();
        let audio_path = audio_path.as_ref();
        
        let cache_key = self.generate_audio_cache_key(video_path)?;
        let cached_audio_path = self.audio_cache_dir.join(format!("{}.wav", cache_key));
        let metadata_file = self.audio_cache_dir.join(format!("{}.json", cache_key));
        
        // Copy audio file to cache
        tokio::fs::copy(audio_path, &cached_audio_path).await
            .map_err(|e| ShuroError::Whisper(format!("Failed to cache audio file: {}", e)))?;
        
        // Create metadata
        let video_modified = std::fs::metadata(video_path)
            .ok()
            .and_then(|m| m.modified().ok())
            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
            .map(|d| d.as_secs());
        
        let cached = AudioCache {
            audio_path: cached_audio_path.to_string_lossy().to_string(),
            video_path: video_path.to_string_lossy().to_string(),
            video_modified,
            cached_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        };

        let content = serde_json::to_string_pretty(&cached)
            .map_err(|e| ShuroError::Whisper(format!("Failed to serialize audio cache: {}", e)))?;
        
        tokio::fs::write(&metadata_file, content).await
            .map_err(|e| ShuroError::Whisper(format!("Failed to write audio cache metadata: {}", e)))?;
        
        debug!("Cached audio: {}", cache_key);
        Ok(cached_audio_path)
    }

    /// List all cached audio files
    pub async fn list_audio_cache(&self) -> Result<Vec<AudioCache>> {
        self.ensure_audio_cache_dir().await?;
        
        let mut entries = tokio::fs::read_dir(&self.audio_cache_dir).await
            .map_err(|e| ShuroError::Whisper(format!("Failed to read audio cache directory: {}", e)))?;
        
        let mut cached_items = Vec::new();
        
        while let Some(entry) = entries.next_entry().await
            .map_err(|e| ShuroError::Whisper(format!("Failed to read directory entry: {}", e)))? {
            
            if let Some(extension) = entry.path().extension() {
                if extension == "json" {
                    match tokio::fs::read_to_string(entry.path()).await {
                        Ok(content) => {
                            if let Ok(cached) = serde_json::from_str::<AudioCache>(&content) {
                                cached_items.push(cached);
                            }
                        }
                        Err(_) => continue, // Skip unreadable files
                    }
                }
            }
        }
        
        // Sort by cache time (newest first)
        cached_items.sort_by(|a, b| b.cached_at.cmp(&a.cached_at));
        
        Ok(cached_items)
    }

    /// Clear all cached audio files
    pub async fn clear_audio_cache(&self) -> Result<u64> {
        self.ensure_audio_cache_dir().await?;
        
        let mut entries = tokio::fs::read_dir(&self.audio_cache_dir).await
            .map_err(|e| ShuroError::Whisper(format!("Failed to read audio cache directory: {}", e)))?;
        
        let mut deleted_count = 0;
        
        while let Some(entry) = entries.next_entry().await
            .map_err(|e| ShuroError::Whisper(format!("Failed to read directory entry: {}", e)))? {
            
            if tokio::fs::remove_file(entry.path()).await.is_ok() {
                deleted_count += 1;
            }
        }
        
        info!("Cleared {} cached audio files", deleted_count / 2); // /2 because we have .wav and .json pairs
        Ok(deleted_count / 2)
    }

    /// Clean old audio cache entries
    pub async fn clean_audio_cache(&self, max_age_days: u64) -> Result<u64> {
        self.ensure_audio_cache_dir().await?;
        
        let cutoff_time = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs()
            .saturating_sub(max_age_days * 24 * 60 * 60);
        
        let mut entries = tokio::fs::read_dir(&self.audio_cache_dir).await
            .map_err(|e| ShuroError::Whisper(format!("Failed to read audio cache directory: {}", e)))?;
        
        let mut deleted_count = 0;
        
        while let Some(entry) = entries.next_entry().await
            .map_err(|e| ShuroError::Whisper(format!("Failed to read directory entry: {}", e)))? {
            
            if let Some(extension) = entry.path().extension() {
                if extension == "json" {
                    // Check if file is old enough to delete
                    if let Ok(content) = tokio::fs::read_to_string(entry.path()).await {
                        if let Ok(cached) = serde_json::from_str::<AudioCache>(&content) {
                            if cached.cached_at < cutoff_time {
                                // Delete both .json and .wav files
                                let json_path = entry.path();
                                let audio_path = json_path.with_extension("wav");
                                
                                let mut deleted_files = 0;
                                if tokio::fs::remove_file(&json_path).await.is_ok() {
                                    deleted_files += 1;
                                }
                                if tokio::fs::remove_file(&audio_path).await.is_ok() {
                                    deleted_files += 1;
                                }
                                
                                if deleted_files == 2 {
                                    deleted_count += 1;
                                }
                            }
                        }
                    }
                }
            }
        }
        
        info!("Cleaned {} old cached audio files (older than {} days)", deleted_count, max_age_days);
        Ok(deleted_count)
    }

    /// Extract audio from video directly to cache
    pub async fn extract_and_cache_audio<P: AsRef<Path>>(&self, video_path: P) -> Result<PathBuf> {
        self.ensure_audio_cache_dir().await?;
        
        let video_path = video_path.as_ref();
        let cache_key = self.generate_audio_cache_key(video_path)?;
        let cached_audio_path = self.audio_cache_dir.join(format!("{}.wav", cache_key));
        let metadata_file = self.audio_cache_dir.join(format!("{}.json", cache_key));
        
        info!("Extracting audio from {} directly to cache", video_path.display());
        
        // Extract audio directly to cache location
        let output = Command::new("ffmpeg")
            .arg("-i").arg(video_path)
            .arg("-vn") // No video
            .arg("-acodec").arg("pcm_s16le") // PCM 16-bit for whisper
            .arg("-ar").arg("16000") // 16kHz sample rate
            .arg("-ac").arg("1") // Mono
            .arg("-y") // Overwrite output
            .arg(&cached_audio_path)
            .output()
            .map_err(|e| ShuroError::FFmpeg(format!("Failed to execute ffmpeg: {}", e)))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(ShuroError::FFmpeg(format!(
                "Audio extraction failed: {}",
                stderr
            )));
        }
        
        // Create metadata
        let video_modified = std::fs::metadata(video_path)
            .ok()
            .and_then(|m| m.modified().ok())
            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
            .map(|d| d.as_secs());
        
        let cached = AudioCache {
            audio_path: cached_audio_path.to_string_lossy().to_string(),
            video_path: video_path.to_string_lossy().to_string(),
            video_modified,
            cached_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        };

        let content = serde_json::to_string_pretty(&cached)
            .map_err(|e| ShuroError::Whisper(format!("Failed to serialize audio cache: {}", e)))?;
        
        tokio::fs::write(&metadata_file, content).await
            .map_err(|e| ShuroError::Whisper(format!("Failed to write audio cache metadata: {}", e)))?;
        
        info!("Audio extracted and cached: {}", cache_key);
        Ok(cached_audio_path)
    }
}

/// Extract audio from video file using ffmpeg
pub async fn extract_audio<P: AsRef<Path>>(
    video_path: P,
    audio_path: P,
    ffmpeg_path: &str,
) -> Result<()> {
    let video_path = video_path.as_ref();
    let audio_path = audio_path.as_ref();

    info!("Extracting audio from {} to {}", video_path.display(), audio_path.display());

    let output = Command::new(ffmpeg_path)
        .arg("-i").arg(video_path)
        .arg("-vn") // No video
        .arg("-acodec").arg("pcm_s16le") // PCM 16-bit for whisper
        .arg("-ar").arg("16000") // 16kHz sample rate
        .arg("-ac").arg("1") // Mono
        .arg("-y") // Overwrite output
        .arg(audio_path)
        .output()
        .map_err(|e| ShuroError::FFmpeg(format!("Failed to execute ffmpeg: {}", e)))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(ShuroError::FFmpeg(format!(
            "Audio extraction failed: {}",
            stderr
        )));
    }

    info!("Audio extraction completed");
    Ok(())
}

/// Format duration in seconds to human readable string
fn format_duration(seconds: u64) -> String {
    if seconds < 60 {
        format!("{}s", seconds)
    } else if seconds < 3600 {
        format!("{}m {}s", seconds / 60, seconds % 60)
    } else {
        format!("{}h {}m", seconds / 3600, (seconds % 3600) / 60)
    }
}