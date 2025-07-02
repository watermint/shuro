// OpenAI Whisper Python implementation
// This provides support for OpenAI's Whisper Python library

use async_trait::async_trait;
use std::path::{Path, PathBuf};
use std::process::Command;
use serde_json;
use serde::{Serialize, Deserialize};
use tracing::{info, debug};
use tempfile;

use crate::config::TranscriberConfig;
use crate::error::{Result, ShuroError};
use crate::quality::{Transcription, QualityValidator};
use super::{TranscriberTrait, TuneResult, TranscriptionCache, AudioCache, CacheInfo, common::{WhisperUtils, AbstractTranscription, AbstractTranscriptionSegment, TranscriptionMapper}};

/// OpenAI Whisper specific JSON output format
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAIWhisperOutput {
    pub text: String,
    pub segments: Vec<OpenAIWhisperSegment>,
    pub language: Option<String>,
}

/// OpenAI Whisper specific segment format
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAIWhisperSegment {
    pub id: u64,
    pub seek: Option<u64>,
    pub start: f64,
    pub end: f64,
    pub text: String,
    pub tokens: Option<Vec<i32>>,
    pub temperature: Option<f64>,
    pub avg_logprob: Option<f64>,
    pub compression_ratio: Option<f64>,
    pub no_speech_prob: Option<f64>,
}

/// Mapper for OpenAI Whisper format to abstract format
pub struct OpenAIWhisperMapper;

impl TranscriptionMapper<OpenAIWhisperOutput> for OpenAIWhisperMapper {
    fn to_abstract_transcription(whisper_output: OpenAIWhisperOutput) -> Result<AbstractTranscription> {
        let segments: Vec<AbstractTranscriptionSegment> = whisper_output.segments
            .into_iter()
            .map(|seg| AbstractTranscriptionSegment {
                id: seg.id as i32,
                start_time: seg.start,
                end_time: seg.end,
                text: seg.text.trim().to_string(),
                confidence: seg.avg_logprob.map(|logprob| {
                    // Convert log probability to confidence score (0.0 to 1.0)
                    (logprob.exp() as f32).clamp(0.0, 1.0)
                }),
                language: whisper_output.language.clone(),
            })
            .collect();

        let duration = segments.last().map(|seg| seg.end_time);

        Ok(AbstractTranscription {
            text: whisper_output.text,
            segments,
            language: whisper_output.language.unwrap_or_else(|| "unknown".to_string()),
            duration,
            model_info: Some("OpenAI Whisper".to_string()),
        })
    }

    fn to_legacy_transcription(abstract_result: AbstractTranscription) -> Transcription {
        abstract_result.into()
    }
}

/// OpenAI Whisper implementation
pub struct OpenAITranscriber {
    config: TranscriberConfig,
    validator: QualityValidator,
    cache_dir: PathBuf,
    audio_cache_dir: PathBuf,
}

impl OpenAITranscriber {
    pub fn new(config: TranscriberConfig, validator: QualityValidator) -> Self {
        let cache_base = std::env::current_dir()
            .unwrap_or_default()
            .join(".shuro")
            .join("cache");
        
        let cache_dir = cache_base.join("transcriptions");
        let audio_cache_dir = cache_base.join("audio");
        
        Self { 
            config, 
            validator, 
            cache_dir, 
            audio_cache_dir 
        }
    }

    /// Check if OpenAI Whisper is available via command line
    pub async fn check_availability() -> Result<()> {
        let output = Command::new("whisper")
            .arg("--help")
            .output()
            .map_err(|e| ShuroError::Transcriber(format!("whisper command not found: {}", e)))?;

        if output.status.success() {
            info!("OpenAI Whisper command-line tool is available");
            Ok(())
        } else {
            let stderr = String::from_utf8_lossy(&output.stderr);
            Err(ShuroError::Transcriber(format!(
                "OpenAI Whisper not available. Install with: brew install openai-whisper\nError: {}",
                stderr
            )))
        }
    }

    /// Execute transcription using OpenAI Whisper command-line tool
    async fn execute_transcription(
        &self,
        audio_path: &Path,
        model: &str,
        language: Option<&str>,
        temperature: f32,
    ) -> Result<Transcription> {
        debug!("Executing OpenAI Whisper transcription with model: {}", model);

        // Create temporary output directory for whisper results
        let temp_dir = tempfile::tempdir()
            .map_err(|e| ShuroError::Transcriber(format!("Failed to create temp directory: {}", e)))?;
        
        let output_dir = temp_dir.path();
        
        // Build whisper command
        let mut cmd = Command::new("whisper");
        cmd.arg(audio_path)
            .arg("--model").arg(model)
            .arg("--output_dir").arg(output_dir)
            .arg("--output_format").arg("json")
            .arg("--temperature").arg(temperature.to_string());

        // Add language if specified
        if let Some(lang) = language {
            cmd.arg("--language").arg(lang);
        }

        // Execute command
        let output = cmd.output()
            .map_err(|e| ShuroError::Transcriber(format!("Failed to execute whisper command: {}", e)))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(ShuroError::Transcriber(format!(
                "OpenAI Whisper transcription failed: {}",
                stderr
            )));
        }

        // Find the JSON output file
        let audio_filename = audio_path.file_stem()
            .ok_or_else(|| ShuroError::Transcriber("Invalid audio filename".to_string()))?;
        let json_file = output_dir.join(format!("{}.json", audio_filename.to_string_lossy()));

        if !json_file.exists() {
            return Err(ShuroError::Transcriber("Whisper JSON output file not found".to_string()));
        }

        // Read and parse JSON output
        let json_content = std::fs::read_to_string(&json_file)
            .map_err(|e| ShuroError::Transcriber(format!("Failed to read JSON output: {}", e)))?;
        
        // Parse into OpenAI-specific format
        let openai_output: OpenAIWhisperOutput = serde_json::from_str(&json_content)
            .map_err(|e| ShuroError::Transcriber(format!("Failed to parse OpenAI Whisper JSON: {}", e)))?;

        // Convert to abstract format
        let abstract_transcription = OpenAIWhisperMapper::to_abstract_transcription(openai_output)?;
        
        // Convert to legacy format for compatibility
        Ok(OpenAIWhisperMapper::to_legacy_transcription(abstract_transcription))
    }

    /// Simple transcription with caching - the original transcribe logic
    async fn transcribe_simple(&self, audio_path: &Path, language: Option<&str>) -> Result<Transcription> {
        // Generate cache key
        let cache_key = WhisperUtils::generate_file_hash(
            audio_path,
            &[&self.config.transcribe_model, &self.config.temperature.to_string(), language.unwrap_or("")]
        )?;

        let cache_file = self.cache_dir.join(format!("{}.json", cache_key));

        // Check cache first
        if cache_file.exists() {
            debug!("Loading transcription from cache: {}", cache_file.display());
            if let Ok(cached_content) = std::fs::read_to_string(&cache_file) {
                if let Ok(cached_transcription) = serde_json::from_str::<Transcription>(&cached_content) {
                    info!("Using cached transcription");
                    return Ok(cached_transcription);
                }
            }
        }

        // Perform transcription
        let transcription = self.execute_transcription(
            audio_path,
            &self.config.transcribe_model,
            language,
            self.config.temperature,
        ).await?;

        // Validate quality
        self.validator.validate_transcription(&transcription)?;

        // Cache the result
        std::fs::create_dir_all(&self.cache_dir)
            .map_err(|e| ShuroError::Cache(format!("Failed to create cache directory: {}", e)))?;
        
        let json_content = serde_json::to_string_pretty(&transcription)
            .map_err(|e| ShuroError::Cache(format!("Failed to serialize transcription: {}", e)))?;
        
        std::fs::write(&cache_file, json_content)
            .map_err(|e| ShuroError::Cache(format!("Failed to write cache file: {}", e)))?;

        info!("OpenAI Whisper transcription completed successfully");
        Ok(transcription)
    }

    /// Internal tuning logic - the original tune_transcription logic
    async fn tune_transcription_internal(&self, audio_path: &Path, language: Option<&str>) -> Result<TuneResult> {
        info!("Tuning OpenAI Whisper transcription for: {}", audio_path.display());
        
        // Test different temperatures
        let mut best_score = f64::MAX;
        let mut best_temperature = self.config.temperature;
        let mut results = Vec::new();

        for &temp in &[0.0, 0.2, 0.4, 0.6, 0.8, 1.0] {
            let transcription = self.execute_transcription(
                audio_path,
                &self.config.transcribe_model,
                language,
                temp,
            ).await?;

            let quality_score = transcription.quality().score();
            results.push((temp, quality_score));

            if quality_score < best_score {
                best_score = quality_score;
                best_temperature = temp;
            }
        }

        Ok(TuneResult {
            best_transcription: self.execute_transcription(
                audio_path,
                &self.config.transcribe_model,
                language,
                best_temperature,
            ).await?,
            best_tempo: 100, // Not applicable for OpenAI Whisper
            best_temperature,
            quality_score: best_score,
            all_attempts: results.iter().map(|(t, s)| ((*t * 10.0) as i32, *s)).collect(), // Convert temperature to tempo-like format
            tested_parameters: results.into_iter().map(|(t, s)| format!("temp={:.1}->score={:.3}", t, s)).collect(),
        })
    }
}

#[async_trait]
impl TranscriberTrait for OpenAITranscriber {
    async fn transcribe(&self, audio_path: &Path, language: Option<&str>) -> Result<Transcription> {
        use crate::config::TranscriptionMode;
        
        info!("Starting OpenAI Whisper transcription of: {}", audio_path.display());
        
        // Check availability first
        Self::check_availability().await?;

        match self.config.mode {
            TranscriptionMode::Simple => {
                info!("Using simple transcription mode");
                self.transcribe_simple(audio_path, language).await
            }
            TranscriptionMode::Tuned => {
                info!("Using tuned transcription mode");
                // For tuned mode with OpenAI, we tune temperature instead of tempo
                // Since this method expects audio path, we'll do a simplified tuning
                let tune_result = self.tune_transcription_internal(audio_path, language).await?;
                Ok(tune_result.best_transcription)
            }
        }
    }

    async fn tune_transcription(&self, video_path: &Path) -> Result<TuneResult> {
        use crate::config::TranscriptionMode;
        
        match self.config.mode {
            TranscriptionMode::Simple => {
                info!("Simple mode tune: single transcription pass");
                
                // Extract audio first
                let audio_path = self.extract_and_cache_audio(video_path).await?;
                let transcription = self.transcribe_simple(&audio_path, None).await?;
                
                Ok(TuneResult {
                    best_transcription: transcription,
                    best_tempo: 100,
                    best_temperature: self.config.temperature,
                    quality_score: 1.0, // Assume reasonable quality for simple mode
                    all_attempts: vec![(100, 1.0)],
                    tested_parameters: vec!["simple-mode-single-pass".to_string()],
                })
            }
            TranscriptionMode::Tuned => {
                info!("Tuned mode: exploring optimal temperature");
                
                // Extract audio first
                let audio_path = self.extract_and_cache_audio(video_path).await?;
                self.tune_transcription_internal(&audio_path, None).await
            }
        }
    }

    async fn extract_and_cache_audio(&self, video_path: &Path) -> Result<PathBuf> {
        // Use the common audio extraction functionality
        let cache_key = WhisperUtils::generate_file_hash(video_path, &["audio_extraction"])?;
        let audio_path = self.audio_cache_dir.join(format!("{}.wav", cache_key));

        if !audio_path.exists() {
            // Create cache directory if needed
            std::fs::create_dir_all(&self.audio_cache_dir)
                .map_err(|e| ShuroError::Cache(format!("Failed to create audio cache directory: {}", e)))?;

            // Extract audio using the common function with proper original file name logging
            let original_name = video_path.file_name()
                .and_then(|n| n.to_str());
            super::common::extract_audio(video_path, &audio_path, "ffmpeg", original_name).await?;
        } else {
            let original_name = video_path.file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("unknown");
            info!("Using cached audio for: {}", original_name);
        }

        Ok(audio_path)
    }

    async fn get_cached_audio(&self, video_path: &Path) -> Result<Option<PathBuf>> {
        let cache_key = WhisperUtils::generate_file_hash(video_path, &[])?;
        let audio_path = self.audio_cache_dir.join(format!("{}.wav", cache_key));

        if audio_path.exists() {
            Ok(Some(audio_path))
        } else {
            Ok(None)
        }
    }
    
    async fn clear_cache(&self) -> Result<u64> {
        WhisperUtils::clean_cache_by_age(&self.cache_dir, 0, "json").await
    }
    
    async fn list_cache(&self) -> Result<Vec<TranscriptionCache>> {
        let mut entries = Vec::new();
        
        if let Ok(mut dir_entries) = tokio::fs::read_dir(&self.cache_dir).await {
            while let Ok(Some(entry)) = dir_entries.next_entry().await {
                if let Some(extension) = entry.path().extension() {
                    if extension == "json" {
                        if let Ok(content) = tokio::fs::read_to_string(entry.path()).await {
                            if let Ok(cache_entry) = serde_json::from_str::<TranscriptionCache>(&content) {
                                entries.push(cache_entry);
                            }
                        }
                    }
                }
            }
        }
        
        entries.sort_by(|a, b| b.cached_at.cmp(&a.cached_at));
        Ok(entries)
    }
    
    async fn cache_info(&self) -> Result<CacheInfo> {
        let (total_files, total_size, oldest_entry, newest_entry) = 
            WhisperUtils::get_cache_stats(&self.cache_dir, "json").await?;
        
        let (audio_files, audio_size, _, _) = 
            WhisperUtils::get_cache_stats(&self.audio_cache_dir, "wav").await?;
        
        // Get models used (simplified for OpenAI)
        let models_used = vec![self.config.transcribe_model.clone()];
        
        Ok(CacheInfo {
            total_files,
            total_size,
            oldest_entry,
            newest_entry,
            models_used,
            audio_files,
            audio_size,
        })
    }
    
    async fn clear_audio_cache(&self) -> Result<u64> {
        WhisperUtils::clean_cache_by_age(&self.audio_cache_dir, 0, "wav").await
    }
    
    async fn list_audio_cache(&self) -> Result<Vec<AudioCache>> {
        let mut entries = Vec::new();
        
        if let Ok(mut dir_entries) = tokio::fs::read_dir(&self.audio_cache_dir).await {
            while let Ok(Some(entry)) = dir_entries.next_entry().await {
                if let Some(extension) = entry.path().extension() {
                    if extension == "wav" {
                        // Create a basic AudioCache entry
                        // Note: This is simplified since we'd need to track the original video paths
                        let cache_entry = AudioCache {
                            audio_path: entry.path().to_string_lossy().to_string(),
                            video_path: "unknown".to_string(), // Would need proper tracking
                            video_modified: None,
                            cached_at: entry.metadata().await
                                .ok()
                                .and_then(|m| m.modified().ok())
                                .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                                .map(|d| d.as_secs())
                                .unwrap_or(0),
                        };
                        entries.push(cache_entry);
                    }
                }
            }
        }
        
        entries.sort_by(|a, b| b.cached_at.cmp(&a.cached_at));
        Ok(entries)
    }
} 