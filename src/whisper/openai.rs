// OpenAI Whisper Python implementation
// This provides support for OpenAI's Whisper Python library

use async_trait::async_trait;
use std::path::{Path, PathBuf};
use std::process::Command;
use serde_json;
use tracing::{info, debug};
use tempfile;

use crate::config::WhisperConfig;
use crate::error::{Result, ShuroError};
use crate::quality::{Transcription, TranscriptionSegment, QualityValidator};
use super::{WhisperTranscriberTrait, TuneResult, TranscriptionCache, AudioCache, CacheInfo, common::WhisperUtils};

/// OpenAI Whisper implementation
pub struct OpenAITranscriber {
    config: WhisperConfig,
    validator: QualityValidator,
    cache_dir: PathBuf,
    audio_cache_dir: PathBuf,
}

impl OpenAITranscriber {
    pub fn new(config: WhisperConfig, validator: QualityValidator) -> Self {
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
            .map_err(|e| ShuroError::Whisper(format!("whisper command not found: {}", e)))?;

        if output.status.success() {
            info!("OpenAI Whisper command-line tool is available");
            Ok(())
        } else {
            let stderr = String::from_utf8_lossy(&output.stderr);
            Err(ShuroError::Whisper(format!(
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
            .map_err(|e| ShuroError::Whisper(format!("Failed to create temp directory: {}", e)))?;
        
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
            .map_err(|e| ShuroError::Whisper(format!("Failed to execute whisper command: {}", e)))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(ShuroError::Whisper(format!(
                "OpenAI Whisper transcription failed: {}",
                stderr
            )));
        }

        // Find the JSON output file
        let audio_filename = audio_path.file_stem()
            .ok_or_else(|| ShuroError::Whisper("Invalid audio filename".to_string()))?;
        let json_file = output_dir.join(format!("{}.json", audio_filename.to_string_lossy()));

        if !json_file.exists() {
            return Err(ShuroError::Whisper("Whisper JSON output file not found".to_string()));
        }

        // Read and parse JSON output
        let json_content = std::fs::read_to_string(&json_file)
            .map_err(|e| ShuroError::Whisper(format!("Failed to read JSON output: {}", e)))?;
        
        let json_output: serde_json::Value = serde_json::from_str(&json_content)
            .map_err(|e| ShuroError::Whisper(format!("Failed to parse transcription JSON: {}", e)))?;

        // Convert to our Transcription format
        let segments: Vec<TranscriptionSegment> = json_output["segments"]
            .as_array()
            .unwrap_or(&Vec::new())
            .iter()
            .map(|seg| TranscriptionSegment {
                id: seg["id"].as_u64().unwrap_or(0) as i32,
                start: seg["start"].as_f64().unwrap_or(0.0),
                end: seg["end"].as_f64().unwrap_or(0.0),
                text: seg["text"].as_str().unwrap_or("").to_string(),
                tokens: seg["tokens"]
                    .as_array()
                    .unwrap_or(&Vec::new())
                    .iter()
                    .filter_map(|t| t.as_u64().map(|n| n as i32))
                    .collect(),
                temperature: seg["temperature"].as_f64().unwrap_or(temperature as f64) as f32,
                avg_logprob: seg["avg_logprob"].as_f64().unwrap_or(0.0) as f32,
                compression_ratio: seg["compression_ratio"].as_f64().unwrap_or(0.0) as f32,
                no_speech_prob: seg["no_speech_prob"].as_f64().unwrap_or(0.0) as f32,
            })
            .collect();

        Ok(Transcription {
            text: json_output["text"].as_str().unwrap_or("").to_string(),
            segments,
            language: json_output["language"].as_str().unwrap_or("unknown").to_string(),
        })
    }
}

#[async_trait]
impl WhisperTranscriberTrait for OpenAITranscriber {
    async fn transcribe(&self, audio_path: &Path, language: Option<&str>) -> Result<Transcription> {
        info!("Starting OpenAI Whisper transcription of: {}", audio_path.display());
        
        // Check availability first
        Self::check_availability().await?;

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

    async fn tune_transcription(&self, audio_path: &Path) -> Result<TuneResult> {
        info!("Tuning OpenAI Whisper transcription for: {}", audio_path.display());
        
        // Test different temperatures
        let mut best_score = f64::MAX;
        let mut best_temperature = self.config.temperature;
        let mut results = Vec::new();

        for &temp in &[0.0, 0.2, 0.4, 0.6, 0.8, 1.0] {
            let transcription = self.execute_transcription(
                audio_path,
                &self.config.transcribe_model,
                None,
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
                None,
                best_temperature,
            ).await?,
            best_tempo: 100, // Not applicable for OpenAI Whisper
            best_temperature,
            quality_score: best_score,
            all_attempts: results.iter().map(|(t, s)| ((*t * 10.0) as i32, *s)).collect(), // Convert temperature to tempo-like format
            tested_parameters: results.into_iter().map(|(t, s)| format!("temp={:.1}->score={:.3}", t, s)).collect(),
        })
    }

    async fn extract_and_cache_audio(&self, video_path: &Path) -> Result<PathBuf> {
        // Use the common audio extraction functionality
        let cache_key = WhisperUtils::generate_file_hash(video_path, &[])?;
        let audio_path = self.audio_cache_dir.join(format!("{}.wav", cache_key));

        if !audio_path.exists() {
            // Extract audio using ffmpeg (placeholder - would need actual implementation)
            std::fs::create_dir_all(&self.audio_cache_dir)
                .map_err(|e| ShuroError::Cache(format!("Failed to create audio cache directory: {}", e)))?;
            // TODO: Implement actual audio extraction
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