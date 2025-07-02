// WhisperCpp implementation placeholder
// TODO: Move the existing whisper.cpp implementation here

use async_trait::async_trait;
use std::path::{Path, PathBuf};
use std::process::Command;
use serde_json;
use tempfile;
use tracing::info;

use crate::config::WhisperConfig;
use crate::error::{Result, ShuroError};
use crate::quality::{Transcription, TranscriptionSegment, QualityValidator};
use super::{WhisperTranscriberTrait, TuneResult, TranscriptionCache, AudioCache, CacheInfo, common::WhisperUtils};

/// Whisper.cpp implementation (uses system whisper command as fallback)
pub struct WhisperCppTranscriber {
    _config: WhisperConfig,
    _validator: QualityValidator,
    cache_dir: PathBuf,
    audio_cache_dir: PathBuf,
}

impl WhisperCppTranscriber {
    pub fn new(config: WhisperConfig, validator: QualityValidator) -> Self {
        let cache_base = std::env::current_dir()
            .unwrap_or_default()
            .join(".shuro")
            .join("cache");
        
        let cache_dir = cache_base.join("transcriptions");
        let audio_cache_dir = cache_base.join("audio");
        
        Self { 
            _config: config, 
            _validator: validator, 
            cache_dir, 
            audio_cache_dir 
        }
    }

    /// Simple transcription using system whisper command (if available)
    async fn simple_transcribe(&self, audio_path: &Path, language: Option<&str>) -> Result<Transcription> {
        info!("Using simple whisper transcription for: {}", audio_path.display());
        
        // Create temporary output directory
        let temp_dir = tempfile::tempdir()
            .map_err(|e| ShuroError::Whisper(format!("Failed to create temp directory: {}", e)))?;
        let output_dir = temp_dir.path();
        
        // Build command - use tiny model to avoid PyTorch issues
        let mut cmd = Command::new("whisper");
        cmd.arg(audio_path)
            .arg("--model").arg("tiny")
            .arg("--output_dir").arg(output_dir)
            .arg("--output_format").arg("json");

        if let Some(lang) = language {
            cmd.arg("--language").arg(lang);
        }

        // Execute command
        let output = cmd.output()
            .map_err(|e| ShuroError::Whisper(format!("Failed to execute whisper: {}", e)))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(ShuroError::Whisper(format!("Whisper failed: {}", stderr)));
        }

        // Find and read JSON output
        let audio_filename = audio_path.file_stem()
            .ok_or_else(|| ShuroError::Whisper("Invalid audio filename".to_string()))?;
        let json_file = output_dir.join(format!("{}.json", audio_filename.to_string_lossy()));

        let json_content = std::fs::read_to_string(&json_file)
            .map_err(|e| ShuroError::Whisper(format!("Failed to read output: {}", e)))?;
        
        let json_output: serde_json::Value = serde_json::from_str(&json_content)
            .map_err(|e| ShuroError::Whisper(format!("Failed to parse JSON: {}", e)))?;

        // Convert to our format
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
                temperature: 0.0,
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
impl WhisperTranscriberTrait for WhisperCppTranscriber {
    async fn transcribe(&self, audio_path: &Path, language: Option<&str>) -> Result<Transcription> {
        self.simple_transcribe(audio_path, language).await
    }

    async fn tune_transcription(&self, audio_path: &Path) -> Result<TuneResult> {
        // For simplicity, just do a single transcription without tuning
        let transcription = self.simple_transcribe(audio_path, None).await?;
        
        Ok(TuneResult {
            best_transcription: transcription,
            best_tempo: 100,
            best_temperature: 0.0,
            quality_score: 8.0, // Assume good quality
            all_attempts: vec![(100, 8.0)],
            tested_parameters: vec!["single-pass".to_string()],
        })
    }

    async fn extract_and_cache_audio(&self, video_path: &Path) -> Result<PathBuf> {
        // Generate cache key and audio path
        let cache_key = WhisperUtils::generate_file_hash(video_path, &["audio_extraction"])?;
        let audio_path = self.audio_cache_dir.join(format!("{}.wav", cache_key));
        
        // Create cache directory if needed
        if let Some(parent) = audio_path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| ShuroError::Cache(format!("Failed to create audio cache dir: {}", e)))?;
        }
        
        // Extract audio using ffmpeg
        let output = Command::new("ffmpeg")
            .args(&[
                "-i", &video_path.to_string_lossy(),
                "-vn",
                "-acodec", "pcm_s16le",
                "-ar", "16000",
                "-ac", "1",
                "-y",
                &audio_path.to_string_lossy()
            ])
            .output()
            .map_err(|e| ShuroError::Whisper(format!("Failed to extract audio: {}", e)))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(ShuroError::Whisper(format!("Audio extraction failed: {}", stderr)));
        }

        Ok(audio_path)
    }

    async fn get_cached_audio(&self, video_path: &Path) -> Result<Option<PathBuf>> {
        let cache_key = WhisperUtils::generate_file_hash(video_path, &["audio_extraction"])?;
        let audio_path = self.audio_cache_dir.join(format!("{}.wav", cache_key));
        
        if audio_path.exists() {
            Ok(Some(audio_path))
        } else {
            Ok(None)
        }
    }

    async fn clear_cache(&self) -> Result<u64> {
        let mut count = 0;
        if let Ok(mut entries) = tokio::fs::read_dir(&self.cache_dir).await {
            while let Ok(Some(entry)) = entries.next_entry().await {
                if tokio::fs::remove_file(entry.path()).await.is_ok() {
                    count += 1;
                }
            }
        }
        Ok(count)
    }

    async fn list_cache(&self) -> Result<Vec<TranscriptionCache>> {
        // Return empty list for simplicity
        Ok(Vec::new())
    }

    async fn cache_info(&self) -> Result<CacheInfo> {
        Ok(CacheInfo {
            total_files: 0,
            total_size: 0,
            audio_files: 0,
            audio_size: 0,
            oldest_entry: None,
            newest_entry: None,
            models_used: vec!["tiny".to_string()],
        })
    }

    async fn clear_audio_cache(&self) -> Result<u64> {
        let mut count = 0;
        if let Ok(mut entries) = tokio::fs::read_dir(&self.audio_cache_dir).await {
            while let Ok(Some(entry)) = entries.next_entry().await {
                if tokio::fs::remove_file(entry.path()).await.is_ok() {
                    count += 1;
                }
            }
        }
        Ok(count)
    }

    async fn list_audio_cache(&self) -> Result<Vec<AudioCache>> {
        // Return empty list for simplicity
        Ok(Vec::new())
    }
} 