// WhisperCpp implementation placeholder
// TODO: Move the existing whisper.cpp implementation here

use async_trait::async_trait;
use std::path::{Path, PathBuf};
use std::process::Command;
use serde_json;
use serde::{Serialize, Deserialize};
use tempfile;
use tracing::{info, warn};

use crate::config::{TranscriberConfig, TranscriptionMode};
use crate::error::{Result, ShuroError};
use crate::quality::{Transcription, QualityValidator};
use super::{TranscriberTrait, TuneResult, TranscriptionCache, AudioCache, CacheInfo, common::{WhisperUtils, AbstractTranscription, AbstractTranscriptionSegment, TranscriptionMapper}};

/// Whisper.cpp specific JSON output format
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WhisperCppOutput {
    pub result: WhisperCppResult,
    pub transcription: Vec<WhisperCppSegment>,
}

/// Whisper.cpp result metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WhisperCppResult {
    pub language: String,
}

/// Whisper.cpp specific segment format
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WhisperCppSegment {
    pub timestamps: WhisperCppTimestamps,
    pub offsets: WhisperCppOffsets,
    pub text: String,
}

/// Timestamp format in whisper-cpp
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WhisperCppTimestamps {
    pub from: String,
    pub to: String,
}

/// Offset format in whisper-cpp (milliseconds)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WhisperCppOffsets {
    pub from: u64,
    pub to: u64,
}

/// Mapper for Whisper.cpp format to abstract format
pub struct WhisperCppMapper;

impl TranscriptionMapper<WhisperCppOutput> for WhisperCppMapper {
    fn to_abstract_transcription(whisper_output: WhisperCppOutput) -> Result<AbstractTranscription> {
        let segments: Vec<AbstractTranscriptionSegment> = whisper_output.transcription
            .into_iter()
            .enumerate()
            .map(|(i, seg)| AbstractTranscriptionSegment {
                id: i as i32,
                start_time: seg.offsets.from as f64 / 1000.0, // Convert ms to seconds
                end_time: seg.offsets.to as f64 / 1000.0,     // Convert ms to seconds
                text: seg.text.trim().to_string(),
                confidence: None, // whisper-cpp doesn't provide confidence in basic output
                language: Some(whisper_output.result.language.clone()),
            })
            .collect();

        let duration = segments.last().map(|seg| seg.end_time);
        
        // Construct full text from segments
        let full_text = segments.iter()
            .map(|seg| seg.text.as_str())
            .collect::<Vec<_>>()
            .join(" ");

        Ok(AbstractTranscription {
            text: full_text,
            segments,
            language: whisper_output.result.language,
            duration,
            model_info: Some("Whisper.cpp".to_string()),
        })
    }

    fn to_legacy_transcription(abstract_result: AbstractTranscription) -> Transcription {
        abstract_result.into()
    }
}

/// Whisper.cpp implementation (uses system whisper command as fallback)
pub struct WhisperCppTranscriber {
    config: TranscriberConfig,
    validator: QualityValidator,
    cache_dir: PathBuf,
    audio_cache_dir: PathBuf,
}

impl WhisperCppTranscriber {
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

    /// Simple transcription using system whisper command (if available)
    async fn simple_transcribe(&self, audio_path: &Path, language: Option<&str>) -> Result<Transcription> {
        info!("Using simple whisper transcription for: {}", audio_path.display());
        
        // Create temporary output directory
        let temp_dir = tempfile::tempdir()
            .map_err(|e| ShuroError::Transcriber(format!("Failed to create temp directory: {}", e)))?;
        let output_dir = temp_dir.path();
        
        // Build command - use configured transcribe model
        let output_file = output_dir.join("transcription");
        let mut cmd = Command::new(&self.config.binary_path);
        cmd.arg("-f").arg(audio_path)
            .arg("-m").arg(&self.config.transcribe_model)
            .arg("-of").arg(&output_file)
            .arg("-oj"); // Output JSON format

        if let Some(lang) = language {
            cmd.arg("-l").arg(lang);
        }

        // Execute command
        let output = cmd.output()
            .map_err(|e| ShuroError::Transcriber(format!("Failed to execute whisper: {}", e)))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(ShuroError::Transcriber(format!("Whisper failed: {}", stderr)));
        }

        // Find and read JSON output
        let json_file = output_dir.join("transcription.json");

        let json_content = std::fs::read_to_string(&json_file)
            .map_err(|e| ShuroError::Transcriber(format!("Failed to read output: {}", e)))?;
        
        // Parse into Whisper.cpp-specific format
        let whisper_output: WhisperCppOutput = serde_json::from_str(&json_content)
            .map_err(|e| ShuroError::Transcriber(format!("Failed to parse Whisper.cpp JSON: {}", e)))?;

        // Convert to abstract format
        let abstract_transcription = WhisperCppMapper::to_abstract_transcription(whisper_output)?;
        
        // Convert to legacy format for compatibility
        Ok(WhisperCppMapper::to_legacy_transcription(abstract_transcription))
    }

    /// Transcribe with specific tempo using exploration model
    async fn transcribe_with_tempo(&self, video_path: &Path, tempo: i32, use_exploration_model: bool) -> Result<Transcription> {
        // Create temporary audio file with adjusted tempo
        let temp_dir = tempfile::tempdir()
            .map_err(|e| ShuroError::Transcriber(format!("Failed to create temp directory: {}", e)))?;
        let temp_audio = temp_dir.path().join("audio_tempo.wav");
        
        // Extract audio with tempo adjustment
        super::common::extract_audio_with_tempo(video_path, &temp_audio, "ffmpeg", tempo).await?;
        
        // Create output directory for transcription
        let output_dir = temp_dir.path().join("output");
        std::fs::create_dir_all(&output_dir)
            .map_err(|e| ShuroError::Transcriber(format!("Failed to create output dir: {}", e)))?;

        // Choose model based on whether this is exploration or final transcription
        let model = if use_exploration_model {
            &self.config.explore_model
        } else {
            &self.config.transcribe_model
        };

        // Build whisper command
        let output_file = output_dir.join("transcription");
        let mut cmd = Command::new(&self.config.binary_path);
        cmd.arg("-f").arg(&temp_audio)
            .arg("-m").arg(model)
            .arg("-of").arg(&output_file)
            .arg("-oj"); // Output JSON format

        // Execute command
        let output = cmd.output()
            .map_err(|e| ShuroError::Transcriber(format!("Failed to execute whisper: {}", e)))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(ShuroError::Transcriber(format!("Whisper failed: {}", stderr)));
        }

        // Find and read JSON output
        let json_file = output_dir.join("transcription.json");

        let json_content = std::fs::read_to_string(&json_file)
            .map_err(|e| ShuroError::Transcriber(format!("Failed to read output: {}", e)))?;
        
        // Parse and convert to legacy format
        let whisper_output: WhisperCppOutput = serde_json::from_str(&json_content)
            .map_err(|e| ShuroError::Transcriber(format!("Failed to parse Whisper.cpp JSON: {}", e)))?;

        let abstract_transcription = WhisperCppMapper::to_abstract_transcription(whisper_output)?;
        Ok(WhisperCppMapper::to_legacy_transcription(abstract_transcription))
    }

    /// Tuned transcription: find best tempo first, then transcribe with optimal settings
    async fn tuned_transcribe(&self, video_path: &Path, language: Option<&str>) -> Result<TuneResult> {
        info!("Starting tuned transcription for: {}", video_path.display());
        
        // Generate tempo test range
        let tempo_range = super::common::generate_tempo_range(
            self.config.explore_range_min,
            self.config.explore_range_max,
            self.config.explore_steps
        );
        
        info!("Testing {} tempo values: {:?}", tempo_range.len(), tempo_range);
        
        let mut best_tempo = 100;
        let mut best_smoothness = f64::MAX;
        let mut all_attempts = Vec::new();
        let mut tested_parameters = Vec::new();
        
        // Test each tempo with exploration model
        for &tempo in &tempo_range {
            info!("Testing tempo {}% with exploration model '{}'", tempo, self.config.explore_model);
            
            match self.transcribe_with_tempo(video_path, tempo, true).await {
                Ok(transcription) => {
                    let smoothness = super::common::calculate_segment_smoothness(&transcription);
                    all_attempts.push((tempo, smoothness));
                    tested_parameters.push(format!("tempo={}%->smoothness={:.3}", tempo, smoothness));
                    
                    info!("Tempo {}%: {} segments, smoothness score: {:.3}", 
                          tempo, transcription.segments.len(), smoothness);
                    
                    if smoothness < best_smoothness {
                        best_smoothness = smoothness;
                        best_tempo = tempo;
                        info!("New best tempo: {}% (smoothness: {:.3})", best_tempo, best_smoothness);
                    }
                }
                Err(e) => {
                    warn!("Failed to transcribe with tempo {}%: {}", tempo, e);
                    all_attempts.push((tempo, f64::MAX));
                    tested_parameters.push(format!("tempo={}%->error", tempo));
                }
            }
        }
        
        info!("Exploration phase complete. Best tempo: {}% (smoothness: {:.3})", best_tempo, best_smoothness);
        
        // Now transcribe with the best tempo using the full model
        info!("Final transcription with tempo {}% using model '{}'", best_tempo, self.config.transcribe_model);
        let final_transcription = self.transcribe_with_tempo(video_path, best_tempo, false).await?;
        
        // Validate quality
        if let Err(e) = self.validator.validate_transcription(&final_transcription) {
            warn!("Quality validation failed for best tempo {}: {}", best_tempo, e);
        }
        
        Ok(TuneResult {
            best_transcription: final_transcription,
            best_tempo,
            best_temperature: self.config.temperature,
            quality_score: best_smoothness,
            all_attempts,
            tested_parameters,
        })
    }
}

#[async_trait]
impl TranscriberTrait for WhisperCppTranscriber {
    async fn transcribe(&self, audio_path: &Path, language: Option<&str>) -> Result<Transcription> {
        use crate::config::TranscriptionMode;
        
        match self.config.mode {
            TranscriptionMode::Simple => {
                info!("Using simple transcription mode");
                self.simple_transcribe(audio_path, language).await
            }
            TranscriptionMode::Tuned => {
                info!("Using tuned transcription mode");
                // For tuned mode, we need to work with video files, but if we only have audio,
                // we'll need to create a temporary "video" file or adjust the approach
                // For now, fall back to simple mode when called with audio path directly
                warn!("Tuned mode called with audio path - falling back to simple mode");
                self.simple_transcribe(audio_path, language).await
            }
        }
    }

    async fn tune_transcription(&self, video_path: &Path) -> Result<TuneResult> {
        match self.config.mode {
            TranscriptionMode::Simple => {
                // Simple mode - just do basic transcription without tempo exploration
                info!("Simple mode tune: single transcription pass");
                
                // Extract audio first
                let audio_path = self.extract_and_cache_audio(video_path).await?;
                let transcription = self.simple_transcribe(&audio_path, None).await?;
                
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
                info!("Tuned mode: exploring optimal tempo");
                self.tuned_transcribe(video_path, None).await
            }
        }
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
            .map_err(|e| ShuroError::Transcriber(format!("Failed to extract audio: {}", e)))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(ShuroError::Transcriber(format!("Audio extraction failed: {}", stderr)));
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