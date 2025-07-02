use std::path::Path;
use std::process::Command;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use serde::{Serialize, Deserialize};
use tracing::info;

use crate::error::{Result, ShuroError};
use crate::quality::{Transcription, TranscriptionSegment};

/// Abstract transcription segment that is service-agnostic
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AbstractTranscriptionSegment {
    pub id: i32,
    pub start_time: f64,
    pub end_time: f64,
    pub text: String,
    pub confidence: Option<f32>,
    pub language: Option<String>,
}

/// Abstract transcription result that is service-agnostic
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AbstractTranscription {
    pub text: String,
    pub segments: Vec<AbstractTranscriptionSegment>,
    pub language: String,
    pub duration: Option<f64>,
    pub model_info: Option<String>,
}

/// Trait for converting service-specific transcription formats to abstract format
pub trait TranscriptionMapper<T> {
    /// Convert service-specific format to abstract transcription
    fn to_abstract_transcription(service_result: T) -> Result<AbstractTranscription>;
    
    /// Convert abstract transcription to legacy format for compatibility
    fn to_legacy_transcription(abstract_result: AbstractTranscription) -> Transcription;
}

/// Convert abstract transcription to the existing Transcription format for backward compatibility
impl From<AbstractTranscription> for Transcription {
    fn from(abstract_transcription: AbstractTranscription) -> Self {
        let segments = abstract_transcription.segments
            .into_iter()
            .map(|seg| TranscriptionSegment {
                id: seg.id,
                start: seg.start_time,
                end: seg.end_time,
                text: seg.text,
                tokens: vec![], // Not always available
                temperature: 0.0, // Default value
                avg_logprob: seg.confidence.unwrap_or(0.0),
                compression_ratio: 1.0, // Default value
                no_speech_prob: 1.0 - seg.confidence.unwrap_or(0.5), // Inverse of confidence
            })
            .collect();

        Transcription {
            text: abstract_transcription.text,
            segments,
            language: abstract_transcription.language,
        }
    }
}

#[derive(Debug)]
pub struct TuneParams {
    pub tempo: i32,
    pub model: String,
    pub temperature: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TuneResult {
    pub best_transcription: Transcription,
    pub best_tempo: i32,
    pub best_temperature: f32,
    pub quality_score: f64,
    pub all_attempts: Vec<(i32, f64)>, // (tempo, quality_score)
    pub tested_parameters: Vec<String>, // Description of tested parameters
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

/// Base utilities for whisper implementations
pub struct WhisperUtils;

impl WhisperUtils {
    /// Format duration in seconds to a human-readable string
    pub fn format_duration(seconds: u64) -> String {
        let days = seconds / (24 * 60 * 60);
        let hours = (seconds % (24 * 60 * 60)) / (60 * 60);
        let minutes = (seconds % (60 * 60)) / 60;
        let secs = seconds % 60;

        if days > 0 {
            format!("{}d {}h", days, hours)
        } else if hours > 0 {
            format!("{}h {}m", hours, minutes)
        } else if minutes > 0 {
            format!("{}m {}s", minutes, secs)
        } else {
            format!("{}s", secs)
        }
    }

    /// Generate file hash for caching
    pub fn generate_file_hash<P: AsRef<Path>>(path: P, additional_data: &[&str]) -> Result<String> {
        let path = path.as_ref();
        
        // Get file metadata
        let metadata = std::fs::metadata(path)
            .map_err(|e| ShuroError::Transcriber(format!("Failed to read file metadata: {}", e)))?;
        
        let modified = metadata.modified()
            .map_err(|e| ShuroError::Transcriber(format!("Failed to get modification time: {}", e)))?
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        
        // Create hash input
        let mut hasher = DefaultHasher::new();
        path.to_string_lossy().hash(&mut hasher);
        modified.hash(&mut hasher);
        
        for data in additional_data {
            data.hash(&mut hasher);
        }
        
        let hash = hasher.finish();
        Ok(format!("{:016x}", hash))
    }

    /// Check if directory exists and create if not
    pub async fn ensure_directory<P: AsRef<Path>>(path: P) -> Result<()> {
        tokio::fs::create_dir_all(path).await
            .map_err(|e| ShuroError::Transcriber(format!("Failed to create directory: {}", e)))?;
        Ok(())
    }

    /// Get file size
    pub fn get_file_size<P: AsRef<Path>>(path: P) -> Result<u64> {
        let metadata = std::fs::metadata(path)
            .map_err(|e| ShuroError::Transcriber(format!("Failed to read file metadata: {}", e)))?;
        Ok(metadata.len())
    }

    /// Clean old cache entries based on age
    pub async fn clean_cache_by_age<P: AsRef<Path>>(
        cache_dir: P,
        max_age_days: u64,
        file_extension: &str,
    ) -> Result<u64> {
        let cache_dir = cache_dir.as_ref();
        let max_age_seconds = max_age_days * 24 * 60 * 60;
        let current_time = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        
        let mut removed_count = 0;
        
        if let Ok(mut entries) = tokio::fs::read_dir(cache_dir).await {
            while let Ok(Some(entry)) = entries.next_entry().await {
                if let Some(extension) = entry.path().extension() {
                    if extension == file_extension {
                        if let Ok(metadata) = entry.metadata().await {
                            if let Ok(modified) = metadata.modified() {
                                if let Ok(duration) = modified.duration_since(std::time::UNIX_EPOCH) {
                                    let age_seconds = current_time.saturating_sub(duration.as_secs());
                                    if age_seconds > max_age_seconds {
                                        if tokio::fs::remove_file(entry.path()).await.is_ok() {
                                            removed_count += 1;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        
        Ok(removed_count)
    }

    /// Get cache statistics
    pub async fn get_cache_stats<P: AsRef<Path>>(
        cache_dir: P,
        file_extension: &str,
    ) -> Result<(u64, u64, Option<u64>, Option<u64>)> {
        let cache_dir = cache_dir.as_ref();
        let mut total_files = 0;
        let mut total_size = 0;
        let mut oldest_entry: Option<u64> = None;
        let mut newest_entry: Option<u64> = None;
        
        if let Ok(mut entries) = tokio::fs::read_dir(cache_dir).await {
            while let Ok(Some(entry)) = entries.next_entry().await {
                if let Some(extension) = entry.path().extension() {
                    if extension == file_extension {
                        total_files += 1;
                        
                        if let Ok(metadata) = entry.metadata().await {
                            total_size += metadata.len();
                            
                            if let Ok(modified) = metadata.modified() {
                                if let Ok(duration) = modified.duration_since(std::time::UNIX_EPOCH) {
                                    let timestamp = duration.as_secs();
                                    
                                    oldest_entry = Some(oldest_entry.map_or(timestamp, |o| o.min(timestamp)));
                                    newest_entry = Some(newest_entry.map_or(timestamp, |n| n.max(timestamp)));
                                }
                            }
                        }
                    }
                }
            }
        }
        
        Ok((total_files, total_size, oldest_entry, newest_entry))
    }
}

/// Extract audio from video using ffmpeg
pub async fn extract_audio<P: AsRef<Path>>(
    video_path: P,
    audio_path: P,
    ffmpeg_path: &str,
    original_file_name: Option<&str>,
) -> Result<()> {
    let video_path = video_path.as_ref();
    let audio_path = audio_path.as_ref();

    let log_message = if let Some(original_name) = original_file_name {
        format!("Extracting audio from {}", original_name)
    } else {
        format!("Extracting audio from {} to {}", video_path.display(), audio_path.display())
    };
    
    info!("{}", log_message);

    let output = Command::new(ffmpeg_path)
        .arg("-i").arg(video_path)
        .arg("-vn") // No video
        .arg("-acodec").arg("pcm_s16le") // PCM 16-bit for whisper
        .arg("-ar").arg("16000") // 16kHz sample rate
        .arg("-ac").arg("1") // Mono
        .arg("-y") // Overwrite output
        .arg(audio_path)
        .output()
        .map_err(|e| ShuroError::Transcriber(format!("Failed to execute ffmpeg: {}", e)))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(ShuroError::Transcriber(format!(
            "Audio extraction failed: {}",
            stderr
        )));
    }

    info!("Audio extraction completed");
    Ok(())
}

/// Extract audio with specific tempo adjustment using ffmpeg
pub async fn extract_audio_with_tempo<P: AsRef<Path>>(
    video_path: P,
    audio_path: P,
    ffmpeg_path: &str,
    tempo_percentage: i32,
    original_file_name: Option<&str>,
) -> Result<()> {
    let video_path = video_path.as_ref();
    let audio_path = audio_path.as_ref();

    let log_message = if let Some(original_name) = original_file_name {
        format!("Extracting audio from {} with tempo {}%", original_name, tempo_percentage)
    } else {
        format!("Extracting audio from {} to {} with tempo {}%", 
                video_path.display(), audio_path.display(), tempo_percentage)
    };
    
    info!("{}", log_message);

    // Convert percentage to ffmpeg atempo value (e.g., 110% -> 1.1, 80% -> 0.8)
    let tempo_factor = tempo_percentage as f64 / 100.0;
    
    let output = Command::new(ffmpeg_path)
        .arg("-i").arg(video_path)
        .arg("-vn") // No video
        .arg("-acodec").arg("pcm_s16le") // PCM 16-bit for whisper
        .arg("-ar").arg("16000") // 16kHz sample rate
        .arg("-ac").arg("1") // Mono
        .arg("-af").arg(format!("atempo={}", tempo_factor)) // Apply tempo adjustment
        .arg("-y") // Overwrite output
        .arg(audio_path)
        .output()
        .map_err(|e| ShuroError::Transcriber(format!("Failed to execute ffmpeg: {}", e)))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(ShuroError::Transcriber(format!(
            "Audio extraction with tempo adjustment failed: {}",
            stderr
        )));
    }

    info!("Audio extraction with tempo adjustment completed");
    Ok(())
}

/// Calculate segment smoothness score - lower score means more evenly distributed segments
pub fn calculate_segment_smoothness(transcription: &Transcription) -> f64 {
    if transcription.segments.len() < 2 {
        return f64::MAX; // Single or no segments are considered poor quality
    }

    // Calculate segment durations
    let mut durations: Vec<f64> = Vec::new();
    for segment in &transcription.segments {
        let duration = segment.end - segment.start;
        if duration > 0.0 {
            durations.push(duration);
        }
    }

    if durations.is_empty() {
        return f64::MAX;
    }

    // Calculate mean duration
    let mean_duration: f64 = durations.iter().sum::<f64>() / durations.len() as f64;
    
    // Calculate coefficient of variation (standard deviation / mean)
    // Lower CV means more evenly distributed segments
    let variance: f64 = durations.iter()
        .map(|&duration| {
            let diff = duration - mean_duration;
            diff * diff
        })
        .sum::<f64>() / durations.len() as f64;
    
    let std_deviation = variance.sqrt();
    
    // Return coefficient of variation - lower is better
    if mean_duration > 0.0 {
        std_deviation / mean_duration
    } else {
        f64::MAX
    }
}

/// Generate tempo test range based on configuration
pub fn generate_tempo_range(min_tempo: i32, max_tempo: i32, steps: i32) -> Vec<i32> {
    if steps <= 1 {
        return vec![100]; // Default tempo if invalid steps
    }
    
    let range = max_tempo - min_tempo;
    let step_size = range as f64 / (steps - 1) as f64;
    
    (0..steps)
        .map(|i| min_tempo + (i as f64 * step_size).round() as i32)
        .collect()
}

/// Format duration in seconds to a human-readable string
pub fn format_duration(seconds: u64) -> String {
    WhisperUtils::format_duration(seconds)
} 