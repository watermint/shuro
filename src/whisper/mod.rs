// Modular whisper architecture
//
// This module provides different whisper implementations through a factory pattern:
// - WhisperCpp: Current whisper.cpp implementation (placeholder)
// - OpenAI: OpenAI Whisper Python implementation

pub mod common;
pub mod whisper_cpp;
pub mod openai;

use async_trait::async_trait;
use std::path::Path;

pub use common::*;
use crate::config::WhisperConfig;
use crate::error::Result;
use crate::quality::{Transcription, QualityValidator};

/// Main trait for whisper transcription operations
#[async_trait]
pub trait WhisperTranscriberTrait: Send + Sync {
    /// Transcribe audio file to text
    async fn transcribe(&self, audio_path: &Path, language: Option<&str>) -> Result<Transcription>;
    
    /// Tune transcription parameters
    async fn tune_transcription(&self, audio_path: &Path) -> Result<TuneResult>;
    
    /// Extract and cache audio from video
    async fn extract_and_cache_audio(&self, video_path: &Path) -> Result<std::path::PathBuf>;
    
    /// Get cached audio file path
    async fn get_cached_audio(&self, video_path: &Path) -> Result<Option<std::path::PathBuf>>;
    
    /// Clear transcription cache
    async fn clear_cache(&self) -> Result<u64>;
    
    /// List cached transcriptions
    async fn list_cache(&self) -> Result<Vec<TranscriptionCache>>;
    
    /// Get cache information
    async fn cache_info(&self) -> Result<CacheInfo>;
    
    /// Clear audio cache
    async fn clear_audio_cache(&self) -> Result<u64>;
    
    /// List audio cache
    async fn list_audio_cache(&self) -> Result<Vec<AudioCache>>;
}

/// Whisper implementation type
#[derive(Debug, Clone)]
pub enum WhisperImplementation {
    WhisperCpp,
    OpenAI,
}

/// Factory for creating whisper transcriber instances
pub struct WhisperTranscriberFactory;

impl WhisperTranscriberFactory {
    /// Create a whisper transcriber based on implementation type
    pub fn create_transcriber(
        implementation: WhisperImplementation,
        config: WhisperConfig,
        validator: QualityValidator,
    ) -> Box<dyn WhisperTranscriberTrait> {
        match implementation {
            WhisperImplementation::WhisperCpp => {
                Box::new(whisper_cpp::WhisperCppTranscriber::new(config, validator))
            }
            WhisperImplementation::OpenAI => {
                Box::new(openai::OpenAITranscriber::new(config, validator))
            }
        }
    }
    
    /// Create with default implementation (WhisperCpp as fallback)
    pub fn create_default(config: WhisperConfig, validator: QualityValidator) -> Box<dyn WhisperTranscriberTrait> {
        // Try OpenAI first, but fallback to WhisperCpp if there are issues
        Self::create_transcriber(WhisperImplementation::WhisperCpp, config, validator)
    }
} 