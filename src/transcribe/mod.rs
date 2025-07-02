// Modular transcription architecture
//
// This module provides different transcription implementations through a factory pattern:
// - WhisperCpp: whisper.cpp implementation
// - OpenAI: OpenAI Whisper Python implementation
//
// To add a new transcription service:
// 1. Create service-specific data structures for parsing JSON
// 2. Implement TranscriptionMapper trait for your service
// 3. Add the service to TranscriberImplementation enum
// 4. Update the factory to create your implementation
// 
// Example for a hypothetical Azure Speech service:
// ```
// #[derive(Serialize, Deserialize)]
// pub struct AzureSpeechOutput { ... }
// 
// pub struct AzureSpeechMapper;
// impl TranscriptionMapper<AzureSpeechOutput> for AzureSpeechMapper {
//     fn to_abstract_transcription(azure_output: AzureSpeechOutput) -> Result<AbstractTranscription> {
//         // Convert Azure format to AbstractTranscription
//     }
// }
// ```

pub mod common;
pub mod whisper_cpp;
pub mod openai;

use async_trait::async_trait;
use std::path::Path;

pub use common::*;
use crate::config::TranscriberConfig;
use crate::error::Result;
use crate::quality::{Transcription, QualityValidator};

/// Main trait for transcription operations
#[async_trait]
pub trait TranscriberTrait: Send + Sync {
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

/// Transcriber implementation type
#[derive(Debug, Clone)]
pub enum TranscriberImplementation {
    WhisperCpp,
    OpenAI,
    // Future implementations can be added here:
    // AssemblyAI,
    // Rev,
    // Azure,
    // Google,
}

/// Factory for creating transcriber instances
pub struct TranscriberFactory;

impl TranscriberFactory {
    /// Create a transcriber based on implementation type
    pub fn create_transcriber(
        implementation: TranscriberImplementation,
        config: TranscriberConfig,
        validator: QualityValidator,
    ) -> Box<dyn TranscriberTrait> {
        match implementation {
            TranscriberImplementation::WhisperCpp => {
                Box::new(whisper_cpp::WhisperCppTranscriber::new(config, validator))
            }
            TranscriberImplementation::OpenAI => {
                Box::new(openai::OpenAITranscriber::new(config, validator))
            }
            // Future implementations:
            // TranscriberImplementation::AssemblyAI => {
            //     Box::new(assembly_ai::AssemblyAITranscriber::new(config, validator))
            // }
        }
    }
    
    /// Create with default implementation (WhisperCpp as fallback)
    pub fn create_default(config: TranscriberConfig, validator: QualityValidator) -> Box<dyn TranscriberTrait> {
        // Try WhisperCpp as default implementation
        Self::create_transcriber(TranscriberImplementation::WhisperCpp, config, validator)
    }
} 