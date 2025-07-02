// Modular media processing architecture
//
// This module provides a clean abstraction over media processing operations:
// - Processor: Main implementation with abstract command building
// - Commands: Command builders and abstractions

pub mod commands;
pub mod processor;

use async_trait::async_trait;
use std::path::Path;

pub use commands::*;
pub use processor::*;

use crate::config::MediaConfig;
use crate::error::Result;

/// Main trait for media processing operations
#[async_trait]
pub trait MediaProcessorTrait: Send + Sync {
    /// Embed subtitles into video file
    async fn embed_subtitles(
        &self,
        video_path: &Path,
        subtitle_path: &Path,
        output_path: &Path,
    ) -> Result<()>;

    /// Extract audio from video
    async fn extract_audio(
        &self,
        video_path: &Path,
        audio_path: &Path,
    ) -> Result<()>;

    /// Check if media processor is available
    fn check_availability(&self) -> Result<()>;

    /// Get media processor version information
    async fn get_version_info(&self) -> Result<String>;

    /// Execute custom media processing command
    async fn execute_command(&self, command: MediaCommand) -> Result<()>;
}

/// Factory for creating media processor instances
pub struct MediaProcessorFactory;

impl MediaProcessorFactory {
    /// Create the default media processor implementation (FFmpeg-based)
    pub fn create_processor(config: MediaConfig) -> Box<dyn MediaProcessorTrait> {
        Box::new(processor::MediaProcessorImpl::new(config))
    }
} 