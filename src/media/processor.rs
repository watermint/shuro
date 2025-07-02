use async_trait::async_trait;
use std::path::Path;
use std::process::Command;
use tracing::{info, debug};

use crate::config::MediaConfig;
use crate::error::{Result, ShuroError};
use super::{MediaProcessorTrait, MediaCommand, MediaCommandBuilder};

/// Concrete implementation of media processor (FFmpeg-based)
pub struct MediaProcessorImpl {
    config: MediaConfig,
    command_builder: MediaCommandBuilder,
}

impl MediaProcessorImpl {
    /// Create a new media processor implementation
    pub fn new(config: MediaConfig) -> Self {
        let command_builder = MediaCommandBuilder::new(&config.binary_path);
        
        Self {
            config,
            command_builder,
        }
    }
}

#[async_trait]
impl MediaProcessorTrait for MediaProcessorImpl {
    /// Embed subtitles into video file
    async fn embed_subtitles(
        &self,
        video_path: &Path,
        subtitle_path: &Path,
        output_path: &Path,
    ) -> Result<()> {
        info!("Embedding subtitles from {} into {} -> {}", 
              subtitle_path.display(), video_path.display(), output_path.display());

        let command = self.command_builder.embed_subtitles(
            video_path,
            subtitle_path,
            output_path,
            &self.config.subtitle_options,
        );

        command.execute().await?;

        info!("Subtitle embedding completed successfully");
        Ok(())
    }

    /// Extract audio from video
    async fn extract_audio(
        &self,
        video_path: &Path,
        audio_path: &Path,
    ) -> Result<()> {
        info!("Extracting audio from {} to {}", video_path.display(), audio_path.display());

        let command = self.command_builder.extract_audio(video_path, audio_path);
        command.execute().await?;

        info!("Audio extraction completed");
        Ok(())
    }

    /// Check if media processor is available
    fn check_availability(&self) -> Result<()> {
        let output = Command::new(&self.config.binary_path)
            .arg("-version")
            .output()
            .map_err(|e| ShuroError::Media(format!("Media processor not found: {}", e)))?;

        if output.status.success() {
            info!("Media processor is available");
            Ok(())
        } else {
            Err(ShuroError::Media("Media processor version check failed".to_string()))
        }
    }

    /// Get media processor version information
    async fn get_version_info(&self) -> Result<String> {
        debug!("Getting media processor version information");

        let output = Command::new(&self.config.binary_path)
            .arg("-version")
            .output()
            .map_err(|e| ShuroError::Media(format!("Failed to execute media processor: {}", e)))?;

        if output.status.success() {
            let version_info = String::from_utf8_lossy(&output.stdout);
            // Extract the first line which typically contains the version
            let first_line = version_info.lines().next().unwrap_or("Unknown version");
            Ok(first_line.to_string())
        } else {
            let stderr = String::from_utf8_lossy(&output.stderr);
            Err(ShuroError::Media(format!("Media processor version check failed: {}", stderr)))
        }
    }

    /// Execute custom media processing command
    async fn execute_command(&self, command: MediaCommand) -> Result<()> {
        info!("Executing custom media processing command: {}", command.description);
        command.execute().await
    }
}

/// Additional utility functions for media operations
impl MediaProcessorImpl {
    /// Create a command for converting video format
    pub fn convert_video_format<P: AsRef<Path>>(
        &self,
        input_path: P,
        output_path: P,
        output_format: &str,
    ) -> MediaCommand {
        self.command_builder
            .custom(format!("Convert to {}", output_format))
            .input(input_path)
            .video_codec("libx264")
            .copy_audio()
            .overwrite()
            .output(output_path)
    }

    /// Create a command for resizing video
    pub fn resize_video<P: AsRef<Path>>(
        &self,
        input_path: P,
        output_path: P,
        width: u32,
        height: u32,
    ) -> MediaCommand {
        self.command_builder
            .custom(format!("Resize to {}x{}", width, height))
            .input(input_path)
            .video_filter(format!("scale={}:{}", width, height))
            .video_codec("libx264")
            .copy_audio()
            .overwrite()
            .output(output_path)
    }

    /// Create a command for extracting video frames
    pub fn extract_frames<P: AsRef<Path>>(
        &self,
        input_path: P,
        output_pattern: P,
        fps: f32,
    ) -> MediaCommand {
        self.command_builder
            .custom("Extract frames")
            .input(input_path)
            .video_filter(format!("fps={}", fps))
            .overwrite()
            .output(output_pattern)
    }

    /// Create a command for creating video from images
    pub fn create_video_from_images<P: AsRef<Path>>(
        &self,
        input_pattern: P,
        output_path: P,
        fps: f32,
    ) -> MediaCommand {
        self.command_builder
            .custom("Create video from images")
            .arg("-framerate").arg(fps.to_string())
            .input(input_pattern)
            .video_codec("libx264")
            .arg("-pix_fmt").arg("yuv420p")
            .overwrite()
            .output(output_path)
    }

    /// Create a command for trimming video
    pub fn trim_video<P: AsRef<Path>>(
        &self,
        input_path: P,
        output_path: P,
        start_time: f32,
        duration: f32,
    ) -> MediaCommand {
        self.command_builder
            .custom(format!("Trim video ({}s from {}s)", duration, start_time))
            .input(input_path)
            .arg("-ss").arg(start_time.to_string())
            .arg("-t").arg(duration.to_string())
            .copy_video()
            .copy_audio()
            .overwrite()
            .output(output_path)
    }

    /// Create a command for concatenating videos
    pub fn concatenate_videos<P: AsRef<Path>>(
        &self,
        input_list_file: P,
        output_path: P,
    ) -> MediaCommand {
        self.command_builder
            .custom("Concatenate videos")
            .arg("-f").arg("concat")
            .arg("-safe").arg("0")
            .input(input_list_file)
            .copy_video()
            .copy_audio()
            .overwrite()
            .output(output_path)
    }
} 