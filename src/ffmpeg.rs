use std::path::Path;
use std::process::Command;
use tracing::{info, debug};

use crate::config::FFmpegConfig;
use crate::error::{Result, ShuroError};

pub struct FFmpegProcessor {
    config: FFmpegConfig,
}

impl FFmpegProcessor {
    pub fn new(config: FFmpegConfig) -> Self {
        Self { config }
    }

    /// Embed subtitles into video file
    pub async fn embed_subtitles<P: AsRef<Path>>(
        &self,
        video_path: P,
        subtitle_path: P,
        output_path: P,
    ) -> Result<()> {
        let video_path = video_path.as_ref();
        let subtitle_path = subtitle_path.as_ref();
        let output_path = output_path.as_ref();

        info!("Embedding subtitles from {} into {} -> {}", 
              subtitle_path.display(), video_path.display(), output_path.display());

        let mut cmd = Command::new(&self.config.binary_path);
        cmd.arg("-y") // Overwrite output
           .arg("-i").arg(video_path)
           .arg("-vf").arg(format!("subtitles={}", subtitle_path.display()))
           .arg("-c:v").arg("libx264") // Use H.264 encoder (required when using filters)
           .arg("-c:a").arg("copy");   // Keep audio unchanged

        // Add user-specified additional options
        for option in &self.config.subtitle_options {
            cmd.arg(option);
        }

        cmd.arg(output_path);

        debug!("Executing ffmpeg command: {:?}", cmd);

        let output = cmd.output()
            .map_err(|e| ShuroError::FFmpeg(format!("Failed to execute ffmpeg: {}", e)))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(ShuroError::FFmpeg(format!(
                "Subtitle embedding failed: {}",
                stderr
            )));
        }

        info!("Subtitle embedding completed successfully");
        Ok(())
    }

    /// Extract audio from video
    pub async fn extract_audio<P: AsRef<Path>>(
        &self,
        video_path: P,
        audio_path: P,
    ) -> Result<()> {
        let video_path = video_path.as_ref();
        let audio_path = audio_path.as_ref();

        info!("Extracting audio from {} to {}", video_path.display(), audio_path.display());

        let output = Command::new(&self.config.binary_path)
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

    /// Check if ffmpeg is available
    pub fn check_availability(&self) -> Result<()> {
        let output = Command::new(&self.config.binary_path)
            .arg("-version")
            .output()
            .map_err(|e| ShuroError::FFmpeg(format!("FFmpeg not found: {}", e)))?;

        if output.status.success() {
            info!("FFmpeg is available");
            Ok(())
        } else {
            Err(ShuroError::FFmpeg("FFmpeg version check failed".to_string()))
        }
    }
} 