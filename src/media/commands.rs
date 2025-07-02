use std::path::Path;
use std::process::Command;
use tracing::debug;

use crate::error::{Result, ShuroError};

/// Abstract media processing command representation
#[derive(Debug, Clone)]
pub struct MediaCommand {
    pub binary_path: String,
    pub args: Vec<String>,
    pub description: String,
}

impl MediaCommand {
    /// Create a new media processing command
    pub fn new<S1: Into<String>, S2: Into<String>>(binary_path: S1, description: S2) -> Self {
        Self {
            binary_path: binary_path.into(),
            args: Vec::new(),
            description: description.into(),
        }
    }

    /// Add an argument
    pub fn arg<S: Into<String>>(mut self, arg: S) -> Self {
        self.args.push(arg.into());
        self
    }

    /// Add multiple arguments
    pub fn args<I, S>(mut self, args: I) -> Self 
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.args.extend(args.into_iter().map(|s| s.into()));
        self
    }

    /// Add input file
    pub fn input<P: AsRef<Path>>(self, path: P) -> Self {
        self.arg("-i").arg(path.as_ref().to_string_lossy().to_string())
    }

    /// Add output file
    pub fn output<P: AsRef<Path>>(self, path: P) -> Self {
        self.arg(path.as_ref().to_string_lossy().to_string())
    }

    /// Force overwrite output
    pub fn overwrite(self) -> Self {
        self.arg("-y")
    }

    /// Set video codec
    pub fn video_codec<S: Into<String>>(self, codec: S) -> Self {
        self.arg("-c:v").arg(codec)
    }

    /// Set audio codec
    pub fn audio_codec<S: Into<String>>(self, codec: S) -> Self {
        self.arg("-c:a").arg(codec)
    }

    /// Copy video stream
    pub fn copy_video(self) -> Self {
        self.video_codec("copy")
    }

    /// Copy audio stream
    pub fn copy_audio(self) -> Self {
        self.audio_codec("copy")
    }

    /// Disable video
    pub fn no_video(self) -> Self {
        self.arg("-vn")
    }

    /// Disable audio
    pub fn no_audio(self) -> Self {
        self.arg("-an")
    }

    /// Set audio sample rate
    pub fn audio_sample_rate(self, rate: u32) -> Self {
        self.arg("-ar").arg(rate.to_string())
    }

    /// Set audio channels
    pub fn audio_channels(self, channels: u32) -> Self {
        self.arg("-ac").arg(channels.to_string())
    }

    /// Add video filter
    pub fn video_filter<S: Into<String>>(self, filter: S) -> Self {
        self.arg("-vf").arg(filter)
    }

    /// Add audio filter
    pub fn audio_filter<S: Into<String>>(self, filter: S) -> Self {
        self.arg("-af").arg(filter)
    }

    /// Execute the command
    pub async fn execute(&self) -> Result<()> {
        debug!("Executing media processing command: {} {:?}", self.binary_path, self.args);
        debug!("Description: {}", self.description);

        let mut cmd = Command::new(&self.binary_path);
        cmd.args(&self.args);

        let output = cmd.output()
            .map_err(|e| ShuroError::Media(format!("Failed to execute media processor: {}", e)))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(ShuroError::Media(format!(
                "{} failed: {}",
                self.description,
                stderr
            )));
        }

        Ok(())
    }
}

/// Builder for common media processing operations
pub struct MediaCommandBuilder {
    binary_path: String,
}

impl MediaCommandBuilder {
    /// Create a new command builder
    pub fn new<S: Into<String>>(binary_path: S) -> Self {
        Self {
            binary_path: binary_path.into(),
        }
    }

    /// Build subtitle embedding command
    pub fn embed_subtitles<P: AsRef<Path>>(
        &self,
        video_path: P,
        subtitle_path: P,
        output_path: P,
        additional_options: &[String],
    ) -> MediaCommand {
        let mut cmd = MediaCommand::new(&self.binary_path, "Subtitle embedding")
            .overwrite()
            .input(&video_path)
            .video_filter(format!("subtitles={}", subtitle_path.as_ref().display()))
            .video_codec("libx264")
            .copy_audio();

        // Add user-specified additional options
        for option in additional_options {
            cmd = cmd.arg(option);
        }

        cmd.output(output_path)
    }

    /// Build audio extraction command
    pub fn extract_audio<P: AsRef<Path>>(
        &self,
        video_path: P,
        audio_path: P,
    ) -> MediaCommand {
        MediaCommand::new(&self.binary_path, "Audio extraction")
            .input(video_path)
            .no_video()
            .audio_codec("pcm_s16le")
            .audio_sample_rate(16000)
            .audio_channels(1)
            .overwrite()
            .output(audio_path)
    }

    /// Build version check command
    pub fn version_check(&self) -> MediaCommand {
        MediaCommand::new(&self.binary_path, "Version check")
            .arg("-version")
    }

    /// Build custom command
    pub fn custom<S: Into<String>>(&self, description: S) -> MediaCommand {
        MediaCommand::new(&self.binary_path, description.into())
    }
}

/// Preset commands for common operations
pub struct MediaPresets;

impl MediaPresets {
    /// Create a high-quality video encoding preset
    pub fn high_quality_video(_builder: &MediaCommandBuilder) -> impl Fn(MediaCommand) -> MediaCommand + '_ {
        move |cmd| {
            cmd.video_codec("libx264")
                .arg("-crf").arg("18")
                .arg("-preset").arg("slow")
        }
    }

    /// Create a fast encoding preset
    pub fn fast_encoding(_builder: &MediaCommandBuilder) -> impl Fn(MediaCommand) -> MediaCommand + '_ {
        move |cmd| {
            cmd.video_codec("libx264")
                .arg("-preset").arg("ultrafast")
        }
    }

    /// Create a web-optimized preset
    pub fn web_optimized(_builder: &MediaCommandBuilder) -> impl Fn(MediaCommand) -> MediaCommand + '_ {
        move |cmd| {
            cmd.video_codec("libx264")
                .arg("-movflags").arg("+faststart")
                .arg("-pix_fmt").arg("yuv420p")
        }
    }
} 