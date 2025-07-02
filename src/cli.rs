use clap::{Parser, Subcommand};
use std::path::PathBuf;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
pub struct Args {
    /// Enable verbose logging
    #[arg(short, long)]
    pub verbose: bool,

    /// Configuration file path
    #[arg(short, long)]
    pub config: Option<PathBuf>,

    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand)]
pub enum Commands {
    /// Process a single video file with subtitle translation
    Process {
        /// Input video file
        #[arg(short, long)]
        input: PathBuf,

        /// Target languages for translation (comma-separated)
        #[arg(short, long, default_value = "ja")]
        target_langs: String,

        /// Output directory for processed files
        #[arg(short, long)]
        output_dir: Option<PathBuf>,

        /// Translation optimization mode
        #[arg(long, default_value = "simple")]
        translation_mode: String,
    },

    /// Process all video files in a directory
    Batch {
        /// Input directory containing video files
        #[arg(short, long)]
        input_dir: PathBuf,

        /// Target languages for translation (comma-separated)
        #[arg(short, long, default_value = "ja")]
        target_langs: String,

        /// Output directory for processed files
        #[arg(short, long)]
        output_dir: Option<PathBuf>,

        /// Translation optimization mode
        #[arg(long, default_value = "simple")]
        translation_mode: String,
    },

    /// List available whisper models and their status
    Models {
        /// Download all missing models
        #[arg(long)]
        download: bool,
    },

    /// Manage transcription cache
    Cache {
        #[command(subcommand)]
        action: CacheAction,
    },

    /// Extract audio from video file
    Extract {
        /// Input video file
        #[arg(short, long)]
        input: PathBuf,

        /// Output audio file
        #[arg(short, long)]
        output: PathBuf,
    },

    /// Transcribe audio to subtitles with hallucination detection
    Transcribe {
        /// Input audio file
        #[arg(short, long)]
        input: PathBuf,

        /// Output transcript file
        #[arg(short, long)]
        output: PathBuf,

        /// Source language hint
        #[arg(short, long)]
        language: Option<String>,
    },

    /// Translate subtitles using LLM
    Translate {
        /// Input transcript file
        #[arg(short, long)]
        input: PathBuf,

        /// Output translated file
        #[arg(short, long)]
        output: PathBuf,

        /// Target languages (comma-separated)
        #[arg(short, long)]
        target_langs: String,
    },

    /// Embed subtitles into video file
    Embed {
        /// Input video file
        #[arg(short, long)]
        video: PathBuf,

        /// Subtitle file
        #[arg(short, long)]
        subtitles: PathBuf,

        /// Output video file
        #[arg(short, long)]
        output: PathBuf,
    },
}

#[derive(Subcommand)]
pub enum CacheAction {
    /// List cached transcriptions and audio files
    List,

    /// Clear all cached transcriptions and audio files
    Clear,

    /// Show cache statistics and size
    Info,

    /// Clear cache entries older than specified days
    Clean {
        /// Age in days (default: 30)
        #[arg(short, long, default_value = "30")]
        days: u64,
    },

    /// List only cached transcriptions
    ListTranscriptions,

    /// List only cached audio files
    ListAudio,

    /// List only cached translations
    ListTranslations,

    /// Clear only cached transcriptions
    ClearTranscriptions,

    /// Clear only cached audio files
    ClearAudio,

    /// Clear only cached translations
    ClearTranslations,
}