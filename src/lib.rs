//! Shuro - Automated Subtitle Translation Workflow
//! 
//! A Rust implementation of an automated workflow for adding translated subtitles
//! to movie files using whisper-cpp, ollama, and ffmpeg.

pub mod cli;
pub mod config;
pub mod workflow;
pub mod transcribe;
pub mod translate;
pub mod subtitle;
pub mod media;
pub mod error;
pub mod quality;
pub mod setup; 