use thiserror::Error;

#[derive(Error, Debug)]
pub enum ShuroError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("JSON serialization error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("TOML parsing error: {0}")]
    Toml(#[from] toml::de::Error),

    #[error("HTTP request error: {0}")]
    Http(#[from] reqwest::Error),

    #[error("Whisper transcription error: {0}")]
    Whisper(String),

    #[error("Translation error: {0}")]
    Translation(String),

    #[error("Media processing error: {0}")]
    Media(String),

    #[error("Quality validation error: {0}")]
    Quality(String),

    #[error("Hallucination detected in transcription")]
    Hallucination,

    #[error("Configuration error: {0}")]
    Config(String),

    #[error("File not found: {0}")]
    FileNotFound(String),

    #[error("Unsupported format: {0}")]
    UnsupportedFormat(String),

    #[error("Cache error: {0}")]
    Cache(String),
}

pub type Result<T> = std::result::Result<T, ShuroError>; 