use serde::{Deserialize, Serialize};
use std::path::Path;
use crate::error::{Result, ShuroError};

// Default values for LLM mode configuration
fn default_llm_window_size() -> usize {
    6
}

fn default_llm_confidence_threshold() -> f64 {
    0.6
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub transcriber: TranscriberConfig,
    pub translate: TranslateConfig,
    pub quality: QualityConfig,
    pub media: MediaConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscriberConfig {
    /// Path to transcriber binary (e.g., whisper-cli)
    pub binary_path: String,
    /// Transcription mode: Simple or Tuned
    pub mode: TranscriptionMode,
    /// Model to use for exploration phase
    pub explore_model: String,
    /// Model to use for transcription
    pub transcribe_model: String,
    /// Acceptable source languages
    pub acceptable_languages: String,
    /// Fallback language when detection fails
    pub fallback_language: String,
    /// Number of steps for tempo exploration
    pub explore_steps: i32,
    /// Maximum tempo range for exploration
    pub explore_range_max: i32,
    /// Minimum tempo range for exploration
    pub explore_range_min: i32,
    /// Temperature for transcription
    pub temperature: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TranscriptionMode {
    /// Simple: Use default settings without tempo optimization
    Simple,
    /// Tuned: Find optimal tempo using smaller model first, then transcribe with best tempo
    Tuned,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranslateConfig {
    /// Ollama endpoint URL
    pub endpoint: String,
    /// LLM model to use for translation
    pub model: String,
    /// Maximum retries for failed translations
    pub max_retries: u32,
    /// Translation optimization mode
    pub mode: TranslationMode,
    /// Minimum gap between segments (seconds) to trigger hard stop in NLP mode
    pub nlp_gap_threshold: f64,
    /// Maximum context window size for context mode
    pub context_window_size: usize,
    /// Window size for LLM mode (number of segments to analyze at once)
    #[serde(default = "default_llm_window_size")]
    pub llm_window_size: usize,
    /// Minimum confidence threshold for sentence boundaries in LLM mode
    #[serde(default = "default_llm_confidence_threshold")]
    pub llm_confidence_threshold: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TranslationMode {
    /// Simple: Translate each segment individually without context
    Simple,
    /// Context: Use surrounding segments as context but only translate the target segment
    Context,
    /// NLP: Reconstruct complete sentences using natural language processing
    Nlp,
    /// LLM: Use sliding window approach with LLM to split segments by contextual sentences
    Llm,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityConfig {
    /// Repetitive segment threshold
    pub repetitive_segment_threshold: f64,
    /// Maximum tokens per segment threshold
    pub max_tokens_threshold: f64,
    /// Minimum quality score required
    pub min_quality_score: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MediaConfig {
    /// Path to ffmpeg binary
    pub binary_path: String,
    /// Additional encoding options for subtitle embedding
    /// Common options: ["-preset", "medium", "-crf", "23", "-pix_fmt", "yuv420p"]
    /// - preset: encoding speed (ultrafast, fast, medium, slow, veryslow)
    /// - crf: quality (0-51, lower = better quality, 23 is default)
    /// - pix_fmt: pixel format for compatibility
    pub subtitle_options: Vec<String>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            transcriber: TranscriberConfig {
                binary_path: "whisper-cli".to_string(),
                mode: TranscriptionMode::Tuned,
                explore_model: "base".to_string(),
                transcribe_model: "medium".to_string(),
                acceptable_languages: "en,ja,ko,zh,fr,de,es,ru,it,pt,pl,nl,tr,ar,hi,th,vi,sv,da,no,fi,he,hu,cs,sk,bg,hr,sl,et,lv,lt,mt,ga,cy,eu,ca,gl,is,mk,sq,be,uk,az,kk,ky,uz,tg,am,ka,hy,ne,si,my,km,lo,ka,gu,pa,ta,te,kn,ml,bn,as,or,mr".to_string(),
                fallback_language: "en".to_string(),
                explore_steps: 10,
                explore_range_max: 110,
                explore_range_min: 80,
                temperature: 0.0,
            },
            translate: TranslateConfig {
                endpoint: "http://localhost:11434".to_string(),
                model: "llama3.2:3b".to_string(),
                max_retries: 3,
                mode: TranslationMode::Simple,
                nlp_gap_threshold: 2.0,
                context_window_size: 2,
                llm_window_size: 15,
                llm_confidence_threshold: 0.6,
            },
            quality: QualityConfig {
                repetitive_segment_threshold: 0.8,
                max_tokens_threshold: 50.0,
                min_quality_score: 0.7,
            },
            media: MediaConfig {
                binary_path: "ffmpeg".to_string(),
                subtitle_options: vec![
                    // Example encoding options users can customize:
                    // "-preset".to_string(), "medium".to_string(),  // Encoding speed (ultrafast, fast, medium, slow, veryslow)
                    // "-crf".to_string(), "23".to_string(),         // Quality (0-51, lower = better quality)
                    // "-pix_fmt".to_string(), "yuv420p".to_string(), // Pixel format for compatibility
                ],
            },
        }
    }
}

impl Config {
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| ShuroError::Config(format!("Failed to read config file: {}", e)))?;
        
        toml::from_str(&content)
            .map_err(|e| ShuroError::Config(format!("Failed to parse config file: {}", e)))
    }

    pub fn save_to_file<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let content = toml::to_string_pretty(self)
            .map_err(|e| ShuroError::Config(format!("Failed to serialize config: {}", e)))?;
        
        std::fs::write(path, content)
            .map_err(|e| ShuroError::Config(format!("Failed to write config file: {}", e)))?;
        
        Ok(())
    }
} 