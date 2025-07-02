// Modular translation architecture
//
// This module provides different translation implementations through a factory pattern:
// - Simple: Direct segment translation  
// - Context: Context-aware translation with quality validation
// - NLP: NLP-based sentence reconstruction and translation

pub mod common;
pub mod simple;
pub mod context;
pub mod nlp;

use async_trait::async_trait;

pub use common::*;
use crate::config::{TranslateConfig, TranslationMode};
use crate::error::Result;
use crate::quality::Transcription;

/// Main trait for translation operations
#[async_trait]
pub trait Translator: Send + Sync {
    /// Translate transcription segments to target language
    async fn translate_transcription(
        &mut self,
        transcription: &mut Transcription,
        target_language: &str,
        context: Option<&str>,
    ) -> Result<()>;
}

/// Factory for creating translator instances
pub struct TranslatorFactory;

impl TranslatorFactory {
    /// Create a translator based on the translation mode
    pub fn create_translator(config: TranslateConfig) -> Box<dyn Translator> {
        match config.mode {
            TranslationMode::Simple => {
                Box::new(simple::SimpleTranslator::new(config))
            }
            TranslationMode::Context => {
                Box::new(context::ContextTranslator::new(config))
            }
            TranslationMode::Nlp => {
                Box::new(nlp::NlpTranslator::new(config))
            }
        }
    }
}

/// Check if Ollama is available and the model is loaded
pub async fn check_ollama_availability(endpoint: &str, model: &str) -> Result<()> {
    common::check_ollama_availability(endpoint, model).await
} 