use async_trait::async_trait;
use tracing::{info, warn};

use crate::config::TranslateConfig;
use crate::error::Result;
use crate::quality::{Transcription, TranscriptionSegment};
use super::{Translator, common::{BaseTranslator, TranslationQuality}};

/// Simple translation: Translate each segment individually without context
pub struct SimpleTranslator {
    base: BaseTranslator,
}

impl SimpleTranslator {
    pub fn new(config: TranslateConfig) -> Self {
        Self {
            base: BaseTranslator::new(config),
        }
    }

    /// Simple segment translation without context or quality validation
    async fn translate_segment_simple(
        &mut self,
        segment: &TranscriptionSegment,
        target_language: &str,
    ) -> Result<String> {
        let cache_key = self.base.generate_cache_key(&segment.text, target_language, "");
        
        // Check persistent cache first
        if let Ok(Some(cached_translation)) = self.base.load_from_persistent_cache(&cache_key).await {
            self.base.cache.insert(cache_key.clone(), cached_translation.clone());
            return Ok(cached_translation);
        }
        
        // Check in-memory cache as fallback
        if let Some(cached) = self.base.cache.get(&cache_key) {
            return Ok(cached.clone());
        }

        // Translate without context
        match self.base.translate_text(&segment.text, target_language, None).await {
            Ok(translation) => {
                // Cache successful translation
                self.base.cache.insert(cache_key.clone(), translation.clone());
                
                // Save to persistent cache
                if let Err(e) = self.base.save_to_persistent_cache(
                    &cache_key,
                    &segment.text,
                    target_language,
                    "",
                    &translation,
                    &TranslationQuality::Good,
                ).await {
                    warn!("Failed to save translation to persistent cache: {}", e);
                }
                
                Ok(translation)
            }
            Err(e) => Err(e),
        }
    }
}

#[async_trait]
impl Translator for SimpleTranslator {
    /// Simple translation: Translate each segment individually without context
    async fn translate_transcription(
        &mut self,
        transcription: &mut Transcription,
        target_language: &str,
        _context: Option<&str>,
    ) -> Result<()> {
        info!("Starting simple translation to {}", target_language);
        
        let total_segments = transcription.segments.len();
        
        for (idx, segment) in transcription.segments.iter_mut().enumerate() {
            let original_text = segment.text.clone();
            
            info!("┌─ Translating segment {}/{} (Simple) ────────", idx + 1, total_segments);
            info!("│ Source: {}", original_text);

            match self.translate_segment_simple(segment, target_language).await {
                Ok(translation) => {
                    info!("│ Target: {}", translation);
                    info!("└─────────────────────────────────────");
                    segment.text = translation;
                }
                Err(e) => {
                    warn!("│ Failed: {}", e);
                    warn!("└─────────────────────────────────────");
                    // Keep original text on failure
                }
            }
        }
        
        Ok(())
    }
} 