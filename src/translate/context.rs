use async_trait::async_trait;
use std::time::Duration;
use tracing::{info, warn};

use crate::config::TranslateConfig;
use crate::error::{Result, ShuroError};
use crate::quality::{Transcription, TranscriptionSegment};
use super::{Translator, common::{BaseTranslator, TranslationQuality}};

/// Context-aware translation: Use surrounding segments as context but only translate the target segment
pub struct ContextTranslator {
    base: BaseTranslator,
}

impl ContextTranslator {
    pub fn new(config: TranslateConfig) -> Self {
        Self {
            base: BaseTranslator::new(config),
        }
    }

    fn build_segment_context(&self, segments: &[TranscriptionSegment], current_idx: usize, context_size: usize) -> String {
        let mut context_parts = Vec::new();
        
        let start_idx = current_idx.saturating_sub(context_size);
        for i in start_idx..current_idx {
            if i < segments.len() {
                context_parts.push(segments[i].text.clone());
            }
        }
        
        let end_idx = (current_idx + 1 + context_size).min(segments.len());
        for i in (current_idx + 1)..end_idx {
            context_parts.push(segments[i].text.clone());
        }
        
        context_parts.join(" ")
    }

    /// Translate a single segment with quality validation and retries
    async fn translate_segment_with_quality(
        &mut self,
        segment: &TranscriptionSegment,
        target_language: &str,
        context: &str,
    ) -> Result<String> {
        let cache_key = self.base.generate_cache_key(&segment.text, target_language, context);
        
        // Check persistent cache first
        if let Ok(Some(cached_translation)) = self.base.load_from_persistent_cache(&cache_key).await {
            // Also store in memory cache for faster access during this session
            self.base.cache.insert(cache_key.clone(), cached_translation.clone());
            return Ok(cached_translation);
        }
        
        // Check in-memory cache as fallback
        if let Some(cached) = self.base.cache.get(&cache_key) {
            return Ok(cached.clone());
        }

        let mut current_context = context.to_string();
        let mut attempts = 0;
        
        while attempts < self.base.config.max_retries {
            attempts += 1;

            match self.base.translate_text(&segment.text, target_language, if current_context.is_empty() { None } else { Some(&current_context) }).await {
                Ok(translation) => {
                    // Length validation - if translation is too long, remove context
                    if segment.text.len() * 5 < translation.len() {
                        info!("│ Translation too long, removing context (attempt {})", attempts);
                        current_context = String::new();
                        continue;
                    }
                    
                    // Validate translation quality
                    let quality = self.base.evaluate_translation_quality(&segment.text, &translation, &current_context, target_language).await;
                    
                    match quality {
                        Ok(TranslationQuality::Perfect | TranslationQuality::Good) => {
                            info!("│ Quality: {} ✓", quality.as_ref().unwrap().to_str());
                            
                            // Save to persistent cache (only good quality translations)
                            if let Err(e) = self.base.save_to_persistent_cache(
                                &cache_key,
                                &segment.text,
                                target_language,
                                &current_context,
                                &translation,
                                quality.as_ref().unwrap(),
                            ).await {
                                warn!("Failed to save translation to persistent cache: {}", e);
                            }
                            
                            // Cache in memory for this session
                            self.base.cache.insert(cache_key, translation.clone());
                            return Ok(translation);
                        }
                        Ok(qual) => {
                            warn!("│ Quality: {} - retrying (attempt {})", qual.to_str(), attempts);
                        }
                        Err(e) => {
                            warn!("│ Quality evaluation failed: {} (attempt {})", e, attempts);
                        }
                    }
                }
                Err(e) => {
                    warn!("│ Attempt {} failed: {}", attempts, e);
                }
            }

            // Brief delay before retry
            tokio::time::sleep(Duration::from_millis(500)).await;
        }

        Err(ShuroError::Translation(format!(
            "Failed to translate after {} attempts", 
            self.base.config.max_retries
        )))
    }
}

#[async_trait]
impl Translator for ContextTranslator {
    /// Context-aware translation: Use surrounding segments as context but only translate the target segment
    async fn translate_transcription(
        &mut self,
        transcription: &mut Transcription,
        target_language: &str,
        _context: Option<&str>,
    ) -> Result<()> {
        info!("Starting context-aware translation to {}", target_language);
        
        let total_segments = transcription.segments.len();
        
        // Pre-build all contexts to avoid borrowing issues
        let contexts: Vec<String> = (0..total_segments)
            .map(|idx| self.build_segment_context(&transcription.segments, idx, self.base.config.context_window_size))
            .collect();
        
        for (idx, (segment, context)) in transcription.segments.iter_mut().zip(contexts.iter()).enumerate() {
            let original_text = segment.text.clone();
            
            info!("┌─ Translating segment {}/{} (Context) ────────", idx + 1, total_segments);
            info!("│ Source: {}", original_text);
            if !context.is_empty() {
                info!("│ Context: {}...", &context[..context.len().min(100)]);
            }

            match self.translate_segment_with_quality(segment, target_language, context).await {
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