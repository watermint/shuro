use async_trait::async_trait;
use tracing::{info, warn};

use crate::config::TranslateConfig;
use crate::error::Result;
use crate::quality::{Transcription, TranscriptionSegment};
use super::{Translator, common::{BaseTranslator, TranslationQuality}};

/// Sentence structure for NLP mode
#[derive(Debug, Clone)]
struct Sentence {
    text: String,
    _start_time: f64,
    _end_time: f64,
    segment_indices: Vec<usize>,
}

/// NLP translation: Reconstruct complete sentences using natural language processing
pub struct NlpTranslator {
    base: BaseTranslator,
}

impl NlpTranslator {
    pub fn new(config: TranslateConfig) -> Self {
        Self {
            base: BaseTranslator::new(config),
        }
    }

    /// Build complete sentences from segments using NLP approach
    fn build_sentences_from_segments(&self, segments: &[TranscriptionSegment]) -> Vec<Sentence> {
        let mut sentences = Vec::new();
        let mut current_sentence = String::new();
        let mut current_start = 0.0;
        let mut current_segment_indices = Vec::new();
        
        for (idx, segment) in segments.iter().enumerate() {
            let text = segment.text.trim();
            if text.is_empty() {
                continue;
            }
            
            // Check for hard stop due to timing gap
            if !current_sentence.is_empty() && idx > 0 {
                let gap = segment.start - segments[idx - 1].end;
                if gap > self.base.config.nlp_gap_threshold {
                    // Save current sentence and start new one
                    if !current_sentence.trim().is_empty() {
                        self.split_and_add_sentences(
                            &current_sentence.trim().to_string(),
                            current_start,
                            segments[*current_segment_indices.last().unwrap_or(&(idx - 1))].end,
                            &current_segment_indices,
                            &mut sentences,
                        );
                    }
                    
                    current_sentence = String::new();
                    current_segment_indices.clear();
                }
            }
            
            // Start new sentence if needed
            if current_sentence.is_empty() {
                current_start = segment.start;
            }
            
            // Add segment to current sentence
            if !current_sentence.is_empty() {
                current_sentence.push(' ');
            }
            current_sentence.push_str(text);
            current_segment_indices.push(idx);
            
            // Check if we should split - be more conservative
            if current_sentence.len() > 800 || 
               (current_sentence.len() > 200 && self.has_strong_sentence_ending(&current_sentence)) {
                self.split_and_add_sentences(
                    &current_sentence.trim().to_string(),
                    current_start,
                    segment.end,
                    &current_segment_indices,
                    &mut sentences,
                );
                
                current_sentence = String::new();
                current_segment_indices.clear();
            }
        }
        
        // Add remaining sentence if any
        if !current_sentence.trim().is_empty() {
            self.split_and_add_sentences(
                &current_sentence.trim().to_string(),
                current_start,
                segments[*current_segment_indices.last().unwrap_or(&(segments.len() - 1))].end,
                &current_segment_indices,
                &mut sentences,
            );
        }
        
        sentences
    }

    /// Check for strong sentence endings that indicate a good split point
    fn has_strong_sentence_ending(&self, text: &str) -> bool {
        // Look for sentence endings followed by space and capital letter (stronger indicators)
        let text_bytes = text.as_bytes();
        for i in 0..text_bytes.len().saturating_sub(2) {
            if matches!(text_bytes[i], b'.' | b'!' | b'?') {
                // Check if followed by space and capital letter
                if text_bytes[i + 1] == b' ' && text_bytes.get(i + 2).map_or(false, |&b| b.is_ascii_uppercase()) {
                    return true;
                }
            }
        }
        
        // Also check for common sentence ending patterns
        text.contains(". ") && (text.contains(". The ") || text.contains(". I ") || 
                               text.contains(". We ") || text.contains(". This ") ||
                               text.contains(". That ") || text.contains(". So "))
    }

    /// Split a reconstructed sentence into individual sentences and add them to the list
    fn split_and_add_sentences(
        &self,
        text: &str,
        start_time: f64,
        end_time: f64,
        segment_indices: &[usize],
        sentences: &mut Vec<Sentence>,
    ) {
        // Split by sentence punctuation while preserving the punctuation
        let individual_sentences = self.split_by_sentences(text);
        
        if individual_sentences.len() <= 1 {
            // Single sentence or no split possible
            sentences.push(Sentence {
                text: text.to_string(),
                _start_time: start_time,
                _end_time: end_time,
                segment_indices: segment_indices.to_vec(),
            });
        } else {
            // Multiple sentences - distribute timing proportionally
            let total_duration = end_time - start_time;
            let total_chars: f64 = individual_sentences.iter().map(|s| s.len()).sum::<usize>() as f64;
            
            let mut current_start = start_time;
            let segments_per_sentence = segment_indices.len() / individual_sentences.len().max(1);
            
            for (idx, sentence_text) in individual_sentences.iter().enumerate() {
                if sentence_text.trim().is_empty() {
                    continue;
                }
                
                // Calculate proportional timing based on sentence length
                let sentence_duration = if total_chars > 0.0 {
                    (sentence_text.len() as f64 / total_chars) * total_duration
                } else {
                    total_duration / individual_sentences.len() as f64
                };
                
                let sentence_end = if idx == individual_sentences.len() - 1 {
                    end_time // Last sentence gets the exact end time
                } else {
                    current_start + sentence_duration
                };
                
                // Distribute segment indices proportionally
                let start_seg_idx = idx * segments_per_sentence;
                let end_seg_idx = ((idx + 1) * segments_per_sentence).min(segment_indices.len());
                let sentence_segments = if start_seg_idx < segment_indices.len() {
                    segment_indices[start_seg_idx..end_seg_idx].to_vec()
                } else {
                    vec![*segment_indices.last().unwrap_or(&0)]
                };
                
                sentences.push(Sentence {
                    text: sentence_text.trim().to_string(),
                    _start_time: current_start,
                    _end_time: sentence_end,
                    segment_indices: sentence_segments,
                });
                
                current_start = sentence_end;
            }
        }
    }

    /// Split text into individual sentences while preserving punctuation
    fn split_by_sentences(&self, text: &str) -> Vec<String> {
        let mut sentences = Vec::new();
        let mut current_sentence = String::new();
        let mut chars = text.chars().peekable();
        
        while let Some(ch) = chars.next() {
            current_sentence.push(ch);
            
            // Check for sentence endings
            if ch == '.' || ch == '!' || ch == '?' {
                // Look ahead to see if this is a real sentence ending
                if let Some(&next_ch) = chars.peek() {
                    if next_ch.is_whitespace() {
                        // Check if the next non-whitespace character is uppercase or if sentence is long enough
                        let sentence = current_sentence.trim().to_string();
                        if sentence.len() > 30 { // Minimum 30 characters for a meaningful sentence
                            // Look further ahead for uppercase
                            let remaining: String = chars.clone().collect();
                            let trimmed_remaining = remaining.trim_start();
                            if trimmed_remaining.is_empty() || 
                               trimmed_remaining.chars().next().map_or(false, |c| c.is_uppercase()) {
                                sentences.push(sentence);
                                current_sentence = String::new();
                            }
                        }
                    }
                } else {
                    // End of text, definitely a sentence ending if long enough
                    let sentence = current_sentence.trim().to_string();
                    if sentence.len() > 15 { // Shorter minimum for end of text
                        sentences.push(sentence);
                        current_sentence = String::new();
                    }
                }
            }
        }
        
        // Add remaining text if any
        if !current_sentence.trim().is_empty() {
            sentences.push(current_sentence.trim().to_string());
        }
        
        // If no sentences were split (no proper punctuation), split by length with larger chunks
        if sentences.is_empty() && !text.trim().is_empty() {
            sentences = self.split_by_length(text, 500); // Larger chunks - 500 chars per sentence
        }
        
        sentences
    }

    /// Split text by length when no sentence punctuation is available
    fn split_by_length(&self, text: &str, max_length: usize) -> Vec<String> {
        let mut sentences = Vec::new();
        let words: Vec<&str> = text.split_whitespace().collect();
        let mut current_sentence = String::new();
        
        for word in words {
            if current_sentence.len() + word.len() + 1 > max_length && 
               current_sentence.len() > 100 { // Ensure minimum length before splitting
                sentences.push(current_sentence.trim().to_string());
                current_sentence = String::new();
            }
            
            if !current_sentence.is_empty() {
                current_sentence.push(' ');
            }
            current_sentence.push_str(word);
        }
        
        if !current_sentence.trim().is_empty() {
            sentences.push(current_sentence.trim().to_string());
        }
        
        sentences
    }

    /// Translate a complete sentence
    async fn translate_sentence(&mut self, sentence: &Sentence, target_language: &str) -> Result<String> {
        let cache_key = self.base.generate_cache_key(&sentence.text, target_language, "nlp");
        
        // Check persistent cache first
        if let Ok(Some(cached_translation)) = self.base.load_from_persistent_cache(&cache_key).await {
            self.base.cache.insert(cache_key.clone(), cached_translation.clone());
            return Ok(cached_translation);
        }
        
        // Check in-memory cache as fallback
        if let Some(cached) = self.base.cache.get(&cache_key) {
            return Ok(cached.clone());
        }

        // Translate the complete sentence
        match self.base.translate_text(&sentence.text, target_language, None).await {
            Ok(translation) => {
                // Cache successful translation
                self.base.cache.insert(cache_key.clone(), translation.clone());
                
                // Save to persistent cache
                if let Err(e) = self.base.save_to_persistent_cache(
                    &cache_key,
                    &sentence.text,
                    target_language,
                    "nlp",
                    &translation,
                    &TranslationQuality::Good,
                ).await {
                    warn!("Failed to save sentence translation to persistent cache: {}", e);
                }
                
                Ok(translation)
            }
            Err(e) => Err(e),
        }
    }

    /// Map translated sentences back to original segments
    fn map_sentences_to_segments(&self, sentences: &[Sentence], segments: &mut [TranscriptionSegment]) {
        for sentence in sentences {
            // Split translated sentence back to words based on original segment count
            let words: Vec<&str> = sentence.text.split_whitespace().collect();
            let segment_count = sentence.segment_indices.len();
            
            if segment_count == 0 || words.is_empty() {
                continue;
            }
            
            // Distribute words across segments proportionally
            let words_per_segment = words.len() / segment_count;
            let extra_words = words.len() % segment_count;
            
            let mut word_idx = 0;
            
            for (seg_pos, &segment_idx) in sentence.segment_indices.iter().enumerate() {
                if segment_idx >= segments.len() {
                    continue;
                }
                
                let words_for_this_segment = words_per_segment + if seg_pos < extra_words { 1 } else { 0 };
                let end_idx = (word_idx + words_for_this_segment).min(words.len());
                
                if word_idx < words.len() {
                    let segment_words = &words[word_idx..end_idx];
                    segments[segment_idx].text = segment_words.join(" ");
                    word_idx = end_idx;
                }
            }
        }
    }
}

#[async_trait]
impl Translator for NlpTranslator {
    /// NLP translation: Reconstruct complete sentences, then translate sentence by sentence
    async fn translate_transcription(
        &mut self,
        transcription: &mut Transcription,
        target_language: &str,
        _context: Option<&str>,
    ) -> Result<()> {
        info!("Starting NLP translation with sentence reconstruction to {}", target_language);
        
        // Step 1: Group segments into sentences based on timing gaps and content
        let sentences = self.build_sentences_from_segments(&transcription.segments);
        
        info!("Reconstructed {} sentences from {} segments", sentences.len(), transcription.segments.len());
        
        // Step 2: Translate each sentence
        let mut translated_sentences = Vec::new();
        
        for (idx, sentence) in sentences.iter().enumerate() {
            info!("┌─ Translating sentence {}/{} (NLP) ────────", idx + 1, sentences.len());
            info!("│ Source: {}", sentence.text);
            
            match self.translate_sentence(sentence, target_language).await {
                Ok(translation) => {
                    info!("│ Target: {}", translation);
                    info!("└─────────────────────────────────────");
                    
                    let mut translated_sentence = sentence.clone();
                    translated_sentence.text = translation;
                    translated_sentences.push(translated_sentence);
                }
                Err(e) => {
                    warn!("│ Failed: {}", e);
                    warn!("└─────────────────────────────────────");
                    // Keep original sentence on failure
                    translated_sentences.push(sentence.clone());
                }
            }
        }
        
        // Step 3: Map translated sentences back to original segments
        self.map_sentences_to_segments(&translated_sentences, &mut transcription.segments);
        
        Ok(())
    }
} 