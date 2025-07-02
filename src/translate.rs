use std::collections::HashMap;
use std::time::Duration;
use std::path::PathBuf;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::json;
use tracing::{info, warn, debug};

use crate::config::{TranslateConfig, TranslationMode};
use crate::error::{Result, ShuroError};
use crate::quality::{Transcription, TranscriptionSegment};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranslationRequest {
    pub model: String,
    pub prompt: String,
    pub stream: bool,
    pub format: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranslationResponse {
    pub response: String,
    pub done: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranslationResult {
    pub text: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityEvaluation {
    pub evaluation: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranslationCacheEntry {
    pub source_text: String,
    pub target_language: String,
    pub context: String,
    pub translation: String,
    pub quality: String,
    pub model: String,
    pub cached_at: u64,
}

#[derive(Debug, Clone)]
pub enum TranslationQuality {
    Perfect,
    Good,
    Bad,
    Invalid,
}

impl TranslationQuality {
    pub fn from_str(s: &str) -> Self {
        match s.to_uppercase().as_str() {
            "PERFECT" => Self::Perfect,
            "GOOD" => Self::Good,
            "BAD" => Self::Bad,
            "INVALID" => Self::Invalid,
            _ => Self::Good,
        }
    }
    
    pub fn to_str(&self) -> &'static str {
        match self {
            Self::Perfect => "PERFECT",
            Self::Good => "GOOD", 
            Self::Bad => "BAD",
            Self::Invalid => "INVALID",
        }
    }
    
    pub fn is_acceptable(&self) -> bool {
        matches!(self, Self::Perfect | Self::Good)
    }
}

/// Sentence structure for NLP mode
#[derive(Debug, Clone)]
struct Sentence {
    text: String,
    start_time: f64,
    end_time: f64,
    segment_indices: Vec<usize>,
}

pub struct OllamaTranslator {
    client: Client,
    config: TranslateConfig,
    cache: HashMap<String, String>,
    cache_dir: PathBuf,
}

impl OllamaTranslator {
    pub fn new(config: TranslateConfig) -> Self {
        let client = Client::builder()
            .timeout(Duration::from_secs(300)) // 5 minute timeout
            .build()
            .expect("Failed to create HTTP client");

        let cache_dir = PathBuf::from(".shuro/cache/translations");
        if let Err(e) = std::fs::create_dir_all(&cache_dir) {
            warn!("Failed to create translation cache directory: {}", e);
        }

        Self {
            client,
            config,
            cache: HashMap::new(),
            cache_dir,
        }
    }

    /// Translate all segments in a transcription
    pub async fn translate_transcription(
        &mut self,
        transcription: &mut Transcription,
        target_language: &str,
        _context: Option<&str>,
    ) -> Result<()> {
        info!("Starting translation to {} using {:?} mode", target_language, self.config.mode);
        
        match self.config.mode {
            TranslationMode::Simple => {
                self.translate_simple(transcription, target_language).await
            }
            TranslationMode::Context => {
                self.translate_with_context(transcription, target_language).await
            }
            TranslationMode::Nlp => {
                self.translate_with_nlp(transcription, target_language).await
            }
        }
    }

    /// Simple translation: Translate each segment individually without context
    async fn translate_simple(
        &mut self,
        transcription: &mut Transcription,
        target_language: &str,
    ) -> Result<()> {
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

    /// Context-aware translation: Use surrounding segments as context but only translate the target segment
    async fn translate_with_context(
        &mut self,
        transcription: &mut Transcription,
        target_language: &str,
    ) -> Result<()> {
        let total_segments = transcription.segments.len();
        
        // Pre-build all contexts to avoid borrowing issues
        let contexts: Vec<String> = (0..total_segments)
            .map(|idx| self.build_segment_context(&transcription.segments, idx, self.config.context_window_size))
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

    /// NLP translation: Reconstruct complete sentences, then translate sentence by sentence
    async fn translate_with_nlp(
        &mut self,
        transcription: &mut Transcription,
        target_language: &str,
    ) -> Result<()> {
        info!("Building sentences from segments using NLP approach");
        
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

    /// Simple segment translation without context or quality validation
    async fn translate_segment_simple(
        &mut self,
        segment: &TranscriptionSegment,
        target_language: &str,
    ) -> Result<String> {
        let cache_key = self.generate_cache_key(&segment.text, target_language, "");
        
        // Check persistent cache first
        if let Ok(Some(cached_translation)) = self.load_from_persistent_cache(&cache_key).await {
            debug!("Using cached translation from disk");
            self.cache.insert(cache_key.clone(), cached_translation.clone());
            return Ok(cached_translation);
        }
        
        // Check in-memory cache as fallback
        if let Some(cached) = self.cache.get(&cache_key) {
            debug!("Using cached translation from memory");
            return Ok(cached.clone());
        }

        // Translate without context
        match self.translate_text(&segment.text, target_language, None).await {
            Ok(translation) => {
                // Cache successful translation
                self.cache.insert(cache_key.clone(), translation.clone());
                
                // Save to persistent cache
                if let Err(e) = self.save_to_persistent_cache(
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
        let cache_key = self.generate_cache_key(&segment.text, target_language, context);
        
        // Check persistent cache first
        if let Ok(Some(cached_translation)) = self.load_from_persistent_cache(&cache_key).await {
            debug!("Using cached translation from disk");
            // Also store in memory cache for faster access during this session
            self.cache.insert(cache_key.clone(), cached_translation.clone());
            return Ok(cached_translation);
        }
        
        // Check in-memory cache as fallback
        if let Some(cached) = self.cache.get(&cache_key) {
            debug!("Using cached translation from memory");
            return Ok(cached.clone());
        }

        let mut current_context = context.to_string();
        let mut attempts = 0;
        
        while attempts < self.config.max_retries {
            attempts += 1;

            match self.translate_text(&segment.text, target_language, if current_context.is_empty() { None } else { Some(&current_context) }).await {
                Ok(translation) => {
                    // Length validation like gearbox - if translation is too long, remove context
                    if segment.text.len() * 5 < translation.len() {
                        info!("│ Translation too long, removing context (attempt {})", attempts);
                        current_context = String::new();
                        continue;
                    }
                    
                    // Validate translation quality
                    let quality = self.evaluate_translation_quality(&segment.text, &translation, &current_context, target_language).await;
                    
                    match quality {
                        Ok(TranslationQuality::Perfect | TranslationQuality::Good) => {
                            info!("│ Quality: {} ✓", quality.as_ref().unwrap().to_str());
                            
                            // Save to persistent cache (only good quality translations)
                            if let Err(e) = self.save_to_persistent_cache(
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
                            self.cache.insert(cache_key, translation.clone());
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
            self.config.max_retries
        )))
    }

    /// Perform the actual translation using Ollama with JSON format
    async fn translate_text(
        &self,
        text: &str,
        target_language: &str,
        context: Option<&str>,
    ) -> Result<String> {
        let prompt = self.build_translation_prompt(text, target_language, context);
        
        let request = TranslationRequest {
            model: self.config.model.clone(),
            prompt,
            stream: false,
            format: "json".to_string(),
        };

        let url = format!("{}/api/generate", self.config.endpoint);
        
        debug!("Sending translation request to: {}", url);
        
        let response = self.client
            .post(&url)
            .json(&request)
            .send()
            .await
            .map_err(|e| ShuroError::Translation(format!("HTTP request failed: {}", e)))?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            return Err(ShuroError::Translation(format!(
                "Ollama API error {}: {}", status, error_text
            )));
        }

        let translation_response: TranslationResponse = response.json().await
            .map_err(|e| ShuroError::Translation(format!("Failed to parse response: {}", e)))?;

        let raw_response = translation_response.response.trim().to_string();
        
        debug!("Raw Ollama response: {}", raw_response);
        
        if raw_response.is_empty() {
            return Err(ShuroError::Translation("Empty translation received".to_string()));
        }

        if let Ok(result) = serde_json::from_str::<TranslationResult>(&raw_response) {
            return Ok(result.text.trim().to_string());
        }

        let cleaned = self.clean_translation_response(&raw_response);
        Ok(cleaned)
    }

    /// Evaluate translation quality using structured evaluation like gearbox
    async fn evaluate_translation_quality(
        &self,
        original: &str,
        translation: &str,
        context: &str,
        target_language: &str,
    ) -> Result<TranslationQuality> {
        let quality_prompt = format!(
            "You are a professional translator.\n\
             Evaluate the following translation quality into {}.\n\
             \n\
             Evaluate translation quality in one of the following levels:\n\
             - [PERFECT]: The translation is perfect and no further improvement is needed.\n\
             - [GOOD]: The translation is good but some minor improvements are needed.\n\
             - [BAD]: The translation is bad and needs to be re-translated.\n\
             - [INVALID]: The translation is invalid or not related to the source.\n\
             \n\
             Please return the evaluation results in JSON format as {{\"evaluation\":\"evaluation result\"}}.\n\
             \n\
             [Source]\n\
             {}\n\
             \n\
             [Translation]\n\
             {}\n\
             \n\
             [Context]\n\
             {}",
            target_language, original, translation, context
        );

        let request = TranslationRequest {
            model: self.config.model.clone(),
            prompt: quality_prompt,
            stream: false,
            format: "json".to_string(),
        };

        let url = format!("{}/api/generate", self.config.endpoint);
        
        let response = self.client
            .post(&url)
            .json(&request)
            .send()
            .await
            .map_err(|e| ShuroError::Translation(format!("Quality evaluation request failed: {}", e)))?;

        if !response.status().is_success() {
            return Err(ShuroError::Translation("Quality evaluation request failed".to_string()));
        }

        let quality_response: TranslationResponse = response.json().await
            .map_err(|e| ShuroError::Translation(format!("Failed to parse quality response: {}", e)))?;

        let raw_response = quality_response.response.trim();
        
        if let Ok(eval_result) = serde_json::from_str::<QualityEvaluation>(raw_response) {
            return Ok(TranslationQuality::from_str(&eval_result.evaluation));
        }
        
        let response_upper = raw_response.to_uppercase();
        if response_upper.contains("PERFECT") {
            Ok(TranslationQuality::Perfect)
        } else if response_upper.contains("GOOD") {
            Ok(TranslationQuality::Good)
        } else if response_upper.contains("BAD") {
            Ok(TranslationQuality::Bad)
        } else {
            Ok(TranslationQuality::Invalid)
        }
    }

    /// Build translation prompt with context, using JSON format like gearbox
    /// FIXED: Context is provided as background info, but only the source text is translated
    fn build_translation_prompt(&self, text: &str, target_language: &str, context: Option<&str>) -> String {
        if text.len() < 50 {
            format!(
                "You are a professional translator.\n\
                 Translate the following text to {}.\n\
                 Return ONLY the translation in JSON format as {{\"text\":\"translated text\"}}.\n\
                 \n\
                 Text to translate: \"{}\"\n",
                target_language, text
            )
        } else {
            let mut prompt = format!(
                "You are a professional translator.\n\
                 Translate the following text to {}.\n\
                 Return ONLY the translation in JSON format as {{\"text\":\"translated text\"}}.\n\
                 \n\
                 [Text to translate]\n\
                 {}\n\
                 \n",
                target_language, text
            );
            
            if let Some(ctx) = context {
                if !ctx.trim().is_empty() {
                    prompt.push_str(&format!(
                        "[Context for reference - DO NOT translate this part]\n\
                         {}\n\n\
                         Remember: Only translate the text in the [Text to translate] section above.\n",
                        ctx
                    ));
                }
            }
            
            prompt
        }
    }

    /// Clean up translation response to extract just the translation
    fn clean_translation_response(&self, response: &str) -> String {
        let lines: Vec<&str> = response.lines().collect();
        
        for &line in &lines {
            let trimmed = line.trim();
            
            if trimmed.is_empty() {
                continue;
            }
            
            if trimmed.starts_with("Here are") ||
               trimmed.starts_with("Option") ||
               trimmed.starts_with("**Option") ||
               trimmed.starts_with("Translation:") ||
               trimmed.starts_with("- ") ||
               trimmed.starts_with("* ") ||
               trimmed.contains("(Captures") ||
               trimmed.contains("maintains") {
                continue;
            }
            
            if trimmed.starts_with("**") && trimmed.ends_with("**") {
                continue;
            }
            
            if trimmed.len() > 3 {
                return trimmed.to_string();
            }
        }
        
        for &line in &lines {
            let trimmed = line.trim();
            if !trimmed.is_empty() {
                return trimmed.to_string();
            }
        }
        
        response.to_string()
    }

    /// Generate cache key for translation
    fn generate_cache_key(&self, source_text: &str, target_language: &str, context: &str) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        source_text.hash(&mut hasher);
        target_language.hash(&mut hasher);
        context.hash(&mut hasher);
        self.config.model.hash(&mut hasher);
        
        let hash = hasher.finish();
        format!("{:016x}", hash)
    }

    /// Load translation from persistent cache
    async fn load_from_persistent_cache(&self, cache_key: &str) -> Result<Option<String>> {
        let cache_file = self.cache_dir.join(format!("{}.json", cache_key));
        
        if !cache_file.exists() {
            return Ok(None);
        }

        match tokio::fs::read_to_string(&cache_file).await {
            Ok(content) => {
                match serde_json::from_str::<TranslationCacheEntry>(&content) {
                    Ok(entry) => {
                        debug!("Translation cache hit: {} (cached {} ago)", 
                              cache_key,
                              format_duration(std::time::SystemTime::now()
                                  .duration_since(std::time::UNIX_EPOCH)
                                  .unwrap_or_default()
                                  .as_secs()
                                  .saturating_sub(entry.cached_at)));
                        Ok(Some(entry.translation))
                    }
                    Err(e) => {
                        warn!("Failed to parse translation cache entry: {}", e);
                        Ok(None)
                    }
                }
            }
            Err(_) => Ok(None),
        }
    }

    /// Save translation to persistent cache
    async fn save_to_persistent_cache(
        &self,
        cache_key: &str,
        source_text: &str,
        target_language: &str,
        context: &str,
        translation: &str,
        quality: &TranslationQuality,
    ) -> Result<()> {
        let entry = TranslationCacheEntry {
            source_text: source_text.to_string(),
            target_language: target_language.to_string(),
            context: context.to_string(),
            translation: translation.to_string(),
            quality: quality.to_str().to_string(),
            model: self.config.model.clone(),
            cached_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        };

        let cache_file = self.cache_dir.join(format!("{}.json", cache_key));
        let content = serde_json::to_string_pretty(&entry)
            .map_err(|e| ShuroError::Translation(format!("Failed to serialize translation cache: {}", e)))?;
        
        if let Err(e) = tokio::fs::write(&cache_file, content).await {
            warn!("Failed to write translation cache: {}", e);
        } else {
            debug!("Saved translation to cache: {}", cache_key);
        }
        
        Ok(())
    }

    /// Clear all translation cache
    pub async fn clear_translation_cache(&self) -> Result<u64> {
        let mut count = 0;
        if let Ok(mut entries) = tokio::fs::read_dir(&self.cache_dir).await {
            while let Ok(Some(entry)) = entries.next_entry().await {
                if let Some(extension) = entry.path().extension() {
                    if extension == "json" {
                        if tokio::fs::remove_file(entry.path()).await.is_ok() {
                            count += 1;
                        }
                    }
                }
            }
        }
        info!("Cleared {} translation cache entries", count);
        Ok(count)
    }

    /// List translation cache entries
    pub async fn list_translation_cache(&self) -> Result<Vec<TranslationCacheEntry>> {
        let mut entries = Vec::new();
        
        if let Ok(mut dir_entries) = tokio::fs::read_dir(&self.cache_dir).await {
            while let Ok(Some(entry)) = dir_entries.next_entry().await {
                if let Some(extension) = entry.path().extension() {
                    if extension == "json" {
                        if let Ok(content) = tokio::fs::read_to_string(entry.path()).await {
                            if let Ok(cache_entry) = serde_json::from_str::<TranslationCacheEntry>(&content) {
                                entries.push(cache_entry);
                            }
                        }
                    }
                }
            }
        }
        
        // Sort by cache time (newest first)
        entries.sort_by(|a, b| b.cached_at.cmp(&a.cached_at));
        
        Ok(entries)
    }

    // NLP Mode Helper Methods
    
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
                if gap > self.config.nlp_gap_threshold {
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

    /// Check if the text contains sentence endings (less strict version)
    fn has_sentence_ending(&self, text: &str) -> bool {
        text.contains(". ") || text.contains("! ") || text.contains("? ") || 
        text.ends_with('.') || text.ends_with('!') || text.ends_with('?')
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
                start_time,
                end_time,
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
                    start_time: current_start,
                    end_time: sentence_end,
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
        let cache_key = self.generate_cache_key(&sentence.text, target_language, "nlp");
        
        // Check persistent cache first
        if let Ok(Some(cached_translation)) = self.load_from_persistent_cache(&cache_key).await {
            debug!("Using cached sentence translation from disk");
            self.cache.insert(cache_key.clone(), cached_translation.clone());
            return Ok(cached_translation);
        }
        
        // Check in-memory cache as fallback
        if let Some(cached) = self.cache.get(&cache_key) {
            debug!("Using cached sentence translation from memory");
            return Ok(cached.clone());
        }

        // Translate the complete sentence
        match self.translate_text(&sentence.text, target_language, None).await {
            Ok(translation) => {
                // Cache successful translation
                self.cache.insert(cache_key.clone(), translation.clone());
                
                // Save to persistent cache
                if let Err(e) = self.save_to_persistent_cache(
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

/// Check if Ollama is available and the model is loaded
pub async fn check_ollama_availability(endpoint: &str, model: &str) -> Result<()> {
    let client = Client::new();
    let url = format!("{}/api/show", endpoint);
    
    let request = json!({
        "name": model
    });

    let response = client
        .post(&url)
        .json(&request)
        .send()
        .await
        .map_err(|e| ShuroError::Translation(format!("Failed to connect to Ollama: {}", e)))?;

    if response.status().is_success() {
        info!("Ollama model '{}' is available", model);
        Ok(())
    } else {
        Err(ShuroError::Translation(format!(
            "Ollama model '{}' not found. Please pull the model first: ollama pull {}",
            model, model
        )))
    }
}

/// Format duration in seconds to a human-readable string
fn format_duration(seconds: u64) -> String {
    let days = seconds / (24 * 60 * 60);
    let hours = (seconds % (24 * 60 * 60)) / (60 * 60);
    let minutes = (seconds % (60 * 60)) / 60;
    let secs = seconds % 60;

    if days > 0 {
        format!("{}d {}h", days, hours)
    } else if hours > 0 {
        format!("{}h {}m", hours, minutes)
    } else if minutes > 0 {
        format!("{}m {}s", minutes, secs)
    } else {
        format!("{}s", secs)
    }
} 