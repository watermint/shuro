use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use tracing::{info, warn, debug};

use crate::config::TranslateConfig;
use crate::error::{Result, ShuroError};
use crate::quality::{Transcription, TranscriptionSegment};
use super::{Translator, common::BaseTranslator};

/// LLM-based translation with sliding window sentence splitting
pub struct LlmTranslator {
    base: BaseTranslator,
    window_size: usize,
    confidence_threshold: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct LlmAnalysisResponse {
    /// List of reconstructed sentences
    sentences: Vec<String>,
}

/// Ollama response wrapper
#[derive(Debug, Deserialize)]
struct OllamaResponse {
    /// The actual response content (may contain JSON)
    response: String,
}

#[derive(Debug, Clone)]
struct SentenceCandidate {
    /// The reconstructed sentence text
    text: String,
    /// Number of windows that detected this sentence (or a variant)
    detection_count: usize,
    /// Total number of windows this sentence was eligible to appear in
    total_windows: usize,
}

#[derive(Debug, Clone)]
struct ReconstructedSentence {
    /// Segment indices that make up this sentence
    segment_indices: Vec<usize>,
    /// Start time of the sentence (from first segment)
    start_time: f64,
    /// End time of the sentence (from last segment)
    end_time: f64,
    /// Combined text of all segments
    text: String,
    /// Translated text
    translated_text: String,
}

impl LlmTranslator {
    pub fn new(config: TranslateConfig) -> Self {
        Self {
            window_size: config.llm_window_size,
            confidence_threshold: config.llm_confidence_threshold,
            base: BaseTranslator::new(config),
        }
    }

    /// Analyze a window of segments using LLM to reconstruct complete sentences
    async fn analyze_segments_window(
        &self,
        segments: &[TranscriptionSegment],
        window_start: usize,
    ) -> Result<Vec<String>> {
        let window_end = (window_start + self.window_size).min(segments.len());
        let window_segments: Vec<String> = segments[window_start..window_end]
            .iter()
            .map(|s| s.text.clone())
            .collect();

        let prompt = self.build_analysis_prompt(&window_segments);
        
        debug!("Analyzing window {}-{} with {} segments", 
               window_start, window_end, window_segments.len());

        match self.base.client
            .post(&format!("{}/api/generate", self.base.config.endpoint))
            .json(&serde_json::json!({
                "model": self.base.config.model,
                "prompt": prompt,
                "stream": false,
                "format": "json"
            }))
            .send()
            .await
        {
            Ok(response) => {
                if !response.status().is_success() {
                    let status = response.status();
                    let error_text = response.text().await.unwrap_or_default();
                    return Err(ShuroError::Translation(format!(
                        "LLM analysis failed {}: {}", status, error_text
                    )));
                }

                let response_text = response.text().await
                    .map_err(|e| ShuroError::Translation(format!("Failed to read response: {}", e)))?;
                
                debug!("Raw LLM analysis response: {}", response_text);
                
                self.parse_analysis_response(&response_text, window_start)
            }
            Err(e) => Err(ShuroError::Translation(format!("HTTP request failed: {}", e)))
        }
    }

    /// Build prompt for LLM analysis to reconstruct complete sentences
    fn build_analysis_prompt(&self, segments: &[String]) -> String {
        let segments_text = segments
            .iter()
            .map(|text| format!("• {}", text))
            .collect::<Vec<_>>()
            .join("\n");

        format!(
            r#"You are a professional editor. Your task is to reconstruct complete sentences from transcribed speech segments.

TASK: Combine the segments below into complete, natural sentences. Preserve the original words exactly - do not rephrase or correct grammar.

SEGMENTS:
{}

INSTRUCTIONS:
1. Join segments that form complete thoughts
2. Keep the original text exactly as written
3. Each sentence should be a complete idea
4. Maintain natural flow and meaning

Return ONLY a JSON array of sentences:
{{
  "sentences": [
    "First complete sentence here",
    "Second complete sentence here",
    "Third complete sentence here"
  ]
}}

Analyze the segments and return only the JSON response:"#,
            segments_text
        )
    }

    /// Parse LLM response to extract reconstructed sentences
    fn parse_analysis_response(&self, response: &str, _window_start: usize) -> Result<Vec<String>> {
        let response = response.trim();
        
        // First, try to parse as Ollama response format
        if let Ok(ollama_response) = serde_json::from_str::<OllamaResponse>(response) {
            let actual_response = ollama_response.response.trim();
            debug!("Extracted inner response from Ollama format: {}", actual_response);
            
            // Try flexible parsing on the inner response
            if let Some(sentences) = self.try_flexible_json_parsing(actual_response) {
                debug!("Successfully parsed {} sentences from Ollama response", sentences.len());
                return Ok(sentences);
            }
        }
        
        // Fallback: Try flexible parsing on the full response (for other LLM providers)
        if let Some(sentences) = self.try_flexible_json_parsing(response) {
            debug!("Successfully parsed {} sentences from direct response", sentences.len());
            return Ok(sentences);
        }

        warn!("Failed to parse LLM analysis response: {}", response);
        Ok(Vec::new()) // Return empty sentences on parse failure
    }

    /// Try to parse JSON with various common formats and wrappers
    fn try_flexible_json_parsing(&self, text: &str) -> Option<Vec<String>> {
        let text = text.trim();
        
        // 1. Try direct JSON parsing first
        if let Ok(parsed) = serde_json::from_str::<LlmAnalysisResponse>(text) {
            return Some(parsed.sentences);
        }
        
        // 2. Try removing markdown code blocks (```json ... ``` or ``` ... ```)
        let cleaned_text = self.remove_markdown_code_blocks(text);
        if cleaned_text != text {
            debug!("Removed markdown code blocks, trying to parse: {}", cleaned_text);
            if let Ok(parsed) = serde_json::from_str::<LlmAnalysisResponse>(&cleaned_text) {
                return Some(parsed.sentences);
            }
        }
        
        // 3. Try extracting JSON from mixed text (find first { to last })
        if let Some(json_start) = text.find("{") {
            if let Some(json_end) = text.rfind("}") {
                let json_str = &text[json_start..=json_end];
                if json_str != text {
                    debug!("Extracted JSON from mixed text: {}", json_str);
                    if let Ok(parsed) = serde_json::from_str::<LlmAnalysisResponse>(json_str) {
                        return Some(parsed.sentences);
                    }
                }
            }
        }
        
        // 4. Try extracting from cleaned text as well
        if let Some(json_start) = cleaned_text.find("{") {
            if let Some(json_end) = cleaned_text.rfind("}") {
                let json_str = &cleaned_text[json_start..=json_end];
                debug!("Extracted JSON from cleaned text: {}", json_str);
                if let Ok(parsed) = serde_json::from_str::<LlmAnalysisResponse>(json_str) {
                    return Some(parsed.sentences);
                }
            }
        }
        
        None
    }

    /// Remove markdown code blocks from text
    fn remove_markdown_code_blocks(&self, text: &str) -> String {
        let text = text.trim();
        
        // Handle ```json ... ``` pattern
        if text.starts_with("```json") && text.ends_with("```") {
            let inner = &text[7..text.len()-3]; // Remove ```json and ```
            return inner.trim().to_string();
        }
        
        // Handle ``` ... ``` pattern
        if text.starts_with("```") && text.ends_with("```") {
            let inner = &text[3..text.len()-3]; // Remove ``` and ```
            return inner.trim().to_string();
        }
        
        // Handle `json ... ` pattern (single backticks)
        if text.starts_with("`json") && text.ends_with("`") {
            let inner = &text[5..text.len()-1]; // Remove `json and `
            return inner.trim().to_string();
        }
        
        // Handle ` ... ` pattern (single backticks)
        if text.starts_with("`") && text.ends_with("`") {
            let inner = &text[1..text.len()-1]; // Remove ` and `
            return inner.trim().to_string();
        }
        
        // Return original text if no markdown patterns found
        text.to_string()
    }

    /// Process all segments using sliding window analysis to collect sentence candidates
    async fn analyze_all_segments(&self, segments: &[TranscriptionSegment]) -> Result<Vec<SentenceCandidate>> {
        let total_segments = segments.len();
        if total_segments < 2 {
            return Ok(Vec::new());
        }

        let mut all_sentences = Vec::new();
        let mut total_windows = 0;

        // Sliding window analysis
        let mut window_start = 0;
        while window_start < total_segments {
            let window_end = (window_start + self.window_size).min(total_segments);
            
            info!("Analyzing window {}-{} of {} segments", 
                  window_start + 1, window_end, total_segments);

            match self.analyze_segments_window(segments, window_start).await {
                Ok(sentences) => {
                    if sentences.is_empty() {
                        info!("   → No sentences detected in this window");
                    } else {
                        info!("   → Detected {} sentence(s):", sentences.len());
                        for sentence in &sentences {
                            if !sentence.trim().is_empty() {
                                info!("     • {}", sentence);
                                all_sentences.push(sentence.trim().to_string());
                            }
                        }
                    }
                    total_windows += 1;
                }
                Err(e) => {
                    warn!("Failed to analyze window {}-{}: {}", window_start, window_end, e);
                }
            }

            // Move window forward by 1 segment for sliding analysis
            window_start += 1;
            
            // Don't slide if remaining segments are less than minimum window
            if window_start + 2 >= total_segments {
                break;
            }
        }

        // Calculate confidence for each unique sentence
        let sentence_candidates = self.calculate_sentence_confidence(all_sentences, total_windows);
        
        // Log summary of analysis results
        info!("");
        info!("🔍 LLM ANALYSIS SUMMARY");
        info!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        info!("📊 Analysis Results:");
        info!("   • Windows analyzed: {}", total_windows);
        info!("   • Unique sentences found: {}", sentence_candidates.len());
        info!("   • Confidence threshold: {:.1}%", self.confidence_threshold * 100.0);
        info!("");
        
        if sentence_candidates.is_empty() {
            info!("⚠️  No sentences detected - will fall back to individual segments");
        } else {
            info!("🎯 Top sentence candidates:");
            for (i, candidate) in sentence_candidates.iter().take(10).enumerate() {
                let confidence = candidate.detection_count as f64 / candidate.total_windows as f64;
                let status = if confidence >= self.confidence_threshold { "✅" } else { "❌" };
                info!("   {}. {} {:.1}% ({}/{}) - {}", 
                      i + 1, status, confidence * 100.0, 
                      candidate.detection_count, candidate.total_windows, candidate.text);
            }
            
            if sentence_candidates.len() > 10 {
                info!("   ... and {} more sentences", sentence_candidates.len() - 10);
            }
        }
        
        info!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        info!("");
        
        Ok(sentence_candidates)
    }
    
    /// Calculate confidence for sentences based on frequency and prefix/suffix relationships
    fn calculate_sentence_confidence(&self, all_sentences: Vec<String>, total_windows: usize) -> Vec<SentenceCandidate> {
        // Count frequency of each exact sentence
        let mut sentence_counts: HashMap<String, usize> = HashMap::new();
        for sentence in &all_sentences {
            *sentence_counts.entry(sentence.clone()).or_insert(0) += 1;
        }
        
        // Create initial candidates
        let mut candidates: Vec<SentenceCandidate> = sentence_counts
            .clone()
            .into_iter()
            .map(|(text, count)| SentenceCandidate {
                text,
                detection_count: count,
                total_windows,
            })
            .collect();
        
        // Handle prefix/suffix relationships
        // Sort by length (descending) to process longer sentences first
        candidates.sort_by(|a, b| b.text.len().cmp(&a.text.len()));
        
        let mut final_candidates: Vec<SentenceCandidate> = Vec::new();
        let mut used_sentences = HashSet::new();
        
        for candidate in candidates {
            if used_sentences.contains(&candidate.text) {
                continue;
            }
            
            // Check if this sentence is a prefix or suffix of an already processed sentence
            let mut is_redundant = false;
            for existing in &final_candidates {
                if self.is_prefix_or_suffix(&candidate.text, &existing.text) {
                    is_redundant = true;
                    break;
                }
            }
            
            if !is_redundant {
                // Mark this sentence and any of its prefixes/suffixes as used
                used_sentences.insert(candidate.text.clone());
                
                // Find and merge any shorter sentences that are prefixes/suffixes
                let mut merged_candidate = candidate.clone();
                for other in &all_sentences {
                    if other != &merged_candidate.text && 
                       !used_sentences.contains(other) && 
                       self.is_prefix_or_suffix(other, &merged_candidate.text) {
                        // This shorter sentence is a prefix/suffix of our candidate
                        // Add its detection count to our candidate
                        merged_candidate.detection_count += sentence_counts.get(other).unwrap_or(&0);
                        used_sentences.insert(other.clone());
                    }
                }
                
                final_candidates.push(merged_candidate);
            }
        }
        
        // Sort by confidence (detection_count / total_windows) descending
        final_candidates.sort_by(|a, b| {
            let conf_a = a.detection_count as f64 / a.total_windows as f64;
            let conf_b = b.detection_count as f64 / b.total_windows as f64;
            conf_b.partial_cmp(&conf_a).unwrap_or(std::cmp::Ordering::Equal)
        });
        
        final_candidates
    }
    
    /// Check if text1 is a prefix or suffix of text2
    fn is_prefix_or_suffix(&self, shorter: &str, longer: &str) -> bool {
        if shorter.len() >= longer.len() {
            return false;
        }
        
        let shorter_normalized = self.normalize_text(shorter);
        let longer_normalized = self.normalize_text(longer);
        
        longer_normalized.starts_with(&shorter_normalized) || 
        longer_normalized.ends_with(&shorter_normalized)
    }
    
    /// Normalize text for comparison (remove extra spaces, punctuation at boundaries)
    fn normalize_text(&self, text: &str) -> String {
        text.trim()
            .replace(&[' ', '\t', '\n'][..], " ")
            .chars()
            .collect::<String>()
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ")
            .to_lowercase()
    }
    


    /// Filter sentence candidates by confidence threshold
    fn finalize_sentences(&self, sentence_candidates: Vec<SentenceCandidate>) -> Vec<SentenceCandidate> {
        let mut qualified_sentences = Vec::new();

        // Filter by confidence threshold
        for candidate in sentence_candidates {
            let confidence = candidate.detection_count as f64 / candidate.total_windows as f64;
            
            if confidence >= self.confidence_threshold {
                qualified_sentences.push(candidate);
            }
        }

        // Log final results
        info!("🎯 FINAL SENTENCE SELECTION");
        info!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        info!("📋 Qualified sentences ({} passed threshold):", qualified_sentences.len());
        
        if qualified_sentences.is_empty() {
            info!("   ⚠️  No sentences met the {:.1}% confidence threshold", self.confidence_threshold * 100.0);
            info!("   → Will fall back to individual segment translation");
        } else {
            for (i, candidate) in qualified_sentences.iter().enumerate() {
                let confidence = candidate.detection_count as f64 / candidate.total_windows as f64;
                info!("   {}. {:.1}% ({}/{}) - {}", 
                      i + 1, confidence * 100.0, 
                      candidate.detection_count, candidate.total_windows, candidate.text);
            }
        }
        
        info!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        info!("");

        qualified_sentences
    }
    


    /// Convert sentence candidates to reconstructed sentences
    fn reconstruct_sentences_from_candidates(
        &self,
        segments: &[TranscriptionSegment],
        sentence_candidates: Vec<SentenceCandidate>,
    ) -> Vec<ReconstructedSentence> {
        let mut sentences = Vec::new();

        // Convert candidates to reconstructed sentences
        for (idx, candidate) in sentence_candidates.iter().enumerate() {
            sentences.push(ReconstructedSentence {
                segment_indices: vec![idx], // Use sentence index as a placeholder
                start_time: segments.first().map(|s| s.start).unwrap_or(0.0),
                end_time: segments.last().map(|s| s.end).unwrap_or(0.0),
                text: candidate.text.clone(),
                translated_text: String::new(),
            });
        }

        // If no high-confidence sentences found, fall back to individual segments
        if sentences.is_empty() {
            info!("No high-confidence sentences found, falling back to individual segments");
            for (idx, segment) in segments.iter().enumerate() {
                sentences.push(ReconstructedSentence {
                    segment_indices: vec![idx],
                    start_time: segment.start,
                    end_time: segment.end,
                    text: segment.text.clone(),
                    translated_text: String::new(),
                });
            }
        }

        sentences
    }



    /// Validate that translated text matches original structure
    fn validate_translation(&self, original: &str, translated: &str) -> bool {
        // Basic validation: check if translation is not empty and not just the original
        if translated.trim().is_empty() {
            return false;
        }

        // Check if translation is identical to original (likely translation failure)
        if original.trim() == translated.trim() {
            return false;
        }

        // Check if translation is suspiciously long (more than 3x original)
        if translated.len() > original.len() * 3 {
            return false;
        }

        // Basic validation passed
        true
    }

    /// Translate reconstructed sentences
    async fn translate_sentences(
        &mut self,
        sentences: &mut [ReconstructedSentence],
        target_language: &str,
    ) -> Result<()> {
        let total_sentences = sentences.len();
        for (idx, sentence) in sentences.iter_mut().enumerate() {
            info!("Translating sentence {}/{}: {}", 
                  idx + 1, total_sentences, 
                  &sentence.text);

            match self.base.translate_text(&sentence.text, target_language, None).await {
                Ok(translation) => {
                    if self.validate_translation(&sentence.text, &translation) {
                        sentence.translated_text = translation;
                        info!("✓ Translation successful");
                    } else {
                        warn!("✗ Translation validation failed, keeping original");
                        sentence.translated_text = sentence.text.clone();
                    }
                }
                Err(e) => {
                    warn!("✗ Translation failed: {}, keeping original", e);
                    sentence.translated_text = sentence.text.clone();
                }
            }
        }

        Ok(())
    }

    /// Update original transcription with translated sentences
    fn update_transcription_with_sentences(
        &self,
        transcription: &mut Transcription,
        sentences: &[ReconstructedSentence],
    ) {
        if sentences.len() == transcription.segments.len() {
            // We fell back to individual segments, update each segment directly
            for (idx, sentence) in sentences.iter().enumerate() {
                if idx < transcription.segments.len() {
                    transcription.segments[idx].text = sentence.translated_text.clone();
                }
            }
        } else {
            // We have reconstructed sentences, replace the entire transcription
            // Create new segments from the reconstructed sentences
            let mut new_segments = Vec::new();
            let segment_duration = if !transcription.segments.is_empty() {
                (transcription.segments.last().unwrap().end - transcription.segments.first().unwrap().start) 
                / transcription.segments.len() as f64
            } else {
                1.0
            };

            for (idx, sentence) in sentences.iter().enumerate() {
                let start_time = idx as f64 * segment_duration;
                let end_time = start_time + segment_duration;
                
                new_segments.push(TranscriptionSegment {
                    id: idx as i32,
                    start: start_time,
                    end: end_time,
                    text: sentence.translated_text.clone(),
                    tokens: vec![],
                    temperature: 0.0,
                    avg_logprob: 0.0,
                    compression_ratio: 1.0,
                    no_speech_prob: 0.0,
                });
            }
            
            transcription.segments = new_segments;
            transcription.text = sentences
                .iter()
                .map(|s| s.translated_text.as_str())
                .collect::<Vec<_>>()
                .join(" ");
        }
    }
}

#[async_trait]
impl Translator for LlmTranslator {
    /// LLM-based translation with sliding window sentence analysis
    async fn translate_transcription(
        &mut self,
        transcription: &mut Transcription,
        target_language: &str,
        _context: Option<&str>,
    ) -> Result<()> {
        info!("Starting LLM translation with sliding window analysis to {}", target_language);
        
        let total_segments = transcription.segments.len();
        if total_segments == 0 {
            return Ok(());
        }

        info!("Step 1: Analyzing {} segments with sliding window (size: {})", 
              total_segments, self.window_size);

        // Step 1: Analyze all segments with sliding window to get sentence candidates
        let sentence_candidates = self.analyze_all_segments(&transcription.segments).await?;
        
        // Step 2: Filter by confidence threshold
        info!("Step 2: Evaluating {} sentence candidates", sentence_candidates.len());
        let final_sentences = self.finalize_sentences(sentence_candidates);
        
        info!("Selected {} high-confidence sentences for translation", final_sentences.len());

        // Step 3: Convert to reconstructed sentences
        info!("Step 3: Creating sentences for translation");
        let mut sentences = self.reconstruct_sentences_from_candidates(&transcription.segments, final_sentences);
        
        info!("Reconstructed {} sentences", sentences.len());

        // Step 4: Translate sentences
        info!("Step 4: Translating reconstructed sentences");
        self.translate_sentences(&mut sentences, target_language).await?;

        // Step 5: Update original transcription
        info!("Step 5: Updating transcription with translated sentences");
        self.update_transcription_with_sentences(transcription, &sentences);

        // Final summary
        let successfully_translated = sentences.iter()
            .filter(|s| s.translated_text != s.text)
            .count();
        let kept_original = sentences.len() - successfully_translated;
        
        info!("");
        info!("✅ LLM TRANSLATION COMPLETED");
        info!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        info!("📊 Final Results:");
        info!("   • Original segments: {}", total_segments);
        info!("   • Reconstructed sentences: {}", sentences.len());
        info!("   • Successfully translated: {}", successfully_translated);
        info!("   • Kept original (failed): {}", kept_original);
        info!("   • Translation success rate: {:.1}%", 
              (successfully_translated as f64 / sentences.len() as f64) * 100.0);
        info!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        info!("");
        
        Ok(())
    }
} 