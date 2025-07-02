use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use crate::error::{Result, ShuroError};

// Structs for parsing whisper.cpp JSON output
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WhisperCppOutput {
    pub result: WhisperCppResult,
    pub transcription: Vec<WhisperCppSegment>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WhisperCppResult {
    pub language: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WhisperCppSegment {
    pub timestamps: WhisperCppTimestamps,
    pub offsets: WhisperCppOffsets,
    pub text: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WhisperCppTimestamps {
    pub from: String,
    pub to: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WhisperCppOffsets {
    pub from: i64,
    pub to: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscriptionQuality {
    pub tokens_max_segment: i32,
    pub repetitive_segments: f64,
    pub hallucination_periods: Vec<HallucinationPeriod>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HallucinationPeriod {
    pub start: f64,
    pub end: f64,
    pub confidence: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscriptionSegment {
    pub id: i32,
    pub start: f64,
    pub end: f64,
    pub text: String,
    pub tokens: Vec<i32>,
    pub temperature: f32,
    pub avg_logprob: f32,
    pub compression_ratio: f32,
    pub no_speech_prob: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Transcription {
    pub text: String,
    pub segments: Vec<TranscriptionSegment>,
    pub language: String,
}

impl TranscriptionQuality {
    pub fn score(&self) -> f64 {
        let mut score = 0.0;
        
        // Factor in token count (higher = worse)
        let token_penalty = (self.tokens_max_segment as f64) / 100.0;
        score += token_penalty;
        
        // Factor in repetitive segments (higher = worse)
        score += self.repetitive_segments;
        
        // Factor in hallucinations (count * severity)
        let hallucination_penalty = self.hallucination_periods.len() as f64 * 0.5;
        score += hallucination_penalty;
        
        score
    }

    pub fn has_hallucinations(&self) -> bool {
        !self.hallucination_periods.is_empty()
    }
}

impl From<WhisperCppOutput> for Transcription {
    fn from(whisper_output: WhisperCppOutput) -> Self {
        let language = whisper_output.result.language;
        
        // Combine all text segments
        let text = whisper_output.transcription
            .iter()
            .map(|seg| seg.text.trim())
            .collect::<Vec<_>>()
            .join(" ");
        
        // Convert segments
        let segments: Vec<TranscriptionSegment> = whisper_output.transcription
            .into_iter()
            .enumerate()
            .map(|(id, seg)| {
                let start = seg.offsets.from as f64 / 1000.0; // Convert ms to seconds
                let end = seg.offsets.to as f64 / 1000.0;
                
                TranscriptionSegment {
                    id: id as i32,
                    start,
                    end,
                    text: seg.text.trim().to_string(),
                    tokens: vec![], // Not available in whisper.cpp output
                    temperature: 0.0, // Not available in whisper.cpp output
                    avg_logprob: 0.0, // Not available in whisper.cpp output
                    compression_ratio: 1.0, // Default safe value
                    no_speech_prob: 0.0, // Default safe value
                }
            })
            .collect();
        
        Transcription {
            text,
            segments,
            language,
        }
    }
}

impl Transcription {
    pub fn quality(&self) -> TranscriptionQuality {
        let mut tokens_max = 0;
        let mut text_counts: HashMap<String, i32> = HashMap::new();
        
        // Analyze segments for quality metrics
        for segment in &self.segments {
            if segment.tokens.len() > tokens_max {
                tokens_max = segment.tokens.len();
            }
            
            // Count text repetitions
            let normalized_text = segment.text.trim().to_lowercase();
            if !normalized_text.is_empty() {
                *text_counts.entry(normalized_text).or_insert(0) += 1;
            }
        }
        
        // Calculate repetitive segment ratio
        let total_segments = self.segments.len() as f64;
        let repetitive_count = text_counts.values()
            .filter(|&&count| count > 1)
            .map(|&count| count - 1) // Only count duplicates
            .sum::<i32>() as f64;
        
        let repetitive_segments = if total_segments > 0.0 {
            repetitive_count / total_segments
        } else {
            0.0
        };
        
        // Detect potential hallucinations based on no_speech_prob and compression_ratio
        let hallucination_periods = self.detect_hallucinations();
        
        TranscriptionQuality {
            tokens_max_segment: tokens_max as i32,
            repetitive_segments,
            hallucination_periods,
        }
    }

    fn detect_hallucinations(&self) -> Vec<HallucinationPeriod> {
        let mut periods = Vec::new();
        
        for segment in &self.segments {
            // High no_speech_prob with low compression_ratio might indicate hallucination
            if segment.no_speech_prob > 0.8 && segment.compression_ratio < 1.5 {
                periods.push(HallucinationPeriod {
                    start: segment.start,
                    end: segment.end,
                    confidence: segment.no_speech_prob as f64,
                });
            }
            
            // Very high compression ratio might also indicate repetitive/hallucinatory content
            if segment.compression_ratio > 3.0 {
                periods.push(HallucinationPeriod {
                    start: segment.start,
                    end: segment.end,
                    confidence: (segment.compression_ratio - 3.0) as f64 / 10.0,
                });
            }
        }
        
        periods
    }
}

pub struct QualityValidator {
    repetitive_threshold: f64,
    max_tokens_threshold: f64,
    min_quality_score: f64,
}

impl QualityValidator {
    pub fn new(repetitive_threshold: f64, max_tokens_threshold: f64, min_quality_score: f64) -> Self {
        Self {
            repetitive_threshold,
            max_tokens_threshold,
            min_quality_score,
        }
    }

    pub fn validate_transcription(&self, transcription: &Transcription) -> Result<()> {
        let quality = transcription.quality();
        
        if quality.has_hallucinations() {
            return Err(ShuroError::Hallucination);
        }
        
        if quality.repetitive_segments > self.repetitive_threshold {
            return Err(ShuroError::Quality(format!(
                "Too many repetitive segments: {} > {}",
                quality.repetitive_segments, self.repetitive_threshold
            )));
        }
        
        if quality.tokens_max_segment as f64 > self.max_tokens_threshold {
            return Err(ShuroError::Quality(format!(
                "Segment too long: {} tokens > {}",
                quality.tokens_max_segment, self.max_tokens_threshold
            )));
        }
        
        if quality.score() > self.min_quality_score {
            return Err(ShuroError::Quality(format!(
                "Quality score too low: {} > {}",
                quality.score(), self.min_quality_score
            )));
        }
        
        Ok(())
    }
} 