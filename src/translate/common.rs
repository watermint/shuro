use std::collections::HashMap;
use std::time::Duration;
use std::path::PathBuf;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::json;
use tracing::{info, warn, debug};

use crate::config::TranslateConfig;
use crate::error::{Result, ShuroError};

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

/// Base translator with common functionality
pub struct BaseTranslator {
    pub client: Client,
    pub config: TranslateConfig,
    pub cache: HashMap<String, String>,
    pub cache_dir: PathBuf,
}

impl BaseTranslator {
    pub fn new(config: TranslateConfig) -> Self {
        let client = Client::builder()
            .timeout(Duration::from_secs(300)) // 5 minute timeout
            .build()
            .expect("HTTP client creation should not fail");

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

    /// Perform the actual translation using Ollama with JSON format
    pub async fn translate_text(
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

    /// Evaluate translation quality using structured evaluation
    pub async fn evaluate_translation_quality(
        &self,
        original: &str,
        translation: &str,
        context: &str,
        target_language: &str,
        source_language: &str,
    ) -> Result<TranslationQuality> {
        let target_language_name = self.language_code_to_name(target_language);
        let source_language_name = self.language_code_to_name(source_language);
        
        let quality_prompt = format!(
            "You are a professional translation quality evaluator.\n\
             \n\
             Evaluate the translation quality from {} to {} ({}).\n\
             \n\
             IMPORTANT CRITERIA:\n\
             1. The translation must be in {} language ONLY\n\
             2. The translation must accurately convey the meaning of the source text\n\
             3. The translation must be grammatically correct in {}\n\
             4. The translation must be natural and fluent in {}\n\
             \n\
             Evaluate translation quality in one of the following levels:\n\
             - [PERFECT]: The translation is perfect, in correct language, and no further improvement is needed.\n\
             - [GOOD]: The translation is good and in correct language, but some minor improvements are needed.\n\
             - [BAD]: The translation is bad, incorrect, or needs to be re-translated.\n\
             - [INVALID]: The translation is in wrong language, invalid, or not related to the source.\n\
             \n\
             Please return the evaluation results in JSON format as {{\"evaluation\":\"evaluation result\"}}.\n\
             \n\
             [Source ({})]\n\
             {}\n\
             \n\
             [Translation (should be in {})]\n\
             {}\n\
             \n\
             [Context]\n\
             {}",
            source_language_name, target_language_name, target_language, 
            target_language_name, target_language_name, target_language_name, 
            source_language_name, original, target_language_name, translation, context
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

    /// Build translation prompt with context, using JSON format
    fn build_translation_prompt(&self, text: &str, target_language: &str, context: Option<&str>) -> String {
        let language_name = self.language_code_to_name(target_language);
        
        if text.len() < 50 {
            format!(
                "You are a professional translator.\n\
                 \n\
                 CRITICAL: You must translate the text to {} ONLY. Do not translate to any other language.\n\
                 The target language is: {} (language code: {})\n\
                 \n\
                 Return ONLY the translation in JSON format as {{\"text\":\"your {} translation here\"}}.\n\
                 Do not include any explanations, alternatives, or text in other languages.\n\
                 \n\
                 Text to translate: \"{}\"\n",
                language_name, language_name, target_language, language_name, text
            )
        } else {
            let mut prompt = format!(
                "You are a professional translator.\n\
                 \n\
                 CRITICAL: You must translate the text to {} ONLY. Do not translate to any other language.\n\
                 The target language is: {} (language code: {})\n\
                 \n\
                 Return ONLY the translation in JSON format as {{\"text\":\"your {} translation here\"}}.\n\
                 Do not include any explanations, alternatives, or text in other languages.\n\
                 \n\
                 [Text to translate]\n\
                 {}\n\
                 \n",
                language_name, language_name, target_language, language_name, text
            );
            
            if let Some(ctx) = context {
                if !ctx.trim().is_empty() {
                    prompt.push_str(&format!(
                        "[Context for reference - DO NOT translate this part]\n\
                         {}\n\n\
                         Remember: Only translate the text in the [Text to translate] section above to {}.\n",
                        ctx, language_name
                    ));
                }
            }
            
            prompt
        }
    }

    /// Convert language code to full language name for clearer prompts
    fn language_code_to_name(&self, code: &str) -> String {
        match code.to_lowercase().as_str() {
            "ja" => "Japanese".to_string(),
            "ko" => "Korean".to_string(), 
            "zh" => "Chinese".to_string(),
            "fr" => "French".to_string(),
            "de" => "German".to_string(),
            "es" => "Spanish".to_string(),
            "ru" => "Russian".to_string(),
            "it" => "Italian".to_string(),
            "pt" => "Portuguese".to_string(),
            "pl" => "Polish".to_string(),
            "nl" => "Dutch".to_string(),
            "tr" => "Turkish".to_string(),
            "ar" => "Arabic".to_string(),
            "hi" => "Hindi".to_string(),
            "th" => "Thai".to_string(),
            "vi" => "Vietnamese".to_string(),
            "sv" => "Swedish".to_string(),
            "da" => "Danish".to_string(),
            "no" => "Norwegian".to_string(),
            "fi" => "Finnish".to_string(),
            "he" => "Hebrew".to_string(),
            "hu" => "Hungarian".to_string(),
            "cs" => "Czech".to_string(),
            "sk" => "Slovak".to_string(),
            "bg" => "Bulgarian".to_string(),
            "hr" => "Croatian".to_string(),
            "sl" => "Slovenian".to_string(),
            "et" => "Estonian".to_string(),
            "lv" => "Latvian".to_string(),
            "lt" => "Lithuanian".to_string(),
            "mt" => "Maltese".to_string(),
            "ga" => "Irish".to_string(),
            "cy" => "Welsh".to_string(),
            "eu" => "Basque".to_string(),
            "ca" => "Catalan".to_string(),
            "gl" => "Galician".to_string(),
            "is" => "Icelandic".to_string(),
            "mk" => "Macedonian".to_string(),
            "sq" => "Albanian".to_string(),
            "be" => "Belarusian".to_string(),
            "uk" => "Ukrainian".to_string(),
            "az" => "Azerbaijani".to_string(),
            "kk" => "Kazakh".to_string(),
            "ky" => "Kyrgyz".to_string(),
            "uz" => "Uzbek".to_string(),
            "tg" => "Tajik".to_string(),
            "am" => "Amharic".to_string(),
            "ka" => "Georgian".to_string(),
            "hy" => "Armenian".to_string(),
            "ne" => "Nepali".to_string(),
            "si" => "Sinhala".to_string(),
            "my" => "Burmese".to_string(),
            "km" => "Khmer".to_string(),
            "lo" => "Lao".to_string(),
            "gu" => "Gujarati".to_string(),
            "pa" => "Punjabi".to_string(),
            "ta" => "Tamil".to_string(),
            "te" => "Telugu".to_string(),
            "kn" => "Kannada".to_string(),
            "ml" => "Malayalam".to_string(),
            "bn" => "Bengali".to_string(),
            "as" => "Assamese".to_string(),
            "or" => "Odia".to_string(),
            "mr" => "Marathi".to_string(),
            "en" => "English".to_string(),
            _ => code.to_string(), // Fallback to the code itself if not found
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
    pub fn generate_cache_key(&self, source_text: &str, target_language: &str, context: &str) -> String {
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
    pub async fn load_from_persistent_cache(&self, cache_key: &str) -> Result<Option<String>> {
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
    pub async fn save_to_persistent_cache(
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