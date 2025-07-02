use std::path::Path;
use tokio::fs;
use tracing::{info, warn};
use walkdir::WalkDir;

use crate::config::Config;
use crate::error::{Result, ShuroError};
use crate::whisper::{WhisperTranscriberTrait, WhisperTranscriberFactory};
use crate::translate::{TranslatorFactory, check_ollama_availability};
use crate::subtitle::generate_srt;
use crate::media::{MediaProcessorTrait, MediaProcessorFactory};
use crate::quality::QualityValidator;

pub struct Workflow {
    config: Config,
    whisper: Box<dyn WhisperTranscriberTrait>,
    media: Box<dyn MediaProcessorTrait>,
}

impl Workflow {
    pub fn new(config: Config) -> Result<Self> {
        let validator = QualityValidator::new(
            config.quality.repetitive_segment_threshold,
            config.quality.max_tokens_threshold,
            config.quality.min_quality_score,
        );
        
        let whisper = WhisperTranscriberFactory::create_default(config.whisper.clone(), validator);
        let media = MediaProcessorFactory::create_processor(config.media.clone());

        // Check dependencies
        media.check_availability()?;

        Ok(Self {
            config,
            whisper,
            media,
        })
    }

    /// Process a single video file with subtitle translation
    pub async fn process_single_file<P: AsRef<Path>, Q: AsRef<Path>>(
        &self,
        input_path: P,
        target_languages: &[String],
        output_dir: Option<Q>,
    ) -> Result<()> {
        let input_path = input_path.as_ref();
        info!("Processing single file: {}", input_path.display());

        // Validate input file
        if !input_path.exists() {
            return Err(ShuroError::FileNotFound(input_path.display().to_string()));
        }

        // Determine output directory
        let output_dir = match output_dir {
            Some(dir) => dir.as_ref().to_path_buf(),
            None => input_path.parent()
                .ok_or_else(|| ShuroError::Config("Cannot determine output directory".to_string()))?
                .to_path_buf(),
        };

        // Create output directory if it doesn't exist
        fs::create_dir_all(&output_dir).await?;

        // Process the file
        self.process_video_file(input_path, &output_dir, target_languages).await
    }

    /// Process all video files in a directory
    pub async fn process_directory<P: AsRef<Path>, Q: AsRef<Path>>(
        &self,
        input_dir: P,
        target_languages: &[String],
        output_dir: Option<Q>,
    ) -> Result<()> {
        let input_dir = input_dir.as_ref();
        info!("Processing directory: {}", input_dir.display());

        if !input_dir.is_dir() {
            return Err(ShuroError::Config("Input path is not a directory".to_string()));
        }

        // Determine output directory
        let output_dir = match output_dir {
            Some(dir) => dir.as_ref().to_path_buf(),
            None => input_dir.to_path_buf(),
        };

        // Create output directory if it doesn't exist
        fs::create_dir_all(&output_dir).await?;

        // Find video files
        let video_extensions = ["mp4", "avi", "mov", "mkv", "wmv", "flv", "webm"];
        let mut video_files = Vec::new();

        for entry in WalkDir::new(input_dir).into_iter().filter_map(|e| e.ok()) {
            if let Some(extension) = entry.path().extension() {
                if let Some(ext_str) = extension.to_str() {
                    if video_extensions.contains(&ext_str.to_lowercase().as_str()) {
                        video_files.push(entry.path().to_path_buf());
                    }
                }
            }
        }

        info!("Found {} video files to process", video_files.len());

        // Process each video file
        for video_path in video_files {
            match self.process_video_file(&video_path, &output_dir, target_languages).await {
                Ok(_) => info!("Successfully processed: {}", video_path.display()),
                Err(e) => warn!("Failed to process {}: {}", video_path.display(), e),
            }
        }

        Ok(())
    }

    async fn process_video_file<P: AsRef<Path>>(
        &self,
        video_path: P,
        output_dir: P,
        target_languages: &[String],
    ) -> Result<()> {
        let video_path = video_path.as_ref();
        let output_dir = output_dir.as_ref();

        let video_stem = video_path.file_stem()
            .ok_or_else(|| ShuroError::Config("Invalid video filename".to_string()))?
            .to_string_lossy();

        // Step 1: Get or extract audio (with caching)
        let audio_path = match self.whisper.get_cached_audio(video_path).await? {
            Some(cached_path) => {
                info!("Using cached audio file");
                cached_path
            }
            None => {
                info!("Extracting audio from video directly to cache");
                self.whisper.extract_and_cache_audio(video_path).await?
            }
        };

        // Step 2: Transcribe with tuning
        info!("Starting transcription with hallucination detection and tempo tuning");
        let tune_result = self.whisper.tune_transcription(&audio_path).await?;
        let transcription = tune_result.best_transcription;

        info!("Transcription completed with quality score: {}", tune_result.quality_score);

        // Step 3: Translate for each target language
        for target_lang in target_languages {
            info!("Translating to {}", target_lang);

            // Check Ollama availability
            check_ollama_availability(&self.config.translate.endpoint, &self.config.translate.model).await?;

            // Create translator and translate
            let mut translator = TranslatorFactory::create_translator(self.config.translate.clone());
            let mut transcription_copy = transcription.clone();
            
            translator.translate_transcription(&mut transcription_copy, target_lang, None).await?;

            // Step 4: Generate SRT file
            let srt_path = output_dir.join(format!("{}_{}.srt", video_stem, target_lang));
            generate_srt(&transcription_copy, &srt_path).await?;

            // Step 5: Embed subtitles into video
            let output_video_path = output_dir.join(format!("{}_{}.mp4", video_stem, target_lang));
            self.media.embed_subtitles(video_path, &srt_path, &output_video_path).await?;

            info!("Completed processing for language: {}", target_lang);
        }

        // Note: We don't clean up the cached audio file - it stays in cache for reuse

        Ok(())
    }

    /// Extract audio from video file
    pub async fn extract_audio<P: AsRef<Path>>(
        &self,
        video_path: P,
        audio_path: P,
    ) -> Result<()> {
        let video_path = video_path.as_ref();
        let audio_path = audio_path.as_ref();
        
        // Check if we have cached audio first
        if let Ok(Some(cached_path)) = self.whisper.get_cached_audio(video_path).await {
            info!("Using cached audio file, copying to requested location");
            fs::copy(&cached_path, audio_path).await?;
            return Ok(());
        }
        
        // Extract audio normally
        self.media.extract_audio(video_path, audio_path).await?;
        
        // Note: Audio caching is handled by extract_and_cache_audio method when needed
        
        Ok(())
    }

    /// Transcribe audio file to text
    pub async fn transcribe_audio<P: AsRef<Path>>(
        &self,
        audio_path: P,
        output_path: P,
        language: Option<&str>,
    ) -> Result<()> {
        let audio_path = audio_path.as_ref();
        let output_path = output_path.as_ref();
        
        let transcription = self.whisper.transcribe(audio_path, language).await?;
        
        // Generate SRT file
        generate_srt(&transcription, output_path).await?;
        
        Ok(())
    }

    /// Translate subtitle file to multiple languages
    pub async fn translate_subtitles<P: AsRef<Path>>(
        &self,
        _input_path: P,
        _output_path: P,
        target_languages: &[String],
    ) -> Result<()> {
        // For now, this is a placeholder
        // In a full implementation, we'd need to:
        // 1. Read the SRT file
        // 2. Convert it to a Transcription
        // 3. Translate it
        // 4. Write back to SRT
        
        for _target_lang in target_languages {
            let _translator = TranslatorFactory::create_translator(self.config.translate.clone());
            // TODO: Implement subtitle file reading/writing
            // translator.translate_transcription(&mut transcription, target_lang, None).await?;
        }
        
        Ok(())
    }

    /// Embed subtitles into video file
    pub async fn embed_subtitles<P: AsRef<Path>>(
        &self,
        video_path: P,
        subtitles_path: P,
        output_path: P,
    ) -> Result<()> {
        let video_path = video_path.as_ref();
        let subtitles_path = subtitles_path.as_ref();
        let output_path = output_path.as_ref();
        
        self.media.embed_subtitles(video_path, subtitles_path, output_path).await
    }
} 