//! Shuro - Automated Subtitle Translation Workflow
//! 
//! This is the main entry point for the Shuro application, which provides
//! an automated workflow for adding translated subtitles to movie files
//! using whisper-cpp, ollama, and ffmpeg.

use anyhow::Result;
use clap::Parser;
use tracing::{info, Level};
use tracing_subscriber::FmtSubscriber;
use std::env;

use shuro::cli::{Args, Commands, CacheAction};
use shuro::config::{Config, TranslationMode};
use shuro::setup::SetupManager;
use shuro::workflow::Workflow;
use shuro::quality::QualityValidator;
use shuro::whisper::WhisperTranscriber;
use shuro::translate::OllamaTranslator;
use shuro::error::ShuroError;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging (only if not already set)
    let _ = FmtSubscriber::builder()
        .with_max_level(Level::INFO)
        .with_target(false)
        .with_thread_ids(true)
        .with_thread_names(true)
        .with_file(true)
        .with_line_number(true)
        .try_init();

    info!("Starting Shuro - Automated Subtitle Translation Workflow");

    // Parse command line arguments
    let args = Args::parse();

    // Set log level based on verbosity
    if args.verbose {
        // Only set if no global subscriber is already set
        if tracing::subscriber::set_global_default(
            FmtSubscriber::builder()
                .with_max_level(Level::DEBUG)
                .with_target(false)
                .with_thread_ids(true)
                .with_thread_names(true)
                .with_file(true)
                .with_line_number(true)
                .finish(),
        ).is_err() {
            eprintln!("Warning: Global tracing subscriber already set, using existing configuration");
        }
    }

    // Load configuration
    let mut config = match &args.config {
        Some(config_path) => Config::from_file(config_path)?,
        None => {
            // Try to load config.toml from current directory first
            if std::path::Path::new("config.toml").exists() {
                info!("Found config.toml in current directory, loading...");
                Config::from_file("config.toml")?
            } else {
                Config::default()
            }
        }
    };

    // Initialize setup manager and ensure all necessary files are available
    info!("Checking and downloading necessary files...");
    let setup_manager = SetupManager::new()?;
    info!("Created setup manager, now initializing...");
    setup_manager.initialize(&mut config).await?;
    info!("Setup manager initialization completed");

    // Create workflow instance
    let workflow = Workflow::new(config.clone())?;

    // Execute command
    match args.command {
        Commands::Models { download } => {
            info!("Listing available whisper models...");
            
            let models = setup_manager.get_available_models();
            println!("\nAvailable Whisper Models:");
            println!("{:<15} {:<20} {:<10} {:<10}", "Name", "Filename", "Size (MB)", "Status");
            println!("{}", "-".repeat(65));
            
            let models_dir = setup_manager.shuro_dir().join("models");
            for model in &models {
                let local_path = models_dir.join(&model.filename);
                let status = if local_path.exists() {
                    "Downloaded"
                } else {
                    "Missing"
                };
                
                println!("{:<15} {:<20} {:<10.1} {:<10}", 
                    model.name, model.filename, model.size_mb, status);
            }
            
            if download {
                info!("Downloading all missing models...");
                for model in &models {
                    let local_path = models_dir.join(&model.filename);
                    if !local_path.exists() {
                        setup_manager.download_model(&model).await?;
                    }
                }
                info!("All models downloaded successfully");
            }
        }
        Commands::Cache { action } => {
            info!("Managing transcription cache...");
            
            // Create a temporary transcriber to manage cache
            let validator = QualityValidator::new(0.1, 100.0, 5.0);
            let transcriber = WhisperTranscriber::new(config.whisper.clone(), validator);
            
            match action {
                CacheAction::List => {
                    let transcription_items = transcriber.list_cache().await?;
                    let audio_items = transcriber.list_audio_cache().await?;
                    
                    if transcription_items.is_empty() && audio_items.is_empty() {
                        println!("No cached files found.");
                    } else {
                        if !transcription_items.is_empty() {
                            println!("\nCached Transcriptions:");
                            println!("{:<15} {:<20} {:<15} {:<50}", "Model", "Language", "Cached", "Audio File");
                            println!("{}", "-".repeat(100));
                            
                            for item in transcription_items {
                                let cached_ago = std::time::SystemTime::now()
                                    .duration_since(std::time::UNIX_EPOCH)
                                    .unwrap_or_default()
                                    .as_secs()
                                    .saturating_sub(item.cached_at);
                                
                                let language = item.language.unwrap_or_else(|| "auto".to_string());
                                let audio_file = std::path::Path::new(&item.audio_path)
                                    .file_name()
                                    .unwrap_or_default()
                                    .to_string_lossy();
                                
                                println!("{:<15} {:<20} {:<15} {:<50}", 
                                    item.model, 
                                    language, 
                                    format_duration(cached_ago),
                                    audio_file
                                );
                            }
                        }
                        
                        if !audio_items.is_empty() {
                            println!("\nCached Audio Files:");
                            println!("{:<15} {:<60}", "Cached", "Video File");
                            println!("{}", "-".repeat(75));
                            
                            for item in audio_items {
                                let cached_ago = std::time::SystemTime::now()
                                    .duration_since(std::time::UNIX_EPOCH)
                                    .unwrap_or_default()
                                    .as_secs()
                                    .saturating_sub(item.cached_at);
                                
                                let video_file = std::path::Path::new(&item.video_path)
                                    .file_name()
                                    .unwrap_or_default()
                                    .to_string_lossy();
                                
                                println!("{:<15} {:<60}", 
                                    format_duration(cached_ago),
                                    video_file
                                );
                            }
                        }
                    }
                }
                CacheAction::Clear => {
                    let transcription_count = transcriber.clear_cache().await?;
                    let audio_count = transcriber.clear_audio_cache().await?;
                    
                    // Also clear translation cache
                    let translator = OllamaTranslator::new(config.translate.clone());
                    let translation_count = translator.clear_translation_cache().await?;
                    
                    println!("Cleared {} cached transcriptions, {} cached audio files, and {} cached translations", 
                             transcription_count, audio_count, translation_count);
                }
                CacheAction::Info => {
                    let info = transcriber.cache_info().await?;
                    
                    // Get translation cache info
                    let translator = OllamaTranslator::new(config.translate.clone());
                    let translation_items = translator.list_translation_cache().await?;
                    let translation_count = translation_items.len();
                    
                    println!("\nCache Statistics:");
                    println!("Transcription files: {}", info.total_files);
                    println!("Transcription size: {:.2} MB", info.total_size as f64 / 1024.0 / 1024.0);
                    println!("Audio files: {}", info.audio_files);
                    println!("Audio size: {:.2} MB", info.audio_size as f64 / 1024.0 / 1024.0);
                    println!("Translation files: {}", translation_count);
                    println!("Total cache size: {:.2} MB", 
                             (info.total_size + info.audio_size) as f64 / 1024.0 / 1024.0);
                    
                    if let Some(oldest) = info.oldest_entry {
                        let oldest_ago = std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .unwrap_or_default()
                            .as_secs()
                            .saturating_sub(oldest);
                        println!("Oldest entry: {} ago", format_duration(oldest_ago));
                    }
                    
                    if let Some(newest) = info.newest_entry {
                        let newest_ago = std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .unwrap_or_default()
                            .as_secs()
                            .saturating_sub(newest);
                        println!("Newest entry: {} ago", format_duration(newest_ago));
                    }
                    
                    if !info.models_used.is_empty() {
                        println!("Models used: {}", info.models_used.join(", "));
                    }
                }
                CacheAction::Clean { days } => {
                    let transcription_count = transcriber.clean_cache(days).await?;
                    let audio_count = transcriber.clean_audio_cache(days).await?;
                    
                    // Clean translation cache - remove entries older than specified days
                    let translator = OllamaTranslator::new(config.translate.clone());
                    let translation_items = translator.list_translation_cache().await?;
                    let cutoff_time = std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_secs()
                        .saturating_sub(days * 24 * 60 * 60);
                    
                    let mut translation_clean_count = 0;
                    for item in translation_items {
                        if item.cached_at < cutoff_time {
                            translation_clean_count += 1;
                        }
                    }
                    
                    // For now, we'll just show the count. In a full implementation,
                    // we'd need a method to clean by age
                    if translation_clean_count > 0 {
                        println!("Note: {} old translation cache entries found (cleaning by age not yet implemented)", 
                                translation_clean_count);
                    }
                    
                    println!("Cleaned {} old cached transcriptions and {} old cached audio files (older than {} days)", 
                             transcription_count, audio_count, days);
                }
                CacheAction::ListTranscriptions => {
                    let cached_items = transcriber.list_cache().await?;
                    
                    if cached_items.is_empty() {
                        println!("No cached transcriptions found.");
                    } else {
                        println!("\nCached Transcriptions:");
                        println!("{:<15} {:<20} {:<15} {:<50}", "Model", "Language", "Cached", "Audio File");
                        println!("{}", "-".repeat(100));
                        
                        for item in cached_items {
                            let cached_ago = std::time::SystemTime::now()
                                .duration_since(std::time::UNIX_EPOCH)
                                .unwrap_or_default()
                                .as_secs()
                                .saturating_sub(item.cached_at);
                            
                            let language = item.language.unwrap_or_else(|| "auto".to_string());
                            let audio_file = std::path::Path::new(&item.audio_path)
                                .file_name()
                                .unwrap_or_default()
                                .to_string_lossy();
                            
                            println!("{:<15} {:<20} {:<15} {:<50}", 
                                item.model, 
                                language, 
                                format_duration(cached_ago),
                                audio_file
                            );
                        }
                    }
                }
                CacheAction::ListAudio => {
                    let cached_items = transcriber.list_audio_cache().await?;
                    
                    if cached_items.is_empty() {
                        println!("No cached audio files found.");
                    } else {
                        println!("\nCached Audio Files:");
                        println!("{:<15} {:<60}", "Cached", "Video File");
                        println!("{}", "-".repeat(75));
                        
                        for item in cached_items {
                            let cached_ago = std::time::SystemTime::now()
                                .duration_since(std::time::UNIX_EPOCH)
                                .unwrap_or_default()
                                .as_secs()
                                .saturating_sub(item.cached_at);
                            
                            let video_file = std::path::Path::new(&item.video_path)
                                .file_name()
                                .unwrap_or_default()
                                .to_string_lossy();
                            
                            println!("{:<15} {:<60}", 
                                format_duration(cached_ago),
                                video_file
                            );
                        }
                    }
                }
                CacheAction::ClearTranscriptions => {
                    let deleted_count = transcriber.clear_cache().await?;
                    println!("Cleared {} cached transcriptions", deleted_count);
                }
                CacheAction::ClearAudio => {
                    let deleted_count = transcriber.clear_audio_cache().await?;
                    println!("Cleared {} cached audio files", deleted_count);
                }
                CacheAction::ListTranslations => {
                    // Create translator instance to access translation cache
                    let translator = OllamaTranslator::new(config.translate.clone());
                    let cached_items = translator.list_translation_cache().await?;
                    
                    if cached_items.is_empty() {
                        println!("No cached translations found.");
                    } else {
                        println!("\nCached Translations:");
                        println!("{:<15} {:<10} {:<15} {:<8} {:<50}", "Model", "Language", "Cached", "Quality", "Source Text");
                        println!("{}", "-".repeat(100));
                        
                        for item in cached_items {
                            let cached_ago = std::time::SystemTime::now()
                                .duration_since(std::time::UNIX_EPOCH)
                                .unwrap_or_default()
                                .as_secs()
                                .saturating_sub(item.cached_at);
                            
                            let source_preview = if item.source_text.len() > 47 {
                                format!("{}...", &item.source_text[..47])
                            } else {
                                item.source_text.clone()
                            };
                            
                            println!("{:<15} {:<10} {:<15} {:<8} {:<50}", 
                                item.model, 
                                item.target_language, 
                                format_duration(cached_ago),
                                item.quality,
                                source_preview
                            );
                        }
                    }
                }
                CacheAction::ClearTranslations => {
                    // Create translator instance to access translation cache
                    let translator = OllamaTranslator::new(config.translate.clone());
                    let deleted_count = translator.clear_translation_cache().await?;
                    println!("Cleared {} cached translations", deleted_count);
                }
            }
        }
        Commands::Process { input, target_langs, output_dir, translation_mode } => {
            info!("Processing video file: {}", input.display());
            
            // Parse translation mode
            let mode = parse_translation_mode(&translation_mode)?;
            config.translate.mode = mode;
            
            let target_languages = target_langs
                .split(',')
                .map(|s| s.trim().to_string())
                .collect::<Vec<_>>();

            // Create new workflow with updated config
            let workflow = Workflow::new(config)?;
            workflow.process_single_file(&input, &target_languages, output_dir.as_ref()).await?;
        }
        Commands::Batch { input_dir, target_langs, output_dir, translation_mode } => {
            info!("Processing directory: {}", input_dir.display());
            
            // Parse translation mode
            let mode = parse_translation_mode(&translation_mode)?;
            config.translate.mode = mode;
            
            let target_languages = target_langs
                .split(',')
                .map(|s| s.trim().to_string())
                .collect::<Vec<_>>();

            // Create new workflow with updated config
            let workflow = Workflow::new(config)?;
            workflow.process_directory(&input_dir, &target_languages, output_dir.as_ref()).await?;
        }
        Commands::Extract { input, output } => {
            info!("Extracting audio from: {}", input.display());
            workflow.extract_audio(&input, &output).await?;
        }
        Commands::Transcribe { input, output, language } => {
            info!("Transcribing audio: {}", input.display());
            workflow.transcribe_audio(&input, &output, language.as_deref()).await?;
        }
        Commands::Translate { input, output, target_langs } => {
            info!("Translating subtitles: {}", input.display());
            
            let target_languages = target_langs
                .split(',')
                .map(|s| s.trim().to_string())
                .collect::<Vec<_>>();

            workflow.translate_subtitles(&input, &output, &target_languages).await?;
        }
        Commands::Embed { video, subtitles, output } => {
            info!("Embedding subtitles into video: {}", video.display());
            workflow.embed_subtitles(&video, &subtitles, &output).await?;
        }
    }

    info!("Shuro workflow completed successfully");
    Ok(())
}

/// Format duration in seconds to human readable string
fn format_duration(seconds: u64) -> String {
    if seconds < 60 {
        format!("{}s", seconds)
    } else if seconds < 3600 {
        format!("{}m {}s", seconds / 60, seconds % 60)
    } else {
        format!("{}h {}m", seconds / 3600, (seconds % 3600) / 60)
    }
}

/// Parse translation mode from string
fn parse_translation_mode(mode: &str) -> Result<TranslationMode> {
    match mode.to_lowercase().as_str() {
        "simple" => Ok(TranslationMode::Simple),
        "context" => Ok(TranslationMode::Context),
        "nlp" => Ok(TranslationMode::Nlp),
        _ => Err(ShuroError::Config(format!(
            "Invalid translation mode '{}'. Valid modes: simple, context, nlp", 
            mode
        )).into()),
    }
}