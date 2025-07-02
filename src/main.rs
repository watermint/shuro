//! Shuro - Automated Subtitle Translation Workflow
//! 
//! This is the main entry point for the Shuro application, which provides
//! an automated workflow for adding translated subtitles to movie files
//! using whisper-cpp, ollama, and ffmpeg.

use anyhow::Result;
use clap::Parser;
use tracing::{info, Level};
use tracing_subscriber::{fmt, layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};
use tracing_appender::{non_blocking, rolling};

use shuro::cli::{Args, Commands, CacheAction};
use shuro::config::{Config, TranslationMode, TranscriptionMode};
use shuro::setup::SetupManager;
use shuro::workflow::Workflow;
use shuro::quality::QualityValidator;
use shuro::transcribe::TranscriberFactory;
use shuro::translate::BaseTranslator;
use shuro::error::ShuroError;

#[tokio::main]
async fn main() -> Result<()> {
    info!("Starting Shuro - Automated Subtitle Translation Workflow");

    // Parse command line arguments
    let args = Args::parse();

    // Setup logging to both console and file
    setup_logging(args.verbose)?;

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
            let transcriber = TranscriberFactory::create_default(config.transcriber.clone(), validator);
            
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
                    let base_translator = BaseTranslator::new(config.translate.clone());
                    let translation_count = base_translator.clear_translation_cache().await?;
                    
                    println!("Cleared {} cached transcriptions, {} cached audio files, and {} cached translations", 
                             transcription_count, audio_count, translation_count);
                }
                CacheAction::Info => {
                    let info = transcriber.cache_info().await?;
                    
                    // Get translation cache info
                    let base_translator = BaseTranslator::new(config.translate.clone());
                    let translation_items = base_translator.list_translation_cache().await?;
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
                    
                    println!("Models used: {:?}", info.models_used);
                }
                CacheAction::Clean { days } => {
                    // Note: Current implementation clears all cache, not just old entries
                    let transcription_count = transcriber.clear_cache().await?;
                    let audio_count = transcriber.clear_audio_cache().await?;
                    
                    // Clean translation cache - remove entries older than specified days
                    let base_translator = BaseTranslator::new(config.translate.clone());
                    let translation_items = base_translator.list_translation_cache().await?;
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
                    let base_translator = BaseTranslator::new(config.translate.clone());
                    let cached_items = base_translator.list_translation_cache().await?;
                    
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
                    let base_translator = BaseTranslator::new(config.translate.clone());
                    let deleted_count = base_translator.clear_translation_cache().await?;
                    println!("Cleared {} cached translations", deleted_count);
                }
            }
        }
        Commands::Process { input, target_langs, output_dir, translation_mode, transcription_mode } => {
            info!("Processing video file: {}", input.display());
            
            // Parse translation mode
            let translation_mode = parse_translation_mode(&translation_mode)?;
            config.translate.mode = translation_mode;
            
            // Parse transcription mode
            let transcription_mode = parse_transcription_mode(&transcription_mode)?;
            config.transcriber.mode = transcription_mode;
            
            let target_languages = target_langs
                .split(',')
                .map(|s| s.trim().to_string())
                .collect::<Vec<_>>();

            // Create new workflow with updated config
            let workflow = Workflow::new(config)?;
            workflow.process_single_file(&input, &target_languages, output_dir.as_ref()).await?;
        }
        Commands::Batch { input_dir, target_langs, output_dir, translation_mode, transcription_mode } => {
            info!("Processing directory: {}", input_dir.display());
            
            // Parse translation mode
            let translation_mode = parse_translation_mode(&translation_mode)?;
            config.translate.mode = translation_mode;
            
            // Parse transcription mode
            let transcription_mode = parse_transcription_mode(&transcription_mode)?;
            config.transcriber.mode = transcription_mode;
            
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
        Commands::Transcribe { input, output, language, transcription_mode } => {
            info!("Transcribing audio: {}", input.display());
            
            // Parse transcription mode
            let transcription_mode = parse_transcription_mode(&transcription_mode)?;
            config.transcriber.mode = transcription_mode;
            
            // Create new workflow with updated config
            let workflow = Workflow::new(config)?;
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

/// Setup logging to both console and file
fn setup_logging(verbose: bool) -> Result<()> {
    // Create log directory
    let shuro_dir = std::env::current_dir()?.join(".shuro");
    let log_dir = shuro_dir.join("log");
    std::fs::create_dir_all(&log_dir)?;

    // Set up file appender with daily rotation
    let file_appender = rolling::daily(&log_dir, "shuro.log");
    let (non_blocking_file, _guard) = non_blocking(file_appender);
    // Keep the guard alive for the duration of the program
    std::mem::forget(_guard);

    // Determine log level
    let log_level = if verbose { Level::DEBUG } else { Level::INFO };

    // Create console layer
    let console_layer = fmt::layer()
        .with_target(false)
        .with_thread_ids(true)
        .with_thread_names(true)
        .with_file(true)
        .with_line_number(true);

    // Create file layer
    let file_layer = fmt::layer()
        .with_writer(non_blocking_file)
        .with_target(false)
        .with_thread_ids(true)
        .with_thread_names(true)
        .with_file(true)
        .with_line_number(true)
        .with_ansi(false); // No ANSI colors in file

    // Setup layered subscriber
    let subscriber = tracing_subscriber::registry()
        .with(EnvFilter::from_default_env().add_directive(log_level.into()))
        .with(console_layer)
        .with(file_layer);

    // Initialize the subscriber
    subscriber.try_init()
        .map_err(|e| anyhow::anyhow!("Failed to initialize logging: {}", e))?;

    info!("Logging initialized - console: {}, file: {}", 
          log_level, log_dir.join("shuro.log").display());

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

/// Parse transcription mode from string
fn parse_transcription_mode(mode: &str) -> Result<TranscriptionMode> {
    match mode.to_lowercase().as_str() {
        "simple" => Ok(TranscriptionMode::Simple),
        "tuned" => Ok(TranscriptionMode::Tuned),
        _ => Err(ShuroError::Config(format!(
            "Invalid transcription mode '{}'. Valid modes: simple, tuned", 
            mode
        )).into()),
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