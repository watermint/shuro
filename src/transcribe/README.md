# Transcription Module Architecture

This module provides a service-agnostic transcription system that supports multiple transcription backends through a common abstract interface.

## Architecture Overview

The transcription system uses an abstract model pattern to decouple the core application logic from specific transcription service implementations. This allows easy addition of new transcription services without modifying existing code.

### Key Components

1. **Abstract Models** (`common.rs`)
   - `AbstractTranscription`: Service-agnostic transcription result
   - `AbstractTranscriptionSegment`: Service-agnostic segment representation
   - `TranscriptionMapper<T>`: Trait for converting service-specific formats

2. **Service Implementations**
   - `whisper_cpp.rs`: Whisper.cpp command-line implementation
   - `openai.rs`: OpenAI Whisper implementation

3. **Factory Pattern** (`mod.rs`)
   - `TranscriberTrait`: Common interface for all transcription services
   - `TranscriberFactory`: Creates appropriate transcriber instances

## Abstract Data Models

### AbstractTranscription
```rust
pub struct AbstractTranscription {
    pub text: String,                    // Complete transcription text
    pub segments: Vec<AbstractTranscriptionSegment>, // Time-stamped segments
    pub language: String,                // Detected language
    pub duration: Option<f64>,           // Audio duration in seconds
    pub model_info: Option<String>,      // Service/model information
}
```

### AbstractTranscriptionSegment
```rust
pub struct AbstractTranscriptionSegment {
    pub id: i32,                         // Segment identifier
    pub start_time: f64,                 // Start time in seconds
    pub end_time: f64,                   // End time in seconds
    pub text: String,                    // Segment text
    pub confidence: Option<f32>,         // Confidence score (0.0-1.0)
    pub language: Option<String>,        // Segment language (if different)
}
```

## Adding New Transcription Services

To add a new transcription service:

### 1. Create Service-Specific Data Structures
```rust
#[derive(Serialize, Deserialize)]
pub struct YourServiceOutput {
    // Define fields matching your service's JSON response
}

#[derive(Serialize, Deserialize)]
pub struct YourServiceSegment {
    // Define segment structure from your service
}
```

### 2. Implement the TranscriptionMapper Trait
```rust
pub struct YourServiceMapper;

impl TranscriptionMapper<YourServiceOutput> for YourServiceMapper {
    fn to_abstract_transcription(service_output: YourServiceOutput) -> Result<AbstractTranscription> {
        // Convert your service's format to AbstractTranscription
        // Handle field mapping, time format conversion, etc.
    }

    fn to_legacy_transcription(abstract_result: AbstractTranscription) -> Transcription {
        abstract_result.into() // Use default conversion
    }
}
```

### 3. Implement the TranscriberTrait
```rust
pub struct YourServiceTranscriber {
    config: TranscriberConfig,
    validator: QualityValidator,
    // Add service-specific fields
}

#[async_trait]
impl TranscriberTrait for YourServiceTranscriber {
    async fn transcribe(&self, audio_path: &Path, language: Option<&str>) -> Result<Transcription> {
        // 1. Call your service API
        let service_output = self.call_your_service_api(audio_path, language).await?;
        
        // 2. Convert to abstract format
        let abstract_transcription = YourServiceMapper::to_abstract_transcription(service_output)?;
        
        // 3. Convert to legacy format for compatibility
        Ok(YourServiceMapper::to_legacy_transcription(abstract_transcription))
    }
    
    // Implement other required methods...
}
```

### 4. Update the Factory
```rust
// In mod.rs, add to TranscriberImplementation enum:
pub enum TranscriberImplementation {
    WhisperCpp,
    OpenAI,
    YourService, // Add your service here
}

// In TranscriberFactory::create_transcriber():
match implementation {
    TranscriberImplementation::WhisperCpp => { /* ... */ }
    TranscriberImplementation::OpenAI => { /* ... */ }
    TranscriberImplementation::YourService => {
        Box::new(your_service::YourServiceTranscriber::new(config, validator))
    }
}
```

## Benefits of This Architecture

1. **Service Independence**: Each service can have its own data format without affecting others
2. **Easy Extension**: Adding new services requires minimal changes to existing code
3. **Consistent Interface**: All services provide the same API to the rest of the application
4. **Type Safety**: Service-specific parsing with compile-time guarantees
5. **Backwards Compatibility**: Automatic conversion to legacy format preserves existing functionality

## Supported Services

- **Whisper.cpp**: Local whisper.cpp command-line tool
- **OpenAI Whisper**: OpenAI's Whisper Python implementation

## Future Service Candidates

- AssemblyAI
- Rev.ai
- Azure Speech Services
- Google Cloud Speech-to-Text
- Amazon Transcribe
- Speechmatics
- Deepgram

Each of these can be added following the pattern described above, with service-specific authentication, API calls, and response format handling. 