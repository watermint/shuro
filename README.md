# Shuro - Automated Subtitle Translation Workflow

An automated workflow for adding translated subtitles to movie files using whisper-cpp, ollama, and ffmpeg.

## Features

- **Hallucination Detection**: Automatically detects and retries transcriptions with hallucinated content
- **Tempo Tuning**: Optimizes audio speed to minimize transcription errors
- **LLM Translation**: Uses local LLM (Ollama) for high-quality translations with validation
- **Quality Assurance**: Validates both transcription and translation quality before proceeding
- **Batch Processing**: Process single files or entire directories
- **Multiple Languages**: Supports translation to multiple target languages simultaneously

## Prerequisites

1. **whisper.cpp**: Install and make available in PATH
   ```bash
   # Clone and build whisper.cpp
   git clone https://github.com/ggerganov/whisper.cpp.git
   cd whisper.cpp
   make
   # Add to PATH or specify full path in config
   ```

2. **Ollama**: Install and run locally
   ```bash
   # Install ollama
   curl -fsSL https://ollama.ai/install.sh | sh
   
   # Pull a model for translation
   ollama pull llama3.2:3b
   ```

3. **FFmpeg**: Install for video/audio processing
   ```bash
   # On macOS
   brew install ffmpeg
   
   # On Ubuntu/Debian
   sudo apt install ffmpeg
   ```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/watermint/shuro.git
   cd shuro
   ```

2. Build the project:
   ```bash
   cargo build --release
   ```

3. Copy the sample configuration and customize:
   ```bash
   cp config.toml my-config.toml
   # Edit my-config.toml to match your system
   ```

## Usage

### Basic Commands

```bash
# Process a single video file
./target/release/shuro process -i video.mp4 -t "ja,ko" -o output/

# Process all videos in a directory
./target/release/shuro batch -i videos/ -t "ja" -o output/

# Use custom configuration
./target/release/shuro -c my-config.toml process -i video.mp4 -t "ja"

# Enable verbose logging
./target/release/shuro -v process -i video.mp4 -t "ja"
```

### Step-by-step Processing

You can also run individual steps:

```bash
# 1. Extract audio from video
./target/release/shuro extract -i video.mp4 -o audio.wav

# 2. Transcribe with hallucination detection
./target/release/shuro transcribe -i audio.wav -o transcript.json

# 3. Translate transcription
./target/release/shuro translate -i transcript.json -o translated.json -t "ja"

# 4. Embed subtitles into video
./target/release/shuro embed -v video.mp4 -s subtitles.srt -o output.mp4
```

### Configuration

Create a `config.toml` file to customize behavior:

```toml
[whisper]
binary_path = "/path/to/whisper"
explore_model = "base"
transcribe_model = "large-v3"
explore_steps = 5
explore_range_max = 120
explore_range_min = 80

[translate]
ollama_endpoint = "http://localhost:11434"
model = "llama3.2:3b"
quality_threshold = "good"
max_retries = 3

[quality]
repetitive_segment_threshold = 0.8
max_tokens_threshold = 50.0
min_quality_score = 0.7

[ffmpeg]
binary_path = "ffmpeg"
subtitle_options = ["-c:v", "copy", "-c:a", "copy"]
```

### Model Management

Shuro automatically downloads required whisper models when they're not available locally. Models are stored in `.shuro/models/` directory.

#### List Available Models

```bash
./target/release/shuro models
```

This shows all available whisper models and their download status:

```
Available Whisper Models:
Name            Filename             Size (MB)  Status    
-----------------------------------------------------------------
tiny            ggml-tiny.bin        39.0       Downloaded
base            ggml-base.bin        142.0      Downloaded
medium          ggml-medium.bin      769.0      Missing
large-v3        ggml-large-v3.bin    1550.0     Missing
```

#### Download All Models

```bash
./target/release/shuro models --download
```

This downloads all available models to your local system. Note that the large models can be several GB in size.

#### Automatic Model Download

When you run any processing command, Shuro will automatically check for the required models (specified in config) and download them if they're missing. This happens transparently during initialization.

## How It Works

1. **Audio Extraction**: Uses FFmpeg to extract audio from video files
2. **Transcription Tuning**: 
   - Tests different audio speeds (80-120%) with a fast model
   - Detects hallucinations and repetitive content
   - Selects optimal parameters for final transcription
3. **Quality Transcription**: Uses a high-quality model with optimized parameters
4. **Translation**: 
   - Translates each segment using local LLM
   - Validates translation quality
   - Retries failed translations automatically
5. **Subtitle Generation**: Creates SRT files with proper timing
6. **Video Embedding**: Uses FFmpeg to embed subtitles into final video

## Project Structure

- `src/main.rs` - Main application entry point
- `src/lib.rs` - Library exports
- `src/cli.rs` - Command-line interface definitions
- `src/config.rs` - Configuration management
- `src/workflow.rs` - Main workflow orchestration
- `src/whisper.rs` - Whisper transcription with tuning
- `src/translate.rs` - LLM translation with validation
- `src/quality.rs` - Quality assessment and validation
- `src/subtitle.rs` - SRT subtitle generation
- `src/ffmpeg.rs` - Video processing
- `src/error.rs` - Error handling

## Project Rules

* Avoid use of syntax sugar / convenience features that reduce code clarity.
* Backward compatibility is not required. Keep the code simple.
* Do not hardcode constants, messages, etc. Define constants, or consider using externalized resource files.
* Don't create one-off script. Use test instead.
* Don't create sample files, create tests instead to demonstrate feature.
* If you find dead codes, just remove it. Instead of mark as dead.
* Once tasks completed, execute tests for the entire project and resolve all issues.
* Refer GLOSSARY.md to keep consistency of project terminology.
* Use the fail-fast approach. Do not create a fallback mechanism unless instructed to do so.
* Maintain trait/struct design immutable
* Read <module>/README.md when you start working with the module.
* Update <module>/README.md when you updated each modules. README.md should include descriptions of key files/folder and all modules, but should not include too much detal like function level.
* When designing traits/structs, design them to be immutable.

## Testing

Run the test suite:

```bash
cargo test
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [whisper.cpp](https://github.com/ggerganov/whisper.cpp) for speech recognition
- [ollama](https://ollama.ai/) for local LLM inference
- [ffmpeg](https://ffmpeg.org/) for video processing 
