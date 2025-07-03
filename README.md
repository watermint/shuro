# Shuro - Automated Subtitle Translation Workflow

An automated workflow for adding translated subtitles to movie files using whisper-cpp, ollama, and ffmpeg.

## Features

- **Hallucination Detection**: Automatically detects and retries transcriptions with hallucinated content
- **Two Transcription Modes**: 
  - **Simple**: Fast transcription with good quality using default settings
  - **Tuned**: Optimizes audio speed using a smaller model first to find the most smooth segments, then transcribes with the best settings
- **LLM Translation**: Uses local LLM (Ollama) for high-quality translations with validation
- **Quality Assurance**: Validates both transcription and translation quality before proceeding
- **Batch Processing**: Process single files or entire directories
- **Multiple Languages**: Supports translation to multiple target languages simultaneously

## Installation (Mac - Recommended)

For Mac users, we recommend using Homebrew for the easiest installation:

### 1. Install Prerequisites with Homebrew

```bash
# Install Homebrew if you don't have it
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install all required tools
brew install whisper-cpp ollama ffmpeg

# Start Ollama service
brew services start ollama

# Pull a model for translation (gemma3n:e4b has better multilingual support)
ollama pull gemma3n:e4b
```

### 2. Install Shuro

```bash
# Clone the repository
git clone https://github.com/watermint/shuro.git
cd shuro

# Build the project
cargo build --release

# Copy the sample configuration and customize
cp config.example.toml my-config.toml
# Edit my-config.toml to match your system if needed
```

### 3. Verify Installation

```bash
# Test that all tools are available
whisper --help
ollama --version
ffmpeg -version

# Test Shuro
./target/release/shuro --help
```

## Prerequisites (Other Platforms)

If you're not on Mac or prefer manual installation:

### 1. whisper.cpp
```bash
# Clone and build whisper.cpp
git clone https://github.com/ggerganov/whisper.cpp.git
cd whisper.cpp
make
# Add to PATH or specify full path in config
```

### 2. Ollama
```bash
# Install ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull a model for translation (gemma3n:e4b has better multilingual support)
ollama pull gemma3n:e4b
```

### 3. FFmpeg
```bash
# On Ubuntu/Debian
sudo apt install ffmpeg

# On other Linux distributions
# Check your package manager documentation
```

## Installation (From Source)

If you want to build from source or customize the build process:

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
   cp config.example.toml my-config.toml
   # Edit my-config.toml to match your system
   ```

## Usage

### Basic Commands

```bash
# Process a single video file with tuned transcription (default)
./target/release/shuro process -i video.mp4 -t "ja,ko" -o output/

# Process with simple transcription mode (faster but no optimization)
./target/release/shuro process -i video.mp4 -t "ja,ko" -o output/ --transcription-mode simple

# Process all videos in a directory with tuned transcription
./target/release/shuro batch -i videos/ -t "ja" -o output/ --transcription-mode tuned

# Use different translation modes
./target/release/shuro process -i video.mp4 -t "ja" --translation-mode context

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

# 2. Transcribe with tuned mode (default, finds optimal speed first)
./target/release/shuro transcribe -i audio.wav -o transcript.json

# 2. Transcribe with simple mode (faster but no optimization)
./target/release/shuro transcribe -i audio.wav -o transcript.json --transcription-mode simple

# 3. Translate transcription
./target/release/shuro translate -i transcript.json -o translated.json -t "ja"

# 4. Embed subtitles into video
./target/release/shuro embed -v video.mp4 -s subtitles.srt -o output.mp4
```

### Configuration

Create a `config.toml` file to customize behavior:

```toml
[transcriber]
binary_path = "whisper"
# Transcription mode: "simple" or "tuned" (default)
mode = "tuned"
# Model for exploration phase (when mode = "tuned")
explore_model = "base"
# Model for final transcription
transcribe_model = "medium"
# Tempo exploration settings (when mode = "tuned")
explore_steps = 10
explore_range_max = 110
explore_range_min = 80
# Acceptable languages
acceptable_languages = "en,ja,ko,zh,fr,de,es,ru,it,pt"
fallback_language = "en"
temperature = 0.0

[translate]
endpoint = "http://localhost:11434"
model = "gemma3n:e4b"
# Translation mode: "simple", "context", or "nlp"
mode = "simple"
max_retries = 3
context_window_size = 2
nlp_gap_threshold = 2.0

[quality]
repetitive_segment_threshold = 0.8
max_tokens_threshold = 50.0
min_quality_score = 0.7

[media]
binary_path = "ffmpeg"
subtitle_options = []
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
2. **Transcription**: Two modes available:
   - **Tuned Mode** (default): 
     - Tests different audio speeds (configurable range, e.g., 80-110%) with a smaller exploration model
     - Calculates segment smoothness (how evenly distributed segment lengths are)
     - Selects the tempo that produces the most evenly distributed segments
     - Performs final transcription using the optimal tempo with the full quality model
   - **Simple Mode**: Direct transcription with configured model and settings (faster)
3. **Quality Validation**: Detects hallucinations, repetitive content, and validates transcription quality
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
