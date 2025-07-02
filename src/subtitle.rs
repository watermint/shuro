use std::path::Path;
// SRT time formatting doesn't require chrono for this implementation
use tokio::fs;
use tracing::info;

use crate::error::{Result, ShuroError};
use crate::quality::Transcription;

/// Generate SRT subtitle file from transcription
pub async fn generate_srt<P: AsRef<Path>>(
    transcription: &Transcription,
    output_path: P,
) -> Result<()> {
    let output_path = output_path.as_ref();
    info!("Generating SRT file: {}", output_path.display());

    let mut srt_content = String::new();
    
    for (index, segment) in transcription.segments.iter().enumerate() {
        let start_time = format_srt_time(segment.start);
        let end_time = format_srt_time(segment.end);
        
        srt_content.push_str(&format!(
            "{}\n{} --> {}\n{}\n\n",
            index + 1,
            start_time,
            end_time,
            segment.text.trim()
        ));
    }

    fs::write(output_path, srt_content).await
        .map_err(|e| ShuroError::Io(e))?;

    info!("SRT file generated successfully");
    Ok(())
}

/// Format time in seconds to SRT time format (HH:MM:SS,mmm)
fn format_srt_time(seconds: f64) -> String {
    let total_milliseconds = (seconds * 1000.0) as u64;
    let hours = total_milliseconds / 3_600_000;
    let minutes = (total_milliseconds % 3_600_000) / 60_000;
    let secs = (total_milliseconds % 60_000) / 1_000;
    let millis = total_milliseconds % 1_000;
    
    format!("{:02}:{:02}:{:02},{:03}", hours, minutes, secs, millis)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_srt_time() {
        assert_eq!(format_srt_time(0.0), "00:00:00,000");
        assert_eq!(format_srt_time(65.123), "00:01:05,123");
        assert_eq!(format_srt_time(3661.500), "01:01:01,500");
    }
} 