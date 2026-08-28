// Modified by Delta-AI under Apache 2.0
//! Input file types and conversion logic for OpenAI-compatible API.
//!
//! This module handles various input file formats including images, audio files, and PDFs.
//! It provides types for deserializing OpenAI-compatible file formats and helper functions
//! for converting them to TensorZero's internal file representations.

use mime::MediaType;
use serde::Deserialize;
use url::Url;

use crate::error::{Error, ErrorDetails};
use crate::inference::types::file::Detail;
use crate::inference::types::{Base64File, File, UrlFile};

/// OpenAI-compatible image URL content block.
///
/// Represents an image that can be either a data URL (base64-encoded) or a remote URL.
/// Supports optional detail level for image processing.
#[derive(Deserialize, Debug)]
#[serde(tag = "type", deny_unknown_fields, rename_all = "snake_case")]
pub struct OpenAICompatibleImageUrl {
    pub url: Url,
    #[serde(rename = "tensorzero::mime_type")]
    pub mime_type: Option<MediaType>,
    #[serde(default)]
    pub detail: Option<Detail>,
}

/// OpenAI-compatible file content block.
///
/// Represents a file with base64-encoded data URL and optional filename.
/// OpenAI supports file_id with their files API, but we require the file data directly.
#[derive(Deserialize, Debug)]
pub struct OpenAICompatibleFile {
    pub file_data: String,
    #[serde(default)]
    pub filename: Option<String>,
    // OpenAI supports file_id with their files API
    // We do not so we require these two fields
}

/// OpenAI-compatible input audio content block.
///
/// `data` is either unprefixed base64 or (Synapse-compatible clients) an HTTP(S) URL.
/// When `data` is a URL we keep it as `File::Url` so Volcengine Ark can fetch it.
#[derive(Deserialize, Debug)]
pub struct OpenAICompatibleInputAudio {
    pub data: String,
    pub format: String,
}

/// Parses a base64-encoded data URL and extracts the MIME type and data.
///
/// Expected format: `data:<mime_type>;base64,<base64_data>`
///
/// # Examples
///
/// ```ignore
/// let (mime_type, data) = parse_base64_file_data_url("data:image/png;base64,SGVsbG8h")?;
/// ```
pub fn parse_base64_file_data_url(data_url: &str) -> Result<(MediaType, &str), Error> {
    let Some(url) = data_url.strip_prefix("data:") else {
        return Err(Error::new(ErrorDetails::InvalidOpenAICompatibleRequest {
            message: "Expected a base64-encoded data URL with MIME type (e.g. `data:image/png;base64,SGVsbG8sIFdvcmxkIQ==`), but got a value without the `data:` prefix.".to_string(),
        }));
    };
    let Some((mime_type, data)) = url.split_once(";base64,") else {
        return Err(Error::new(ErrorDetails::InvalidOpenAICompatibleRequest {
            message: "Expected a base64-encoded data URL with MIME type (e.g. `data:image/png;base64,SGVsbG8sIFdvcmxkIQ==`), but got a value without the `;base64,` separator.".to_string(),
        }));
    };
    let file_type: MediaType = mime_type.parse().map_err(|_| {
        Error::new(ErrorDetails::InvalidOpenAICompatibleRequest {
            message: format!("Unknown MIME type `{mime_type}` in data URL"),
        })
    })?;
    Ok((file_type, data))
}

/// Converts an OpenAI-compatible image URL to TensorZero's File type.
///
/// Handles both data URLs (base64-encoded images) and remote URLs.
pub fn convert_image_url_to_file(image_url: OpenAICompatibleImageUrl) -> Result<File, Error> {
    if image_url.url.scheme() == "data" {
        let image_url_str = image_url.url.to_string();
        let (mime_type, data) = parse_base64_file_data_url(&image_url_str)?;
        let base64_file = Base64File::new(
            None,
            Some(mime_type),
            data.to_string(),
            image_url.detail,
            None,
        )?;
        Ok(File::Base64(base64_file))
    } else {
        Ok(File::Url(UrlFile {
            url: image_url.url,
            mime_type: image_url.mime_type,
            detail: image_url.detail,
            filename: None,
        }))
    }
}

/// Converts an OpenAI-compatible file to TensorZero's Base64File.
///
/// Parses the data URL and extracts MIME type and base64 data.
pub fn convert_file_to_base64(file: OpenAICompatibleFile) -> Result<File, Error> {
    let (mime_type, data) = parse_base64_file_data_url(&file.file_data)?;
    let base64_file =
        Base64File::new(None, Some(mime_type), data.to_string(), None, file.filename)?;
    Ok(File::Base64(base64_file))
}

/// Converts OpenAI-compatible input audio to TensorZero's File type.
///
/// Trusts the client `format` field (Synapse passthrough). Validates that
/// `data` is base64 (or an HTTP URL). Does not sniff magic bytes — truncated
/// or nonstandard payloads are forwarded to the provider.
pub fn convert_input_audio_to_file(input_audio: OpenAICompatibleInputAudio) -> Result<File, Error> {
    let mime_type = mime_from_audio_format(&input_audio.format).ok_or_else(|| {
        Error::new(ErrorDetails::InvalidOpenAICompatibleRequest {
            message: format!(
                "Unknown audio format `{}` for input_audio",
                input_audio.format
            ),
        })
    })?;

    if looks_like_http_url(&input_audio.data) {
        let url = Url::parse(&input_audio.data).map_err(|e| {
            Error::new(ErrorDetails::InvalidOpenAICompatibleRequest {
                message: format!("Invalid audio URL in input_audio.data: {e}"),
            })
        })?;
        return Ok(File::Url(UrlFile {
            url,
            mime_type: Some(mime_type),
            detail: None,
            filename: None,
        }));
    }

    base64::Engine::decode(
        &base64::engine::general_purpose::STANDARD,
        &input_audio.data,
    )
    .map_err(|e| {
        Error::new(ErrorDetails::InvalidOpenAICompatibleRequest {
            message: format!("Invalid base64 data in input_audio: {e}"),
        })
    })?;

    let base64_file = Base64File::new(None, Some(mime_type), input_audio.data, None, None)?;
    Ok(File::Base64(base64_file))
}

fn looks_like_http_url(value: &str) -> bool {
    let value = value.trim();
    value.starts_with("http://") || value.starts_with("https://")
}

fn mime_from_audio_format(format: &str) -> Option<MediaType> {
    match format.to_ascii_lowercase().as_str() {
        "wav" => "audio/wav".parse().ok(),
        "mp3" | "mpeg" => "audio/mpeg".parse().ok(),
        "ogg" => "audio/ogg".parse().ok(),
        "m4a" | "mp4" => "audio/mp4".parse().ok(),
        "webm" => "audio/webm".parse().ok(),
        "flac" => "audio/flac".parse().ok(),
        other => format!("audio/{other}").parse().ok(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use googletest::gtest;
    use googletest::prelude::{eq, expect_that};
    use serde_json::json;

    use crate::endpoints::openai_compatible::types::chat_completions::{
        OpenAICompatibleContentBlock, OpenAICompatibleMessage, OpenAICompatibleUserMessage,
        convert_openai_message_content, openai_messages_to_input,
    };
    use crate::inference::types::{Input, InputMessageContent, Role};

    #[test]
    fn test_input_audio_content_block() {
        // Test valid WAV audio (magic bytes: RIFF....WAVE)
        let wav_bytes = b"RIFF\x00\x00\x00\x00WAVEfmt ";
        let wav_base64 =
            base64::Engine::encode(&base64::engine::general_purpose::STANDARD, wav_bytes);

        let content = json!([{
            "type": "input_audio",
            "input_audio": {
                "data": wav_base64,
                "format": "wav"
            }
        }]);

        let result = convert_openai_message_content("user".to_string(), content).unwrap();
        assert_eq!(result.len(), 1);

        match &result[0] {
            InputMessageContent::File(File::Base64(base64_file)) => {
                assert_eq!(
                    base64_file.mime_type,
                    "audio/wav".parse::<MediaType>().unwrap()
                );
                assert_eq!(base64_file.data(), wav_base64);
            }
            _ => panic!("Expected File(Base64(...))"),
        }

        // Test valid MP3 audio (magic bytes: FF FB)
        let mp3_bytes = [0xFF, 0xFB, 0x90, 0x44, 0x00, 0x00];
        let mp3_base64 =
            base64::Engine::encode(&base64::engine::general_purpose::STANDARD, mp3_bytes);

        let content = json!([{
            "type": "input_audio",
            "input_audio": {
                "data": mp3_base64,
                "format": "mp3"
            }
        }]);

        let result = convert_openai_message_content("user".to_string(), content).unwrap();
        assert_eq!(result.len(), 1);

        match &result[0] {
            InputMessageContent::File(File::Base64(base64_file)) => {
                assert_eq!(
                    base64_file.mime_type,
                    "audio/mpeg".parse::<MediaType>().unwrap()
                );
                assert_eq!(base64_file.data(), mp3_base64);
            }
            _ => panic!("Expected File(Base64(...))"),
        }

        // Test invalid base64 data
        let content = json!([{
            "type": "input_audio",
            "input_audio": {
                "data": "not-valid-base64!!!",
                "format": "wav"
            }
        }]);

        let error = convert_openai_message_content("user".to_string(), content).unwrap_err();
        let details = error.get_details();
        match details {
            ErrorDetails::InvalidOpenAICompatibleRequest { message } => {
                assert!(message.contains("Invalid base64 data"));
            }
            _ => panic!("Expected InvalidOpenAICompatibleRequest error"),
        }

        // Non-audio magic bytes still pass: the gateway trusts `format`.
        let jpeg_bytes = [0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10];
        let jpeg_base64 =
            base64::Engine::encode(&base64::engine::general_purpose::STANDARD, jpeg_bytes);

        let content = json!([{
            "type": "input_audio",
            "input_audio": {
                "data": jpeg_base64,
                "format": "wav"
            }
        }]);

        let result = convert_openai_message_content("user".to_string(), content)
            .expect("gateway should not sniff input_audio magic bytes");
        match &result[0] {
            InputMessageContent::File(File::Base64(base64_file)) => {
                assert_eq!(
                    base64_file.mime_type,
                    "audio/wav".parse::<MediaType>().unwrap()
                );
            }
            other => panic!("Expected File(Base64(...)), got {other:?}"),
        }
    }

    #[gtest]
    fn input_audio_trusts_format_when_bytes_are_not_sniffable() {
        let unknown_base64 = base64::Engine::encode(
            &base64::engine::general_purpose::STANDARD,
            [0x00, 0x01, 0x02, 0x03],
        );

        let content = json!([{
            "type": "input_audio",
            "input_audio": {
                "data": unknown_base64,
                "format": "mp3"
            }
        }]);

        let result = convert_openai_message_content("user".to_string(), content)
            .expect("truncated or nonstandard audio should be forwarded");
        expect_that!(result.len(), eq(1));
        let InputMessageContent::File(File::Base64(base64_file)) = &result[0] else {
            panic!("Expected File(Base64(...))");
        };
        expect_that!(base64_file.mime_type.as_ref(), eq("audio/mpeg"));
        expect_that!(base64_file.data(), eq(unknown_base64.as_str()));
    }

    #[gtest]
    fn input_audio_uses_declared_format_even_if_magic_bytes_differ() {
        let wav_base64 = base64::Engine::encode(
            &base64::engine::general_purpose::STANDARD,
            b"RIFF\x00\x00\x00\x00WAVEfmt ",
        );

        let content = json!([{
            "type": "input_audio",
            "input_audio": {
                "data": wav_base64,
                "format": "mp3"
            }
        }]);

        let result = convert_openai_message_content("user".to_string(), content)
            .expect("declared format should win over magic bytes");
        let InputMessageContent::File(File::Base64(base64_file)) = &result[0] else {
            panic!("Expected File(Base64(...))");
        };
        expect_that!(base64_file.mime_type.as_ref(), eq("audio/mpeg"));
    }

    #[test]
    fn test_input_audio_http_url_stays_url_file() {
        let content = json!([{
            "type": "input_audio",
            "input_audio": {
                "data": "https://audio.test/a.wav",
                "format": "wav"
            }
        }]);
        let result = convert_openai_message_content("user".to_string(), content).unwrap();
        match &result[0] {
            InputMessageContent::File(File::Url(url_file)) => {
                assert_eq!(url_file.url.as_str(), "https://audio.test/a.wav");
                assert_eq!(
                    url_file
                        .mime_type
                        .as_ref()
                        .map(ToString::to_string)
                        .as_deref(),
                    Some("audio/wav")
                );
            }
            other => panic!("Expected File::Url, got {other:?}"),
        }
    }

    #[test]
    fn test_parse_base64_file_data_url() {
        assert_eq!(
            (mime::IMAGE_JPEG, "YWJjCg=="),
            parse_base64_file_data_url("data:image/jpeg;base64,YWJjCg==").unwrap()
        );
        assert_eq!(
            (mime::IMAGE_PNG, "YWJjCg=="),
            parse_base64_file_data_url("data:image/png;base64,YWJjCg==").unwrap()
        );
        assert_eq!(
            ("image/webp".parse().unwrap(), "YWJjCg=="),
            parse_base64_file_data_url("data:image/webp;base64,YWJjCg==").unwrap()
        );
        assert_eq!(
            ("application/pdf".parse().unwrap(), "JVBERi0xLjQK"),
            parse_base64_file_data_url("data:application/pdf;base64,JVBERi0xLjQK").unwrap()
        );

        // Test error when prefix is missing
        let result = parse_base64_file_data_url("YWJjCg==");
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("without the `data:` prefix")
        );

        // Test error when base64 separator is missing
        let result = parse_base64_file_data_url("data:image/png,YWJjCg==");
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("without the `;base64,` separator")
        );
    }

    #[test]
    fn test_deserialize_image_url_with_detail() {
        // Test deserialization with detail: low
        let json_low = json!({
            "type": "image_url",
            "image_url": {
                "url": "https://example.com/image.png",
                "detail": "low"
            }
        });
        let block: OpenAICompatibleContentBlock = serde_json::from_value(json_low).unwrap();
        match block {
            OpenAICompatibleContentBlock::ImageUrl { image_url } => {
                assert_eq!(image_url.url.as_str(), "https://example.com/image.png");
                assert_eq!(image_url.detail, Some(Detail::Low));
            }
            _ => panic!("Expected ImageUrl variant"),
        }

        // Test deserialization with detail: high
        let json_high = json!({
            "type": "image_url",
            "image_url": {
                "url": "https://example.com/image.png",
                "detail": "high"
            }
        });
        let block: OpenAICompatibleContentBlock = serde_json::from_value(json_high).unwrap();
        match block {
            OpenAICompatibleContentBlock::ImageUrl { image_url } => {
                assert_eq!(image_url.detail, Some(Detail::High));
            }
            _ => panic!("Expected ImageUrl variant"),
        }

        // Test deserialization with detail: auto
        let json_auto = json!({
            "type": "image_url",
            "image_url": {
                "url": "https://example.com/image.png",
                "detail": "auto"
            }
        });
        let block: OpenAICompatibleContentBlock = serde_json::from_value(json_auto).unwrap();
        match block {
            OpenAICompatibleContentBlock::ImageUrl { image_url } => {
                assert_eq!(image_url.detail, Some(Detail::Auto));
            }
            _ => panic!("Expected ImageUrl variant"),
        }

        // Test deserialization without detail (should default to None)
        let json_none = json!({
            "type": "image_url",
            "image_url": {
                "url": "https://example.com/image.png"
            }
        });
        let block: OpenAICompatibleContentBlock = serde_json::from_value(json_none).unwrap();
        match block {
            OpenAICompatibleContentBlock::ImageUrl { image_url } => {
                assert_eq!(image_url.detail, None);
            }
            _ => panic!("Expected ImageUrl variant"),
        }

        // Test deserialization with invalid detail should fail
        let json_invalid = json!({
            "type": "image_url",
            "image_url": {
                "url": "https://example.com/image.png",
                "detail": "invalid"
            }
        });
        let result: Result<OpenAICompatibleContentBlock, _> = serde_json::from_value(json_invalid);
        assert!(result.is_err());
    }

    #[test]
    fn test_openai_file_with_custom_filename() {
        // Test deserialization with custom filename
        let json = json!({
            "type": "file",
            "file": {
                "file_data": "data:text/plain;base64,SGVsbG8h",
                "filename": "my_config.txt"
            }
        });
        let block: OpenAICompatibleContentBlock = serde_json::from_value(json).unwrap();
        match block {
            OpenAICompatibleContentBlock::File { file } => {
                assert_eq!(file.filename, Some("my_config.txt".to_string()));
                assert_eq!(file.file_data, "data:text/plain;base64,SGVsbG8h");
            }
            _ => panic!("Expected File variant"),
        }

        // Test deserialization without filename (should be None)
        let json_no_filename = json!({
            "type": "file",
            "file": {
                "file_data": "data:application/pdf;base64,JVBERi0xLjQ="
            }
        });
        let block: OpenAICompatibleContentBlock = serde_json::from_value(json_no_filename).unwrap();
        match block {
            OpenAICompatibleContentBlock::File { file } => {
                assert_eq!(file.filename, None);
                assert_eq!(file.file_data, "data:application/pdf;base64,JVBERi0xLjQ=");
            }
            _ => panic!("Expected File variant"),
        }
    }

    #[test]
    fn test_filename_propagated_through_openai_to_tensorzero_conversion() {
        // Test that filename flows from OpenAI API format to TensorZero Input type
        let messages = vec![OpenAICompatibleMessage::User(OpenAICompatibleUserMessage {
            content: json!([
                {
                    "type": "text",
                    "text": "Please analyze this file"
                },
                {
                    "type": "file",
                    "file": {
                        "file_data": "data:text/plain;base64,SGVsbG8h",
                        "filename": "important_data.txt"
                    }
                }
            ]),
        })];

        let input: Input = openai_messages_to_input(messages).unwrap();

        assert_eq!(input.messages.len(), 1);
        assert_eq!(input.messages[0].role, Role::User);
        assert_eq!(input.messages[0].content.len(), 2);

        // Check text content
        match &input.messages[0].content[0] {
            InputMessageContent::Text(text) => {
                assert_eq!(text.text, "Please analyze this file");
            }
            _ => panic!("Expected Text content"),
        }

        // Check file content with filename
        match &input.messages[0].content[1] {
            InputMessageContent::File(File::Base64(base64_file)) => {
                assert_eq!(base64_file.filename, Some("important_data.txt".to_string()));
                assert_eq!(base64_file.mime_type, mime::TEXT_PLAIN);
            }
            _ => panic!("Expected Base64File with filename"),
        }

        // Test without filename
        let messages_no_filename =
            vec![OpenAICompatibleMessage::User(OpenAICompatibleUserMessage {
                content: json!([
                    {
                        "type": "file",
                        "file": {
                            "file_data": "data:application/pdf;base64,JVBERi0xLjQ="
                        }
                    }
                ]),
            })];

        let input_no_filename: Input = openai_messages_to_input(messages_no_filename).unwrap();
        match &input_no_filename.messages[0].content[0] {
            InputMessageContent::File(File::Base64(base64_file)) => {
                assert_eq!(base64_file.filename, None);
                assert_eq!(base64_file.mime_type, mime::APPLICATION_PDF);
            }
            _ => panic!("Expected Base64File without filename"),
        }
    }
}
