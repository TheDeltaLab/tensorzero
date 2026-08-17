// Modified by Delta-AI under Apache 2.0
//! Synapse `x-synapse-stream-aggregate`: merge consecutive thinking/content deltas.
//!
//! Clients that bill or buffer per SSE event blow up when providers emit
//! one-character reasoning deltas. This module matches the Synapse
//! `x-synapse-stream-aggregate` contract: warm-up window, interval flush, maxChars cap.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use serde::Deserialize;
use serde_json::{Value, json};

use crate::error::{Error, ErrorDetails};

pub const STREAM_AGGREGATE_HEADER: &str = "x-synapse-stream-aggregate";

const DEFAULT_START_DELAY_MS: u64 = 200;
const DEFAULT_INTERVAL_MS: u64 = 200;
const DEFAULT_MAX_CHARS: u64 = 500;
const MAX_START_DELAY_MS: u64 = 10_000;
const MAX_INTERVAL_MS: u64 = 10_000;
const MAX_MAX_CHARS: u64 = 100_000;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AggregatePart {
    Thinking,
    Content,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StreamAggregateRule {
    pub part: AggregatePart,
    pub start_delay_ms: u64,
    pub interval_ms: u64,
    pub max_chars: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum WireStyle {
    OpenAI,
    Anthropic,
}

#[derive(Clone, Debug)]
struct ClassifiedDelta {
    part: AggregatePart,
    field: String,
    index: Option<i64>,
    text: String,
    skeleton: Value,
    event_name: Option<String>,
    style: WireStyle,
}

#[derive(Debug)]
struct PendingBuffer {
    part: AggregatePart,
    field: String,
    index: Option<i64>,
    skeleton: Value,
    event_name: Option<String>,
    style: WireStyle,
    text: String,
    flush_at: Instant,
}

pub struct StreamAggregator {
    rules: HashMap<AggregatePart, StreamAggregateRule>,
    pending: Option<PendingBuffer>,
    first_seen: HashMap<AggregatePart, Instant>,
}

impl StreamAggregator {
    pub fn new(rules: Vec<StreamAggregateRule>) -> Self {
        let mut map = HashMap::new();
        for rule in rules {
            map.insert(rule.part, rule);
        }
        Self {
            rules: map,
            pending: None,
            first_seen: HashMap::new(),
        }
    }

    pub fn next_deadline(&self) -> Option<Instant> {
        self.pending.as_ref().map(|pending| pending.flush_at)
    }

    /// Push one SSE payload. Returns frames that should be emitted now
    /// (`event_name`, `data` JSON or the raw `[DONE]` sentinel as a string Value).
    pub fn push(
        &mut self,
        event_name: Option<&str>,
        data: &str,
        now: Instant,
    ) -> Vec<(Option<String>, Value)> {
        if data == "[DONE]" {
            let mut out = self.take_pending();
            out.push((None, Value::String("[DONE]".to_string())));
            return out;
        }
        let Ok(json) = serde_json::from_str::<Value>(data) else {
            let mut out = self.take_pending();
            out.push((
                event_name.map(str::to_string),
                Value::String(data.to_string()),
            ));
            return out;
        };
        let Some(classified) = classify(event_name, json) else {
            let mut out = self.take_pending();
            out.push((
                event_name.map(str::to_string),
                serde_json::from_str(data).unwrap_or(Value::String(data.to_string())),
            ));
            return out;
        };
        let Some(rule) = self.rules.get(&classified.part).cloned() else {
            let mut out = self.take_pending();
            out.push((classified.event_name, classified.skeleton));
            return out;
        };
        let first = *self.first_seen.entry(classified.part).or_insert(now);
        if now.duration_since(first) < Duration::from_millis(rule.start_delay_ms) {
            let mut out = self.take_pending();
            out.push((classified.event_name, classified.skeleton));
            return out;
        }
        let mut out = Vec::new();
        if let Some(pending) = self.pending.as_ref()
            && (pending.part != classified.part
                || pending.field != classified.field
                || pending.index != classified.index)
        {
            out.extend(self.take_pending());
        }
        if let Some(pending) = self.pending.as_mut() {
            pending.text.push_str(&classified.text);
            if pending.text.len() as u64 >= rule.max_chars {
                out.extend(self.take_pending());
            }
            return out;
        }
        self.pending = Some(PendingBuffer {
            part: classified.part,
            field: classified.field,
            index: classified.index,
            skeleton: classified.skeleton,
            event_name: classified.event_name,
            style: classified.style,
            text: classified.text,
            flush_at: now + Duration::from_millis(rule.interval_ms),
        });
        out
    }

    pub fn flush_if_due(&mut self, now: Instant) -> Option<(Option<String>, Value)> {
        let due = self
            .pending
            .as_ref()
            .is_some_and(|pending| now >= pending.flush_at);
        if due {
            self.take_pending().into_iter().next()
        } else {
            None
        }
    }

    pub fn finish(&mut self) -> Vec<(Option<String>, Value)> {
        self.take_pending()
    }

    fn take_pending(&mut self) -> Vec<(Option<String>, Value)> {
        let Some(pending) = self.pending.take() else {
            return Vec::new();
        };
        vec![emit_merged(pending)]
    }
}

fn emit_merged(pending: PendingBuffer) -> (Option<String>, Value) {
    let mut json = pending.skeleton;
    match pending.style {
        WireStyle::OpenAI => {
            if let Some(delta) = json
                .pointer_mut("/choices/0/delta")
                .and_then(Value::as_object_mut)
            {
                delta.insert(pending.field, json!(pending.text));
            }
        }
        WireStyle::Anthropic => {
            if let Some(delta) = json.get_mut("delta").and_then(Value::as_object_mut) {
                delta.insert(pending.field, json!(pending.text));
            }
        }
    }
    (pending.event_name, json)
}

fn classify(event_name: Option<&str>, json: Value) -> Option<ClassifiedDelta> {
    if event_name == Some("content_block_delta")
        || json.get("type").and_then(Value::as_str) == Some("content_block_delta")
    {
        return classify_anthropic(json);
    }
    classify_openai(json)
}

fn classify_openai(json: Value) -> Option<ClassifiedDelta> {
    let choices = json.get("choices")?.as_array()?;
    if choices.len() != 1 {
        return None;
    }
    let delta = choices[0].get("delta")?.as_object()?;
    if delta.contains_key("tool_calls") && !delta["tool_calls"].is_null() {
        return None;
    }
    let fields = ["reasoning_content", "reasoning", "content"];
    let present: Vec<&str> = fields
        .iter()
        .copied()
        .filter(|field| delta.get(*field).and_then(Value::as_str).is_some())
        .collect();
    if present.len() != 1 {
        return None;
    }
    let field = present[0];
    let text = delta.get(field)?.as_str()?.to_string();
    Some(ClassifiedDelta {
        part: if field == "content" {
            AggregatePart::Content
        } else {
            AggregatePart::Thinking
        },
        field: field.to_string(),
        index: None,
        text,
        skeleton: json,
        event_name: None,
        style: WireStyle::OpenAI,
    })
}

fn classify_anthropic(json: Value) -> Option<ClassifiedDelta> {
    if json.get("type").and_then(Value::as_str) != Some("content_block_delta") {
        return None;
    }
    let index = json.get("index").and_then(Value::as_i64);
    let delta = json.get("delta")?.as_object()?;
    let (part, field, text) = if delta.get("type").and_then(Value::as_str) == Some("thinking_delta")
    {
        (
            AggregatePart::Thinking,
            "thinking",
            delta.get("thinking")?.as_str()?.to_string(),
        )
    } else if delta.get("type").and_then(Value::as_str) == Some("text_delta") {
        (
            AggregatePart::Content,
            "text",
            delta.get("text")?.as_str()?.to_string(),
        )
    } else {
        return None;
    };
    Some(ClassifiedDelta {
        part,
        field: field.to_string(),
        index,
        text,
        skeleton: json,
        event_name: Some("content_block_delta".to_string()),
        style: WireStyle::Anthropic,
    })
}

pub fn parse_stream_aggregate_header(raw: &str) -> Result<Vec<StreamAggregateRule>, Error> {
    let parsed: Value = serde_json::from_str(raw).map_err(|_| {
        Error::new(ErrorDetails::InvalidOpenAICompatibleRequest {
            message: format!(
                "Invalid {STREAM_AGGREGATE_HEADER} header: expected a JSON array of rules"
            ),
        })
    })?;
    let Value::Array(items) = parsed else {
        return Err(Error::new(ErrorDetails::InvalidOpenAICompatibleRequest {
            message: format!(
                "Invalid {STREAM_AGGREGATE_HEADER} header: expected a non-empty JSON array of rules"
            ),
        }));
    };
    if items.is_empty() {
        return Err(Error::new(ErrorDetails::InvalidOpenAICompatibleRequest {
            message: format!(
                "Invalid {STREAM_AGGREGATE_HEADER} header: expected a non-empty JSON array of rules"
            ),
        }));
    }
    let mut by_part: HashMap<AggregatePart, StreamAggregateRule> = HashMap::new();
    for item in items {
        let Value::Object(obj) = item else {
            return Err(Error::new(ErrorDetails::InvalidOpenAICompatibleRequest {
                message: format!(
                    "Invalid {STREAM_AGGREGATE_HEADER} header: each rule must be an object"
                ),
            }));
        };
        let part = match obj.get("part").and_then(Value::as_str) {
            Some("thinking") => AggregatePart::Thinking,
            Some("content") => AggregatePart::Content,
            _ => {
                return Err(Error::new(ErrorDetails::InvalidOpenAICompatibleRequest {
                    message: format!(
                        "Invalid {STREAM_AGGREGATE_HEADER} header: part must be \"thinking\" or \"content\""
                    ),
                }));
            }
        };
        by_part.insert(
            part,
            StreamAggregateRule {
                part,
                start_delay_ms: read_bounded(
                    obj.get("startDelayMs"),
                    DEFAULT_START_DELAY_MS,
                    0,
                    MAX_START_DELAY_MS,
                    "startDelayMs",
                )?,
                interval_ms: read_bounded(
                    obj.get("intervalMs"),
                    DEFAULT_INTERVAL_MS,
                    1,
                    MAX_INTERVAL_MS,
                    "intervalMs",
                )?,
                max_chars: read_bounded(
                    obj.get("maxChars"),
                    DEFAULT_MAX_CHARS,
                    1,
                    MAX_MAX_CHARS,
                    "maxChars",
                )?,
            },
        );
    }
    Ok(by_part.into_values().collect())
}

fn read_bounded(
    value: Option<&Value>,
    fallback: u64,
    min: u64,
    max: u64,
    name: &str,
) -> Result<u64, Error> {
    let Some(value) = value else {
        return Ok(fallback);
    };
    let Some(n) = value.as_f64() else {
        return Err(invalid_bound(name, min, max));
    };
    if !n.is_finite() || n < min as f64 || n > max as f64 {
        return Err(invalid_bound(name, min, max));
    }
    Ok(n.floor() as u64)
}

fn invalid_bound(name: &str, min: u64, max: u64) -> Error {
    Error::new(ErrorDetails::InvalidOpenAICompatibleRequest {
        message: format!(
            "Invalid {STREAM_AGGREGATE_HEADER} header: {name} must be a number in [{min}, {max}]"
        ),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_defaults_and_override() {
        let rules = parse_stream_aggregate_header(
            r#"[{"part":"thinking"},{"part":"content","intervalMs":50,"maxChars":10}]"#,
        )
        .unwrap();
        assert_eq!(rules.len(), 2);
        let content = rules
            .iter()
            .find(|r| r.part == AggregatePart::Content)
            .unwrap();
        assert_eq!(content.interval_ms, 50);
        assert_eq!(content.max_chars, 10);
        let thinking = rules
            .iter()
            .find(|r| r.part == AggregatePart::Thinking)
            .unwrap();
        assert_eq!(thinking.start_delay_ms, 200);
    }

    #[test]
    fn parse_rejects_empty_and_bad_part() {
        assert!(parse_stream_aggregate_header("[]").is_err());
        assert!(parse_stream_aggregate_header(r#"[{"part":"nope"}]"#).is_err());
        assert!(parse_stream_aggregate_header("not-json").is_err());
    }

    #[test]
    fn merges_openai_content_after_warmup() {
        let rules = vec![StreamAggregateRule {
            part: AggregatePart::Content,
            start_delay_ms: 0,
            interval_ms: 10_000,
            max_chars: 500,
        }];
        let mut agg = StreamAggregator::new(rules);
        let now = Instant::now();
        let a = agg.push(None, r#"{"choices":[{"delta":{"content":"Hel"}}]}"#, now);
        assert!(a.is_empty(), "first aggregatable chunk is buffered");
        let b = agg.push(None, r#"{"choices":[{"delta":{"content":"lo"}}]}"#, now);
        assert!(b.is_empty());
        let flushed = agg.finish();
        assert_eq!(flushed.len(), 1);
        assert_eq!(flushed[0].1["choices"][0]["delta"]["content"], "Hello");
    }

    #[test]
    fn warmup_passes_through() {
        let rules = vec![StreamAggregateRule {
            part: AggregatePart::Content,
            start_delay_ms: 10_000,
            interval_ms: 200,
            max_chars: 500,
        }];
        let mut agg = StreamAggregator::new(rules);
        let now = Instant::now();
        let out = agg.push(None, r#"{"choices":[{"delta":{"content":"Hi"}}]}"#, now);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].1["choices"][0]["delta"]["content"], "Hi");
    }

    #[test]
    fn tool_calls_flush_and_passthrough() {
        let rules = vec![StreamAggregateRule {
            part: AggregatePart::Content,
            start_delay_ms: 0,
            interval_ms: 10_000,
            max_chars: 500,
        }];
        let mut agg = StreamAggregator::new(rules);
        let now = Instant::now();
        let _ = agg.push(None, r#"{"choices":[{"delta":{"content":"A"}}]}"#, now);
        let out = agg.push(
            None,
            r#"{"choices":[{"delta":{"tool_calls":[{"index":0}]}}]}"#,
            now,
        );
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].1["choices"][0]["delta"]["content"], "A");
    }

    #[test]
    fn merges_anthropic_thinking_delta() {
        let rules = vec![StreamAggregateRule {
            part: AggregatePart::Thinking,
            start_delay_ms: 0,
            interval_ms: 10_000,
            max_chars: 500,
        }];
        let mut agg = StreamAggregator::new(rules);
        let now = Instant::now();
        let _ = agg.push(
            Some("content_block_delta"),
            r#"{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"Hel"}}"#,
            now,
        );
        let _ = agg.push(
            Some("content_block_delta"),
            r#"{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"lo"}}"#,
            now,
        );
        let flushed = agg.finish();
        assert_eq!(flushed.len(), 1);
        assert_eq!(flushed[0].1["delta"]["thinking"], "Hello");
    }
}
