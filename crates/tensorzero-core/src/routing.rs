// Modified by Delta-AI under Apache 2.0
//! Request-scoped routing policy for Synapse-compatible failover.
//!
//! Alias multi-target lists are materialized as `ModelConfig.routing`. This
//! module truncates that list when `x-synapse-fallback: false`, skips slow
//! candidates via the throughput tracker, and records the winner for
//! `x-synapse-served-by` / `x-synapse-fallback-count`.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use crate::error::Error;
use crate::inference::types::Latency;
use crate::inference::types::usage::Usage;
use crate::throughput_tracker::{ThroughputTracker, throughput_key};

tokio::task_local! {
    static ROUTING_SESSION: Arc<RoutingSession>;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RoutingOutcome {
    pub served_by: String,
    pub fallback_count: u32,
}

#[derive(Debug)]
pub struct RoutingSession {
    pub fallback_disabled: bool,
    min_tokens_per_sec: Mutex<Option<f64>>,
    outcome: Mutex<Option<RoutingOutcome>>,
}

impl RoutingSession {
    pub fn new(fallback_disabled: bool) -> Arc<Self> {
        Arc::new(Self {
            fallback_disabled,
            min_tokens_per_sec: Mutex::new(None),
            outcome: Mutex::new(None),
        })
    }

    pub fn current() -> Option<Arc<Self>> {
        ROUTING_SESSION.try_with(Clone::clone).ok()
    }

    pub async fn scope<F, T>(session: Arc<Self>, fut: F) -> T
    where
        F: std::future::Future<Output = T>,
    {
        ROUTING_SESSION.scope(session, fut).await
    }

    pub fn set_min_tokens_per_sec(&self, value: Option<f64>) {
        if let Ok(mut slot) = self.min_tokens_per_sec.lock() {
            *slot = value;
        }
    }

    pub fn min_tokens_per_sec(&self) -> Option<f64> {
        self.min_tokens_per_sec.lock().ok().and_then(|slot| *slot)
    }

    pub fn record_provider(&self, routing: &[Arc<str>], provider_name: &str, model_name: &str) {
        let fallback_count = routing
            .iter()
            .position(|name| name.as_ref() == provider_name)
            .unwrap_or(0) as u32;
        // Merged alias targets use `provider::model` as the routing key.
        // Native shorthand still keys providers as just `provider`.
        let served_by = if provider_name.contains("::") {
            served_by_from_provider_name(provider_name)
        } else if model_name.contains("::") {
            served_by_from_model_name(model_name)
        } else {
            served_by_from_provider_name(provider_name)
        };
        if let Ok(mut slot) = self.outcome.lock() {
            *slot = Some(RoutingOutcome {
                served_by,
                fallback_count,
            });
        }
    }

    pub fn take_outcome(&self) -> Option<RoutingOutcome> {
        self.outcome.lock().ok().and_then(|mut slot| slot.take())
    }
}

fn served_by_from_provider_name(provider_name: &str) -> String {
    match provider_name.split_once("::") {
        Some((provider, model)) => format!("{provider}/{model}"),
        None => provider_name.to_string(),
    }
}

fn served_by_from_model_name(model_name: &str) -> String {
    match model_name.split_once("::") {
        Some((provider, model)) => format!("{provider}/{model}"),
        None => model_name.to_string(),
    }
}

/// Ordered provider names to actually try for this request.
pub fn effective_routing(routing: &[Arc<str>]) -> Vec<Arc<str>> {
    if routing.is_empty() {
        return Vec::new();
    }
    let session = RoutingSession::current();
    let fallback_disabled = session
        .as_ref()
        .is_some_and(|session| session.fallback_disabled);
    let candidates: Vec<Arc<str>> = if fallback_disabled {
        routing.iter().take(1).cloned().collect()
    } else {
        routing.to_vec()
    };
    let threshold = session
        .as_ref()
        .and_then(|session| session.min_tokens_per_sec());
    filter_slow_candidates(&candidates, threshold)
}

fn filter_slow_candidates(candidates: &[Arc<str>], threshold: Option<f64>) -> Vec<Arc<str>> {
    let Some(threshold) = threshold else {
        return candidates.to_vec();
    };
    let tracker = ThroughputTracker::global();
    let passing: Vec<Arc<str>> = candidates
        .iter()
        .filter(|name| !tracker.is_below(&throughput_key(name, ""), threshold))
        .cloned()
        .collect();
    if passing.is_empty() {
        // Degrade: never skip every candidate.
        candidates.iter().take(1).cloned().collect()
    } else {
        passing
    }
}

/// Synapse `isFallbackableStatus`: 5xx, 401, 402, 403, 429, plus network/timeouts.
pub fn is_failoverable(error: &Error) -> bool {
    let code = error
        .underlying_status_code()
        .unwrap_or_else(|| error.status_code())
        .as_u16();
    matches!(code, 401 | 402 | 403 | 408 | 429 | 500..=599)
}

/// Native TensorZero routing retries every provider error. Synapse OpenAI
/// requests (a live `RoutingSession`) only fail over on the statuses above.
pub fn should_failover(error: &Error) -> bool {
    RoutingSession::current().is_none() || is_failoverable(error)
}

pub fn record_tokens_per_sec(
    provider_name: &str,
    model_name: &str,
    output_tokens: Option<u32>,
    generation: Duration,
) {
    let Some(output_tokens) = output_tokens else {
        return;
    };
    let secs = generation.as_secs_f64();
    if output_tokens == 0 || secs <= 0.0 {
        return;
    }
    let tps = f64::from(output_tokens) / secs;
    ThroughputTracker::global().record(&throughput_key(provider_name, model_name), tps);
}

pub fn record_tokens_per_sec_from_latency(
    provider_name: &str,
    model_name: &str,
    usage: &Usage,
    latency: &Latency,
) {
    let generation = match latency {
        Latency::NonStreaming { response_time } => *response_time,
        Latency::Streaming {
            ttft,
            response_time,
        } => response_time.saturating_sub(*ttft),
        Latency::Batch => return,
    };
    record_tokens_per_sec(provider_name, model_name, usage.output_tokens, generation);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::ErrorDetails;
    use reqwest::StatusCode;

    #[test]
    fn failoverable_matches_synapse_statuses() {
        let five_hundred = Error::new(ErrorDetails::InferenceClient {
            message: "upstream".into(),
            status_code: Some(StatusCode::INTERNAL_SERVER_ERROR),
            provider_type: "dummy".into(),
            api_type: crate::inference::types::ApiType::ChatCompletions,
            raw_request: None,
            raw_response: None,
        });
        assert!(is_failoverable(&five_hundred));

        let bad_request = Error::new(ErrorDetails::InvalidOpenAICompatibleRequest {
            message: "nope".into(),
        });
        assert!(!is_failoverable(&bad_request));
    }

    #[tokio::test]
    async fn fallback_disabled_keeps_head_only() {
        let session = RoutingSession::new(true);
        let routing = vec![
            Arc::<str>::from("dummy::error"),
            Arc::<str>::from("dummy::good"),
        ];
        let effective = RoutingSession::scope(session, async { effective_routing(&routing) }).await;
        assert_eq!(effective.len(), 1);
        assert_eq!(effective[0].as_ref(), "dummy::error");
    }

    #[test]
    fn record_provider_uses_shorthand_model_name() {
        let session = RoutingSession::new(false);
        session.record_provider(&[Arc::<str>::from("dummy")], "dummy", "dummy::good");
        let outcome = session.take_outcome().unwrap();
        assert_eq!(outcome.served_by, "dummy/good");
        assert_eq!(outcome.fallback_count, 0);
    }

    #[test]
    fn record_provider_uses_merged_routing_key() {
        let session = RoutingSession::new(false);
        let routing = vec![
            Arc::<str>::from("dummy::error"),
            Arc::<str>::from("dummy::good"),
        ];
        session.record_provider(&routing, "dummy::good", "flash");
        let outcome = session.take_outcome().unwrap();
        assert_eq!(outcome.served_by, "dummy/good");
        assert_eq!(outcome.fallback_count, 1);
    }
}
