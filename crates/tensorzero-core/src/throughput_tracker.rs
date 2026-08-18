// Modified by Delta-AI under Apache 2.0
//! Recent output-token throughput tracker used to skip slow routing candidates.
//!
//! Mirrors Synapse `throughput-tracker.ts`: a per-key ring of samples with a
//! TTL. A candidate is below threshold only after `min_samples` fresh points
//! and a strict mean `< threshold`. Missing data never gates.

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

const DEFAULT_WINDOW_SIZE: usize = 20;
const DEFAULT_MIN_SAMPLES: usize = 5;
const DEFAULT_SAMPLE_TTL: Duration = Duration::from_secs(600);
const DEFAULT_MAX_ENTRIES: usize = 10_000;

#[derive(Clone, Copy, Debug)]
struct Sample {
    tps: f64,
    at: Instant,
}

#[derive(Debug)]
struct Ring {
    samples: Vec<Sample>,
    next: usize,
}

impl Ring {
    fn new(capacity: usize) -> Self {
        Self {
            samples: Vec::with_capacity(capacity),
            next: 0,
        }
    }

    fn push(&mut self, sample: Sample, capacity: usize) {
        if self.samples.len() < capacity {
            self.samples.push(sample);
            self.next = self.samples.len() % capacity;
        } else {
            self.samples[self.next] = sample;
            self.next = (self.next + 1) % capacity;
        }
    }
}

/// Process-wide tracker. Synapse kept this in gateway memory; we do the same.
pub struct ThroughputTracker {
    inner: Mutex<HashMap<String, Ring>>,
    window_size: usize,
    min_samples: usize,
    sample_ttl: Duration,
    max_entries: usize,
}

impl ThroughputTracker {
    pub fn new(
        window_size: usize,
        min_samples: usize,
        sample_ttl: Duration,
        max_entries: usize,
    ) -> Self {
        Self {
            inner: Mutex::new(HashMap::new()),
            window_size: window_size.max(1),
            min_samples: min_samples.max(1),
            sample_ttl,
            max_entries: max_entries.max(1),
        }
    }

    pub fn global() -> &'static Self {
        static GLOBAL: OnceLock<ThroughputTracker> = OnceLock::new();
        GLOBAL.get_or_init(|| {
            let window_size = env_usize("TENSORZERO_THROUGHPUT_TRACKER_WINDOW_SIZE")
                .unwrap_or(DEFAULT_WINDOW_SIZE);
            let min_samples = env_usize("TENSORZERO_THROUGHPUT_TRACKER_MIN_SAMPLES")
                .unwrap_or(DEFAULT_MIN_SAMPLES);
            let sample_ttl = env_usize("TENSORZERO_THROUGHPUT_TRACKER_SAMPLE_TTL_MS")
                .map(|ms| Duration::from_millis(ms as u64))
                .unwrap_or(DEFAULT_SAMPLE_TTL);
            let max_entries = env_usize("TENSORZERO_THROUGHPUT_TRACKER_MAX_ENTRIES")
                .unwrap_or(DEFAULT_MAX_ENTRIES);
            Self::new(window_size, min_samples, sample_ttl, max_entries)
        })
    }

    pub fn record(&self, key: &str, tps: f64) {
        if !tps.is_finite() || tps <= 0.0 {
            return;
        }
        let Ok(mut map) = self.inner.lock() else {
            return;
        };
        if !map.contains_key(key) && map.len() >= self.max_entries {
            return;
        }
        let ring = map
            .entry(key.to_string())
            .or_insert_with(|| Ring::new(self.window_size));
        ring.push(
            Sample {
                tps,
                at: Instant::now(),
            },
            self.window_size,
        );
    }

    /// True when we have enough fresh samples and their mean is strictly below `threshold`.
    pub fn is_below(&self, key: &str, threshold: f64) -> bool {
        if !threshold.is_finite() || threshold <= 0.0 {
            return false;
        }
        let Ok(mut map) = self.inner.lock() else {
            return false;
        };
        let Some(ring) = map.get_mut(key) else {
            return false;
        };
        let cutoff = Instant::now().checked_sub(self.sample_ttl);
        let fresh: Vec<f64> = ring
            .samples
            .iter()
            .filter(|sample| cutoff.is_none_or(|cutoff| sample.at >= cutoff))
            .map(|sample| sample.tps)
            .collect();
        if fresh.len() < self.min_samples {
            return false;
        }
        let avg = fresh.iter().sum::<f64>() / fresh.len() as f64;
        avg < threshold
    }

    #[cfg(test)]
    pub fn clear(&self) {
        if let Ok(mut map) = self.inner.lock() {
            map.clear();
        }
    }
}

fn env_usize(name: &str) -> Option<usize> {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
}

/// Tracker key: `provider:model`. Alias routing keys use `provider::model`.
pub fn throughput_key(provider_name: &str, model_name: &str) -> String {
    if let Some((provider, model)) = provider_name.split_once("::") {
        format!("{provider}:{model}")
    } else {
        format!("{provider_name}:{model_name}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::time::Duration;

    #[test]
    fn ignores_non_positive_samples() {
        let tracker = ThroughputTracker::new(20, 1, Duration::from_secs(60), 100);
        tracker.record("a:b", 0.0);
        tracker.record("a:b", f64::NAN);
        assert!(!tracker.is_below("a:b", 10.0));
    }

    #[test]
    fn does_not_gate_before_min_samples() {
        let tracker = ThroughputTracker::new(20, 5, Duration::from_secs(60), 100);
        for _ in 0..4 {
            tracker.record("p:m", 1.0);
        }
        assert!(!tracker.is_below("p:m", 10.0));
        tracker.record("p:m", 1.0);
        assert!(tracker.is_below("p:m", 10.0));
        assert!(!tracker.is_below("p:m", 0.5));
    }

    #[test]
    fn expired_samples_are_ignored() {
        let tracker = ThroughputTracker::new(20, 2, Duration::from_millis(20), 100);
        tracker.record("p:m", 1.0);
        tracker.record("p:m", 1.0);
        assert!(tracker.is_below("p:m", 10.0));
        thread::sleep(Duration::from_millis(30));
        assert!(!tracker.is_below("p:m", 10.0));
    }

    #[test]
    fn throughput_key_from_alias_route() {
        assert_eq!(throughput_key("dummy::error", "alias"), "dummy:error");
        assert_eq!(
            throughput_key("good", "test_fallback"),
            "good:test_fallback"
        );
    }
}
