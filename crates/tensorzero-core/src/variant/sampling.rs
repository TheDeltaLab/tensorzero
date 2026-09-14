// Modified by Delta-AI under Apache 2.0
//! Deterministic variant sampling for multi-variant functions.
//!
//! Sampling is done in-memory using a SHA-256 hash of `(function_name, episode_id)`,
//! so the chosen variant is stable within an episode. If any active variant carries
//! an explicit `weight`, weighted sampling is used; otherwise all active variants
//! are sampled uniformly.

use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::sync::Arc;
use uuid::Uuid;

use crate::error::{Error, ErrorDetails, IMPOSSIBLE_ERROR_MESSAGE};
use crate::variant::VariantInfo;

/// Samples and removes a variant from `active_variants`.
///
/// This never touches the database: variant selection is a pure function of the
/// function name, episode id, and the active variants' weights.
pub fn sample_variant(
    function_name: &str,
    episode_id: &Uuid,
    active_variants: &mut BTreeMap<String, Arc<VariantInfo>>,
) -> Result<(String, Arc<VariantInfo>), Error> {
    if active_variants.is_empty() {
        return Err(Error::new(ErrorDetails::InvalidFunctionVariants {
            message: format!(
                "No active variants to sample from for function `{function_name}`. {IMPOSSIBLE_ERROR_MESSAGE}"
            ),
        }));
    }

    let uniform = get_uniform_value(function_name, episode_id);
    let sampled_name = if active_variants
        .values()
        .any(|variant| variant.inner.weight().is_some())
    {
        sample_weighted(active_variants, uniform)?
    } else {
        sample_uniform(active_variants, uniform)
    };

    let (name, variant) = active_variants
        .remove_entry(&sampled_name)
        .ok_or_else(|| {
            Error::new(ErrorDetails::InvalidFunctionVariants {
                message: format!(
                    "Function `{function_name}` has no variant for the sampled variant `{sampled_name}`. {IMPOSSIBLE_ERROR_MESSAGE}"
                ),
            })
        })?;
    Ok((name, variant))
}

fn sample_uniform(
    active_variants: &BTreeMap<String, Arc<VariantInfo>>,
    uniform_sample: f64,
) -> String {
    let pool: Vec<&String> = active_variants.keys().collect();
    // `uniform_sample` is strictly less than 1.0, so this index is in bounds.
    let index = (uniform_sample * pool.len() as f64) as usize;
    pool[index.min(pool.len() - 1)].clone()
}

fn sample_weighted(
    active_variants: &BTreeMap<String, Arc<VariantInfo>>,
    uniform_sample: f64,
) -> Result<String, Error> {
    let weight_of = |variant: &VariantInfo| -> f64 {
        variant.inner.weight().unwrap_or_default()
    };
    let total_weight: f64 = active_variants.values().map(|v| weight_of(v)).sum();

    if total_weight <= 0.0 {
        // All weighted variants have been consumed by earlier rounds of the
        // retry loop; fall back to the first remaining active variant.
        return active_variants
            .keys()
            .next()
            .cloned()
            .ok_or_else(|| {
                Error::new(ErrorDetails::InvalidFunctionVariants {
                    message: format!(
                        "No active variants with positive weight remain. {IMPOSSIBLE_ERROR_MESSAGE}"
                    ),
                })
            });
    }

    let random_threshold = uniform_sample * total_weight;
    let mut cumulative_weight = 0.0;
    let variant_name: Option<&String> = active_variants
        .iter()
        .find(|(_name, variant)| {
            cumulative_weight += weight_of(variant);
            cumulative_weight > random_threshold
        })
        .map(|(name, _variant)| name);

    if let Some(name) = variant_name {
        return Ok(name.clone());
    }
    // Floating-point edge case where `cumulative_weight` never exceeds
    // `random_threshold`: pick the last variant with positive weight.
    active_variants
        .iter()
        .filter(|(_name, variant)| weight_of(variant) > 0.0)
        .map(|(name, _variant)| name)
        .next_back()
        .or_else(|| active_variants.keys().next())
        .cloned()
        .ok_or_else(|| {
            Error::new(ErrorDetails::InvalidFunctionVariants {
                message: format!(
                    "No active variants available. {IMPOSSIBLE_ERROR_MESSAGE}"
                ),
            })
        })
}

/// Implements a uniform distribution over the interval [0, 1) using a hash function.
/// This function is deterministic but should have good statistical properties.
pub(crate) fn get_uniform_value(function_name: &str, episode_id: &Uuid) -> f64 {
    let mut hasher = Sha256::new();
    hasher.update(function_name.as_bytes());
    hasher.update(episode_id.as_bytes());
    let hash_value = hasher.finalize();
    let truncated_hash =
        u32::from_be_bytes([hash_value[0], hash_value[1], hash_value[2], hash_value[3]]);
    // Divide by 2^32 (not u32::MAX) so the result is strictly in [0, 1).
    truncated_hash as f64 / (u32::MAX as f64 + 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::variant::VariantConfig;
    use std::collections::HashMap;

    fn make_variant(weight: Option<f64>) -> Arc<VariantInfo> {
        let mut inner = VariantConfig::ChatCompletion(Default::default());
        inner.set_weight(weight);
        Arc::new(VariantInfo {
            inner,
            timeouts: Default::default(),
            namespace: None,
        })
    }

    #[test]
    fn test_sampling_is_stable_per_episode() {
        let mut variants: BTreeMap<String, Arc<VariantInfo>> = BTreeMap::new();
        variants.insert("a".to_string(), make_variant(None));
        variants.insert("b".to_string(), make_variant(None));
        variants.insert("c".to_string(), make_variant(None));

        let episode_id = Uuid::now_v7();
        let first = sample_variant("fn", &episode_id, &mut variants.clone()).unwrap();
        let second = sample_variant("fn", &episode_id, &mut variants).unwrap();
        assert_eq!(first.0, second.0);
    }

    #[test]
    fn test_weighted_sampling_prefers_heavy_variant() {
        // With weight 1000 vs 1, the heavy variant should win for almost all
        // uniform values; a fixed episode id pins down one deterministic choice.
        let mut variants: BTreeMap<String, Arc<VariantInfo>> = BTreeMap::new();
        variants.insert("heavy".to_string(), make_variant(Some(1000.0)));
        variants.insert("light".to_string(), make_variant(Some(1.0)));

        let mut hits = HashMap::new();
        for i in 0..100u32 {
            let episode_id = Uuid::from_u64_pair(u64::from(i), 0);
            let (name, _) =
                sample_variant("fn", &episode_id, &mut variants.clone()).unwrap();
            *hits.entry(name).or_insert(0) += 1;
        }
        assert!(hits.get("heavy").copied().unwrap_or(0) > 90);
    }

    #[test]
    fn test_sample_removes_variant() {
        let mut variants: BTreeMap<String, Arc<VariantInfo>> = BTreeMap::new();
        variants.insert("only".to_string(), make_variant(None));
        let (name, _) = sample_variant("fn", &Uuid::now_v7(), &mut variants).unwrap();
        assert_eq!(name, "only");
        assert!(variants.is_empty());
        assert!(sample_variant("fn", &Uuid::now_v7(), &mut variants).is_err());
    }
}
