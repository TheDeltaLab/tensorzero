// Modified by Delta-AI under Apache 2.0
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use tensorzero_types::{
    CostPointerConfig, PointerList, TierMode, UnifiedCostPointerConfig, UninitializedCostConfig,
    UninitializedCostConfigEntry, UninitializedCostRate, UninitializedCostTier,
    UninitializedPeakWindow, UninitializedPeakWindows, UninitializedTierWhen,
    UninitializedUnifiedCostConfig, UsageField,
};

// --- Cost config (provider-level `cost` field) ---

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StoredCostConfig {
    pub entries: Vec<StoredCostConfigEntry>,
}

impl From<&UninitializedCostConfig> for StoredCostConfig {
    fn from(config: &UninitializedCostConfig) -> Self {
        StoredCostConfig {
            entries: config
                .iter()
                .map(|entry| StoredCostConfigEntry {
                    pointer: entry.pointer.pointer.clone(),
                    pointer_nonstreaming: entry.pointer.pointer_nonstreaming.clone(),
                    pointer_streaming: entry.pointer.pointer_streaming.clone(),
                    cost_per_million: entry.rate.cost_per_million,
                    cost_per_unit: entry.rate.cost_per_unit,
                    required: Some(entry.required),
                    usage: entry.usage,
                    peak: entry.peak.as_ref().map(StoredPeakWindows::from),
                    skip_if_pointer: entry.skip_if_pointer.clone(),
                    tiers: stored_tiers(&entry.tiers),
                    tier_mode: stored_tier_mode(entry.tier_mode),
                    tier_by: entry.tier_by.clone(),
                })
                .collect(),
        }
    }
}

impl From<StoredCostConfig> for UninitializedCostConfig {
    fn from(stored: StoredCostConfig) -> Self {
        stored
            .entries
            .into_iter()
            .map(|entry| UninitializedCostConfigEntry {
                pointer: CostPointerConfig {
                    pointer: entry.pointer,
                    pointer_nonstreaming: entry.pointer_nonstreaming,
                    pointer_streaming: entry.pointer_streaming,
                },
                rate: UninitializedCostRate {
                    cost_per_million: entry.cost_per_million,
                    cost_per_unit: entry.cost_per_unit,
                },
                required: entry.required.unwrap_or_default(),
                usage: entry.usage,
                peak: entry.peak.map(UninitializedPeakWindows::from),
                skip_if_pointer: entry.skip_if_pointer,
                tiers: entry
                    .tiers
                    .unwrap_or_default()
                    .into_iter()
                    .map(UninitializedCostTier::from)
                    .collect(),
                tier_mode: entry.tier_mode.unwrap_or_default(),
                tier_by: entry.tier_by,
            })
            .collect()
    }
}

/// Stored equivalent of `UninitializedCostConfigEntry<CostPointerConfig>`.
/// Flattened fields are represented explicitly (no `#[serde(flatten)]`).
#[serde_with::skip_serializing_none]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StoredCostConfigEntry {
    pub pointer: Option<PointerList>,
    pub pointer_nonstreaming: Option<PointerList>,
    pub pointer_streaming: Option<PointerList>,
    pub cost_per_million: Option<Decimal>,
    pub cost_per_unit: Option<Decimal>,
    pub required: Option<bool>,
    #[serde(default)]
    pub usage: Option<UsageField>,
    #[serde(default)]
    pub peak: Option<StoredPeakWindows>,
    #[serde(default)]
    pub skip_if_pointer: Option<PointerList>,
    #[serde(default)]
    pub tiers: Option<Vec<StoredCostTier>>,
    #[serde(default)]
    pub tier_mode: Option<TierMode>,
    #[serde(default)]
    pub tier_by: Option<PointerList>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum StoredPeakWindows {
    One(StoredPeakWindow),
    Many(Vec<StoredPeakWindow>),
}

impl From<&UninitializedPeakWindows> for StoredPeakWindows {
    fn from(peak: &UninitializedPeakWindows) -> Self {
        match peak {
            UninitializedPeakWindows::One(window) => Self::One(StoredPeakWindow::from(window)),
            UninitializedPeakWindows::Many(windows) => {
                Self::Many(windows.iter().map(StoredPeakWindow::from).collect())
            }
        }
    }
}

impl From<StoredPeakWindows> for UninitializedPeakWindows {
    fn from(stored: StoredPeakWindows) -> Self {
        match stored {
            StoredPeakWindows::One(window) => Self::One(window.into()),
            StoredPeakWindows::Many(windows) => Self::Many(
                windows
                    .into_iter()
                    .map(UninitializedPeakWindow::from)
                    .collect(),
            ),
        }
    }
}

#[serde_with::skip_serializing_none]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StoredPeakWindow {
    pub start: String,
    pub end: String,
    pub days: Option<Vec<String>>,
    pub timezone: Option<String>,
    pub cost_per_million: Option<Decimal>,
    pub cost_per_unit: Option<Decimal>,
}

impl From<&UninitializedPeakWindow> for StoredPeakWindow {
    fn from(peak: &UninitializedPeakWindow) -> Self {
        StoredPeakWindow {
            start: peak.start.clone(),
            end: peak.end.clone(),
            days: if peak.days.is_empty() {
                None
            } else {
                Some(peak.days.clone())
            },
            timezone: peak.timezone.clone(),
            cost_per_million: peak.rate.cost_per_million,
            cost_per_unit: peak.rate.cost_per_unit,
        }
    }
}

impl From<StoredPeakWindow> for UninitializedPeakWindow {
    fn from(stored: StoredPeakWindow) -> Self {
        UninitializedPeakWindow {
            start: stored.start,
            end: stored.end,
            days: stored.days.unwrap_or_default(),
            timezone: stored.timezone,
            rate: UninitializedCostRate {
                cost_per_million: stored.cost_per_million,
                cost_per_unit: stored.cost_per_unit,
            },
        }
    }
}

#[serde_with::skip_serializing_none]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StoredCostTier {
    pub up_to: Option<u64>,
    #[serde(default)]
    pub when: Option<Vec<StoredTierWhen>>,
    pub cost_per_million: Option<Decimal>,
    pub cost_per_unit: Option<Decimal>,
}

impl From<&UninitializedCostTier> for StoredCostTier {
    fn from(tier: &UninitializedCostTier) -> Self {
        StoredCostTier {
            up_to: tier.up_to,
            when: if tier.when.is_empty() {
                None
            } else {
                Some(tier.when.iter().map(StoredTierWhen::from).collect())
            },
            cost_per_million: tier.rate.cost_per_million,
            cost_per_unit: tier.rate.cost_per_unit,
        }
    }
}

impl From<StoredCostTier> for UninitializedCostTier {
    fn from(stored: StoredCostTier) -> Self {
        UninitializedCostTier {
            up_to: stored.up_to,
            when: stored
                .when
                .unwrap_or_default()
                .into_iter()
                .map(UninitializedTierWhen::from)
                .collect(),
            rate: UninitializedCostRate {
                cost_per_million: stored.cost_per_million,
                cost_per_unit: stored.cost_per_unit,
            },
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StoredTierWhen {
    pub pointer: PointerList,
    pub up_to: u64,
}

impl From<&UninitializedTierWhen> for StoredTierWhen {
    fn from(when: &UninitializedTierWhen) -> Self {
        StoredTierWhen {
            pointer: when.pointer.clone(),
            up_to: when.up_to,
        }
    }
}

impl From<StoredTierWhen> for UninitializedTierWhen {
    fn from(stored: StoredTierWhen) -> Self {
        UninitializedTierWhen {
            pointer: stored.pointer,
            up_to: stored.up_to,
        }
    }
}

fn stored_tiers(tiers: &[UninitializedCostTier]) -> Option<Vec<StoredCostTier>> {
    if tiers.is_empty() {
        None
    } else {
        Some(tiers.iter().map(StoredCostTier::from).collect())
    }
}

fn stored_tier_mode(mode: TierMode) -> Option<TierMode> {
    if mode == TierMode::default() {
        None
    } else {
        Some(mode)
    }
}

// --- Unified cost config (provider-level `batch_cost` field) ---

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StoredUnifiedCostConfig {
    pub entries: Vec<StoredUnifiedCostConfigEntry>,
}

impl From<&UninitializedUnifiedCostConfig> for StoredUnifiedCostConfig {
    fn from(config: &UninitializedUnifiedCostConfig) -> Self {
        StoredUnifiedCostConfig {
            entries: config
                .iter()
                .map(|entry| StoredUnifiedCostConfigEntry {
                    pointer: entry.pointer.pointer.clone(),
                    cost_per_million: entry.rate.cost_per_million,
                    cost_per_unit: entry.rate.cost_per_unit,
                    required: Some(entry.required),
                    usage: entry.usage,
                    peak: entry.peak.as_ref().map(StoredPeakWindows::from),
                    skip_if_pointer: entry.skip_if_pointer.clone(),
                    tiers: stored_tiers(&entry.tiers),
                    tier_mode: stored_tier_mode(entry.tier_mode),
                    tier_by: entry.tier_by.clone(),
                })
                .collect(),
        }
    }
}

impl From<StoredUnifiedCostConfig> for UninitializedUnifiedCostConfig {
    fn from(stored: StoredUnifiedCostConfig) -> Self {
        stored
            .entries
            .into_iter()
            .map(|entry| UninitializedCostConfigEntry {
                pointer: UnifiedCostPointerConfig {
                    pointer: entry.pointer,
                },
                rate: UninitializedCostRate {
                    cost_per_million: entry.cost_per_million,
                    cost_per_unit: entry.cost_per_unit,
                },
                required: entry.required.unwrap_or_default(),
                usage: entry.usage,
                peak: entry.peak.map(UninitializedPeakWindows::from),
                skip_if_pointer: entry.skip_if_pointer,
                tiers: entry
                    .tiers
                    .unwrap_or_default()
                    .into_iter()
                    .map(UninitializedCostTier::from)
                    .collect(),
                tier_mode: entry.tier_mode.unwrap_or_default(),
                tier_by: entry.tier_by,
            })
            .collect()
    }
}

/// Stored equivalent of `UninitializedCostConfigEntry<UnifiedCostPointerConfig>`.
/// Flattened fields are represented explicitly (no `#[serde(flatten)]`).
#[serde_with::skip_serializing_none]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StoredUnifiedCostConfigEntry {
    pub pointer: PointerList,
    pub cost_per_million: Option<Decimal>,
    pub cost_per_unit: Option<Decimal>,
    pub required: Option<bool>,
    #[serde(default)]
    pub usage: Option<UsageField>,
    #[serde(default)]
    pub peak: Option<StoredPeakWindows>,
    #[serde(default)]
    pub skip_if_pointer: Option<PointerList>,
    #[serde(default)]
    pub tiers: Option<Vec<StoredCostTier>>,
    #[serde(default)]
    pub tier_mode: Option<TierMode>,
    #[serde(default)]
    pub tier_by: Option<PointerList>,
}
