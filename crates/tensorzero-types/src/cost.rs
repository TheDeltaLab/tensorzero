// Modified by Delta-AI under Apache 2.0
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};

pub type UninitializedCostConfig = Vec<UninitializedCostConfigEntry>;

pub type UninitializedUnifiedCostConfig =
    Vec<UninitializedCostConfigEntry<UnifiedCostPointerConfig>>;

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct UninitializedCostConfigEntry<P = CostPointerConfig> {
    #[serde(flatten)]
    pub pointer: P,
    #[serde(flatten)]
    pub rate: UninitializedCostRate,
    #[serde(default)]
    pub required: bool,
    /// When set, the extracted pointer value is also written onto this usage field
    /// (`input` / `output` / `cache_read` / `cache_write`).
    #[serde(default)]
    pub usage: Option<UsageField>,
    /// Optional peak-hours rate(s). A single table or an array of windows.
    /// Outside every window the base `cost_per_*` / `tiers` schedule is used.
    #[serde(default)]
    pub peak: Option<UninitializedPeakWindows>,
    /// Skip this entry when any of these pointers resolve to a number `> 0`
    /// (e.g. Omni: skip text-output billing when audio output is present).
    #[serde(default)]
    pub skip_if_pointer: Option<PointerList>,
    /// Token-length schedule. When set, do not also set a base `cost_per_*`.
    #[serde(default)]
    pub tiers: Vec<UninitializedCostTier>,
    /// How `tiers` are applied. Defaults to `bucket` (whole request uses one band).
    #[serde(default)]
    pub tier_mode: TierMode,
    /// Pointer(s) whose value selects the `bucket` band. Defaults to this entry's billed pointer.
    /// Use this so output/cache rates can follow input length (GLM-5.1).
    #[serde(default)]
    pub tier_by: Option<PointerList>,
}

impl<P: Default> Default for UninitializedCostConfigEntry<P> {
    fn default() -> Self {
        Self {
            pointer: P::default(),
            rate: UninitializedCostRate::default(),
            required: false,
            usage: None,
            peak: None,
            skip_if_pointer: None,
            tiers: Vec::new(),
            tier_mode: TierMode::default(),
            tier_by: None,
        }
    }
}

/// Which TensorZero usage field a cost pointer should populate.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum UsageField {
    Input,
    Output,
    CacheRead,
    CacheWrite,
}

/// One JSON Pointer or a list tried in order (first present wins).
///
/// Use a list when Chat Completions and Responses (or streaming envelopes)
/// expose the same quantity at different paths.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(untagged)]
pub enum PointerList {
    One(String),
    Many(Vec<String>),
}

impl PointerList {
    pub fn one(value: impl Into<String>) -> Self {
        Self::One(value.into())
    }

    pub fn into_vec(self) -> Vec<String> {
        match self {
            Self::One(value) => vec![value],
            Self::Many(values) => values,
        }
    }

    pub fn as_slice(&self) -> &[String] {
        match self {
            Self::One(value) => std::slice::from_ref(value),
            Self::Many(values) => values,
        }
    }
}

/// Peak window list: a single inline table or an array of windows.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(untagged)]
pub enum UninitializedPeakWindows {
    One(UninitializedPeakWindow),
    Many(Vec<UninitializedPeakWindow>),
}

impl UninitializedPeakWindows {
    pub fn into_vec(self) -> Vec<UninitializedPeakWindow> {
        match self {
            Self::One(window) => vec![window],
            Self::Many(windows) => windows,
        }
    }

    pub fn as_slice(&self) -> &[UninitializedPeakWindow] {
        match self {
            Self::One(window) => std::slice::from_ref(window),
            Self::Many(windows) => windows,
        }
    }
}

impl From<UninitializedPeakWindow> for UninitializedPeakWindows {
    fn from(window: UninitializedPeakWindow) -> Self {
        Self::One(window)
    }
}

/// Peak / off-peak window. `start` is inclusive, `end` is exclusive (`HH:MM`).
/// If `start` is later than `end`, the window wraps midnight.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct UninitializedPeakWindow {
    pub start: String,
    pub end: String,
    /// Weekdays this window applies to (`mon`…`sun`, `weekday`, `weekend`).
    /// Empty means every day.
    #[serde(default)]
    pub days: Vec<String>,
    /// IANA timezone (e.g. `Asia/Shanghai`). Defaults to `UTC`.
    #[serde(default)]
    pub timezone: Option<String>,
    #[serde(flatten)]
    pub rate: UninitializedCostRate,
}

/// One band in a token-length schedule.
///
/// `up_to` is an exclusive upper bound (tokens). Omit it on the last band for "and above".
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct UninitializedCostTier {
    #[serde(default)]
    pub up_to: Option<u64>,
    /// Extra AND conditions (e.g. GLM-4.7 output band that also depends on completion length).
    #[serde(default)]
    pub when: Vec<UninitializedTierWhen>,
    #[serde(flatten)]
    pub rate: UninitializedCostRate,
}

/// Additional exclusive upper bound on another pointer, ANDed with the tier's `up_to`.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct UninitializedTierWhen {
    pub pointer: PointerList,
    pub up_to: u64,
}

/// How length bands are applied.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TierMode {
    /// Whole billed quantity uses the matching band (GLM-5.1, 通义按「单次请求输入长度」).
    #[default]
    Bucket,
    /// Each token is billed at the band it falls into (marginal / 累进).
    Progressive,
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct UninitializedCostRate {
    #[serde(default, with = "crate::serde_utils::decimal_float_option")]
    pub cost_per_million: Option<Decimal>,
    #[serde(default, with = "crate::serde_utils::decimal_float_option")]
    pub cost_per_unit: Option<Decimal>,
}

impl UninitializedCostRate {
    pub fn is_empty(&self) -> bool {
        self.cost_per_million.is_none() && self.cost_per_unit.is_none()
    }
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct CostPointerConfig {
    #[serde(default)]
    pub pointer: Option<PointerList>,
    #[serde(default)]
    pub pointer_nonstreaming: Option<PointerList>,
    #[serde(default)]
    pub pointer_streaming: Option<PointerList>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct UnifiedCostPointerConfig {
    pub pointer: PointerList,
}

impl Default for UnifiedCostPointerConfig {
    fn default() -> Self {
        Self {
            pointer: PointerList::One(String::new()),
        }
    }
}

/// ISO 4217 alphabetic currency code (e.g. `USD`, `CNY`).
///
/// Stored as three ASCII letters so [`crate::Usage`] can stay `Copy`.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct Currency([u8; 3]);

impl Currency {
    pub const USD: Self = Self(*b"USD");
    pub const CNY: Self = Self(*b"CNY");
    pub const EUR: Self = Self(*b"EUR");

    /// Parse a 3-letter alphabetic code. `RMB` is accepted as an alias for `CNY`.
    pub fn parse(value: &str) -> Result<Self, String> {
        let normalized = value.trim().to_ascii_uppercase();
        let code = match normalized.as_str() {
            "RMB" => "CNY",
            other => other,
        };
        if code.len() != 3 || !code.bytes().all(|byte| byte.is_ascii_alphabetic()) {
            return Err(format!(
                "invalid currency `{value}`: expected a 3-letter ISO 4217 code (e.g. `USD`, `CNY`)"
            ));
        }
        let mut bytes = [0u8; 3];
        bytes.copy_from_slice(code.as_bytes());
        Ok(Self(bytes))
    }

    pub fn as_str(&self) -> &str {
        std::str::from_utf8(&self.0).unwrap_or("USD")
    }
}

impl Default for Currency {
    fn default() -> Self {
        Self::USD
    }
}

impl std::fmt::Display for Currency {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl serde::Serialize for Currency {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(self.as_str())
    }
}

impl<'de> serde::Deserialize<'de> for Currency {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let value = String::deserialize(deserializer)?;
        Self::parse(&value).map_err(serde::de::Error::custom)
    }
}

#[cfg(test)]
mod tests {
    use super::Currency;

    #[test]
    fn parse_iso_codes_and_rmb_alias() {
        assert_eq!(
            Currency::parse("usd").expect("USD should parse"),
            Currency::USD
        );
        assert_eq!(
            Currency::parse("CNY").expect("CNY should parse"),
            Currency::CNY
        );
        assert_eq!(
            Currency::parse("rmb").expect("RMB should alias to CNY"),
            Currency::CNY
        );
        assert_eq!(
            Currency::parse("EUR").expect("EUR should parse").as_str(),
            "EUR"
        );
    }

    #[test]
    fn reject_invalid_currency_codes() {
        assert!(
            Currency::parse("US").is_err(),
            "2-letter codes should be rejected"
        );
        assert!(
            Currency::parse("USDT").is_err(),
            "4-letter codes should be rejected"
        );
        assert!(Currency::parse("US1").is_err(), "digits should be rejected");
    }

    #[test]
    fn default_currency_is_usd() {
        assert_eq!(Currency::default(), Currency::USD);
    }
}
