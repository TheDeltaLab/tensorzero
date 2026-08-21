// Modified by Delta-AI under Apache 2.0
use chrono::{DateTime, Datelike, NaiveTime, Utc, Weekday};
use chrono_tz::Tz;
use json_pointer::JsonPointer;
use rust_decimal::Decimal;
use rust_decimal::prelude::ToPrimitive;
use serde_json::Value;
use tensorzero_types::{
    CostPointerConfig, Currency, PointerList, TierMode, UninitializedCostConfig,
    UninitializedCostConfigEntry, UninitializedCostRate, UninitializedCostTier,
    UninitializedPeakWindow, UninitializedPeakWindows, UninitializedTierWhen,
    UninitializedUnifiedCostConfig, Usage, UsageField,
};

use crate::error::{Error, ErrorDetails};

/// Decimal type alias for cost values.
pub type Cost = Decimal;

/// Whether the response being costed came from streaming or non-streaming inference.
/// Used to select the correct pointer for `Split` pointer configurations.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResponseMode {
    Streaming,
    NonStreaming,
}

/// Instant used to select peak vs off-peak rates.
#[derive(Clone, Copy, Debug)]
pub struct CostClock {
    pub at: DateTime<Utc>,
}

impl CostClock {
    pub fn now() -> Self {
        Self { at: Utc::now() }
    }
}

/// Usage fields extracted from cost pointers (`usage = "input"` etc.).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct UsageOverrides {
    pub input_tokens: Option<u32>,
    pub output_tokens: Option<u32>,
    pub cache_read: Option<u32>,
    pub cache_write: Option<u32>,
}

impl UsageOverrides {
    pub fn apply_to(&self, usage: &mut Usage) {
        if let Some(value) = self.input_tokens {
            usage.input_tokens = Some(value);
        }
        if let Some(value) = self.output_tokens {
            usage.output_tokens = Some(value);
        }
        if let Some(value) = self.cache_read {
            usage.provider_cache_read_input_tokens = Some(value);
        }
        if let Some(value) = self.cache_write {
            usage.provider_cache_write_input_tokens = Some(value);
        }
    }

    fn set_field(&mut self, field: UsageField, value: Decimal) {
        let Some(tokens) = decimal_to_u32(value) else {
            return;
        };
        match field {
            UsageField::Input => self.input_tokens = Some(tokens),
            UsageField::Output => self.output_tokens = Some(tokens),
            UsageField::CacheRead => self.cache_read = Some(tokens),
            UsageField::CacheWrite => self.cache_write = Some(tokens),
        }
    }
}

fn decimal_to_u32(value: Decimal) -> Option<u32> {
    if value.is_sign_negative() {
        return None;
    }
    value.round().to_u32()
}

/// Compute the cost of a provider response by resolving JSON pointers in the raw response
/// and multiplying extracted values by configured rates.
pub fn compute_cost(
    raw_response: &str,
    cost_config: &CostConfig,
    mode: ResponseMode,
) -> Result<Cost, Error> {
    compute_cost_at(raw_response, cost_config, mode, CostClock::now()).map(|(cost, _)| cost)
}

/// Like [`compute_cost`], and also returns usage fields tagged with `usage = "input"` etc.
pub fn compute_cost_at(
    raw_response: &str,
    cost_config: &CostConfig,
    mode: ResponseMode,
    clock: CostClock,
) -> Result<(Cost, UsageOverrides), Error> {
    let json: Value = serde_json::from_str(raw_response).map_err(|e| {
        Error::new(ErrorDetails::CostComputation {
            message: format!("raw response is not valid JSON: {e}"),
        })
    })?;
    let mut total = Decimal::ZERO;
    let mut usage = UsageOverrides::default();
    let lookup = |pointer: &str| lookup_in_json(&json, pointer);

    for entry in cost_config {
        if should_skip(entry, &lookup)? {
            continue;
        }
        let pointers = pointers_for_mode(entry, mode);
        match first_numeric(&lookup, pointers)? {
            Some(numeric) => {
                apply_extracted_value(entry, numeric, clock, &lookup, &mut total, &mut usage)?;
            }
            None => {
                if entry.required {
                    return Err(missing_required_pointer(pointers));
                }
            }
        }
    }

    finalize_cost(total, usage)
}

/// Compute cost from multiple streaming chunks by scanning all chunks per cost config pointer.
///
/// For each config entry, resolves the pointer against every chunk and takes the maximum value found.
/// This correctly handles both:
/// - Cumulative providers (e.g. OpenAI): the same pointer appears in multiple chunks with increasing values → max is correct.
/// - Split-usage providers (e.g. Anthropic): different pointers resolve from different chunks → each max is that pointer's only value.
pub fn compute_cost_from_streaming_chunks(
    raw_chunks: &[&str],
    cost_config: &CostConfig,
) -> Result<Cost, Error> {
    compute_cost_from_streaming_chunks_at(raw_chunks, cost_config, CostClock::now())
        .map(|(cost, _)| cost)
}

/// Like [`compute_cost_from_streaming_chunks`], with an explicit clock and usage overlays.
pub fn compute_cost_from_streaming_chunks_at(
    raw_chunks: &[&str],
    cost_config: &CostConfig,
    clock: CostClock,
) -> Result<(Cost, UsageOverrides), Error> {
    let parsed_chunks: Vec<Value> = raw_chunks
        .iter()
        .filter_map(|raw_chunk| serde_json::from_str(raw_chunk).ok())
        .collect();
    let lookup = |pointer: &str| max_numeric_in_values(&parsed_chunks, pointer);
    let mut total = Decimal::ZERO;
    let mut usage = UsageOverrides::default();

    for entry in cost_config {
        if should_skip(entry, &lookup)? {
            continue;
        }
        let pointers = pointers_for_mode(entry, ResponseMode::Streaming);
        match first_numeric(&lookup, pointers)? {
            Some(numeric) => {
                apply_extracted_value(entry, numeric, clock, &lookup, &mut total, &mut usage)?;
            }
            None => {
                if entry.required {
                    return Err(missing_required_pointer(pointers));
                }
            }
        }
    }

    finalize_cost(total, usage)
}

/// Compute cost and overlay `usage = "..."` fields onto `usage`.
///
/// On computation failure the existing usage values are left unchanged (same as
/// the previous `compute_cost(...).ok()` call sites).
pub fn apply_computed_cost(
    usage: &mut Usage,
    raw_response: &str,
    cost_config: &CostConfig,
    mode: ResponseMode,
) {
    apply_computed_cost_at(usage, raw_response, cost_config, mode, CostClock::now());
}

pub fn apply_computed_cost_at(
    usage: &mut Usage,
    raw_response: &str,
    cost_config: &CostConfig,
    mode: ResponseMode,
    clock: CostClock,
) {
    if let Ok((cost, overrides)) = compute_cost_at(raw_response, cost_config, mode, clock) {
        usage.cost = Some(cost);
        usage.currency = currency_of(cost_config);
        overrides.apply_to(usage);
    }
}

pub fn apply_computed_cost_from_streaming_chunks(
    usage: &mut Usage,
    raw_chunks: &[&str],
    cost_config: &CostConfig,
) {
    if let Ok((cost, overrides)) =
        compute_cost_from_streaming_chunks_at(raw_chunks, cost_config, CostClock::now())
    {
        usage.cost = Some(cost);
        usage.currency = currency_of(cost_config);
        overrides.apply_to(usage);
    }
}

fn currency_of(cost_config: &CostConfig) -> Option<Currency> {
    cost_config.first().map(|entry| entry.currency)
}

fn pointers_for_mode(entry: &CostConfigEntry, mode: ResponseMode) -> &[String] {
    match &entry.pointer {
        NormalizedCostPointerConfig::Unified { pointers } => pointers,
        NormalizedCostPointerConfig::Split {
            pointer_nonstreaming,
            pointer_streaming,
        } => match mode {
            ResponseMode::NonStreaming => pointer_nonstreaming,
            ResponseMode::Streaming => pointer_streaming,
        },
    }
}

fn lookup_in_json(json: &Value, pointer: &str) -> Result<Option<Decimal>, Error> {
    match json.pointer(pointer) {
        Some(value) => {
            let numeric = value_to_decimal(value).ok_or_else(|| {
                Error::new(ErrorDetails::CostComputation {
                    message: format!("value at JSON pointer `{pointer}` is not numeric"),
                })
            })?;
            Ok(Some(numeric))
        }
        None => Ok(None),
    }
}

fn max_numeric_in_values(values: &[Value], pointer: &str) -> Result<Option<Decimal>, Error> {
    let mut max_value: Option<Decimal> = None;
    for json in values {
        if let Some(numeric) = lookup_in_json(json, pointer)? {
            max_value = Some(match max_value {
                Some(current) if current >= numeric => current,
                _ => numeric,
            });
        }
    }
    Ok(max_value)
}

fn first_numeric(
    lookup: &impl Fn(&str) -> Result<Option<Decimal>, Error>,
    pointers: &[String],
) -> Result<Option<Decimal>, Error> {
    for pointer in pointers {
        if let Some(numeric) = lookup(pointer)? {
            return Ok(Some(numeric));
        }
    }
    Ok(None)
}

fn should_skip(
    entry: &CostConfigEntry,
    lookup: &impl Fn(&str) -> Result<Option<Decimal>, Error>,
) -> Result<bool, Error> {
    for pointer in &entry.skip_if {
        if lookup(pointer)?.is_some_and(|value| value > Decimal::ZERO) {
            return Ok(true);
        }
    }
    Ok(false)
}

fn missing_required_pointer(pointers: &[String]) -> Error {
    Error::new(ErrorDetails::CostComputation {
        message: format!(
            "required field not found at JSON pointer `{}`",
            pointers.join("` or `")
        ),
    })
}

fn apply_extracted_value(
    entry: &CostConfigEntry,
    value: Decimal,
    clock: CostClock,
    lookup: &impl Fn(&str) -> Result<Option<Decimal>, Error>,
    total: &mut Decimal,
    usage: &mut UsageOverrides,
) -> Result<(), Error> {
    *total += billed_amount(entry, value, clock, lookup)?;
    if let Some(field) = entry.usage {
        usage.set_field(field, value);
    }
    Ok(())
}

fn billed_amount(
    entry: &CostConfigEntry,
    value: Decimal,
    clock: CostClock,
    lookup: &impl Fn(&str) -> Result<Option<Decimal>, Error>,
) -> Result<Decimal, Error> {
    if let Some(peak) = entry.peak.iter().find(|window| window.contains(clock)) {
        return apply_schedule(Some(&peak.rate), &[], TierMode::Bucket, value, lookup, &[]);
    }
    apply_schedule(
        entry.rate.as_ref(),
        &entry.tiers,
        entry.tier_mode,
        value,
        lookup,
        &entry.tier_by,
    )
}

fn apply_schedule(
    rate: Option<&CostRate>,
    tiers: &[CostTier],
    mode: TierMode,
    value: Decimal,
    lookup: &impl Fn(&str) -> Result<Option<Decimal>, Error>,
    tier_by: &[String],
) -> Result<Decimal, Error> {
    if tiers.is_empty() {
        let Some(rate) = rate else {
            return Err(Error::new(ErrorDetails::CostComputation {
                message: "cost entry has neither a flat rate nor tiers".to_string(),
            }));
        };
        return Ok(value * rate.cost_per_unit);
    }
    match mode {
        TierMode::Bucket => bucket_cost(tiers, value, lookup, tier_by),
        TierMode::Progressive => progressive_cost(tiers, value),
    }
}

fn bucket_cost(
    tiers: &[CostTier],
    value: Decimal,
    lookup: &impl Fn(&str) -> Result<Option<Decimal>, Error>,
    tier_by: &[String],
) -> Result<Decimal, Error> {
    let key = if tier_by.is_empty() {
        value
    } else {
        first_numeric(lookup, tier_by)?.unwrap_or(value)
    };
    for tier in tiers {
        if !bucket_matches(tier, key, lookup)? {
            continue;
        }
        return Ok(value * tier.rate.cost_per_unit);
    }
    Err(Error::new(ErrorDetails::CostComputation {
        message: format!("no cost tier matched billed quantity {key}"),
    }))
}

fn bucket_matches(
    tier: &CostTier,
    key: Decimal,
    lookup: &impl Fn(&str) -> Result<Option<Decimal>, Error>,
) -> Result<bool, Error> {
    if let Some(up_to) = tier.up_to
        && key >= up_to
    {
        return Ok(false);
    }
    for condition in &tier.when {
        let Some(other) = first_numeric(lookup, &condition.pointers)? else {
            return Ok(false);
        };
        if other >= condition.up_to {
            return Ok(false);
        }
    }
    Ok(true)
}

fn progressive_cost(tiers: &[CostTier], value: Decimal) -> Result<Decimal, Error> {
    let mut remaining = value.max(Decimal::ZERO);
    let mut previous_bound = Decimal::ZERO;
    let mut total = Decimal::ZERO;
    for tier in tiers {
        if remaining <= Decimal::ZERO {
            break;
        }
        let cap = tier.up_to.unwrap_or(Decimal::MAX);
        let width = (cap - previous_bound).max(Decimal::ZERO);
        let billed = remaining.min(width);
        total += billed * tier.rate.cost_per_unit;
        remaining -= billed;
        previous_bound = cap;
    }
    if remaining > Decimal::ZERO {
        return Err(Error::new(ErrorDetails::CostComputation {
            message: format!(
                "progressive cost tiers do not cover billed quantity {value} (unbilled remainder {remaining})"
            ),
        }));
    }
    Ok(total)
}

fn finalize_cost(total: Decimal, usage: UsageOverrides) -> Result<(Cost, UsageOverrides), Error> {
    if total < Decimal::ZERO {
        return Err(Error::new(ErrorDetails::CostComputation {
            message: format!(
                "computed total cost is negative ({total}), which likely indicates a problematic cost configuration"
            ),
        }));
    }
    Ok((total, usage))
}

/// Convert a JSON value to a Decimal. Handles integers, floats, and string representations.
fn value_to_decimal(value: &Value) -> Option<Decimal> {
    match value {
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Some(Decimal::from(i))
            } else if let Some(u) = n.as_u64() {
                Some(Decimal::from(u))
            } else if let Some(f) = n.as_f64() {
                Decimal::try_from(f).ok()
            } else {
                None
            }
        }
        Value::String(s) => s.parse::<Decimal>().ok(),
        _ => None,
    }
}

// ============================================================================
// Normalized (runtime) types — after validation and rate normalization
// ============================================================================

pub type CostConfig = Vec<CostConfigEntry>;

#[derive(Clone, Debug, Default)]
pub struct CostConfigEntry {
    pub pointer: NormalizedCostPointerConfig,
    pub rate: Option<CostRate>,
    pub required: bool,
    pub usage: Option<UsageField>,
    pub peak: Vec<PeakWindow>,
    pub skip_if: Vec<String>,
    pub tier_by: Vec<String>,
    pub tier_mode: TierMode,
    pub tiers: Vec<CostTier>,
    pub currency: Currency,
}

#[derive(Clone, Debug)]
pub struct CostTier {
    pub up_to: Option<Decimal>,
    pub when: Vec<CostTierWhen>,
    pub rate: CostRate,
}

#[derive(Clone, Debug)]
pub struct CostTierWhen {
    pub pointers: Vec<String>,
    pub up_to: Decimal,
}

#[derive(Clone, Debug)]
pub struct CostRate {
    pub cost_per_unit: Decimal,
}

/// Inclusive start / exclusive end. If `start` is later than `end`, the window wraps midnight.
#[derive(Clone, Debug)]
pub struct PeakWindow {
    pub start: NaiveTime,
    pub end: NaiveTime,
    pub days: PeakDays,
    pub timezone: Tz,
    pub rate: CostRate,
}

impl PeakWindow {
    fn contains(&self, clock: CostClock) -> bool {
        let local = clock.at.with_timezone(&self.timezone);
        if !self.days.contains(local.weekday()) {
            return false;
        }
        in_time_window(local.time(), self.start, self.end)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PeakDays {
    All,
    /// Bitmask: Monday = bit 0 … Sunday = bit 6.
    Selected(u8),
}

impl PeakDays {
    fn contains(self, weekday: Weekday) -> bool {
        match self {
            PeakDays::All => true,
            PeakDays::Selected(mask) => {
                let bit = 1u8 << weekday.num_days_from_monday();
                mask & bit != 0
            }
        }
    }
}

fn in_time_window(time: NaiveTime, start: NaiveTime, end: NaiveTime) -> bool {
    if start == end {
        // A 24-hour window.
        return true;
    }
    if start < end {
        time >= start && time < end
    } else {
        time >= start || time < end
    }
}

#[derive(Clone, Debug)]
pub enum NormalizedCostPointerConfig {
    Unified {
        pointers: Vec<String>,
    },
    Split {
        pointer_nonstreaming: Vec<String>,
        pointer_streaming: Vec<String>,
    },
}

impl Default for NormalizedCostPointerConfig {
    fn default() -> Self {
        Self::Unified {
            pointers: Vec::new(),
        }
    }
}

// ============================================================================
// Validation and normalization
// ============================================================================

fn validate_pointer(pointer: &str) -> Result<(), Error> {
    pointer
        .parse::<JsonPointer<String, Vec<String>>>()
        .map_err(|e| {
            Error::new(ErrorDetails::Config {
                message: format!("invalid JSON pointer `{pointer}`: {e:?}"),
            })
        })?;
    Ok(())
}

pub fn load_cost_config(config: UninitializedCostConfig) -> Result<CostConfig, Error> {
    load_cost_config_with_provider_defaults(config, None, None)
}

pub fn load_cost_config_with_timezone(
    config: UninitializedCostConfig,
    default_timezone: Option<&str>,
) -> Result<CostConfig, Error> {
    load_cost_config_with_provider_defaults(config, default_timezone, None)
}

pub fn load_cost_config_with_provider_defaults(
    config: UninitializedCostConfig,
    default_timezone: Option<&str>,
    currency: Option<&str>,
) -> Result<CostConfig, Error> {
    let currency = parse_currency(currency)?;
    config
        .into_iter()
        .map(|entry| load_cost_config_entry(entry, default_timezone, currency))
        .collect()
}

/// Load a cost config that only allows unified (non-split) pointers.
///
/// Used for embedding models (which don't support streaming) and batch cost configs.
pub fn load_unified_cost_config(
    config: UninitializedUnifiedCostConfig,
) -> Result<CostConfig, Error> {
    load_unified_cost_config_with_provider_defaults(config, None, None)
}

pub fn load_unified_cost_config_with_timezone(
    config: UninitializedUnifiedCostConfig,
    default_timezone: Option<&str>,
) -> Result<CostConfig, Error> {
    load_unified_cost_config_with_provider_defaults(config, default_timezone, None)
}

pub fn load_unified_cost_config_with_provider_defaults(
    config: UninitializedUnifiedCostConfig,
    default_timezone: Option<&str>,
    currency: Option<&str>,
) -> Result<CostConfig, Error> {
    let currency = parse_currency(currency)?;
    config
        .into_iter()
        .map(|entry| load_unified_cost_config_entry(entry, default_timezone, currency))
        .collect()
}

fn parse_pointer_config(config: CostPointerConfig) -> Result<NormalizedCostPointerConfig, Error> {
    match (
        config.pointer,
        config.pointer_nonstreaming,
        config.pointer_streaming,
    ) {
        (Some(pointer), None, None) => Ok(NormalizedCostPointerConfig::Unified {
            pointers: parse_pointer_list(pointer)?,
        }),
        (None, Some(pointer_nonstreaming), Some(pointer_streaming)) => {
            Ok(NormalizedCostPointerConfig::Split {
                pointer_nonstreaming: parse_pointer_list(pointer_nonstreaming)?,
                pointer_streaming: parse_pointer_list(pointer_streaming)?,
            })
        }
        _ => Err(Error::new(ErrorDetails::Config {
            message: "invalid pointer configuration: specify either `pointer` alone, or both `pointer_nonstreaming` and `pointer_streaming`".to_string(),
        })),
    }
}

fn parse_pointer_list(list: PointerList) -> Result<Vec<String>, Error> {
    let pointers = list.into_vec();
    if pointers.is_empty() {
        return Err(Error::new(ErrorDetails::Config {
            message: "pointer list must not be empty".to_string(),
        }));
    }
    for pointer in &pointers {
        validate_pointer(pointer)?;
    }
    Ok(pointers)
}

fn parse_optional_pointer_list(list: Option<PointerList>) -> Result<Vec<String>, Error> {
    match list {
        Some(list) => parse_pointer_list(list),
        None => Ok(Vec::new()),
    }
}

fn parse_rate(rate: UninitializedCostRate) -> Result<CostRate, Error> {
    let cost_per_unit = match (rate.cost_per_million, rate.cost_per_unit) {
        (Some(cost_per_million), None) => cost_per_million / Decimal::from(1_000_000),
        (None, Some(cost_per_unit)) => cost_per_unit,
        _ => {
            return Err(Error::new(ErrorDetails::Config {
                message: "must specify exactly one of `cost_per_million` or `cost_per_unit`"
                    .to_string(),
            }));
        }
    };
    Ok(CostRate { cost_per_unit })
}

fn parse_optional_rate(rate: UninitializedCostRate) -> Result<Option<CostRate>, Error> {
    if rate.is_empty() {
        Ok(None)
    } else {
        Ok(Some(parse_rate(rate)?))
    }
}

fn load_cost_config_entry(
    entry: UninitializedCostConfigEntry,
    default_timezone: Option<&str>,
    currency: Currency,
) -> Result<CostConfigEntry, Error> {
    let pointer = parse_pointer_config(entry.pointer.clone())?;
    assemble_cost_config_entry(pointer, entry, default_timezone, currency)
}

fn load_unified_cost_config_entry(
    entry: UninitializedCostConfigEntry<tensorzero_types::UnifiedCostPointerConfig>,
    default_timezone: Option<&str>,
    currency: Currency,
) -> Result<CostConfigEntry, Error> {
    let pointer = NormalizedCostPointerConfig::Unified {
        pointers: parse_pointer_list(entry.pointer.pointer.clone())?,
    };
    assemble_cost_config_entry(pointer, entry, default_timezone, currency)
}

fn assemble_cost_config_entry<P>(
    pointer: NormalizedCostPointerConfig,
    entry: UninitializedCostConfigEntry<P>,
    default_timezone: Option<&str>,
    currency: Currency,
) -> Result<CostConfigEntry, Error> {
    let tiers = load_tiers(&entry.tiers)?;
    let rate = parse_optional_rate(entry.rate)?;
    if rate.is_none() && tiers.is_empty() {
        return Err(Error::new(ErrorDetails::Config {
            message: "must specify exactly one of `cost_per_million` or `cost_per_unit`"
                .to_string(),
        }));
    }
    if rate.is_some() && !tiers.is_empty() {
        return Err(Error::new(ErrorDetails::Config {
            message: "cannot combine a base `cost_per_million`/`cost_per_unit` with `tiers`"
                .to_string(),
        }));
    }
    Ok(CostConfigEntry {
        pointer,
        rate,
        required: entry.required,
        usage: entry.usage,
        peak: load_peak_windows(entry.peak, default_timezone)?,
        skip_if: parse_optional_pointer_list(entry.skip_if_pointer)?,
        tier_by: parse_optional_pointer_list(entry.tier_by)?,
        tier_mode: entry.tier_mode,
        tiers,
        currency,
    })
}

fn load_tiers(tiers: &[UninitializedCostTier]) -> Result<Vec<CostTier>, Error> {
    tiers.iter().map(load_tier).collect()
}

fn load_tier(tier: &UninitializedCostTier) -> Result<CostTier, Error> {
    Ok(CostTier {
        up_to: tier.up_to.map(Decimal::from),
        when: tier
            .when
            .iter()
            .map(load_tier_when)
            .collect::<Result<_, _>>()?,
        rate: parse_rate(tier.rate.clone())?,
    })
}

fn load_tier_when(when: &UninitializedTierWhen) -> Result<CostTierWhen, Error> {
    Ok(CostTierWhen {
        pointers: parse_pointer_list(when.pointer.clone())?,
        up_to: Decimal::from(when.up_to),
    })
}

fn load_peak_windows(
    peak: Option<UninitializedPeakWindows>,
    default_timezone: Option<&str>,
) -> Result<Vec<PeakWindow>, Error> {
    let Some(peak) = peak else {
        return Ok(Vec::new());
    };
    peak.into_vec()
        .into_iter()
        .map(|window| load_peak_window(window, default_timezone))
        .collect()
}

fn load_peak_window(
    peak: UninitializedPeakWindow,
    default_timezone: Option<&str>,
) -> Result<PeakWindow, Error> {
    let start = parse_clock_time(&peak.start)?;
    let end = parse_clock_time(&peak.end)?;
    let days = parse_peak_days(&peak.days)?;
    let tz_name = peak
        .timezone
        .as_deref()
        .or(default_timezone)
        .unwrap_or("UTC");
    let timezone = parse_timezone(tz_name)?;
    let rate = parse_rate(peak.rate)?;
    Ok(PeakWindow {
        start,
        end,
        days,
        timezone,
        rate,
    })
}

fn parse_timezone(name: &str) -> Result<Tz, Error> {
    name.parse::<Tz>().map_err(|_| {
        Error::new(ErrorDetails::Config {
            message: format!(
                "invalid timezone `{name}`: expected an IANA name (e.g. `UTC`, `Asia/Shanghai`)"
            ),
        })
    })
}

fn parse_currency(value: Option<&str>) -> Result<Currency, Error> {
    let Some(value) = value else {
        return Ok(Currency::USD);
    };
    Currency::parse(value).map_err(|message| Error::new(ErrorDetails::Config { message }))
}

fn parse_clock_time(value: &str) -> Result<NaiveTime, Error> {
    NaiveTime::parse_from_str(value, "%H:%M:%S")
        .or_else(|_| NaiveTime::parse_from_str(value, "%H:%M"))
        .map_err(|_| {
            Error::new(ErrorDetails::Config {
                message: format!("invalid peak time `{value}`: expected `HH:MM` or `HH:MM:SS`"),
            })
        })
}

const PEAK_DAY_MON: u8 = 1 << 0;
const PEAK_DAY_TUE: u8 = 1 << 1;
const PEAK_DAY_WED: u8 = 1 << 2;
const PEAK_DAY_THU: u8 = 1 << 3;
const PEAK_DAY_FRI: u8 = 1 << 4;
const PEAK_DAY_SAT: u8 = 1 << 5;
const PEAK_DAY_SUN: u8 = 1 << 6;
const PEAK_DAY_WEEKDAY: u8 =
    PEAK_DAY_MON | PEAK_DAY_TUE | PEAK_DAY_WED | PEAK_DAY_THU | PEAK_DAY_FRI;
const PEAK_DAY_WEEKEND: u8 = PEAK_DAY_SAT | PEAK_DAY_SUN;

fn parse_peak_days(days: &[String]) -> Result<PeakDays, Error> {
    if days.is_empty() {
        return Ok(PeakDays::All);
    }
    let mut mask: u8 = 0;
    for day in days {
        mask |= peak_day_mask(day)?;
    }
    Ok(PeakDays::Selected(mask))
}

fn peak_day_mask(day: &str) -> Result<u8, Error> {
    match day.trim().to_ascii_lowercase().as_str() {
        "mon" | "monday" => Ok(PEAK_DAY_MON),
        "tue" | "tues" | "tuesday" => Ok(PEAK_DAY_TUE),
        "wed" | "wednesday" => Ok(PEAK_DAY_WED),
        "thu" | "thur" | "thurs" | "thursday" => Ok(PEAK_DAY_THU),
        "fri" | "friday" => Ok(PEAK_DAY_FRI),
        "sat" | "saturday" => Ok(PEAK_DAY_SAT),
        "sun" | "sunday" => Ok(PEAK_DAY_SUN),
        "weekday" | "weekdays" => Ok(PEAK_DAY_WEEKDAY),
        "weekend" | "weekends" => Ok(PEAK_DAY_WEEKEND),
        other => Err(Error::new(ErrorDetails::Config {
            message: format!(
                "invalid peak day `{other}`: expected `mon`…`sun`, `weekday`, or `weekend`"
            ),
        })),
    }
}

#[cfg(test)]
mod tests {
    use serde::Deserialize;

    use super::*;

    fn unified(pointer: &str) -> CostPointerConfig {
        CostPointerConfig {
            pointer: Some(PointerList::one(pointer)),
            pointer_nonstreaming: None,
            pointer_streaming: None,
        }
    }

    fn split(nonstreaming: &str, streaming: &str) -> CostPointerConfig {
        CostPointerConfig {
            pointer: None,
            pointer_nonstreaming: Some(PointerList::one(nonstreaming)),
            pointer_streaming: Some(PointerList::one(streaming)),
        }
    }

    fn loaded_rate(entry: &CostConfigEntry) -> Decimal {
        entry
            .rate
            .as_ref()
            .expect("entry should have a flat rate")
            .cost_per_unit
    }

    fn per_million(value: Decimal) -> UninitializedCostRate {
        UninitializedCostRate {
            cost_per_million: Some(value),
            cost_per_unit: None,
        }
    }

    fn per_unit(value: Decimal) -> UninitializedCostRate {
        UninitializedCostRate {
            cost_per_million: None,
            cost_per_unit: Some(value),
        }
    }

    #[derive(Deserialize)]
    struct UninitializedCostConfigWrapper {
        cost: UninitializedCostConfig,
    }

    // ========================================================================
    // Rate conversion tests
    // ========================================================================

    #[test]
    fn test_per_million_to_per_unit_normalization() {
        let config = vec![UninitializedCostConfigEntry {
            pointer: unified("/usage/input_tokens"),
            rate: per_million(Decimal::from(3)),
            required: false,
            usage: None,
            peak: None,
            ..Default::default()
        }];
        let result = load_cost_config(config).expect("should load successfully");
        assert_eq!(result.len(), 1, "should have one entry");
        let expected = Decimal::from(3) / Decimal::from(1_000_000);
        assert_eq!(
            loaded_rate(&result[0]),
            expected,
            "per_million rate should be divided by 1,000,000"
        );
    }

    #[test]
    fn test_per_unit_passthrough() {
        let config = vec![UninitializedCostConfigEntry {
            pointer: unified("/usage/total_cost"),
            rate: per_unit(Decimal::new(5, 2)), // 0.05
            required: true,
            usage: None,
            peak: None,
            ..Default::default()
        }];
        let result = load_cost_config(config).expect("should load successfully");
        assert_eq!(result.len(), 1, "should have one entry");
        assert_eq!(
            loaded_rate(&result[0]),
            Decimal::new(5, 2),
            "per_unit rate should pass through unchanged"
        );
        assert!(result[0].required, "required flag should be preserved");
    }

    // ========================================================================
    // TOML deserialization tests
    // ========================================================================

    #[test]
    fn test_deserialize_unified_pointer_per_million() {
        let toml_str = r#"
[[cost]]
pointer = "/usage/input_tokens"
cost_per_million = 3.0
"#;

        let wrapper: UninitializedCostConfigWrapper =
            toml::from_str(toml_str).expect("should deserialize");
        assert_eq!(wrapper.cost.len(), 1, "should have one cost entry");
        assert!(
            wrapper.cost[0].pointer.pointer.is_some(),
            "should have a unified pointer"
        );
        assert!(
            wrapper.cost[0].rate.cost_per_million.is_some(),
            "should have a per-million rate"
        );
        assert!(
            wrapper.cost[0].rate.cost_per_unit.is_none(),
            "should not have a per-unit rate"
        );
    }

    #[test]
    fn test_deserialize_per_unit_rate() {
        let toml_str = r#"
[[cost]]
pointer = "/cost"
cost_per_unit = 0.05
"#;

        let wrapper: UninitializedCostConfigWrapper =
            toml::from_str(toml_str).expect("should deserialize");
        assert!(
            wrapper.cost[0].rate.cost_per_unit.is_some(),
            "should have a per-unit rate"
        );
        assert!(
            wrapper.cost[0].rate.cost_per_million.is_none(),
            "should not have a per-million rate"
        );
    }

    #[test]
    fn test_deserialize_split_pointers() {
        let toml_str = r#"
[[cost]]
pointer_nonstreaming = "/usage/total"
pointer_streaming = "/usage/stream_total"
cost_per_million = 1.5
"#;

        let wrapper: UninitializedCostConfigWrapper =
            toml::from_str(toml_str).expect("should deserialize");
        assert!(
            wrapper.cost[0].pointer.pointer_nonstreaming.is_some(),
            "should have nonstreaming pointer"
        );
        assert!(
            wrapper.cost[0].pointer.pointer_streaming.is_some(),
            "should have streaming pointer"
        );
        assert!(
            wrapper.cost[0].pointer.pointer.is_none(),
            "should not have unified pointer"
        );
    }

    #[test]
    fn test_invalid_pointer_rejected() {
        let config = vec![UninitializedCostConfigEntry {
            pointer: unified("no_leading_slash"),
            rate: per_million(Decimal::from(1)),
            required: false,
            usage: None,
            peak: None,
            ..Default::default()
        }];
        let err = load_cost_config(config).expect_err("should fail on invalid pointer");
        let msg = err.to_string();
        assert!(
            msg.contains("invalid JSON pointer"),
            "error should mention invalid JSON pointer: {msg}"
        );
    }

    #[test]
    fn test_invalid_split_pointer_rejected() {
        let config = vec![UninitializedCostConfigEntry {
            pointer: split("/valid", "invalid_no_slash"),
            rate: per_unit(Decimal::from(1)),
            required: false,
            usage: None,
            peak: None,
            ..Default::default()
        }];
        let err = load_cost_config(config).expect_err("should fail on invalid split pointer");
        let msg = err.to_string();
        assert!(
            msg.contains("invalid JSON pointer"),
            "error should mention invalid JSON pointer: {msg}"
        );
    }

    #[test]
    fn test_missing_rate_rejected() {
        let toml_str = r#"
[[cost]]
pointer = "/usage/tokens"
"#;

        let wrapper: UninitializedCostConfigWrapper = toml::from_str(toml_str)
            .expect("should deserialize with both rates defaulting to None");
        let err =
            load_cost_config(wrapper.cost).expect_err("should fail when neither rate is specified");
        let msg = err.to_string();
        assert!(
            msg.contains("must specify exactly one of"),
            "error should mention missing rate: {msg}"
        );
    }

    #[test]
    fn test_missing_pointer_rejected() {
        let toml_str = r"
[[cost]]
cost_per_million = 3.0
";

        let wrapper: UninitializedCostConfigWrapper = toml::from_str(toml_str)
            .expect("should deserialize with all pointers defaulting to None");
        let err =
            load_cost_config(wrapper.cost).expect_err("should fail when no pointer is specified");
        let msg = err.to_string();
        assert!(
            msg.contains("invalid pointer configuration"),
            "error should mention invalid pointer configuration: {msg}"
        );
    }

    #[test]
    fn test_exact_decimal_precision() {
        let toml_str = r#"
[[cost]]
pointer = "/a"
cost_per_unit = 0.1

[[cost]]
pointer = "/b"
cost_per_unit = 0.3
"#;

        let wrapper: UninitializedCostConfigWrapper =
            toml::from_str(toml_str).expect("should deserialize");
        let result = load_cost_config(wrapper.cost).expect("should load");
        assert_eq!(
            loaded_rate(&result[0]),
            Decimal::new(1, 1),
            "0.1 should deserialize with exact decimal precision"
        );
        assert_eq!(
            loaded_rate(&result[1]),
            Decimal::new(3, 1),
            "0.3 should deserialize with exact decimal precision"
        );
    }

    #[test]
    fn test_negative_cost_allowed() {
        let config = vec![UninitializedCostConfigEntry {
            pointer: unified("/discount"),
            rate: per_unit(Decimal::new(-5, 2)), // -0.05
            required: false,
            usage: None,
            peak: None,
            ..Default::default()
        }];
        let result = load_cost_config(config).expect("negative costs should be allowed");
        assert_eq!(
            loaded_rate(&result[0]),
            Decimal::new(-5, 2),
            "negative cost should pass through"
        );
    }

    #[test]
    fn test_both_rates_rejected() {
        let toml_str = r#"
[[cost]]
pointer = "/usage/tokens"
cost_per_million = 3.0
cost_per_unit = 0.5
"#;

        let wrapper: UninitializedCostConfigWrapper =
            toml::from_str(toml_str).expect("should deserialize");
        let err = load_cost_config(wrapper.cost)
            .expect_err("should reject config with both rates specified");
        let msg = err.to_string();
        assert!(
            msg.contains("must specify exactly one of"),
            "error should mention rate requirement: {msg}"
        );
    }

    #[test]
    fn test_mixed_pointer_and_split_rejected() {
        let toml_str = r#"
[[cost]]
pointer = "/usage/tokens"
pointer_nonstreaming = "/usage/ns_tokens"
pointer_streaming = "/usage/s_tokens"
cost_per_million = 3.0
"#;

        let wrapper: UninitializedCostConfigWrapper =
            toml::from_str(toml_str).expect("should deserialize");
        let err = load_cost_config(wrapper.cost)
            .expect_err("should reject config with both unified and split pointers");
        let msg = err.to_string();
        assert!(
            msg.contains("invalid pointer configuration"),
            "error should mention invalid pointer configuration: {msg}"
        );
    }

    #[test]
    fn test_partial_split_pointer_rejected() {
        let config = vec![UninitializedCostConfigEntry {
            pointer: CostPointerConfig {
                pointer: None,
                pointer_nonstreaming: Some(PointerList::one("/usage/ns")),
                pointer_streaming: None,
            },
            rate: per_unit(Decimal::from(1)),
            required: false,
            usage: None,
            peak: None,
            ..Default::default()
        }];
        let err =
            load_cost_config(config).expect_err("should reject config with only one split pointer");
        let msg = err.to_string();
        assert!(
            msg.contains("invalid pointer configuration"),
            "error should mention invalid pointer configuration: {msg}"
        );
    }

    #[test]
    fn test_empty_pointer_string_is_valid_root_pointer() {
        // Per RFC 6901, "" is the root JSON pointer and is valid.
        let config = vec![UninitializedCostConfigEntry {
            pointer: unified(""),
            rate: per_unit(Decimal::from(1)),
            required: false,
            usage: None,
            peak: None,
            ..Default::default()
        }];
        assert!(
            load_cost_config(config).is_ok(),
            "empty string is a valid root JSON pointer per RFC 6901"
        );
    }

    // ========================================================================
    // Config normalization tests
    // ========================================================================

    #[test]
    fn test_load_cost_config_multi_entry() {
        let config = vec![
            UninitializedCostConfigEntry {
                pointer: unified("/usage/input_tokens"),
                rate: per_million(Decimal::from(3)),
                required: true,
                usage: None,
                peak: None,
                ..Default::default()
            },
            UninitializedCostConfigEntry {
                pointer: split("/usage/output_tokens", "/usage/stream_output_tokens"),
                rate: per_unit(Decimal::new(15, 6)), // 0.000015
                required: false,
                usage: None,
                peak: None,
                ..Default::default()
            },
        ];

        let result = load_cost_config(config).expect("should load multi-entry config");
        assert_eq!(result.len(), 2, "should have two entries");

        // First entry: per_million normalized
        let expected_first = Decimal::from(3) / Decimal::from(1_000_000);
        assert_eq!(
            loaded_rate(&result[0]),
            expected_first,
            "first entry rate should be normalized from per_million"
        );
        assert!(result[0].required, "first entry should be required");
        assert!(
            matches!(
                result[0].pointer,
                NormalizedCostPointerConfig::Unified { .. }
            ),
            "first entry should have a unified pointer"
        );

        // Second entry: per_unit passthrough
        assert_eq!(
            loaded_rate(&result[1]),
            Decimal::new(15, 6),
            "second entry rate should pass through unchanged"
        );
        assert!(!result[1].required, "second entry should not be required");
        assert!(
            matches!(result[1].pointer, NormalizedCostPointerConfig::Split { .. }),
            "second entry should have split pointers"
        );
    }

    // ========================================================================
    // JSON Pointer validation tests
    // ========================================================================

    #[test]
    fn test_valid_json_pointer() {
        let config = vec![UninitializedCostConfigEntry {
            pointer: unified("/foo/bar"),
            rate: per_unit(Decimal::from(1)),
            required: false,
            usage: None,
            peak: None,
            ..Default::default()
        }];
        assert!(
            load_cost_config(config).is_ok(),
            "valid JSON pointer `/foo/bar` should be accepted"
        );
    }

    #[test]
    fn test_invalid_json_pointer_no_leading_slash() {
        let config = vec![UninitializedCostConfigEntry {
            pointer: unified("foo/bar"),
            rate: per_unit(Decimal::from(1)),
            required: false,
            usage: None,
            peak: None,
            ..Default::default()
        }];
        assert!(
            load_cost_config(config).is_err(),
            "JSON pointer without leading `/` should be rejected"
        );
    }

    // ========================================================================
    // compute_cost tests
    // ========================================================================

    fn make_config_entry(
        pointer: NormalizedCostPointerConfig,
        cost_per_unit: Decimal,
        required: bool,
    ) -> CostConfigEntry {
        CostConfigEntry {
            pointer,
            rate: Some(CostRate { cost_per_unit }),
            required,
            ..Default::default()
        }
    }

    fn unified_config(pointer: &str, cost_per_unit: Decimal, required: bool) -> CostConfigEntry {
        make_config_entry(
            NormalizedCostPointerConfig::Unified {
                pointers: vec![pointer.to_string()],
            },
            cost_per_unit,
            required,
        )
    }

    fn split_config(
        nonstreaming: &str,
        streaming: &str,
        cost_per_unit: Decimal,
        required: bool,
    ) -> CostConfigEntry {
        make_config_entry(
            NormalizedCostPointerConfig::Split {
                pointer_nonstreaming: vec![nonstreaming.to_string()],
                pointer_streaming: vec![streaming.to_string()],
            },
            cost_per_unit,
            required,
        )
    }

    #[test]
    fn test_compute_cost_per_unit_unified() {
        let raw = r#"{"usage": {"prompt_tokens": 100, "completion_tokens": 50}}"#;
        let config = vec![
            unified_config(
                "/usage/prompt_tokens",
                Decimal::from(3) / Decimal::from(1_000_000),
                false,
            ),
            unified_config(
                "/usage/completion_tokens",
                Decimal::from(15) / Decimal::from(1_000_000),
                false,
            ),
        ];
        let cost = compute_cost(raw, &config, ResponseMode::NonStreaming)
            .expect("should compute cost successfully");
        let expected = Decimal::from(100) * Decimal::from(3) / Decimal::from(1_000_000)
            + Decimal::from(50) * Decimal::from(15) / Decimal::from(1_000_000);
        assert_eq!(
            cost, expected,
            "cost should be sum of (tokens * rate) for each entry"
        );
    }

    #[test]
    fn test_compute_cost_per_unit_direct() {
        let raw = r#"{"cost": 0.05}"#;
        let config = vec![unified_config(
            "/cost",
            Decimal::from(1), // cost_per_unit = 1 means use value directly
            true,
        )];
        let cost = compute_cost(raw, &config, ResponseMode::NonStreaming)
            .expect("should compute cost successfully");
        assert_eq!(
            cost,
            Decimal::new(5, 2),
            "should extract cost directly when rate is 1"
        );
    }

    #[test]
    fn test_compute_cost_split_pointers_nonstreaming() {
        let raw = r#"{"usage": {"total": 200, "stream_total": 999}}"#;
        let config = vec![split_config(
            "/usage/total",
            "/usage/stream_total",
            Decimal::from(1) / Decimal::from(1_000_000),
            false,
        )];
        let cost = compute_cost(raw, &config, ResponseMode::NonStreaming)
            .expect("should compute cost successfully");
        let expected = Decimal::from(200) / Decimal::from(1_000_000);
        assert_eq!(
            cost, expected,
            "non-streaming mode should use nonstreaming pointer"
        );
    }

    #[test]
    fn test_compute_cost_split_pointers_streaming() {
        let raw = r#"{"usage": {"total": 200, "stream_total": 300}}"#;
        let config = vec![split_config(
            "/usage/total",
            "/usage/stream_total",
            Decimal::from(1) / Decimal::from(1_000_000),
            false,
        )];
        let cost = compute_cost(raw, &config, ResponseMode::Streaming)
            .expect("should compute cost successfully");
        let expected = Decimal::from(300) / Decimal::from(1_000_000);
        assert_eq!(
            cost, expected,
            "streaming mode should use streaming pointer"
        );
    }

    #[test]
    fn test_compute_cost_required_field_missing_returns_err() {
        let raw = r#"{"usage": {"prompt_tokens": 100}}"#;
        let config = vec![
            unified_config("/usage/prompt_tokens", Decimal::from(1), false),
            unified_config("/usage/completion_tokens", Decimal::from(1), true), // required but missing
        ];
        let err = compute_cost(raw, &config, ResponseMode::NonStreaming)
            .expect_err("should return Err when a required field is missing");
        assert!(
            err.to_string().contains("required field not found"),
            "should mention missing required field: {err}"
        );
    }

    #[test]
    fn test_compute_cost_nonrequired_field_missing() {
        let raw = r#"{"usage": {"prompt_tokens": 100}}"#;
        let config = vec![
            unified_config("/usage/prompt_tokens", Decimal::from(1), false),
            unified_config("/usage/completion_tokens", Decimal::from(1), false), // not required
        ];
        let cost = compute_cost(raw, &config, ResponseMode::NonStreaming)
            .expect("should compute cost successfully");
        assert_eq!(
            cost,
            Decimal::from(100),
            "should skip non-required missing fields"
        );
    }

    #[test]
    fn test_compute_cost_invalid_json_returns_err() {
        let raw = "not valid json";
        let config = vec![unified_config("/usage/tokens", Decimal::from(1), false)];
        let err = compute_cost(raw, &config, ResponseMode::NonStreaming)
            .expect_err("should return Err for invalid JSON");
        assert!(
            err.to_string().contains("not valid JSON"),
            "should mention invalid JSON: {err}"
        );
    }

    #[test]
    fn test_compute_cost_negative_total_returns_err() {
        // Set up a config where the total will be negative
        let raw = r#"{"tokens": 10}"#;
        let config = vec![unified_config(
            "/tokens",
            Decimal::new(-5, 0), // -5 per unit → total = -50
            false,
        )];
        let err = compute_cost(raw, &config, ResponseMode::NonStreaming)
            .expect_err("should return Err when total cost is negative");
        assert!(
            err.to_string().contains("negative"),
            "should mention negative total: {err}"
        );
    }

    #[test]
    fn test_compute_cost_empty_config() {
        let raw = r#"{"usage": {"tokens": 100}}"#;
        let config: CostConfig = vec![];
        let cost = compute_cost(raw, &config, ResponseMode::NonStreaming)
            .expect("should compute cost successfully");
        assert_eq!(cost, Decimal::ZERO, "empty config should return Ok(0)");
    }

    #[test]
    fn test_compute_cost_non_numeric_value_returns_err() {
        let raw = r#"{"usage": {"tokens": "not_a_number"}}"#;
        let config = vec![unified_config("/usage/tokens", Decimal::from(1), false)];
        let err = compute_cost(raw, &config, ResponseMode::NonStreaming)
            .expect_err("should return Err for non-numeric field values");
        assert!(
            err.to_string().contains("not numeric"),
            "should mention non-numeric value: {err}"
        );
    }

    #[test]
    fn test_compute_cost_string_numeric_value() {
        let raw = r#"{"usage": {"tokens": "100"}}"#;
        let config = vec![unified_config("/usage/tokens", Decimal::from(1), false)];
        let cost = compute_cost(raw, &config, ResponseMode::NonStreaming)
            .expect("should compute cost successfully");
        assert_eq!(cost, Decimal::from(100), "should parse numeric strings");
    }

    #[test]
    fn test_compute_cost_boolean_value_returns_err() {
        let raw = r#"{"usage": {"tokens": true}}"#;
        let config = vec![unified_config("/usage/tokens", Decimal::from(1), false)];
        let err = compute_cost(raw, &config, ResponseMode::NonStreaming)
            .expect_err("should return Err for boolean field values");
        assert!(
            err.to_string().contains("not numeric"),
            "should mention non-numeric value: {err}"
        );
    }

    // ========================================================================
    // compute_cost_from_streaming_chunks tests
    // ========================================================================

    #[test]
    fn test_streaming_chunks_split_usage_anthropic_style() {
        // Anthropic sends input_tokens in message_start and output_tokens in message_delta
        let chunk1 = r#"{"usage": {"input_tokens": 69}}"#;
        let chunk2 = r#"{"usage": {"output_tokens": 100}}"#;
        let input_rate = Decimal::from(3) / Decimal::from(1_000_000);
        let output_rate = Decimal::from(15) / Decimal::from(1_000_000);
        let config = vec![
            split_config(
                "/usage/input_tokens",
                "/usage/input_tokens",
                input_rate,
                false,
            ),
            split_config(
                "/usage/output_tokens",
                "/usage/output_tokens",
                output_rate,
                false,
            ),
        ];
        let chunks: Vec<&str> = vec![chunk1, chunk2];
        let cost = compute_cost_from_streaming_chunks(&chunks, &config)
            .expect("should compute cost successfully");
        let expected = Decimal::from(69) * input_rate + Decimal::from(100) * output_rate;
        assert_eq!(
            cost, expected,
            "should sum costs from different chunks for split-usage providers"
        );
    }

    #[test]
    fn test_streaming_chunks_single_chunk_openai_style() {
        // OpenAI sends all usage in a single final chunk
        let chunk = r#"{"usage": {"prompt_tokens": 100, "completion_tokens": 50}}"#;
        let config = vec![
            unified_config(
                "/usage/prompt_tokens",
                Decimal::from(3) / Decimal::from(1_000_000),
                false,
            ),
            unified_config(
                "/usage/completion_tokens",
                Decimal::from(15) / Decimal::from(1_000_000),
                false,
            ),
        ];
        let chunks: Vec<&str> = vec![chunk];
        let cost = compute_cost_from_streaming_chunks(&chunks, &config)
            .expect("should compute cost successfully");
        let expected = Decimal::from(100) * Decimal::from(3) / Decimal::from(1_000_000)
            + Decimal::from(50) * Decimal::from(15) / Decimal::from(1_000_000);
        assert_eq!(
            cost, expected,
            "should compute correct cost from a single chunk"
        );
    }

    #[test]
    fn test_streaming_chunks_cumulative_values() {
        // Provider sends cumulative token counts across chunks
        let chunk1 = r#"{"usage": {"tokens": 50}}"#;
        let chunk2 = r#"{"usage": {"tokens": 100}}"#;
        let chunk3 = r#"{"usage": {"tokens": 150}}"#;
        let config = vec![unified_config("/usage/tokens", Decimal::from(1), false)];
        let chunks: Vec<&str> = vec![chunk1, chunk2, chunk3];
        let cost = compute_cost_from_streaming_chunks(&chunks, &config)
            .expect("should compute cost successfully");
        assert_eq!(
            cost,
            Decimal::from(150),
            "should take the max value across cumulative chunks"
        );
    }

    #[test]
    fn test_streaming_chunks_required_field_missing_from_all() {
        let chunk1 = r#"{"usage": {"other": 10}}"#;
        let chunk2 = r#"{"usage": {"other": 20}}"#;
        let config = vec![unified_config("/usage/tokens", Decimal::from(1), true)];
        let chunks: Vec<&str> = vec![chunk1, chunk2];
        let err = compute_cost_from_streaming_chunks(&chunks, &config)
            .expect_err("should return Err when a required field is not found in any chunk");
        assert!(
            err.to_string().contains("required field not found"),
            "should mention missing required field: {err}"
        );
    }

    #[test]
    fn test_streaming_chunks_empty_chunks_list() {
        let config = vec![unified_config("/usage/tokens", Decimal::from(1), false)];
        let chunks: Vec<&str> = vec![];
        let cost = compute_cost_from_streaming_chunks(&chunks, &config)
            .expect("should compute cost successfully");
        assert_eq!(
            cost,
            Decimal::ZERO,
            "empty chunks with non-required fields should return Ok(0)"
        );
    }

    #[test]
    fn test_streaming_chunks_empty_chunks_required_field() {
        let config = vec![unified_config("/usage/tokens", Decimal::from(1), true)];
        let chunks: Vec<&str> = vec![];
        let err = compute_cost_from_streaming_chunks(&chunks, &config)
            .expect_err("should return Err for empty chunks with required fields");
        assert!(
            err.to_string().contains("required field not found"),
            "should mention missing required field: {err}"
        );
    }

    #[test]
    fn test_streaming_chunks_invalid_json_skipped() {
        let chunk1 = "not valid json";
        let chunk2 = r#"{"usage": {"tokens": 42}}"#;
        let config = vec![unified_config("/usage/tokens", Decimal::from(1), false)];
        let chunks: Vec<&str> = vec![chunk1, chunk2];
        let cost = compute_cost_from_streaming_chunks(&chunks, &config)
            .expect("should compute cost successfully");
        assert_eq!(
            cost,
            Decimal::from(42),
            "should skip invalid JSON chunks and use valid ones"
        );
    }

    #[test]
    fn test_streaming_chunks_negative_total_returns_err() {
        let chunk = r#"{"tokens": 10}"#;
        let config = vec![unified_config(
            "/tokens",
            Decimal::new(-5, 0), // -5 per unit → total = -50
            false,
        )];
        let chunks: Vec<&str> = vec![chunk];
        let err = compute_cost_from_streaming_chunks(&chunks, &config)
            .expect_err("should return Err when total cost is negative");
        assert!(
            err.to_string().contains("negative"),
            "should mention negative total: {err}"
        );
    }

    // ========================================================================
    // load_unified_cost_config tests
    // ========================================================================

    fn unified_entry(
        pointer: &str,
        rate: UninitializedCostRate,
        required: bool,
    ) -> UninitializedCostConfigEntry<tensorzero_types::UnifiedCostPointerConfig> {
        UninitializedCostConfigEntry {
            pointer: tensorzero_types::UnifiedCostPointerConfig {
                pointer: PointerList::one(pointer),
            },
            rate,
            required,
            ..Default::default()
        }
    }

    #[test]
    fn test_load_unified_cost_config_valid() {
        let config = vec![
            unified_entry("/usage/input_tokens", per_million(Decimal::from(1)), true),
            unified_entry("/usage/output_tokens", per_unit(Decimal::new(5, 6)), false),
        ];
        let result =
            load_unified_cost_config(config).expect("should load valid unified cost config");
        assert_eq!(result.len(), 2, "should have two entries");

        assert!(
            matches!(
                result[0].pointer,
                NormalizedCostPointerConfig::Unified { .. }
            ),
            "unified cost entries should always be unified pointers"
        );
        assert!(result[0].required, "first entry should be required");

        let expected_rate = Decimal::from(1) / Decimal::from(1_000_000);
        assert_eq!(
            loaded_rate(&result[0]),
            expected_rate,
            "per_million rate should be normalized"
        );
    }

    #[test]
    fn test_load_unified_cost_config_invalid_pointer_rejected() {
        let config = vec![unified_entry(
            "no_leading_slash",
            per_million(Decimal::from(1)),
            false,
        )];
        let err = load_unified_cost_config(config)
            .expect_err("should fail on invalid unified cost pointer");
        let msg = err.to_string();
        assert!(
            msg.contains("invalid JSON pointer"),
            "error should mention invalid JSON pointer: {msg}"
        );
    }

    #[test]
    fn test_load_unified_cost_config_missing_rate_rejected() {
        let config = vec![unified_entry(
            "/usage/tokens",
            UninitializedCostRate {
                cost_per_million: None,
                cost_per_unit: None,
            },
            false,
        )];
        let err = load_unified_cost_config(config)
            .expect_err("should fail when neither rate is specified");
        let msg = err.to_string();
        assert!(
            msg.contains("must specify exactly one of"),
            "error should mention missing rate: {msg}"
        );
    }

    #[derive(Deserialize)]
    struct UninitializedUnifiedCostConfigWrapper {
        batch_cost: UninitializedUnifiedCostConfig,
    }

    #[test]
    fn test_deserialize_unified_cost_config() {
        let toml_str = r#"
[[batch_cost]]
pointer = "/usage/input_tokens"
cost_per_million = 1.5

[[batch_cost]]
pointer = "/usage/output_tokens"
cost_per_million = 6.0
required = true
"#;

        let wrapper: UninitializedUnifiedCostConfigWrapper =
            toml::from_str(toml_str).expect("should deserialize unified cost config");
        assert_eq!(
            wrapper.batch_cost.len(),
            2,
            "should have two unified cost entries"
        );
        assert_eq!(
            wrapper.batch_cost[0].pointer.pointer,
            PointerList::one("/usage/input_tokens"),
            "first pointer should match"
        );
        assert!(
            !wrapper.batch_cost[0].required,
            "required should default to false"
        );
        assert!(
            wrapper.batch_cost[1].required,
            "required should be true when set"
        );
    }

    fn clock_rfc3339(value: &str) -> CostClock {
        CostClock {
            at: DateTime::parse_from_rfc3339(value)
                .expect("valid RFC 3339 timestamp")
                .with_timezone(&Utc),
        }
    }

    fn shanghai_peak() -> UninitializedPeakWindow {
        UninitializedPeakWindow {
            start: "08:00".to_string(),
            end: "22:00".to_string(),
            days: vec!["weekday".to_string()],
            timezone: Some("Asia/Shanghai".to_string()),
            rate: per_million(Decimal::from(2)),
        }
    }

    #[test]
    fn test_deserialize_usage_and_peak() {
        let toml_str = r#"
[[cost]]
pointer = "/usage/prompt_tokens"
cost_per_million = 0.8
usage = "input"
required = true
peak = { start = "08:00", end = "22:00", days = ["weekday"], timezone = "Asia/Shanghai", cost_per_million = 2.0 }
"#;
        let wrapper: UninitializedCostConfigWrapper =
            toml::from_str(toml_str).expect("should deserialize usage and peak");
        assert_eq!(
            wrapper.cost[0].usage,
            Some(UsageField::Input),
            "usage field should deserialize"
        );
        let peak = wrapper.cost[0]
            .peak
            .as_ref()
            .expect("peak window should deserialize")
            .as_slice();
        assert_eq!(
            peak.len(),
            1,
            "single peak table should deserialize as one window"
        );
        assert_eq!(peak[0].start, "08:00", "peak start should match");
        assert_eq!(peak[0].end, "22:00", "peak end should match");
        assert_eq!(
            peak[0].days,
            vec!["weekday".to_string()],
            "peak days should match"
        );
        assert_eq!(
            peak[0].timezone.as_deref(),
            Some("Asia/Shanghai"),
            "peak timezone should match"
        );
        assert_eq!(
            peak[0].rate.cost_per_million,
            Some(Decimal::new(20, 1)),
            "peak rate should deserialize as 2.0"
        );
    }

    #[test]
    fn test_usage_overlay_from_pointer() {
        let config = load_cost_config(vec![UninitializedCostConfigEntry {
            pointer: unified("/usage/prompt_tokens"),
            rate: per_million(Decimal::from(1)),
            required: true,
            usage: Some(UsageField::Input),
            peak: None,
            ..Default::default()
        }])
        .expect("should load");
        let raw = r#"{"usage": {"prompt_tokens": 42, "completion_tokens": 7}}"#;
        let (cost, usage) =
            compute_cost_at(raw, &config, ResponseMode::NonStreaming, CostClock::now())
                .expect("should compute");
        assert_eq!(
            cost,
            Decimal::from(42) / Decimal::from(1_000_000),
            "cost should use the extracted prompt token count"
        );
        assert_eq!(
            usage.input_tokens,
            Some(42),
            "usage=input should overlay prompt_tokens"
        );
        assert_eq!(usage.output_tokens, None, "output should be untouched");
    }

    #[test]
    fn test_peak_weekday_uses_peak_rate() {
        let config = load_cost_config(vec![UninitializedCostConfigEntry {
            pointer: unified("/usage/prompt_tokens"),
            rate: per_million(Decimal::from(1)),
            required: true,
            usage: Some(UsageField::Input),
            peak: Some(shanghai_peak().into()),
            ..Default::default()
        }])
        .expect("should load");
        // Thursday 10:00 Asia/Shanghai
        let clock = clock_rfc3339("2026-08-20T02:00:00Z");
        let (cost, _) = compute_cost_at(
            r#"{"usage": {"prompt_tokens": 1000000}}"#,
            &config,
            ResponseMode::NonStreaming,
            clock,
        )
        .expect("should compute");
        assert_eq!(
            cost,
            Decimal::from(2),
            "weekday 10:00 Shanghai should use the peak rate of $2 / million"
        );
    }

    #[test]
    fn test_peak_weekend_uses_offpeak_rate() {
        let config = load_cost_config(vec![UninitializedCostConfigEntry {
            pointer: unified("/usage/prompt_tokens"),
            rate: per_million(Decimal::from(1)),
            required: true,
            usage: None,
            peak: Some(shanghai_peak().into()),
            ..Default::default()
        }])
        .expect("should load");
        // Saturday 10:00 Asia/Shanghai
        let clock = clock_rfc3339("2026-08-22T02:00:00Z");
        let (cost, _) = compute_cost_at(
            r#"{"usage": {"prompt_tokens": 1000000}}"#,
            &config,
            ResponseMode::NonStreaming,
            clock,
        )
        .expect("should compute");
        assert_eq!(
            cost,
            Decimal::from(1),
            "weekend 10:00 Shanghai should use the off-peak rate of $1 / million"
        );
    }

    #[test]
    fn test_peak_outside_hours_uses_offpeak_rate() {
        let config = load_cost_config(vec![UninitializedCostConfigEntry {
            pointer: unified("/usage/prompt_tokens"),
            rate: per_million(Decimal::from(1)),
            required: true,
            usage: None,
            peak: Some(shanghai_peak().into()),
            ..Default::default()
        }])
        .expect("should load");
        // Thursday 23:00 Asia/Shanghai
        let clock = clock_rfc3339("2026-08-20T15:00:00Z");
        let (cost, _) = compute_cost_at(
            r#"{"usage": {"prompt_tokens": 1000000}}"#,
            &config,
            ResponseMode::NonStreaming,
            clock,
        )
        .expect("should compute");
        assert_eq!(
            cost,
            Decimal::from(1),
            "weekday 23:00 Shanghai should use the off-peak rate"
        );
    }

    #[test]
    fn test_overnight_peak_window() {
        let config = load_cost_config(vec![UninitializedCostConfigEntry {
            pointer: unified("/tokens"),
            rate: per_unit(Decimal::from(1)),
            required: true,
            usage: None,
            peak: Some(
                UninitializedPeakWindow {
                    start: "22:00".to_string(),
                    end: "08:00".to_string(),
                    days: vec![],
                    timezone: Some("UTC".to_string()),
                    rate: per_unit(Decimal::from(10)),
                }
                .into(),
            ),
            ..Default::default()
        }])
        .expect("should load");
        let (night, _) = compute_cost_at(
            r#"{"tokens": 1}"#,
            &config,
            ResponseMode::NonStreaming,
            clock_rfc3339("2026-08-20T23:00:00Z"),
        )
        .expect("night should compute");
        let (day, _) = compute_cost_at(
            r#"{"tokens": 1}"#,
            &config,
            ResponseMode::NonStreaming,
            clock_rfc3339("2026-08-20T10:00:00Z"),
        )
        .expect("day should compute");
        assert_eq!(
            night,
            Decimal::from(10),
            "23:00 should be inside overnight peak"
        );
        assert_eq!(
            day,
            Decimal::from(1),
            "10:00 should be outside overnight peak"
        );
    }

    #[test]
    fn test_provider_timezone_default_for_peak() {
        let config = load_cost_config_with_timezone(
            vec![UninitializedCostConfigEntry {
                pointer: unified("/usage/prompt_tokens"),
                rate: per_million(Decimal::from(1)),
                required: true,
                usage: None,
                peak: Some(
                    UninitializedPeakWindow {
                        start: "08:00".to_string(),
                        end: "22:00".to_string(),
                        days: vec!["weekday".to_string()],
                        timezone: None,
                        rate: per_million(Decimal::from(2)),
                    }
                    .into(),
                ),
                ..Default::default()
            }],
            Some("Asia/Shanghai"),
        )
        .expect("should load with provider timezone");
        let (cost, _) = compute_cost_at(
            r#"{"usage": {"prompt_tokens": 1000000}}"#,
            &config,
            ResponseMode::NonStreaming,
            clock_rfc3339("2026-08-20T02:00:00Z"),
        )
        .expect("should compute");
        assert_eq!(
            cost,
            Decimal::from(2),
            "peak window should inherit provider timezone Asia/Shanghai"
        );
    }

    #[test]
    fn test_invalid_timezone_rejected() {
        let err = load_cost_config(vec![UninitializedCostConfigEntry {
            pointer: unified("/tokens"),
            rate: per_unit(Decimal::from(1)),
            required: false,
            usage: None,
            peak: Some(
                UninitializedPeakWindow {
                    start: "08:00".to_string(),
                    end: "22:00".to_string(),
                    days: vec![],
                    timezone: Some("Not/AZone".to_string()),
                    rate: per_unit(Decimal::from(2)),
                }
                .into(),
            ),
            ..Default::default()
        }])
        .expect_err("invalid timezone should be rejected");
        assert!(
            err.to_string().contains("invalid timezone"),
            "error should mention invalid timezone: {err}"
        );
    }

    #[test]
    fn test_apply_computed_cost_overlays_usage() {
        let config = load_cost_config(vec![
            UninitializedCostConfigEntry {
                pointer: unified("/usage/prompt_tokens"),
                rate: per_million(Decimal::from(1)),
                required: true,
                usage: Some(UsageField::Input),
                peak: None,
                ..Default::default()
            },
            UninitializedCostConfigEntry {
                pointer: unified("/usage/completion_tokens"),
                rate: per_million(Decimal::from(2)),
                required: true,
                usage: Some(UsageField::Output),
                peak: None,
                ..Default::default()
            },
        ])
        .expect("should load");
        let mut usage = Usage::zero();
        apply_computed_cost(
            &mut usage,
            r#"{"usage": {"prompt_tokens": 10, "completion_tokens": 20}}"#,
            &config,
            ResponseMode::NonStreaming,
        );
        assert_eq!(usage.input_tokens, Some(10), "input overlay");
        assert_eq!(usage.output_tokens, Some(20), "output overlay");
        assert!(usage.cost.is_some(), "cost should be set");
    }

    #[test]
    fn test_deserialize_pointer_list_and_multi_peak() {
        let toml_str = r#"
[[cost]]
pointer = ["/usage/prompt_cache_miss_tokens", "/usage/input_tokens"]
cost_per_million = 1.5
peak = [
  { start = "09:00", end = "12:00", cost_per_million = 3.0 },
  { start = "14:00", end = "18:00", cost_per_million = 3.0 },
]
"#;
        let wrapper: UninitializedCostConfigWrapper =
            toml::from_str(toml_str).expect("should deserialize pointer list and peak array");
        assert_eq!(
            wrapper.cost[0].pointer.pointer,
            Some(PointerList::Many(vec![
                "/usage/prompt_cache_miss_tokens".to_string(),
                "/usage/input_tokens".to_string(),
            ])),
            "pointer should deserialize as an ordered list"
        );
        let peaks = wrapper.cost[0]
            .peak
            .as_ref()
            .expect("peak array should deserialize")
            .as_slice();
        assert_eq!(peaks.len(), 2, "two peak windows should deserialize");
        assert_eq!(peaks[0].start, "09:00", "first peak start");
        assert_eq!(peaks[1].start, "14:00", "second peak start");
    }

    #[test]
    fn test_deepseek_style_two_peak_windows() {
        let config = load_cost_config_with_timezone(
            vec![UninitializedCostConfigEntry {
                pointer: unified("/usage/prompt_tokens"),
                rate: per_million(Decimal::new(15, 1)),
                required: true,
                peak: Some(UninitializedPeakWindows::Many(vec![
                    UninitializedPeakWindow {
                        start: "09:00".to_string(),
                        end: "12:00".to_string(),
                        days: vec![],
                        timezone: None,
                        rate: per_million(Decimal::from(3)),
                    },
                    UninitializedPeakWindow {
                        start: "14:00".to_string(),
                        end: "18:00".to_string(),
                        days: vec![],
                        timezone: None,
                        rate: per_million(Decimal::from(3)),
                    },
                ])),
                ..Default::default()
            }],
            Some("Asia/Shanghai"),
        )
        .expect("should load two peak windows");
        let raw = r#"{"usage": {"prompt_tokens": 1000000}}"#;
        let morning = compute_cost_at(
            raw,
            &config,
            ResponseMode::NonStreaming,
            clock_rfc3339("2026-08-20T02:00:00Z"),
        )
        .expect("10:00 Shanghai should compute")
        .0;
        let lunch = compute_cost_at(
            raw,
            &config,
            ResponseMode::NonStreaming,
            clock_rfc3339("2026-08-20T05:00:00Z"),
        )
        .expect("13:00 Shanghai should compute")
        .0;
        let afternoon = compute_cost_at(
            raw,
            &config,
            ResponseMode::NonStreaming,
            clock_rfc3339("2026-08-20T07:00:00Z"),
        )
        .expect("15:00 Shanghai should compute")
        .0;
        assert_eq!(
            morning,
            Decimal::from(3),
            "09:00–12:00 Shanghai should use the peak rate"
        );
        assert_eq!(
            lunch,
            Decimal::new(15, 1),
            "12:00–14:00 Shanghai should use the off-peak rate"
        );
        assert_eq!(
            afternoon,
            Decimal::from(3),
            "14:00–18:00 Shanghai should use the peak rate"
        );
    }

    #[test]
    fn test_pointer_list_first_present_wins() {
        let config = load_cost_config(vec![UninitializedCostConfigEntry {
            pointer: CostPointerConfig {
                pointer: Some(PointerList::Many(vec![
                    "/usage/prompt_cache_miss_tokens".to_string(),
                    "/usage/input_tokens".to_string(),
                ])),
                pointer_nonstreaming: None,
                pointer_streaming: None,
            },
            rate: per_unit(Decimal::from(1)),
            required: true,
            ..Default::default()
        }])
        .expect("should load pointer list");
        let chat = compute_cost(
            r#"{"usage": {"prompt_cache_miss_tokens": 11, "completion_tokens": 3}}"#,
            &config,
            ResponseMode::NonStreaming,
        )
        .expect("chat usage should compute");
        let responses = compute_cost(
            r#"{"usage": {"input_tokens": 7, "output_tokens": 2}}"#,
            &config,
            ResponseMode::NonStreaming,
        )
        .expect("responses usage should compute");
        assert_eq!(
            chat,
            Decimal::from(11),
            "chat path should use the first present pointer"
        );
        assert_eq!(
            responses,
            Decimal::from(7),
            "responses path should fall through to the second pointer"
        );
    }

    #[test]
    fn test_glm51_style_bucket_tiers() {
        let toml_str = r#"
[[cost]]
pointer = "/usage/prompt_tokens"
usage = "input"
required = true
tiers = [
  { up_to = 32000, cost_per_million = 6 },
  { cost_per_million = 8 },
]

[[cost]]
pointer = "/usage/completion_tokens"
usage = "output"
required = true
tier_by = "/usage/prompt_tokens"
tiers = [
  { up_to = 32000, cost_per_million = 24 },
  { cost_per_million = 28 },
]
"#;
        let wrapper: UninitializedCostConfigWrapper =
            toml::from_str(toml_str).expect("should deserialize GLM-5.1 style tiers");
        let config = load_cost_config(wrapper.cost).expect("should load bucket tiers");
        let cheap = compute_cost(
            r#"{"usage": {"prompt_tokens": 31999, "completion_tokens": 1000000}}"#,
            &config,
            ResponseMode::NonStreaming,
        )
        .expect("short prompt should compute");
        let expensive = compute_cost(
            r#"{"usage": {"prompt_tokens": 32000, "completion_tokens": 1000000}}"#,
            &config,
            ResponseMode::NonStreaming,
        )
        .expect("long prompt should compute");
        let expected_cheap =
            Decimal::from(31999) * Decimal::from(6) / Decimal::from(1_000_000) + Decimal::from(24);
        let expected_expensive =
            Decimal::from(32000) * Decimal::from(8) / Decimal::from(1_000_000) + Decimal::from(28);
        assert_eq!(
            cheap, expected_cheap,
            "input below 32k should use the cheap input and output rates"
        );
        assert_eq!(
            expensive, expected_expensive,
            "input at 32k should use the expensive input and output rates"
        );
    }

    #[test]
    fn test_progressive_tiers() {
        let config = load_cost_config(vec![UninitializedCostConfigEntry {
            pointer: unified("/usage/prompt_tokens"),
            tiers: vec![
                UninitializedCostTier {
                    up_to: Some(32_000),
                    when: vec![],
                    rate: per_unit(Decimal::from(1)),
                },
                UninitializedCostTier {
                    up_to: None,
                    when: vec![],
                    rate: per_unit(Decimal::from(2)),
                },
            ],
            tier_mode: TierMode::Progressive,
            required: true,
            ..Default::default()
        }])
        .expect("should load progressive tiers");
        let cost = compute_cost(
            r#"{"usage": {"prompt_tokens": 40000}}"#,
            &config,
            ResponseMode::NonStreaming,
        )
        .expect("progressive cost should compute");
        assert_eq!(
            cost,
            Decimal::from(48_000),
            "40k tokens should bill 32k at 1 plus 8k at 2"
        );
    }

    #[test]
    fn test_skip_if_pointer() {
        let config = load_cost_config(vec![
            UninitializedCostConfigEntry {
                pointer: unified("/usage/completion_tokens"),
                rate: per_unit(Decimal::from(1)),
                skip_if_pointer: Some(PointerList::one(
                    "/usage/completion_tokens_details/audio_tokens",
                )),
                required: true,
                ..Default::default()
            },
            UninitializedCostConfigEntry {
                pointer: unified("/usage/completion_tokens_details/audio_tokens"),
                rate: per_unit(Decimal::from(10)),
                ..Default::default()
            },
        ])
        .expect("should load skip_if");
        let with_audio = compute_cost(
            r#"{"usage": {"completion_tokens": 20, "completion_tokens_details": {"audio_tokens": 5}}}"#,
            &config,
            ResponseMode::NonStreaming,
        )
        .expect("audio output should compute");
        let text_only = compute_cost(
            r#"{"usage": {"completion_tokens": 20}}"#,
            &config,
            ResponseMode::NonStreaming,
        )
        .expect("text output should compute");
        assert_eq!(
            with_audio,
            Decimal::from(50),
            "audio output should skip the text line and bill audio tokens"
        );
        assert_eq!(
            text_only,
            Decimal::from(20),
            "text-only output should bill the text line"
        );
    }

    #[test]
    fn test_tier_when_condition() {
        let config = load_cost_config(vec![UninitializedCostConfigEntry {
            pointer: unified("/usage/prompt_tokens"),
            required: true,
            tiers: vec![
                UninitializedCostTier {
                    up_to: Some(32_000),
                    when: vec![UninitializedTierWhen {
                        pointer: PointerList::one("/usage/completion_tokens"),
                        up_to: 8_000,
                    }],
                    rate: per_unit(Decimal::from(1)),
                },
                UninitializedCostTier {
                    up_to: Some(32_000),
                    when: vec![],
                    rate: per_unit(Decimal::from(2)),
                },
                UninitializedCostTier {
                    up_to: None,
                    when: vec![],
                    rate: per_unit(Decimal::from(3)),
                },
            ],
            ..Default::default()
        }])
        .expect("should load when conditions");
        let short_out = compute_cost(
            r#"{"usage": {"prompt_tokens": 10000, "completion_tokens": 100}}"#,
            &config,
            ResponseMode::NonStreaming,
        )
        .expect("short completion should compute");
        let long_out = compute_cost(
            r#"{"usage": {"prompt_tokens": 10000, "completion_tokens": 9000}}"#,
            &config,
            ResponseMode::NonStreaming,
        )
        .expect("long completion should compute");
        let long_in = compute_cost(
            r#"{"usage": {"prompt_tokens": 40000, "completion_tokens": 100}}"#,
            &config,
            ResponseMode::NonStreaming,
        )
        .expect("long prompt should compute");
        assert_eq!(
            short_out,
            Decimal::from(10_000),
            "short input and output should match the first band"
        );
        assert_eq!(
            long_out,
            Decimal::from(20_000),
            "short input and long output should match the second band"
        );
        assert_eq!(
            long_in,
            Decimal::from(120_000),
            "input at 40k should match the unbounded band"
        );
    }

    #[test]
    fn test_base_rate_and_tiers_rejected() {
        let toml_str = r#"
[[cost]]
pointer = "/usage/prompt_tokens"
cost_per_million = 6
tiers = [{ cost_per_million = 8 }]
"#;
        let wrapper: UninitializedCostConfigWrapper =
            toml::from_str(toml_str).expect("should deserialize mixed rate and tiers");
        let err = load_cost_config(wrapper.cost)
            .expect_err("should reject combining a base rate with tiers");
        assert!(
            err.to_string().contains("cannot combine"),
            "error should mention combining rate with tiers: {err}"
        );
    }

    #[test]
    fn test_provider_currency_stamped_on_entries() {
        let config = load_cost_config_with_provider_defaults(
            vec![UninitializedCostConfigEntry {
                pointer: unified("/usage/prompt_tokens"),
                rate: per_million(Decimal::from(6)),
                required: true,
                usage: Some(UsageField::Input),
                peak: None,
                ..Default::default()
            }],
            None,
            Some("rmb"),
        )
        .expect("RMB should load as CNY");
        assert_eq!(
            config[0].currency,
            Currency::CNY,
            "provider currency RMB should normalize to CNY"
        );

        let mut usage = Usage::default();
        apply_computed_cost_at(
            &mut usage,
            r#"{"usage": {"prompt_tokens": 1000000}}"#,
            &config,
            ResponseMode::NonStreaming,
            CostClock::now(),
        );
        assert_eq!(
            usage.cost,
            Some(Decimal::from(6)),
            "computed cost should use the configured rate"
        );
        assert_eq!(
            usage.currency,
            Some(Currency::CNY),
            "computed usage should carry the provider currency"
        );
    }

    #[test]
    fn test_default_currency_is_usd() {
        let config = load_cost_config(vec![UninitializedCostConfigEntry {
            pointer: unified("/usage/prompt_tokens"),
            rate: per_million(Decimal::from(1)),
            required: true,
            usage: None,
            peak: None,
            ..Default::default()
        }])
        .expect("should load");
        assert_eq!(
            config[0].currency,
            Currency::USD,
            "unset provider currency should default to USD"
        );
    }

    #[test]
    fn test_invalid_currency_rejected() {
        let err = load_cost_config_with_provider_defaults(
            vec![UninitializedCostConfigEntry {
                pointer: unified("/usage/prompt_tokens"),
                rate: per_million(Decimal::from(1)),
                required: true,
                usage: None,
                peak: None,
                ..Default::default()
            }],
            None,
            Some("US"),
        )
        .expect_err("invalid currency should fail load");
        assert!(
            err.to_string().contains("invalid currency"),
            "error should mention invalid currency: {err}"
        );
    }
}
