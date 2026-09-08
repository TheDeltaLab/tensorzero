// Modified by Delta-AI under Apache 2.0
use std::{collections::HashMap, sync::Arc};

use crate::{
    config::with_skip_credential_validation,
    error::{Error, ErrorDetails},
    model::UninitializedProviderConfig,
    model_alias::{ModelAlias, ModelAliasTable, ModelAliasTarget},
    relay::TensorzeroRelay,
};
use lazy_static::lazy_static;
use serde::Serialize;
use strum::VariantNames;

// Reserve prefixes for all supported providers, regardless of whether or not a particular `BaseModelTable`
// currently supports them.
lazy_static! {
    pub static ref RESERVED_MODEL_PREFIXES: Vec<String> = {
        let mut prefixes: Vec<String> = UninitializedProviderConfig::VARIANTS
            .iter()
            .map(|&v| format!("{v}::"))
            .collect();
        prefixes.push("tensorzero::".to_string());
        // OpenAI-compatible Chinese providers are shorthand-only (they reuse
        // OpenAIProvider) so they are not UninitializedProviderConfig variants.
        for extra in ["alibaba::", "siliconflow::", "volcengine::"] {
            if !prefixes.iter().any(|prefix| prefix == extra) {
                prefixes.push(extra.to_string());
            }
        }
        prefixes
    };
}

pub use tensorzero_inference_types::credentials::ProviderType;
// Modified by Delta-AI under Apache 2.0
// Provider credential machinery moved to `tensorzero-providers` (Delta-AI fork).
pub use tensorzero_providers::default_credentials::{
    AnthropicKind, AzureKind, DeepSeekKind, FireworksKind, GCPVertexAnthropicKind,
    GCPVertexGeminiKind, GoogleAIStudioGeminiKind, GroqKind, HyperbolicKind, LazyAsyncCredential,
    LazyCredential, MistralKind, OpenAIKind, OpenRouterKind, ProviderKind, SGLangKind, TGIKind,
    TogetherKind, VLLMKind, XAIKind,
};
pub use tensorzero_providers::default_credentials::{
    ProviderTypeDefaultCredentials, load_tensorzero_relay_credential,
};

#[derive(ts_rs::TS, Serialize, Debug)]
#[ts(export)]
// TODO: investigate why derive(TS) doesn't work if we add bounds to BaseModelTable itself
// #[serde(bound(deserialize = "T: ShorthandModelConfig + Deserialize<'de>"))]
// #[serde(try_from = "HashMap<Arc<str>, T>")]
pub struct BaseModelTable<T> {
    /// The underlying HashMap of explicitly configured models.
    ///
    /// **WARNING:** This does NOT contain shorthand models (e.g. `openai::gpt-5`).
    /// Shorthand models are constructed dynamically at lookup time.
    /// Use `BaseModelTable::get()` instead, which handles both explicit and shorthand models.
    pub table: HashMap<Arc<str>, T>,
    #[serde(skip)]
    #[ts(skip)]
    pub default_credentials: Arc<ProviderTypeDefaultCredentials>,
    global_outbound_http_timeout: chrono::Duration,
    pub model_aliases: Arc<ModelAliasTable>,
}

pub trait ShorthandModelConfig: Sized {
    const SHORTHAND_MODEL_PREFIXES: &[&str];
    /// Used in error messages (e.g. 'Model' or 'Embedding model')
    const MODEL_TYPE: &str;
    /// Task type for alias resolution: "chat", "embedding", or "rerank"
    const TASK_TYPE: &str;
    async fn from_shorthand(
        provider_type: &str,
        model_name: &str,
        default_credentials: &ProviderTypeDefaultCredentials,
    ) -> Result<Self, Error>;
    /// Combine one-provider shorthand configs into a multi-target routing chain.
    /// `parts` is `(routing_key, config)` in try-order. Each config must have
    /// exactly one routing entry.
    fn merge_shorthand_targets(parts: Vec<(Arc<str>, Self)>) -> Result<Self, Error>;
    fn validate(
        &self,
        key: &str,
        global_outbound_http_timeout: &chrono::Duration,
    ) -> Result<(), Error>;
    /// Copy `cost` / `batch_cost` from a matching configured provider onto a
    /// `from_shorthand` model so `provider::model` lookups bill like the table entry.
    fn inherit_configured_provider_settings(
        &mut self,
        table: &HashMap<Arc<str>, Self>,
        requested_model_name: &str,
    ) {
        let _ = (table, requested_model_name);
    }
    /// Whether this table entry can serve `provider_type::…` without rebuilding
    /// providers from shorthand (same routing key or `provider::` prefix).
    fn covers_shorthand_provider(&self, provider_type: &str) -> bool {
        let _ = provider_type;
        false
    }
}

pub use tensorzero_http::CowNoClone;

pub struct Shorthand<'a> {
    pub provider_type: &'a str,
    pub model_name: &'a str,
}

fn check_shorthand<'a>(prefixes: &[&'a str], key: &'a str) -> Option<Shorthand<'a>> {
    for prefix in prefixes {
        if let Some(model_name) = key.strip_prefix(prefix) {
            // Remove the last two characters of the prefix to get the provider type
            let provider_type = &prefix[..prefix.len() - 2];
            return Some(Shorthand {
                provider_type,
                model_name,
            });
        }
    }
    None
}

fn rotated_alias_targets<'a>(
    alias: &'a ModelAlias,
    requested: Option<&Shorthand<'_>>,
) -> Vec<&'a ModelAliasTarget> {
    let mut targets: Vec<&ModelAliasTarget> = alias.targets.iter().collect();
    if let Some(shorthand) = requested
        && let Some(idx) = targets.iter().position(|target| {
            target.provider_type.as_ref() == shorthand.provider_type
                && target.model_name.as_ref() == shorthand.model_name
        })
        && idx != 0
    {
        let head = targets.remove(idx);
        targets.insert(0, head);
    }
    targets
}

impl<T: ShorthandModelConfig> Default for BaseModelTable<T> {
    fn default() -> Self {
        Self {
            table: HashMap::new(),
            default_credentials: Arc::new(ProviderTypeDefaultCredentials::default()),
            global_outbound_http_timeout: chrono::Duration::seconds(120),
            model_aliases: Arc::new(ModelAliasTable::default()),
        }
    }
}

impl<T: ShorthandModelConfig> BaseModelTable<T> {
    pub fn new(
        models: HashMap<Arc<str>, T>,
        provider_type_default_credentials: Arc<ProviderTypeDefaultCredentials>,
        global_outbound_http_timeout: chrono::Duration,
        model_aliases: Arc<ModelAliasTable>,
    ) -> Result<Self, String> {
        for key in models.keys() {
            if RESERVED_MODEL_PREFIXES
                .iter()
                .any(|name| key.starts_with(name))
            {
                return Err(format!(
                    "{} name '{}' contains a reserved prefix",
                    T::MODEL_TYPE,
                    key
                ));
            }
        }

        Ok(Self {
            table: models,
            default_credentials: provider_type_default_credentials,
            global_outbound_http_timeout,
            model_aliases,
        })
    }

    pub async fn get(
        &self,
        key: &str,
        relay: Option<&TensorzeroRelay>,
    ) -> Result<Option<CowNoClone<'_, T>>, Error> {
        if let Some(model_config) = self.table.get(key) {
            return Ok(Some(CowNoClone::Borrowed(model_config)));
        }

        let requested_shorthand = check_shorthand(T::SHORTHAND_MODEL_PREFIXES, key);
        let alias = self
            .model_aliases
            .resolve(key, Some(T::TASK_TYPE))
            .or_else(|| {
                requested_shorthand.as_ref().and_then(|shorthand| {
                    self.model_aliases.find_containing(
                        shorthand.provider_type,
                        shorthand.model_name,
                        Some(T::TASK_TYPE),
                    )
                })
            });

        if let Some(alias) = alias {
            if let Some(session) = crate::routing::RoutingSession::current() {
                session.set_min_tokens_per_sec(alias.min_tokens_per_sec);
            }
            // `deepseek-v4-pro` hits the table (and its cost config). The Synapse
            // rewrite `deepseek::deepseek-v4-pro` used to rebuild via shorthand
            // with `cost: None`. Reuse the configured model when it already lists
            // that provider so billing / timeouts / upstream IDs stay in sync.
            if let Some(configured) = self.table.get(alias.name.as_ref()) {
                let covered = requested_shorthand
                    .as_ref()
                    .is_none_or(|sh| configured.covers_shorthand_provider(sh.provider_type));
                if covered {
                    if let Some(session) = crate::routing::RoutingSession::current()
                        && let Some(sh) = requested_shorthand.as_ref()
                    {
                        session.set_requested_provider(sh.provider_type);
                    }
                    return Ok(Some(CowNoClone::Borrowed(configured)));
                }
            }
            let targets = rotated_alias_targets(alias, requested_shorthand.as_ref());
            let mut parts = Vec::new();
            for target in targets {
                let shorthand_key = format!("{}::{}", target.provider_type, target.model_name);
                let Some(sh) = check_shorthand(T::SHORTHAND_MODEL_PREFIXES, &shorthand_key) else {
                    continue;
                };
                let model = self
                    .load_shorthand(sh.provider_type, sh.model_name, relay)
                    .await?;
                parts.push((Arc::<str>::from(shorthand_key), model));
            }
            if parts.is_empty() {
                return Ok(None);
            }
            return Ok(Some(CowNoClone::Owned(T::merge_shorthand_targets(parts)?)));
        }

        if let Some(shorthand) = requested_shorthand {
            let model = self
                .load_shorthand(shorthand.provider_type, shorthand.model_name, relay)
                .await?;
            return Ok(Some(CowNoClone::Owned(model)));
        }
        Ok(None)
    }

    async fn load_shorthand(
        &self,
        provider_type: &str,
        model_name: &str,
        relay: Option<&TensorzeroRelay>,
    ) -> Result<T, Error> {
        let mut model = if relay.is_some() {
            let creds = self.default_credentials.clone();
            let provider_type = provider_type.to_string();
            let model_name = model_name.to_string();
            with_skip_credential_validation(async move {
                T::from_shorthand(&provider_type, &model_name, &creds).await
            })
            .await?
        } else {
            T::from_shorthand(provider_type, model_name, &self.default_credentials).await?
        };
        model.inherit_configured_provider_settings(&self.table, model_name);
        Ok(model)
    }
    /// Check that a model name is valid
    /// This is either true because it's in the table, because it resolves via alias,
    /// or because it's a valid shorthand name.
    pub fn validate(&self, key: &str) -> Result<(), Error> {
        if let Some(model_config) = self.table.get(key) {
            model_config.validate(key, &self.global_outbound_http_timeout)?;
            return Ok(());
        }

        // Aliases checked before shorthands, matching `get()` order
        if let Some(alias) = self.model_aliases.resolve(key, Some(T::TASK_TYPE)) {
            // Verify at least one target matches a supported shorthand prefix
            let any_valid = alias.targets.iter().any(|target| {
                let shorthand_key = format!("{}::{}", target.provider_type, target.model_name);
                check_shorthand(T::SHORTHAND_MODEL_PREFIXES, &shorthand_key).is_some()
            });
            if any_valid {
                return Ok(());
            }
            return Err(ErrorDetails::Config {
                message: format!(
                    "Model alias '{key}' has no targets matching a supported shorthand prefix"
                ),
            }
            .into());
        }

        if check_shorthand(T::SHORTHAND_MODEL_PREFIXES, key).is_some() {
            return Ok(());
        }

        Err(ErrorDetails::Config {
            message: format!("Model name '{key}' not found in model table"),
        }
        .into())
    }

    #[cfg(any(test, feature = "e2e_tests"))]
    pub fn static_model_len(&self) -> usize {
        self.table.len()
    }

    pub fn iter_static_models(&self) -> impl Iterator<Item = (&Arc<str>, &T)> {
        self.table.iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cost::{CostConfigEntry, CostRate, NormalizedCostPointerConfig};
    use crate::model::{ModelConfig, ModelProvider, ProviderConfig};
    use crate::providers::dummy::DummyProvider;
    use googletest::prelude::*;
    use rust_decimal::Decimal;
    use std::collections::HashMap;
    use std::sync::Arc;

    fn dummy_model_config(model_name: &str, provider_name: &str, with_cost: bool) -> ModelConfig {
        let cost = with_cost.then(|| {
            vec![CostConfigEntry {
                pointer: NormalizedCostPointerConfig::Unified {
                    pointers: vec!["/usage/input_tokens".to_string()],
                },
                rate: Some(CostRate {
                    cost_per_unit: Decimal::from(1) / Decimal::from(1_000_000),
                }),
                ..Default::default()
            }]
        });
        ModelConfig {
            routing: vec![provider_name.into()],
            providers: HashMap::from([(
                Arc::from(provider_name),
                ModelProvider {
                    name: provider_name.into(),
                    config: ProviderConfig::Dummy(DummyProvider {
                        model_name: model_name.to_string(),
                        ..Default::default()
                    }),
                    extra_body: Default::default(),
                    extra_headers: Default::default(),
                    timeouts: Default::default(),
                    discard_unknown_chunks: false,
                    cost,
                    batch_cost: None,
                },
            )]),
            timeouts: Default::default(),
            skip_relay: false,
            namespace: None,
        }
    }

    fn flash_alias() -> ModelAliasTable {
        ModelAliasTable {
            aliases: vec![ModelAlias {
                name: Arc::from("flash"),
                task: Some(Arc::from("chat")),
                targets: vec![
                    ModelAliasTarget {
                        provider_type: Arc::from("dummy"),
                        model_name: Arc::from("error"),
                    },
                    ModelAliasTarget {
                        provider_type: Arc::from("dummy"),
                        model_name: Arc::from("good"),
                    },
                ],
                min_tokens_per_sec: Some(10.0),
            }],
        }
    }

    #[tokio::test]
    async fn alias_get_merges_all_shorthand_targets() {
        let table = BaseModelTable::<ModelConfig>::new(
            HashMap::new(),
            Arc::new(ProviderTypeDefaultCredentials::default()),
            chrono::Duration::seconds(120),
            Arc::new(flash_alias()),
        )
        .unwrap();
        let model = table.get("flash", None).await.unwrap().unwrap();
        assert_eq!(
            model
                .routing
                .iter()
                .map(std::convert::AsRef::as_ref)
                .collect::<Vec<_>>(),
            vec!["dummy::error", "dummy::good"]
        );
    }

    #[tokio::test]
    async fn find_containing_rotates_requested_shorthand_to_head() {
        let table = BaseModelTable::<ModelConfig>::new(
            HashMap::new(),
            Arc::new(ProviderTypeDefaultCredentials::default()),
            chrono::Duration::seconds(120),
            Arc::new(flash_alias()),
        )
        .unwrap();
        let model = table.get("dummy::good", None).await.unwrap().unwrap();
        assert_eq!(model.routing[0].as_ref(), "dummy::good");
        assert_eq!(model.routing[1].as_ref(), "dummy::error");
    }

    #[gtest]
    #[tokio::test]
    async fn shorthand_reuses_configured_model_when_alias_name_is_in_table() {
        let mut models = HashMap::new();
        models.insert(
            Arc::from("flash"),
            dummy_model_config("good", "dummy", true),
        );
        let table = BaseModelTable::<ModelConfig>::new(
            models,
            Arc::new(ProviderTypeDefaultCredentials::default()),
            chrono::Duration::seconds(120),
            Arc::new(flash_alias()),
        )
        .unwrap();
        let model = table
            .get("dummy::good", None)
            .await
            .expect("lookup should succeed")
            .expect("model should exist");
        expect_eq!(
            model
                .routing
                .iter()
                .map(std::convert::AsRef::as_ref)
                .collect::<Vec<_>>(),
            vec!["dummy"]
        );
        expect_true!(
            model
                .providers
                .get("dummy")
                .expect("dummy provider")
                .cost
                .is_some()
        );
    }

    #[gtest]
    #[tokio::test]
    async fn shorthand_inherits_cost_from_configured_model() {
        let mut models = HashMap::new();
        models.insert(Arc::from("good"), dummy_model_config("good", "dummy", true));
        let table = BaseModelTable::<ModelConfig>::new(
            models,
            Arc::new(ProviderTypeDefaultCredentials::default()),
            chrono::Duration::seconds(120),
            Arc::new(ModelAliasTable::default()),
        )
        .unwrap();
        let model = table
            .get("dummy::good", None)
            .await
            .expect("lookup should succeed")
            .expect("model should exist");
        expect_that!(model.routing[0].as_ref(), eq("dummy"));
        expect_true!(
            model
                .providers
                .get("dummy")
                .expect("dummy provider")
                .cost
                .is_some()
        );
    }

    #[gtest]
    #[tokio::test]
    async fn alias_shorthand_merges_when_configured_model_lacks_provider() {
        let mut models = HashMap::new();
        models.insert(
            Arc::from("flash"),
            dummy_model_config("good", "other", true),
        );
        let table = BaseModelTable::<ModelConfig>::new(
            models,
            Arc::new(ProviderTypeDefaultCredentials::default()),
            chrono::Duration::seconds(120),
            Arc::new(flash_alias()),
        )
        .unwrap();
        let model = table
            .get("dummy::good", None)
            .await
            .expect("lookup should succeed")
            .expect("model should exist");
        expect_that!(model.routing[0].as_ref(), eq("dummy::good"));
        expect_that!(model.routing[1].as_ref(), eq("dummy::error"));
    }
}
