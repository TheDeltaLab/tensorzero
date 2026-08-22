// Modified by Delta-AI under Apache 2.0
//! Cost configuration for standalone `/v1/rerank` (not routed through `[models]`).

use std::collections::HashMap;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use tensorzero_types::UninitializedUnifiedCostConfig;

use crate::cost::{CostConfig, load_unified_cost_config_with_provider_defaults};
use crate::error::{Error, ErrorDetails};

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct UninitializedRerankModelConfig {
    #[serde(default)]
    pub providers: HashMap<String, UninitializedRerankProviderConfig>,
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct UninitializedRerankProviderConfig {
    #[serde(default)]
    pub cost: Option<UninitializedUnifiedCostConfig>,
    /// Default IANA timezone for `cost` peak windows that omit `timezone`.
    #[serde(default)]
    pub timezone: Option<String>,
    /// ISO 4217 code for `cost` rates. Defaults to `USD`.
    #[serde(default)]
    pub currency: Option<String>,
}

#[derive(Clone, Debug, Default)]
pub struct RerankModelTable {
    models: HashMap<Arc<str>, HashMap<Arc<str>, CostConfig>>,
}

impl RerankModelTable {
    pub fn load(
        uninitialized: HashMap<Arc<str>, UninitializedRerankModelConfig>,
    ) -> Result<Self, Error> {
        let mut models = HashMap::new();
        for (model_name, model) in uninitialized {
            let mut providers = HashMap::new();
            for (provider_name, provider) in model.providers {
                let Some(cost) = provider.cost else {
                    continue;
                };
                let cost = load_unified_cost_config_with_provider_defaults(
                    cost,
                    provider.timezone.as_deref(),
                    provider.currency.as_deref(),
                )
                .map_err(|e| {
                    Error::new(ErrorDetails::Config {
                        message: format!(
                            "rerank_models.{model_name}.providers.{provider_name}.cost: {e}"
                        ),
                    })
                })?;
                providers.insert(Arc::from(provider_name), cost);
            }
            if !providers.is_empty() {
                models.insert(model_name, providers);
            }
        }
        Ok(Self { models })
    }

    pub fn cost(&self, model: &str, provider: &str) -> Option<&CostConfig> {
        self.models.get(model)?.get(provider)
    }
}
