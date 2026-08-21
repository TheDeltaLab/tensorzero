// Modified by Delta-AI under Apache 2.0
//! Endpoint for returning the gateway config to the UI.
//!
//! This endpoint returns a UI-safe subset of the Config for use by the TensorZero UI.

use std::collections::HashMap;
use std::sync::Arc;

use axum::Json;
use axum::extract::{Path, State};
use serde::Serialize;

use crate::{
    config::snapshot::{ConfigSnapshot, SnapshotHash},
    config::{Config, MetricConfig, UninitializedConfig, UninitializedModelAlias},
    db::ConfigQueries,
    embeddings::{EmbeddingModelConfig, UninitializedEmbeddingModelConfig},
    error::{Error, ErrorDetails},
    evaluations::EvaluationConfig,
    function::FunctionConfig,
    model::{ModelConfig, UninitializedModelConfig},
    model_alias::{ModelAlias, ModelAliasTable},
    tool::StaticToolConfig,
    utils::gateway::AppState,
};

/// UI-safe alias target: provider + model the alias routes to.
#[derive(ts_rs::TS, Clone, Debug, PartialEq, Eq, Serialize)]
#[ts(export)]
pub struct UiModelAliasTarget {
    pub provider: String,
    pub model: String,
}

/// UI-safe model alias: name plus optional task filter (`chat`, `embedding`, `rerank`).
#[derive(ts_rs::TS, Clone, Debug, PartialEq, Eq, Serialize)]
#[ts(export, optional_fields)]
pub struct UiModelAlias {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub task: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub targets: Option<Vec<UiModelAliasTarget>>,
}

/// Response type for GET /internal/ui_config
///
/// Contains only UI-safe fields from the gateway config, excluding sensitive
/// information like provider credentials, API keys, and internal settings.
#[derive(ts_rs::TS, Debug, Serialize)]
#[ts(export)]
pub struct UiConfig {
    pub functions: HashMap<String, Arc<FunctionConfig>>,
    pub metrics: HashMap<String, MetricConfig>,
    pub tools: HashMap<String, Arc<StaticToolConfig>>,
    pub evaluations: HashMap<String, Arc<EvaluationConfig>>,
    pub model_names: Vec<String>,
    pub embedding_model_names: Vec<String>,
    /// Configured chat model name => routing provider names (no credentials).
    pub model_providers: HashMap<String, Vec<String>>,
    /// Configured embedding model name => routing provider names (no credentials).
    pub embedding_model_providers: HashMap<String, Vec<String>>,
    pub model_aliases: Vec<UiModelAlias>,
    pub config_hash: String,
    /// Whether the gateway config was loaded from the database (as opposed to a file on disk).
    /// Used by the UI to decide whether to show the config editor.
    pub config_in_database: bool,
    /// Whether the gateway is enforcing API key authentication (`[gateway.auth] enabled = true`).
    /// Used by the UI to decide whether routes that bypass the gateway (e.g. the API keys page)
    /// must validate the caller's key themselves.
    pub auth_enabled: bool,
}

fn ui_alias_from_model_alias(alias: &ModelAlias) -> UiModelAlias {
    let targets: Vec<UiModelAliasTarget> = alias
        .targets
        .iter()
        .map(|target| UiModelAliasTarget {
            provider: target.provider_type.to_string(),
            model: target.model_name.to_string(),
        })
        .collect();
    UiModelAlias {
        name: alias.name.to_string(),
        task: alias.task.as_ref().map(std::string::ToString::to_string),
        targets: if targets.is_empty() {
            None
        } else {
            Some(targets)
        },
    }
}

fn ui_aliases_from_table(table: &ModelAliasTable) -> Vec<UiModelAlias> {
    let mut aliases: Vec<UiModelAlias> = table
        .aliases
        .iter()
        .map(ui_alias_from_model_alias)
        .collect();
    aliases.sort_by(|left, right| left.name.cmp(&right.name));
    aliases
}

fn ui_aliases_from_uninit(map: HashMap<String, UninitializedModelAlias>) -> Vec<UiModelAlias> {
    let mut aliases: Vec<UiModelAlias> = map
        .into_iter()
        .map(|(name, alias)| {
            let targets: Vec<UiModelAliasTarget> = alias
                .targets
                .into_iter()
                .map(|target| UiModelAliasTarget {
                    provider: target.provider,
                    model: target.model,
                })
                .collect();
            UiModelAlias {
                name,
                task: alias.task,
                targets: if targets.is_empty() {
                    None
                } else {
                    Some(targets)
                },
            }
        })
        .collect();
    aliases.sort_by(|left, right| left.name.cmp(&right.name));
    aliases
}

fn sorted_provider_names(mut names: Vec<String>) -> Vec<String> {
    names.sort();
    names.dedup();
    names
}

fn ui_providers_from_chat_models(
    table: &HashMap<Arc<str>, ModelConfig>,
) -> HashMap<String, Vec<String>> {
    table
        .iter()
        .map(|(name, config)| {
            let providers: Vec<String> = if config.routing.is_empty() {
                config.providers.keys().map(|key| key.to_string()).collect()
            } else {
                config.routing.iter().map(|key| key.to_string()).collect()
            };
            (name.to_string(), sorted_provider_names(providers))
        })
        .collect()
}

fn ui_providers_from_embedding_models(
    table: &HashMap<Arc<str>, EmbeddingModelConfig>,
) -> HashMap<String, Vec<String>> {
    table
        .iter()
        .map(|(name, config)| {
            let providers: Vec<String> = if config.routing.is_empty() {
                config.providers.keys().map(|key| key.to_string()).collect()
            } else {
                config.routing.iter().map(|key| key.to_string()).collect()
            };
            (name.to_string(), sorted_provider_names(providers))
        })
        .collect()
}

fn ui_providers_from_uninit_chat_models(
    models: &HashMap<Arc<str>, UninitializedModelConfig>,
) -> HashMap<String, Vec<String>> {
    models
        .iter()
        .map(|(name, config)| {
            let providers: Vec<String> = if config.routing.is_empty() {
                config.providers.keys().map(|key| key.to_string()).collect()
            } else {
                config.routing.iter().map(|key| key.to_string()).collect()
            };
            (name.to_string(), sorted_provider_names(providers))
        })
        .collect()
}

fn ui_providers_from_uninit_embedding_models(
    models: &HashMap<Arc<str>, UninitializedEmbeddingModelConfig>,
) -> HashMap<String, Vec<String>> {
    models
        .iter()
        .map(|(name, config)| {
            let providers: Vec<String> = if config.routing.is_empty() {
                config.providers.keys().map(|key| key.to_string()).collect()
            } else {
                config.routing.iter().map(|key| key.to_string()).collect()
            };
            (name.to_string(), sorted_provider_names(providers))
        })
        .collect()
}

fn sorted_names<K: ToString>(keys: impl IntoIterator<Item = K>) -> Vec<String> {
    let mut names: Vec<String> = keys.into_iter().map(|key| key.to_string()).collect();
    names.sort();
    names
}

impl UiConfig {
    pub fn from_config(config: &Config, config_in_database: bool) -> Self {
        Self {
            functions: config
                .functions
                .iter()
                .map(|(k, v)| (k.clone(), Arc::clone(v)))
                .collect(),
            metrics: config.metrics.clone(),
            tools: config
                .tools
                .iter()
                .map(|(k, v)| (k.clone(), Arc::clone(v)))
                .collect(),
            evaluations: config
                .evaluations
                .iter()
                .map(|(k, v)| (k.clone(), Arc::clone(v)))
                .collect(),
            model_names: config.models.table.keys().map(|s| s.to_string()).collect(),
            embedding_model_names: sorted_names(config.embedding_models.table.keys()),
            model_providers: ui_providers_from_chat_models(&config.models.table),
            embedding_model_providers: ui_providers_from_embedding_models(
                &config.embedding_models.table,
            ),
            model_aliases: ui_aliases_from_table(&config.models.model_aliases),
            config_hash: config.hash.to_string(),
            config_in_database,
            auth_enabled: config.gateway.auth.enabled,
        }
    }

    /// Creates a `UiConfig` from a historical config snapshot.
    ///
    /// This initializes only the parts needed by the UI (functions, tools, evaluations,
    /// metrics, model names), skipping heavy initialization like model credentials, HTTP
    /// clients, gateway config, object store, and rate limiting.
    ///
    /// `auth_enabled` is sourced from the live gateway config rather than the snapshot —
    /// auth state is a deployment-level concern, not a snapshot-level one.
    pub fn from_snapshot(snapshot: ConfigSnapshot, auth_enabled: bool) -> Result<Self, Error> {
        let hash = snapshot.hash.to_string();
        let uninit_config: UninitializedConfig =
            snapshot.config.try_into().map_err(|e: &'static str| {
                Error::new(ErrorDetails::Config {
                    message: e.to_string(),
                })
            })?;

        let UninitializedConfig {
            models,
            embedding_models,
            model_aliases,
            functions,
            metrics,
            tools,
            evaluations,
            gateway: _,
            clickhouse: _,
            postgres: _,
            rate_limiting: _,
            object_storage: _,
            provider_types: _,
            optimizers: _,
            autopilot: _,
        } = uninit_config;

        // Load functions (sync, no FS/network — file data embedded in ResolvedTomlPathData)
        let mut all_functions = HashMap::new();
        let mut all_metrics = metrics.unwrap_or_default();
        for (name, func) in functions.unwrap_or_default() {
            let loaded = func.load(&name, &all_metrics)?;
            for (fn_name, fn_config) in loaded.evaluator_functions {
                all_functions.insert(fn_name, Arc::new(fn_config));
            }
            all_metrics.extend(loaded.evaluator_metrics);
            all_functions.insert(name, Arc::new(loaded.function_config));
        }

        // Load tools (sync, same reason)
        let loaded_tools: HashMap<String, Arc<StaticToolConfig>> = tools
            .unwrap_or_default()
            .into_iter()
            .map(|(name, tool)| tool.load(name.clone()).map(|c| (name, Arc::new(c))))
            .collect::<Result<_, _>>()?;

        // Load evaluations (sync, needs loaded functions)
        // Also collects generated evaluation functions and metrics
        let mut loaded_evaluations = HashMap::new();
        for (name, eval_config) in evaluations.unwrap_or_default() {
            let (eval, eval_functions, eval_metrics) = eval_config.load(&all_functions, &name)?;
            loaded_evaluations.insert(name, Arc::new(EvaluationConfig::Inference(eval)));
            all_functions.extend(eval_functions);
            all_metrics.extend(eval_metrics);
        }

        // Model names — just keys, no initialization (only inference models, matching from_config)
        let uninit_models = models.unwrap_or_default();
        let uninit_embedding_models = embedding_models.unwrap_or_default();
        let model_providers = ui_providers_from_uninit_chat_models(&uninit_models);
        let embedding_model_providers =
            ui_providers_from_uninit_embedding_models(&uninit_embedding_models);
        let model_names: Vec<String> = uninit_models.keys().map(|s| s.to_string()).collect();
        let embedding_model_names = sorted_names(uninit_embedding_models.keys());
        let model_aliases = ui_aliases_from_uninit(model_aliases.unwrap_or_default());

        Ok(Self {
            functions: all_functions,
            metrics: all_metrics,
            tools: loaded_tools,
            evaluations: loaded_evaluations,
            model_names,
            embedding_model_names,
            model_providers,
            embedding_model_providers,
            model_aliases,
            config_hash: hash,
            config_in_database: false,
            auth_enabled,
        })
    }
}

/// Handler for GET /internal/ui_config
///
/// Returns a UI-safe subset of the Config.
#[expect(clippy::unused_async)]
pub async fn ui_config_handler(State(app_state): AppState) -> Json<UiConfig> {
    Json(UiConfig::from_config(
        &app_state.config,
        app_state.config_in_database,
    ))
}

/// Handler for GET /internal/ui_config/{hash}
///
/// Returns a UI-safe subset of the Config for a historical config snapshot.
pub async fn ui_config_by_hash_handler(
    State(app_state): AppState,
    Path(hash): Path<String>,
) -> Result<Json<UiConfig>, Error> {
    let snapshot_hash: SnapshotHash = hash.parse().map_err(|_| {
        Error::new(ErrorDetails::ConfigSnapshotNotFound {
            snapshot_hash: hash.clone(),
        })
    })?;

    let db = app_state.get_delegating_database();
    let snapshot = db.get_config_snapshot(snapshot_hash).await?;

    Ok(Json(UiConfig::from_snapshot(
        snapshot,
        app_state.config.gateway.auth.enabled,
    )?))
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::Arc;

    use crate::config::{
        Config, MetricConfig, MetricConfigLevel, MetricConfigOptimize, MetricConfigType,
    };
    use crate::function::{FunctionConfig, FunctionConfigChat};

    use super::*;

    #[test]
    fn test_ui_config_with_functions_and_metrics() {
        let function_config = FunctionConfig::Chat(FunctionConfigChat {
            variants: HashMap::new(),
            schemas: Default::default(),
            tools: vec![],
            tool_choice: Default::default(),
            parallel_tool_calls: None,
            description: Some("Test function".to_string()),
            experimentation: Default::default(),
            all_explicit_templates_names: Default::default(),
            evaluators: HashMap::new(),
        });

        let metric_config = MetricConfig {
            r#type: MetricConfigType::Boolean,
            optimize: MetricConfigOptimize::Max,
            level: MetricConfigLevel::Inference,
            description: None,
        };

        let mut config = Config::default();
        config
            .functions
            .insert("test_function".to_string(), Arc::new(function_config));
        config
            .metrics
            .insert("test_metric".to_string(), metric_config);

        let ui_config = UiConfig::from_config(&config, false);

        assert_eq!(ui_config.functions.len(), 1);
        assert!(ui_config.functions.contains_key("test_function"));
        let returned_function = ui_config.functions.get("test_function").unwrap();

        if let FunctionConfig::Chat(chat_config) = returned_function.as_ref() {
            assert_eq!(chat_config.description, Some("Test function".to_string()));
        } else {
            panic!("Expected Chat function config");
        }

        assert_eq!(ui_config.metrics.len(), 1);
        assert!(ui_config.metrics.contains_key("test_metric"));
        let returned_metric = ui_config.metrics.get("test_metric").unwrap();
        assert_eq!(returned_metric.r#type, MetricConfigType::Boolean);
        assert_eq!(returned_metric.optimize, MetricConfigOptimize::Max);
        assert_eq!(returned_metric.level, MetricConfigLevel::Inference);

        assert!(ui_config.model_names.is_empty());
        assert!(ui_config.embedding_model_names.is_empty());
        assert!(ui_config.model_providers.is_empty());
        assert!(ui_config.embedding_model_providers.is_empty());
        assert!(ui_config.model_aliases.is_empty());
        assert!(ui_config.tools.is_empty());
        assert!(ui_config.evaluations.is_empty());
        assert!(!ui_config.config_hash.is_empty());
        assert!(!ui_config.auth_enabled);
    }

    #[test]
    fn test_ui_config_propagates_auth_enabled() {
        let mut config = Config::default();
        config.gateway.auth.enabled = true;

        let ui_config = UiConfig::from_config(&config, false);
        assert!(ui_config.auth_enabled);
    }

    #[test]
    fn test_ui_config_from_config_extracts_correct_fields() {
        // Create a function config
        let function_config = FunctionConfig::Chat(FunctionConfigChat {
            variants: HashMap::new(),
            schemas: Default::default(),
            tools: vec![],
            tool_choice: Default::default(),
            parallel_tool_calls: None,
            description: Some("My function".to_string()),
            experimentation: Default::default(),
            all_explicit_templates_names: Default::default(),
            evaluators: HashMap::new(),
        });

        // Create a metric config
        let metric_config = MetricConfig {
            r#type: MetricConfigType::Float,
            optimize: MetricConfigOptimize::Min,
            level: MetricConfigLevel::Episode,
            description: None,
        };

        let mut config = Config::default();
        config
            .functions
            .insert("my_function".to_string(), Arc::new(function_config));
        config
            .metrics
            .insert("my_metric".to_string(), metric_config);

        let ui_config = UiConfig::from_config(&config, false);

        // Verify functions are copied correctly
        assert_eq!(ui_config.functions.len(), 1);
        let func = ui_config.functions.get("my_function").unwrap();
        if let FunctionConfig::Chat(chat_config) = func.as_ref() {
            assert_eq!(chat_config.description, Some("My function".to_string()));
        } else {
            panic!("Expected Chat function config");
        }

        // Verify metrics are copied correctly
        assert_eq!(ui_config.metrics.len(), 1);
        let metric = ui_config.metrics.get("my_metric").unwrap();
        assert_eq!(metric.r#type, MetricConfigType::Float);
        assert_eq!(metric.optimize, MetricConfigOptimize::Min);
        assert_eq!(metric.level, MetricConfigLevel::Episode);

        // Verify config_hash is present
        assert!(!ui_config.config_hash.is_empty());
    }

    #[test]
    fn test_ui_aliases_from_uninit_sorts_by_name() {
        let mut map = HashMap::new();
        map.insert(
            "zeta".to_string(),
            crate::config::UninitializedModelAlias {
                task: Some("chat".to_string()),
                targets: vec![crate::config::UninitializedModelAliasTarget {
                    provider: "synapse".to_string(),
                    model: "deepseek-v4-flash".to_string(),
                }],
                min_tokens_per_sec: None,
            },
        );
        map.insert(
            "alpha".to_string(),
            crate::config::UninitializedModelAlias {
                task: Some("rerank".to_string()),
                targets: vec![],
                min_tokens_per_sec: None,
            },
        );

        let aliases = ui_aliases_from_uninit(map);
        assert_eq!(aliases.len(), 2);
        assert_eq!(aliases[0].name, "alpha");
        assert_eq!(aliases[0].task.as_deref(), Some("rerank"));
        assert_eq!(aliases[1].name, "zeta");
        assert_eq!(aliases[1].task.as_deref(), Some("chat"));
        assert_eq!(aliases[1].targets.as_ref().map(Vec::len), Some(1));
        assert_eq!(
            aliases[1]
                .targets
                .as_ref()
                .map(|targets| targets[0].provider.as_str()),
            Some("synapse")
        );
        assert_eq!(
            aliases[1]
                .targets
                .as_ref()
                .map(|targets| targets[0].model.as_str()),
            Some("deepseek-v4-flash")
        );
    }
}
