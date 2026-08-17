// Modified by Delta-AI under Apache 2.0
use std::{
    collections::HashMap,
    env,
    fmt::Display,
    fs,
    sync::{Arc, OnceLock},
};

use crate::{
    config::{
        e2e_skip_credential_validation, provider_types::ProviderTypesConfig,
        skip_credential_validation, with_skip_credential_validation,
    },
    error::{Error, ErrorDetails},
    model::{
        Credential, CredentialLocation, CredentialLocationWithFallback, UninitializedProviderConfig,
    },
    model_alias::{ModelAlias, ModelAliasTable, ModelAliasTarget},
    providers::{
        anthropic::AnthropicCredentials,
        azure::AzureCredentials,
        deepseek::DeepSeekCredentials,
        fireworks::FireworksCredentials,
        gcp_vertex_anthropic::make_gcp_sdk_credentials,
        gcp_vertex_gemini::{GCPVertexCredentials, build_gcp_non_sdk_credentials},
        google_ai_studio_gemini::GoogleAIStudioCredentials,
        groq::GroqCredentials,
        hyperbolic::HyperbolicCredentials,
        mistral::MistralCredentials,
        openai::OpenAICredentials,
        openrouter::OpenRouterCredentials,
        sglang::SGLangCredentials,
        tgi::TGICredentials,
        together::TogetherCredentials,
        vllm::VLLMCredentials,
        xai::XAICredentials,
    },
    relay::TensorzeroRelay,
};
use lazy_static::lazy_static;
use secrecy::SecretString;
use serde::Serialize;
use strum::VariantNames;
use tokio::sync::OnceCell;

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

pub trait ProviderKind {
    type Credential: Clone;
    fn get_provider_type(&self) -> ProviderType;
    async fn get_credential_field(
        &self,
        default_credentials: &ProviderTypeDefaultCredentials,
    ) -> Result<Self::Credential, Error>;
    async fn get_defaulted_credential(
        &self,
        api_key_location: Option<&CredentialLocationWithFallback>,
        default_credentials: &ProviderTypeDefaultCredentials,
    ) -> Result<Self::Credential, Error>
    where
        Self::Credential: TryFrom<Credential, Error = Error>,
    {
        let provider_type = self.get_provider_type();
        if let Some(api_key_location) = api_key_location {
            return load_credential_with_fallback(api_key_location, provider_type)?.try_into();
        }

        Ok(self
            .get_credential_field(default_credentials)
            .await?
            .clone())
    }
}

pub use tensorzero_inference_types::credentials::ProviderType;

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
        if relay.is_some() {
            let creds = self.default_credentials.clone();
            let provider_type = provider_type.to_string();
            let model_name = model_name.to_string();
            with_skip_credential_validation(async move {
                T::from_shorthand(&provider_type, &model_name, &creds).await
            })
            .await
        } else {
            T::from_shorthand(provider_type, model_name, &self.default_credentials).await
        }
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

pub struct LazyCredential<T: Clone> {
    cell: OnceLock<Result<T, Error>>,
    loader: Box<dyn Fn() -> Result<T, Error> + Send + Sync>,
}

impl<T: Clone> LazyCredential<T> {
    pub fn new<F>(loader: F) -> Self
    where
        F: Fn() -> Result<T, Error> + Send + Sync + 'static,
    {
        Self {
            cell: OnceLock::new(),
            loader: Box::new(loader),
        }
    }

    pub fn get(&self) -> Result<&T, &Error> {
        self.cell.get_or_init(|| (self.loader)()).as_ref()
    }

    pub fn get_cloned(&self) -> Result<T, Error>
    where
        Error: Clone,
    {
        self.get().cloned().map_err(std::clone::Clone::clone)
    }
}

type AsyncCredentialLoader<T> = Box<
    dyn Fn() -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<T, Error>> + Send>>
        + Send
        + Sync,
>;

pub struct LazyAsyncCredential<T: Clone> {
    cell: OnceCell<Result<T, Error>>,
    loader: AsyncCredentialLoader<T>,
}

impl<T: Clone> LazyAsyncCredential<T> {
    pub fn new<F, Fut>(loader: F) -> Self
    where
        F: Fn() -> Fut + Send + Sync + 'static,
        Fut: std::future::Future<Output = Result<T, Error>> + Send + 'static,
    {
        Self {
            cell: OnceCell::new(),
            loader: Box::new(move || Box::pin(loader())),
        }
    }

    pub async fn get(&self) -> Result<&T, &Error> {
        self.cell
            .get_or_init(|| async { (self.loader)().await })
            .await
            .as_ref()
    }

    pub async fn get_cloned(&self) -> Result<T, Error>
    where
        T: Clone,
        Error: Clone,
    {
        self.get().await.cloned().map_err(std::clone::Clone::clone)
    }
}

pub struct ProviderTypeDefaultCredentials {
    anthropic: LazyCredential<AnthropicCredentials>,
    // Note: we currently do not support shorthand for either AWS Bedrock or AWS Sagemaker
    // aws_bedrock:
    // aws_sagemaker:
    azure: LazyCredential<AzureCredentials>,
    deepseek: LazyCredential<DeepSeekCredentials>,
    fireworks: LazyCredential<FireworksCredentials>,
    gcp_vertex_anthropic: LazyAsyncCredential<GCPVertexCredentials>,
    gcp_vertex_gemini: LazyAsyncCredential<GCPVertexCredentials>,
    google_ai_studio_gemini: LazyCredential<GoogleAIStudioCredentials>,
    groq: LazyCredential<GroqCredentials>,
    hyperbolic: LazyCredential<HyperbolicCredentials>,
    mistral: LazyCredential<MistralCredentials>,
    openai: LazyCredential<OpenAICredentials>,
    openrouter: LazyCredential<OpenRouterCredentials>,
    sglang: LazyCredential<SGLangCredentials>,
    tgi: LazyCredential<TGICredentials>,
    together: LazyCredential<TogetherCredentials>,
    vllm: LazyCredential<VLLMCredentials>,
    xai: LazyCredential<XAICredentials>,
}

impl ProviderTypeDefaultCredentials {
    pub fn new(provider_types_config: &ProviderTypesConfig) -> Self {
        let anthropic_location = provider_types_config
            .anthropic
            .as_ref()
            .and_then(|a| a.defaults.as_ref())
            .cloned()
            .unwrap_or_default()
            .api_key_location;
        let azure_location = provider_types_config
            .azure
            .as_ref()
            .and_then(|a| a.defaults.as_ref())
            .cloned()
            .unwrap_or_default()
            .api_key_location;
        let deepseek_location = provider_types_config
            .deepseek
            .as_ref()
            .and_then(|a| a.defaults.as_ref())
            .cloned()
            .unwrap_or_default()
            .api_key_location;
        let fireworks_location = provider_types_config
            .fireworks
            .as_ref()
            .and_then(|a| a.defaults.as_ref())
            .cloned()
            .unwrap_or_default()
            .api_key_location;
        let google_ai_studio_gemini_location = provider_types_config
            .google_ai_studio_gemini
            .as_ref()
            .and_then(|a| a.defaults.as_ref())
            .cloned()
            .unwrap_or_default()
            .api_key_location;
        let gcp_vertex_anthropic_location = provider_types_config
            .gcp_vertex_anthropic
            .as_ref()
            .and_then(|a| a.defaults.as_ref())
            .cloned()
            .unwrap_or_default()
            .credential_location;
        let gcp_vertex_gemini_location = provider_types_config
            .gcp_vertex_gemini
            .as_ref()
            .and_then(|a| a.defaults.as_ref())
            .cloned()
            .unwrap_or_default()
            .credential_location;
        let groq_location = provider_types_config
            .groq
            .as_ref()
            .and_then(|a| a.defaults.as_ref())
            .cloned()
            .unwrap_or_default()
            .api_key_location;
        let hyperbolic_location = provider_types_config
            .hyperbolic
            .as_ref()
            .and_then(|a| a.defaults.as_ref())
            .cloned()
            .unwrap_or_default()
            .api_key_location;
        let mistral_location = provider_types_config
            .mistral
            .as_ref()
            .and_then(|a| a.defaults.as_ref())
            .cloned()
            .unwrap_or_default()
            .api_key_location;
        let openai_location = provider_types_config
            .openai
            .as_ref()
            .and_then(|a| a.defaults.as_ref())
            .cloned()
            .unwrap_or_default()
            .api_key_location;
        let openrouter_location = provider_types_config
            .openrouter
            .as_ref()
            .and_then(|a| a.defaults.as_ref())
            .cloned()
            .unwrap_or_default()
            .api_key_location;
        let sglang_location = provider_types_config
            .sglang
            .as_ref()
            .and_then(|a| a.defaults.as_ref())
            .cloned()
            .unwrap_or_default()
            .api_key_location;
        let tgi_location = provider_types_config
            .tgi
            .as_ref()
            .and_then(|a| a.defaults.as_ref())
            .cloned()
            .unwrap_or_default()
            .api_key_location;
        let together_location = provider_types_config
            .together
            .as_ref()
            .and_then(|a| a.defaults.as_ref())
            .cloned()
            .unwrap_or_default()
            .api_key_location;
        let vllm_location = provider_types_config
            .vllm
            .as_ref()
            .and_then(|a| a.defaults.as_ref())
            .cloned()
            .unwrap_or_default()
            .api_key_location;
        let xai_location = provider_types_config
            .xai
            .as_ref()
            .and_then(|a| a.defaults.as_ref())
            .cloned()
            .unwrap_or_default()
            .api_key_location;

        ProviderTypeDefaultCredentials {
            anthropic: LazyCredential::new(move || {
                load_credential_with_fallback(&anthropic_location, ProviderType::Anthropic)?
                    .try_into()
            }),
            azure: LazyCredential::new(move || {
                load_azure_credential_with_legacy_fallback(&azure_location)?.try_into()
            }),
            deepseek: LazyCredential::new(move || {
                load_credential_with_fallback(&deepseek_location, ProviderType::Deepseek)?
                    .try_into()
            }),
            fireworks: LazyCredential::new(move || {
                load_credential_with_fallback(&fireworks_location, ProviderType::Fireworks)?
                    .try_into()
            }),
            google_ai_studio_gemini: LazyCredential::new(move || {
                load_credential_with_fallback(
                    &google_ai_studio_gemini_location,
                    ProviderType::GoogleAIStudioGemini,
                )?
                .try_into()
            }),
            gcp_vertex_anthropic: LazyAsyncCredential::new(move || {
                let location = gcp_vertex_anthropic_location.clone();
                async move {
                    make_gcp_credentials_with_fallback(ProviderType::GCPVertexAnthropic, &location)
                        .await
                }
            }),
            gcp_vertex_gemini: LazyAsyncCredential::new(move || {
                let location = gcp_vertex_gemini_location.clone();
                async move {
                    make_gcp_credentials_with_fallback(ProviderType::GCPVertexGemini, &location)
                        .await
                }
            }),

            groq: LazyCredential::new(move || {
                load_credential_with_fallback(&groq_location, ProviderType::Groq)?.try_into()
            }),
            hyperbolic: LazyCredential::new(move || {
                load_credential_with_fallback(&hyperbolic_location, ProviderType::Hyperbolic)?
                    .try_into()
            }),
            mistral: LazyCredential::new(move || {
                load_credential_with_fallback(&mistral_location, ProviderType::Mistral)?.try_into()
            }),
            openai: LazyCredential::new(move || {
                load_credential_with_fallback(&openai_location, ProviderType::OpenAI)?.try_into()
            }),
            openrouter: LazyCredential::new(move || {
                load_credential_with_fallback(&openrouter_location, ProviderType::OpenRouter)?
                    .try_into()
            }),
            sglang: LazyCredential::new(move || {
                load_credential_with_fallback(&sglang_location, ProviderType::SGLang)?.try_into()
            }),
            tgi: LazyCredential::new(move || {
                load_credential_with_fallback(&tgi_location, ProviderType::TGI)?.try_into()
            }),
            together: LazyCredential::new(move || {
                load_credential_with_fallback(&together_location, ProviderType::Together)?
                    .try_into()
            }),
            vllm: LazyCredential::new(move || {
                load_credential_with_fallback(&vllm_location, ProviderType::VLLM)?.try_into()
            }),
            xai: LazyCredential::new(move || {
                load_credential_with_fallback(&xai_location, ProviderType::XAI)?.try_into()
            }),
        }
    }
}

async fn make_gcp_credentials_with_fallback(
    provider_type: ProviderType,
    location: &CredentialLocationWithFallback,
) -> Result<GCPVertexCredentials, Error> {
    // Build default credential
    let default_cred = match location.default_location() {
        CredentialLocation::Sdk => make_gcp_sdk_credentials(provider_type).await?,
        loc => build_gcp_non_sdk_credentials(load_credential(loc, provider_type)?, &provider_type)?,
    };

    // If fallback location is specified, construct a WithFallback credential
    if let Some(fallback_location) = location.fallback_location() {
        let fallback_cred = match fallback_location {
            CredentialLocation::Sdk => make_gcp_sdk_credentials(provider_type).await?,
            fallback_loc => build_gcp_non_sdk_credentials(
                load_credential(fallback_loc, provider_type)?,
                &provider_type,
            )?,
        };
        Ok(GCPVertexCredentials::WithFallback {
            default: Box::new(default_cred),
            fallback: Box::new(fallback_cred),
        })
    } else {
        Ok(default_cred)
    }
}

impl Default for ProviderTypeDefaultCredentials {
    fn default() -> Self {
        Self::new(&ProviderTypesConfig::default())
    }
}

impl std::fmt::Debug for ProviderTypeDefaultCredentials {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ProviderTypeDefaultCredentials")
            .finish_non_exhaustive()
    }
}

fn load_credential(
    location: &CredentialLocation,
    provider_type: impl Display,
) -> Result<Credential, Error> {
    match location {
        CredentialLocation::Env(key_name) => match env::var(key_name) {
            Ok(value) => Ok(Credential::Static(SecretString::from(value))),
            Err(_) => {
                if skip_credential_validation() {
                    if e2e_skip_credential_validation() {
                        tracing::warn!(
                            "You are missing the credentials required for a model provider of type {provider_type} (environment variable `{key_name}` is unset), so the associated tests will likely fail.",
                        );
                    }
                    Ok(Credential::Missing)
                } else {
                    Err(Error::new(ErrorDetails::ApiKeyMissing {
                        provider_name: provider_type.to_string(),
                        message: format!("Environment variable `{key_name}` is missing"),
                    }))
                }
            }
        },
        CredentialLocation::PathFromEnv(env_key) => {
            // First get the path from environment variable
            let path = match env::var(env_key) {
                Ok(path) => path,
                Err(_) => {
                    if skip_credential_validation() {
                        if e2e_skip_credential_validation() {
                            tracing::warn!(
                                "Environment variable {} is required for a model provider of type {} but is missing, so the associated tests will likely fail.",
                                env_key,
                                provider_type
                            );
                        }
                        return Ok(Credential::Missing);
                    } else {
                        return Err(Error::new(ErrorDetails::ApiKeyMissing {
                            provider_name: provider_type.to_string(),
                            message: format!(
                                "Environment variable `{env_key}` for credentials path is missing"
                            ),
                        }));
                    }
                }
            };
            // Then read the file contents
            match fs::read_to_string(path) {
                Ok(contents) => Ok(Credential::FileContents(SecretString::from(contents))),
                Err(e) => {
                    if skip_credential_validation() {
                        if e2e_skip_credential_validation() {
                            tracing::warn!(
                                "Failed to read credentials file for a model provider of type {}, so the associated tests will likely fail: {}",
                                provider_type,
                                e
                            );
                        }
                        Ok(Credential::Missing)
                    } else {
                        Err(Error::new(ErrorDetails::ApiKeyMissing {
                            provider_name: provider_type.to_string(),
                            message: format!("Failed to read credentials file - {e}"),
                        }))
                    }
                }
            }
        }
        CredentialLocation::Path(path) => match fs::read_to_string(path) {
            Ok(contents) => Ok(Credential::FileContents(SecretString::from(contents))),
            Err(e) => {
                if skip_credential_validation() {
                    if e2e_skip_credential_validation() {
                        tracing::warn!(
                            "Failed to read credentials file for a model provider of type {}, so the associated tests will likely fail: {}",
                            provider_type,
                            e
                        );
                    }
                    Ok(Credential::Missing)
                } else {
                    Err(Error::new(ErrorDetails::ApiKeyMissing {
                        provider_name: provider_type.to_string(),
                        message: format!("Failed to read credentials file - {e}"),
                    }))
                }
            }
        },
        CredentialLocation::Dynamic(key_name) => Ok(Credential::Dynamic(key_name.clone())),
        CredentialLocation::Sdk => Ok(Credential::Sdk),
        CredentialLocation::None => Ok(Credential::None),
    }
}

pub fn load_tensorzero_relay_credential(
    location_with_fallback: &crate::model::CredentialLocationWithFallback,
) -> Result<Credential, Error> {
    load_credential_with_fallback(location_with_fallback, "tensorzero::relay")
}

/// Load credential with fallback support
/// Constructs a WithFallback credential that will be resolved at inference time
fn load_credential_with_fallback(
    location_with_fallback: &crate::model::CredentialLocationWithFallback,
    provider_type: impl Display + Copy,
) -> Result<Credential, Error> {
    let default_credential =
        load_credential(location_with_fallback.default_location(), provider_type)?;

    // If fallback location is specified, construct a WithFallback credential
    if let Some(fallback_location) = location_with_fallback.fallback_location() {
        let fallback_credential = load_credential(fallback_location, provider_type)?;
        Ok(Credential::WithFallback {
            default: Box::new(default_credential),
            fallback: Box::new(fallback_credential),
        })
    } else {
        Ok(default_credential)
    }
}

/// Load Azure credential with legacy `AZURE_OPENAI_API_KEY` fallback support.
/// Only applies fallback when using the default location (`AZURE_API_KEY`).
fn load_azure_credential_with_legacy_fallback(
    location: &CredentialLocationWithFallback,
) -> Result<Credential, Error> {
    // Check if using the default location (AZURE_API_KEY)
    let is_default_location = matches!(
        location.default_location(),
        CredentialLocation::Env(key) if key == "AZURE_API_KEY"
    );

    // For the default location, check legacy key BEFORE attempting primary load
    // to avoid logging an error when fallback will succeed
    if is_default_location
        && env::var("AZURE_API_KEY").is_err()
        && let Ok(value) = env::var("AZURE_OPENAI_API_KEY")
    {
        crate::utils::deprecation_warning(
            "The environment variable `AZURE_OPENAI_API_KEY` is deprecated and will be removed in a future release. Please set `AZURE_API_KEY` instead. The legacy value will be removed in 2026.4+ (#5530).",
        );
        return Ok(Credential::Static(SecretString::from(value)));
    }

    load_credential_with_fallback(location, ProviderType::Azure)
}

pub struct AnthropicKind;

impl ProviderKind for AnthropicKind {
    type Credential = AnthropicCredentials;
    fn get_provider_type(&self) -> ProviderType {
        ProviderType::Anthropic
    }

    async fn get_credential_field(
        &self,
        default_credentials: &ProviderTypeDefaultCredentials,
    ) -> Result<Self::Credential, Error> {
        default_credentials.anthropic.get_cloned()
    }
}

pub struct OpenAIKind;

impl ProviderKind for OpenAIKind {
    type Credential = OpenAICredentials;
    fn get_provider_type(&self) -> ProviderType {
        ProviderType::OpenAI
    }

    async fn get_credential_field(
        &self,
        default_credentials: &ProviderTypeDefaultCredentials,
    ) -> Result<Self::Credential, Error> {
        default_credentials.openai.get_cloned()
    }
}

pub struct AzureKind;

impl ProviderKind for AzureKind {
    type Credential = AzureCredentials;
    fn get_provider_type(&self) -> ProviderType {
        ProviderType::Azure
    }

    async fn get_credential_field(
        &self,
        default_credentials: &ProviderTypeDefaultCredentials,
    ) -> Result<Self::Credential, Error> {
        default_credentials.azure.get_cloned()
    }
}

pub struct DeepSeekKind;

impl ProviderKind for DeepSeekKind {
    type Credential = DeepSeekCredentials;
    fn get_provider_type(&self) -> ProviderType {
        ProviderType::Deepseek
    }

    async fn get_credential_field(
        &self,
        default_credentials: &ProviderTypeDefaultCredentials,
    ) -> Result<Self::Credential, Error> {
        default_credentials.deepseek.get_cloned()
    }
}

pub struct FireworksKind;

impl ProviderKind for FireworksKind {
    type Credential = FireworksCredentials;
    fn get_provider_type(&self) -> ProviderType {
        ProviderType::Fireworks
    }

    async fn get_credential_field(
        &self,
        default_credentials: &ProviderTypeDefaultCredentials,
    ) -> Result<Self::Credential, Error> {
        default_credentials.fireworks.get_cloned()
    }
}

pub struct GCPVertexAnthropicKind;

impl ProviderKind for GCPVertexAnthropicKind {
    type Credential = GCPVertexCredentials;
    fn get_provider_type(&self) -> ProviderType {
        ProviderType::GCPVertexAnthropic
    }

    async fn get_credential_field(
        &self,
        default_credentials: &ProviderTypeDefaultCredentials,
    ) -> Result<Self::Credential, Error> {
        default_credentials.gcp_vertex_anthropic.get_cloned().await
    }
}

impl GCPVertexAnthropicKind {
    pub async fn get_defaulted_credential(
        &self,
        api_key_location: Option<&CredentialLocationWithFallback>,
        default_credentials: &ProviderTypeDefaultCredentials,
    ) -> Result<GCPVertexCredentials, Error> {
        if let Some(api_key_location) = api_key_location {
            return make_gcp_credentials_with_fallback(
                ProviderType::GCPVertexAnthropic,
                api_key_location,
            )
            .await;
        }

        Ok(self
            .get_credential_field(default_credentials)
            .await?
            .clone())
    }
}

pub struct GCPVertexGeminiKind;

impl ProviderKind for GCPVertexGeminiKind {
    type Credential = GCPVertexCredentials;
    fn get_provider_type(&self) -> ProviderType {
        ProviderType::GCPVertexGemini
    }

    async fn get_credential_field(
        &self,
        default_credentials: &ProviderTypeDefaultCredentials,
    ) -> Result<Self::Credential, Error> {
        default_credentials.gcp_vertex_gemini.get_cloned().await
    }
}

impl GCPVertexGeminiKind {
    pub async fn get_defaulted_credential(
        &self,
        api_key_location: Option<&CredentialLocationWithFallback>,
        default_credentials: &ProviderTypeDefaultCredentials,
    ) -> Result<GCPVertexCredentials, Error> {
        if let Some(api_key_location) = api_key_location {
            return make_gcp_credentials_with_fallback(
                ProviderType::GCPVertexGemini,
                api_key_location,
            )
            .await;
        }

        Ok(self
            .get_credential_field(default_credentials)
            .await?
            .clone())
    }
}

pub struct GoogleAIStudioGeminiKind;

impl ProviderKind for GoogleAIStudioGeminiKind {
    type Credential = GoogleAIStudioCredentials;
    fn get_provider_type(&self) -> ProviderType {
        ProviderType::GoogleAIStudioGemini
    }

    async fn get_credential_field(
        &self,
        default_credentials: &ProviderTypeDefaultCredentials,
    ) -> Result<Self::Credential, Error> {
        default_credentials.google_ai_studio_gemini.get_cloned()
    }
}

pub struct GroqKind;

impl ProviderKind for GroqKind {
    type Credential = GroqCredentials;
    fn get_provider_type(&self) -> ProviderType {
        ProviderType::Groq
    }

    async fn get_credential_field(
        &self,
        default_credentials: &ProviderTypeDefaultCredentials,
    ) -> Result<Self::Credential, Error> {
        default_credentials.groq.get_cloned()
    }
}

pub struct HyperbolicKind;

impl ProviderKind for HyperbolicKind {
    type Credential = HyperbolicCredentials;
    fn get_provider_type(&self) -> ProviderType {
        ProviderType::Hyperbolic
    }

    async fn get_credential_field(
        &self,
        default_credentials: &ProviderTypeDefaultCredentials,
    ) -> Result<Self::Credential, Error> {
        default_credentials.hyperbolic.get_cloned()
    }
}

pub struct MistralKind;

impl ProviderKind for MistralKind {
    type Credential = MistralCredentials;
    fn get_provider_type(&self) -> ProviderType {
        ProviderType::Mistral
    }

    async fn get_credential_field(
        &self,
        default_credentials: &ProviderTypeDefaultCredentials,
    ) -> Result<Self::Credential, Error> {
        default_credentials.mistral.get_cloned()
    }
}

pub struct OpenRouterKind;

impl ProviderKind for OpenRouterKind {
    type Credential = OpenRouterCredentials;
    fn get_provider_type(&self) -> ProviderType {
        ProviderType::OpenRouter
    }

    async fn get_credential_field(
        &self,
        default_credentials: &ProviderTypeDefaultCredentials,
    ) -> Result<Self::Credential, Error> {
        default_credentials.openrouter.get_cloned()
    }
}

pub struct SGLangKind;

impl ProviderKind for SGLangKind {
    type Credential = SGLangCredentials;
    fn get_provider_type(&self) -> ProviderType {
        ProviderType::SGLang
    }

    async fn get_credential_field(
        &self,
        default_credentials: &ProviderTypeDefaultCredentials,
    ) -> Result<Self::Credential, Error> {
        default_credentials.sglang.get_cloned()
    }
}

pub struct TGIKind;

impl ProviderKind for TGIKind {
    type Credential = TGICredentials;
    fn get_provider_type(&self) -> ProviderType {
        ProviderType::TGI
    }

    async fn get_credential_field(
        &self,
        default_credentials: &ProviderTypeDefaultCredentials,
    ) -> Result<Self::Credential, Error> {
        default_credentials.tgi.get_cloned()
    }
}

pub struct TogetherKind;

impl ProviderKind for TogetherKind {
    type Credential = TogetherCredentials;
    fn get_provider_type(&self) -> ProviderType {
        ProviderType::Together
    }

    async fn get_credential_field(
        &self,
        default_credentials: &ProviderTypeDefaultCredentials,
    ) -> Result<Self::Credential, Error> {
        default_credentials.together.get_cloned()
    }
}

pub struct VLLMKind;

impl ProviderKind for VLLMKind {
    type Credential = VLLMCredentials;
    fn get_provider_type(&self) -> ProviderType {
        ProviderType::VLLM
    }

    async fn get_credential_field(
        &self,
        default_credentials: &ProviderTypeDefaultCredentials,
    ) -> Result<Self::Credential, Error> {
        default_credentials.vllm.get_cloned()
    }
}

pub struct XAIKind;

impl ProviderKind for XAIKind {
    type Credential = XAICredentials;
    fn get_provider_type(&self) -> ProviderType {
        ProviderType::XAI
    }

    async fn get_credential_field(
        &self,
        default_credentials: &ProviderTypeDefaultCredentials,
    ) -> Result<Self::Credential, Error> {
        default_credentials.xai.get_cloned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::ModelConfig;
    use std::collections::HashMap;
    use std::sync::Arc;

    #[tokio::test]
    async fn alias_get_merges_all_shorthand_targets() {
        let aliases = ModelAliasTable {
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
        };
        let table = BaseModelTable::<ModelConfig>::new(
            HashMap::new(),
            Arc::new(ProviderTypeDefaultCredentials::default()),
            chrono::Duration::seconds(120),
            Arc::new(aliases),
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
        let aliases = ModelAliasTable {
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
                min_tokens_per_sec: None,
            }],
        };
        let table = BaseModelTable::<ModelConfig>::new(
            HashMap::new(),
            Arc::new(ProviderTypeDefaultCredentials::default()),
            chrono::Duration::seconds(120),
            Arc::new(aliases),
        )
        .unwrap();
        let model = table.get("dummy::good", None).await.unwrap().unwrap();
        assert_eq!(model.routing[0].as_ref(), "dummy::good");
        assert_eq!(model.routing[1].as_ref(), "dummy::error");
    }
}
