// Modified by Delta-AI under Apache 2.0
//! `EmbeddingProviderConfig` extracted from `tensorzero-core`'s `embeddings.rs`.

use serde::Serialize;
use tensorzero_error::Error;
use tensorzero_http::TensorzeroHttpClient;
use tensorzero_inference_types::embeddings::{
    EmbeddingProvider, EmbeddingProviderRequestInfo, EmbeddingProviderResponse, EmbeddingRequest,
};
use tensorzero_types::inference_params::InferenceCredentials;

use crate::providers::azure::AzureProvider;
#[cfg(any(test, feature = "e2e_tests"))]
use crate::providers::dummy::DummyProvider;
use crate::providers::openai::OpenAIProvider;
use crate::providers::openrouter::OpenRouterProvider;

#[derive(ts_rs::TS, Debug, Serialize)]
#[ts(export)]
pub enum EmbeddingProviderConfig {
    OpenAI(OpenAIProvider),
    Azure(AzureProvider),
    OpenRouter(OpenRouterProvider),
    #[cfg(any(test, feature = "e2e_tests"))]
    Dummy(DummyProvider),
}

impl EmbeddingProviderConfig {
    pub fn provider_type(&self) -> &'static str {
        match self {
            EmbeddingProviderConfig::OpenAI(_) => crate::providers::openai::PROVIDER_TYPE,
            EmbeddingProviderConfig::Azure(_) => crate::providers::azure::PROVIDER_TYPE,
            EmbeddingProviderConfig::OpenRouter(_) => crate::providers::openrouter::PROVIDER_TYPE,
            #[cfg(any(test, feature = "e2e_tests"))]
            EmbeddingProviderConfig::Dummy(_) => crate::providers::dummy::PROVIDER_TYPE,
        }
    }
}

impl EmbeddingProvider for EmbeddingProviderConfig {
    async fn embed(
        &self,
        request: &EmbeddingRequest,
        client: &TensorzeroHttpClient,
        dynamic_api_keys: &InferenceCredentials,
        model_provider_data: &EmbeddingProviderRequestInfo,
    ) -> Result<EmbeddingProviderResponse, Error> {
        match self {
            EmbeddingProviderConfig::OpenAI(provider) => {
                provider
                    .embed(request, client, dynamic_api_keys, model_provider_data)
                    .await
            }
            EmbeddingProviderConfig::Azure(provider) => {
                provider
                    .embed(request, client, dynamic_api_keys, model_provider_data)
                    .await
            }
            EmbeddingProviderConfig::OpenRouter(provider) => {
                provider
                    .embed(request, client, dynamic_api_keys, model_provider_data)
                    .await
            }
            #[cfg(any(test, feature = "e2e_tests"))]
            EmbeddingProviderConfig::Dummy(provider) => {
                provider
                    .embed(request, client, dynamic_api_keys, model_provider_data)
                    .await
            }
        }
    }
}
