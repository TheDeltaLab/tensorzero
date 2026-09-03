// Modified by Delta-AI under Apache 2.0
//! OpenAI-compatible API endpoints.
//!
//! This module provides compatibility with OpenAI's API format, supporting
//! chat completions, embeddings, completions, and responses. Routes are
//! registered at both `/openai/v1/...` (TensorZero native) and `/v1/...`
//! (Synapse / OpenAI SDK default).

pub mod anthropic_messages;
pub mod async_inference;
pub mod async_inference_types;
pub mod chat_completions;
pub mod completions;
pub mod embeddings;
pub mod error;
pub mod infer;
pub mod rerank;
pub mod responses;
pub mod stream_aggregator;
pub mod synapse;
pub mod types;

pub use error::{OpenAICompatibleError, OpenAIStructuredJson};

use anthropic_messages::messages_handler;
use async_inference::{
    chat_completions_async_handler, get_async_task_handler, messages_async_handler,
    responses_async_handler, stream_async_task_handler,
};
use chat_completions::chat_completions_handler;
use completions::completions_handler;
use embeddings::embeddings_handler;
use rerank::rerank_handler;
use responses::responses_handler;

use axum::Router;
use axum::routing::{get, post};

use crate::endpoints::RouteHandlers;
#[expect(
    clippy::disallowed_types,
    reason = "router extension trait must be implemented for Router<SwappableAppStateData>"
)]
use crate::utils::gateway::SwappableAppStateData;

/// Constructs (but does not register) all of our OpenAI-compatible endpoints.
/// The `RouterExt::register_openai_compatible_routes` is a convenience method
/// to register all of the routes on a router.
///
/// Alternatively, the returned `RouteHandlers` can be inspected (e.g. to allow middleware to see the route paths)
/// and then manually registered on a router.
pub fn build_openai_compatible_routes() -> RouteHandlers {
    RouteHandlers {
        routes: vec![
            (
                "/openai/v1/chat/completions",
                post(chat_completions_handler),
            ),
            ("/v1/chat/completions", post(chat_completions_handler)),
            ("/openai/v1/embeddings", post(embeddings_handler)),
            ("/v1/embeddings", post(embeddings_handler)),
            ("/openai/v1/completions", post(completions_handler)),
            ("/v1/completions", post(completions_handler)),
            ("/openai/v1/responses", post(responses_handler)),
            ("/v1/responses", post(responses_handler)),
            ("/openai/v1/rerank", post(rerank_handler)),
            ("/v1/rerank", post(rerank_handler)),
            ("/openai/v1/reranks", post(rerank_handler)),
            ("/v1/reranks", post(rerank_handler)),
            ("/openai/v1/messages", post(messages_handler)),
            ("/v1/messages", post(messages_handler)),
            ("/anthropic/v1/messages", post(messages_handler)),
            // Async inference
            (
                "/openai/v1/chat/completions/async",
                post(chat_completions_async_handler),
            ),
            (
                "/v1/chat/completions/async",
                post(chat_completions_async_handler),
            ),
            ("/openai/v1/responses/async", post(responses_async_handler)),
            ("/v1/responses/async", post(responses_async_handler)),
            ("/openai/v1/messages/async", post(messages_async_handler)),
            ("/v1/messages/async", post(messages_async_handler)),
            ("/anthropic/v1/messages/async", post(messages_async_handler)),
            (
                "/openai/v1/async_tasks/{task_id}",
                get(get_async_task_handler),
            ),
            ("/v1/async_tasks/{task_id}", get(get_async_task_handler)),
            (
                "/openai/v1/async_tasks/{task_id}/stream",
                get(stream_async_task_handler),
            ),
            (
                "/v1/async_tasks/{task_id}/stream",
                get(stream_async_task_handler),
            ),
        ],
    }
}

pub trait RouterExt {
    /// Applies our OpenAI-compatible endpoints to the router.
    /// This is used by the the gateway for the patched OpenAI python client (`start_openai_compatible_gateway`),
    /// as well as the normal standalone TensorZero gateway.
    fn register_openai_compatible_routes(self) -> Self;
}

#[expect(
    clippy::disallowed_types,
    reason = "router extension trait must be implemented for Router<SwappableAppStateData>"
)]
impl RouterExt for Router<SwappableAppStateData> {
    fn register_openai_compatible_routes(mut self) -> Self {
        for (path, handler) in build_openai_compatible_routes().routes {
            self = self.route(path, handler);
        }
        self
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_openai_compatible_routes_include_v1_aliases() {
        let paths: Vec<_> = super::build_openai_compatible_routes()
            .routes
            .iter()
            .map(|(path, _)| *path)
            .collect();
        for path in [
            "/openai/v1/chat/completions",
            "/v1/chat/completions",
            "/openai/v1/embeddings",
            "/v1/embeddings",
            "/openai/v1/completions",
            "/v1/completions",
            "/openai/v1/responses",
            "/v1/responses",
            "/openai/v1/rerank",
            "/v1/rerank",
            "/openai/v1/reranks",
            "/v1/reranks",
            "/openai/v1/messages",
            "/v1/messages",
            "/anthropic/v1/messages",
            "/openai/v1/chat/completions/async",
            "/v1/chat/completions/async",
            "/openai/v1/responses/async",
            "/v1/responses/async",
            "/openai/v1/messages/async",
            "/v1/messages/async",
            "/anthropic/v1/messages/async",
            "/openai/v1/async_tasks/{task_id}",
            "/v1/async_tasks/{task_id}",
            "/openai/v1/async_tasks/{task_id}/stream",
            "/v1/async_tasks/{task_id}/stream",
        ] {
            assert!(paths.contains(&path), "missing route {path}");
        }
    }
}
