// Modified by Delta-AI under Apache 2.0
//! Persist standalone embeddings/rerank HTTP calls as TensorZero inferences.
//!
//! Chat completions already go through `write_inference`. Embeddings and rerank
//! historically incremented Prometheus counters (embeddings) or wrote nothing
//! (rerank). Observability → Inferences and Synapse analytics both read
//! `chat_inferences` / `model_inferences`, so these endpoints need the same
//! write path.
//!
//! Rows still live in `chat_inferences` (Postgres/ClickHouse only distinguish
//! chat vs json). We use dedicated function names (`tensorzero::embedding` /
//! `tensorzero::rerank`) and structured output JSON so the UI can show them as
//! their own types instead of chat completions.
//!
//! Embedding vectors are not stored in `chat_inferences.output` (too large);
//! we keep count/dimensions and put provider `raw_request` / `raw_response` on
//! the model-inference row.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use serde_json::{Value, json};
use tokio_util::task::TaskTracker;
use tracing_futures::Instrument;
use uuid::Uuid;

use crate::config::Config;
use crate::db::clickhouse::ClickHouseConnectionInfo;
use crate::db::delegating_connection::{DelegatingDatabaseConnection, PrimaryDatastore};
use crate::db::inferences::InferenceQueries;
use crate::db::model_inferences::ModelInferenceQueries;
use crate::db::postgres::PostgresConnectionInfo;
use crate::embeddings::{Embedding, EmbeddingInput};
use crate::endpoints::inference::InferenceParams;
use crate::function::{DEFAULT_FUNCTION_NAME, EMBEDDING_FUNCTION_NAME, RERANK_FUNCTION_NAME};
use crate::inference::types::extra_body::UnfilteredInferenceExtraBody;
use crate::inference::types::{
    ChatInferenceDatabaseInsert, ContentBlockChatOutput, ContentBlockOutput, Latency,
    ModelInferenceResponseWithMetadata, RequestMessage, RequestMessagesOrBatch, Role, StoredInput,
    StoredInputMessage, StoredInputMessageContent, StoredModelInference, System, Text, Usage,
};
use crate::observability_tags::apply_usage_observability_tags;

pub(crate) const ENDPOINT_TAG: &str = "tensorzero::endpoint";
pub(crate) const EMBEDDINGS_ENDPOINT: &str = "embeddings";
pub(crate) const RERANK_ENDPOINT: &str = "rerank";

#[derive(Clone, Debug)]
pub(crate) enum StandaloneInput {
    Embeddings {
        texts: Vec<String>,
    },
    Rerank {
        query: String,
        documents: Vec<String>,
    },
}

#[derive(Clone, Debug)]
pub(crate) struct StandaloneInferenceRecord {
    pub endpoint: &'static str,
    pub variant_name: String,
    pub model_name: String,
    pub model_provider_name: String,
    pub provider_type: String,
    pub input: StandaloneInput,
    pub output_text: String,
    pub raw_request: String,
    pub raw_response: String,
    pub usage: Usage,
    pub latency: Latency,
    pub cached: bool,
    pub extra_internal_tags: HashMap<String, String>,
    pub tags: HashMap<String, String>,
    pub episode_id: Option<Uuid>,
}

pub(crate) async fn maybe_write_standalone_inference(
    config: Arc<Config>,
    clickhouse_connection_info: ClickHouseConnectionInfo,
    postgres_connection_info: PostgresConnectionInfo,
    deferred_tasks: TaskTracker,
    dryrun: bool,
    record: StandaloneInferenceRecord,
) {
    if dryrun || !config.gateway.observability.writes_enabled() {
        return;
    }
    let primary_datastore = match PrimaryDatastore::resolve(
        &config.gateway.observability,
        &clickhouse_connection_info,
        &postgres_connection_info,
    ) {
        Ok(primary_datastore) => primary_datastore,
        Err(error) => {
            let _ = error.log();
            return;
        }
    };
    if primary_datastore == PrimaryDatastore::Disabled {
        return;
    }

    let inference_id = Uuid::now_v7();
    let episode_id = record.episode_id.unwrap_or_else(Uuid::now_v7);
    let async_writes = config.gateway.observability.async_writes();
    let parent_span = tracing::Span::current();
    let write_future = async move {
        if let Err(error) = write_standalone_inference(
            &config,
            clickhouse_connection_info,
            postgres_connection_info,
            primary_datastore,
            inference_id,
            episode_id,
            record,
        )
        .await
        {
            tracing::error!(
                %error,
                %inference_id,
                "Failed to persist standalone inference"
            );
        }
    }
    .instrument(tracing::debug_span!(
        parent: &parent_span,
        "write_inference",
        otel.name = "write_inference",
        stream = false,
        inference_id = %inference_id,
        async_writes = async_writes
    ));
    if async_writes {
        deferred_tasks.spawn(write_future);
    } else {
        write_future.await;
    }
}

async fn write_standalone_inference(
    config: &Config,
    clickhouse_connection_info: ClickHouseConnectionInfo,
    postgres_connection_info: PostgresConnectionInfo,
    primary_datastore: PrimaryDatastore,
    inference_id: Uuid,
    episode_id: Uuid,
    record: StandaloneInferenceRecord,
) -> Result<(), crate::error::Error> {
    let database = DelegatingDatabaseConnection::new(
        clickhouse_connection_info,
        postgres_connection_info,
        primary_datastore,
    );
    let stored_input = stored_input_from_standalone(&record.input);
    let input_messages = request_messages_from_standalone(&record.input);
    let processing_time = latency_duration(&record.latency);
    let mut tags = record.extra_internal_tags;
    tags.extend(record.tags);
    tags.insert(ENDPOINT_TAG.to_string(), record.endpoint.to_string());
    crate::routing::apply_cached_observability_tag(&mut tags, record.cached);
    crate::routing::RoutingSession::apply_observability_tags(&mut tags);
    let function_name = function_name_for_endpoint(record.endpoint).to_string();
    let output_text = record.output_text.clone();

    let model_result = ModelInferenceResponseWithMetadata {
        id: inference_id,
        output: vec![ContentBlockOutput::Text(Text {
            text: record.output_text,
        })],
        system: None,
        input_messages: RequestMessagesOrBatch::Message(input_messages),
        raw_request: record.raw_request,
        raw_response: record.raw_response,
        usage: record.usage,
        latency: record.latency,
        model_provider_name: Arc::from(record.model_provider_name),
        provider_type: Arc::from(record.provider_type),
        model_name: Arc::from(record.model_name),
        cached: record.cached,
        finish_reason: None,
        raw_usage: None,
        relay_raw_response: None,
        failed_raw_response: vec![],
    };
    let model_inference = StoredModelInference::new(
        model_result,
        inference_id,
        function_name.clone(),
        record.variant_name.clone(),
        config.hash.clone(),
    )
    .await?;
    apply_usage_observability_tags(
        &mut tags,
        [(
            model_inference.input_tokens,
            model_inference.output_tokens,
            model_inference.cost,
            model_inference.currency.as_deref(),
        )],
    );
    let chat_inference = ChatInferenceDatabaseInsert {
        id: inference_id,
        function_name,
        variant_name: record.variant_name.clone(),
        episode_id,
        input: Some(stored_input),
        output: Some(vec![ContentBlockChatOutput::Text(Text {
            text: output_text,
        })]),
        tool_params: None,
        inference_params: Some(InferenceParams::default()),
        processing_time_ms: processing_time.map(|duration| duration.as_millis() as u32),
        ttft_ms: None,
        tags,
        extra_body: Some(UnfilteredInferenceExtraBody::default()),
        snapshot_hash: Some(config.hash.clone()),
    };

    let _ = database.insert_model_inferences(&[model_inference]).await;
    let _ = database.insert_chat_inferences(&[chat_inference]).await;
    Ok(())
}

pub(crate) fn function_name_for_endpoint(endpoint: &str) -> &'static str {
    match endpoint {
        EMBEDDINGS_ENDPOINT => EMBEDDING_FUNCTION_NAME,
        RERANK_ENDPOINT => RERANK_FUNCTION_NAME,
        _ => DEFAULT_FUNCTION_NAME,
    }
}

pub(crate) fn embedding_input_to_texts(input: &EmbeddingInput) -> Vec<String> {
    match input {
        EmbeddingInput::Single(text) => vec![text.clone()],
        EmbeddingInput::Batch(texts) => texts.clone(),
        EmbeddingInput::SingleTokens(tokens) => vec![format!("{tokens:?}")],
        EmbeddingInput::BatchTokens(batches) => {
            batches.iter().map(|tokens| format!("{tokens:?}")).collect()
        }
    }
}

pub(crate) fn embedding_output_summary(embeddings: &[Embedding]) -> String {
    let n = embeddings.len();
    let dims = embeddings.first().map(Embedding::ndims).unwrap_or(0);
    if n == 1 {
        format!("Generated 1 embedding ({dims} dimensions)")
    } else {
        format!("Generated {n} embeddings ({dims} dimensions)")
    }
}

pub(crate) fn embedding_output_payload(embeddings: &[Embedding]) -> String {
    let count = embeddings.len();
    let dimensions = embeddings.first().map(Embedding::ndims).unwrap_or(0);
    json!({
        "kind": "embedding",
        "count": count,
        "dimensions": dimensions,
        "vectors_omitted": true,
        "summary": embedding_output_summary(embeddings),
    })
    .to_string()
}

pub(crate) fn rerank_output_summary(body: &Value) -> String {
    let n = rerank_results(body).len();
    format!("Reranked {n} documents")
}

pub(crate) fn rerank_output_payload(body: &Value) -> String {
    let results = rerank_results(body);
    json!({
        "kind": "rerank",
        "count": results.len(),
        "results": results,
        "summary": rerank_output_summary(body),
    })
    .to_string()
}

pub(crate) fn usage_from_json(value: &Value) -> Usage {
    let Some(usage) = value.get("usage") else {
        return Usage::default();
    };
    let as_u32 = |key: &str| usage.get(key).and_then(Value::as_u64).map(|n| n as u32);
    Usage {
        input_tokens: as_u32("prompt_tokens").or_else(|| as_u32("total_tokens")),
        output_tokens: as_u32("completion_tokens"),
        provider_cache_read_input_tokens: None,
        provider_cache_write_input_tokens: None,
        cost: None,
        currency: None,
    }
}

fn rerank_results(body: &Value) -> Vec<Value> {
    body.get("results")
        .and_then(Value::as_array)
        .map(|results| {
            results
                .iter()
                .filter_map(|item| {
                    let index = item.get("index")?.as_u64()?;
                    let score = item
                        .get("relevance_score")
                        .or_else(|| item.get("score"))
                        .and_then(Value::as_f64);
                    Some(json!({
                        "index": index,
                        "relevance_score": score,
                    }))
                })
                .collect()
        })
        .unwrap_or_default()
}

fn stored_input_from_standalone(input: &StandaloneInput) -> StoredInput {
    match input {
        StandaloneInput::Embeddings { texts } => texts_to_stored_input(texts),
        StandaloneInput::Rerank { query, documents } => StoredInput {
            system: Some(System::Text(query.clone())),
            messages: documents
                .iter()
                .map(|text| StoredInputMessage {
                    role: Role::User,
                    content: vec![StoredInputMessageContent::Text(Text { text: text.clone() })],
                })
                .collect(),
        },
    }
}

fn request_messages_from_standalone(input: &StandaloneInput) -> Vec<RequestMessage> {
    match input {
        StandaloneInput::Embeddings { texts } => texts_to_request_messages(texts),
        StandaloneInput::Rerank { query, documents } => {
            let mut texts = Vec::with_capacity(documents.len() + 1);
            texts.push(query.clone());
            texts.extend(documents.iter().cloned());
            texts_to_request_messages(&texts)
        }
    }
}

fn texts_to_stored_input(texts: &[String]) -> StoredInput {
    StoredInput {
        system: None,
        messages: texts
            .iter()
            .map(|text| StoredInputMessage {
                role: Role::User,
                content: vec![StoredInputMessageContent::Text(Text { text: text.clone() })],
            })
            .collect(),
    }
}

fn texts_to_request_messages(texts: &[String]) -> Vec<RequestMessage> {
    texts
        .iter()
        .map(|text| RequestMessage {
            role: Role::User,
            content: vec![crate::inference::types::ContentBlock::Text(Text {
                text: text.clone(),
            })],
        })
        .collect()
}

fn latency_duration(latency: &Latency) -> Option<Duration> {
    match latency {
        Latency::NonStreaming { response_time } | Latency::Streaming { response_time, .. } => {
            Some(*response_time)
        }
        Latency::Batch => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use googletest::prelude::*;

    #[gtest]
    fn embedding_input_to_texts_single_and_batch() {
        expect_eq!(
            embedding_input_to_texts(&EmbeddingInput::Single("hello".into())),
            vec!["hello".to_string()]
        );
        expect_eq!(
            embedding_input_to_texts(&EmbeddingInput::Batch(vec!["a".into(), "b".into()])),
            vec!["a".to_string(), "b".to_string()]
        );
        expect_eq!(
            embedding_input_to_texts(&EmbeddingInput::SingleTokens(vec![1, 2, 3])),
            vec!["[1, 2, 3]".to_string()]
        );
    }

    #[gtest]
    fn embedding_output_payload_avoids_vectors() {
        let embeddings = vec![Embedding::Float(vec![0.1, 0.2, 0.3])];
        let payload = embedding_output_payload(&embeddings);
        expect_that!(payload.contains("0.1"), eq(false));
        let value: Value =
            serde_json::from_str(&payload).expect("embedding payload should be JSON");
        expect_eq!(value["kind"], "embedding");
        expect_eq!(value["count"], 1);
        expect_eq!(value["dimensions"], 3);
        expect_eq!(value["vectors_omitted"], true);
        expect_eq!(value["summary"], "Generated 1 embedding (3 dimensions)");
    }

    #[gtest]
    fn stored_input_keeps_all_batch_texts() {
        let input = texts_to_stored_input(&["one".into(), "two".into()]);
        expect_that!(input.messages.len(), eq(2));
        match &input.messages[1].content[0] {
            StoredInputMessageContent::Text(text) => expect_that!(text.text, eq("two")),
            other => panic!("expected text content, got {other:?}"),
        }
    }

    #[gtest]
    fn rerank_stored_input_keeps_query_and_documents() {
        let input = stored_input_from_standalone(&StandaloneInput::Rerank {
            query: "capital".into(),
            documents: vec!["Paris".into(), "London".into()],
        });
        match input.system {
            Some(System::Text(query)) => expect_that!(query, eq("capital")),
            other => panic!("expected text system query, got {other:?}"),
        }
        expect_that!(input.messages.len(), eq(2));
    }

    #[gtest]
    fn usage_from_json_reads_prompt_or_total_tokens() {
        expect_that!(
            usage_from_json(&json!({ "usage": { "prompt_tokens": 12 } })).input_tokens,
            eq(Some(12))
        );
        expect_that!(
            usage_from_json(&json!({ "usage": { "total_tokens": 7 } })).input_tokens,
            eq(Some(7))
        );
        expect_that!(usage_from_json(&json!({})).input_tokens, eq(None));
    }

    #[gtest]
    fn rerank_output_payload_keeps_index_and_score() {
        let payload = rerank_output_payload(&json!({
            "results": [
                { "index": 1, "relevance_score": 0.9, "document": { "text": "secret" } },
                { "index": 0, "score": 0.2 }
            ]
        }));
        expect_that!(payload.contains("secret"), eq(false));
        let value: Value = serde_json::from_str(&payload).expect("rerank payload should be JSON");
        expect_eq!(value["kind"], "rerank");
        expect_eq!(value["count"], 2);
        expect_eq!(value["results"][0]["index"], 1);
        expect_eq!(value["results"][0]["relevance_score"], 0.9);
        expect_eq!(value["results"][1]["relevance_score"], 0.2);
        expect_eq!(value["summary"], "Reranked 2 documents");
    }

    #[gtest]
    fn function_names_match_endpoint() {
        expect_eq!(
            function_name_for_endpoint(EMBEDDINGS_ENDPOINT),
            EMBEDDING_FUNCTION_NAME
        );
        expect_eq!(
            function_name_for_endpoint(RERANK_ENDPOINT),
            RERANK_FUNCTION_NAME
        );
    }
}
