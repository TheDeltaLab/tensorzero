// Modified by Delta-AI under Apache 2.0
//! Lightweight spawn-only client.

use chrono::{DateTime, Utc};
use durable::{Durable, DurableBuilder, SpawnOptions, SpawnResult};
use secrecy::{ExposeSecret, SecretString};
use serde_json::Value as JsonValue;
use sqlx::{Executor, PgPool, Postgres, QueryBuilder};
use uuid::Uuid;

use crate::error::SpawnError;
use crate::params::TaskToolParams;
use crate::types::{TaskPollResult, TaskStatus, TaskTiming};

/// A lightweight client for spawning durable tasks.
///
/// Unlike the full `ToolExecutor` in `durable-tools`, this client only
/// supports spawning tasks by name. It does not include tool registration,
/// workers, or inference capabilities.
///
/// # Example
///
/// ```ignore
/// use durable_tools_spawn::{SpawnClient, SpawnClientBuilder};
/// use secrecy::SecretString;
/// use uuid::Uuid;
///
/// let client = SpawnClient::builder()
///     .database_url(database_url)
///     .queue_name("tools")
///     .build()
///     .await?;
///
/// let episode_id = Uuid::now_v7();
/// client.spawn_tool_by_name(
///     "research",
///     serde_json::json!({"topic": "rust"}),
///     serde_json::json!(null),  // side_info
///     episode_id,
/// ).await?;
/// ```
pub struct SpawnClient {
    durable: Durable<()>,
}

impl SpawnClient {
    /// Create a builder for custom configuration.
    pub fn builder() -> SpawnClientBuilder {
        SpawnClientBuilder::new()
    }

    /// Spawn a task by name with JSON parameters.
    ///
    /// # Arguments
    ///
    /// * `tool_name` - The registered name of the tool to spawn
    /// * `llm_params` - Parameters visible to the LLM
    /// * `side_info` - Hidden parameters (use `json!(null)` if not needed)
    /// * `episode_id` - The episode ID for this execution
    /// * `options` - Options for spawning the task
    ///
    /// # Errors
    ///
    /// Returns an error if spawning the task fails.
    pub async fn spawn_tool_by_name(
        &self,
        tool_name: &str,
        llm_params: JsonValue,
        side_info: JsonValue,
        episode_id: Uuid,
        options: SpawnOptions,
    ) -> Result<SpawnResult, SpawnError> {
        let wrapped_params = TaskToolParams {
            llm_params,
            side_info,
            episode_id,
        };

        self.durable
            .spawn_by_name_unchecked(tool_name, serde_json::to_value(wrapped_params)?, options)
            .await
            .map_err(Into::into)
    }

    /// Spawn a task by name using a custom executor (e.g., a transaction).
    ///
    /// This allows you to atomically enqueue a task as part of a larger transaction.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let mut tx = client.pool().begin().await?;
    ///
    /// sqlx::query("INSERT INTO orders (id) VALUES ($1)")
    ///     .bind(order_id)
    ///     .execute(&mut *tx)
    ///     .await?;
    ///
    /// client.spawn_tool_by_name_with(
    ///     &mut *tx,
    ///     "process_order",
    ///     serde_json::json!({"order_id": order_id}),
    ///     serde_json::json!(null),
    ///     episode_id,
    /// ).await?;
    ///
    /// tx.commit().await?;
    /// ```
    ///
    /// # Arguments
    ///
    /// * `executor` - The executor to use (e.g., `&mut *tx` for a transaction)
    /// * `tool_name` - The registered name of the tool to spawn
    /// * `llm_params` - Parameters visible to the LLM
    /// * `side_info` - Hidden parameters (use `json!(null)` if not needed)
    /// * `episode_id` - The episode ID for this execution
    /// * `options` - Options for spawning the task
    ///
    /// # Errors
    ///
    /// Returns an error if spawning the task fails.
    pub async fn spawn_tool_by_name_with<'e, E>(
        &self,
        executor: E,
        tool_name: &str,
        llm_params: JsonValue,
        side_info: JsonValue,
        episode_id: Uuid,
        options: SpawnOptions,
    ) -> Result<SpawnResult, SpawnError>
    where
        E: Executor<'e, Database = Postgres>,
    {
        let wrapped_params = TaskToolParams {
            llm_params,
            side_info,
            episode_id,
        };

        self.durable
            .spawn_by_name_unchecked_with(
                executor,
                tool_name,
                serde_json::to_value(wrapped_params)?,
                options,
            )
            .await
            .map_err(Into::into)
    }

    /// Spawn a task by name with raw JSON parameters, without wrapping them in
    /// [`TaskToolParams`].
    ///
    /// Use this for tasks whose `Params` type is not the tool-call envelope
    /// (e.g. internal gateway tasks like async inference). The task's `Params`
    /// type must deserialize from `params` as-is.
    ///
    /// # Errors
    ///
    /// Returns an error if spawning the task fails.
    pub async fn spawn_task_by_name(
        &self,
        task_name: &str,
        params: JsonValue,
        options: SpawnOptions,
    ) -> Result<SpawnResult, SpawnError> {
        self.durable
            .spawn_by_name_unchecked(task_name, params, options)
            .await
            .map_err(Into::into)
    }

    /// Get the queue name.
    pub fn queue_name(&self) -> &str {
        self.durable.queue_name()
    }

    /// Get the database pool.
    pub fn pool(&self) -> &PgPool {
        self.durable.pool()
    }

    /// Poll the status and result of a durable task by ID.
    ///
    /// Returns the current status of the task. If the task has completed,
    /// the result includes the task's output payload. If the task has failed,
    /// the result includes error information from the last run.
    ///
    /// # Errors
    ///
    /// Returns `SpawnError::TaskNotFound` if no task with the given ID exists.
    /// Returns `SpawnError::InvalidState` if the database returns an unrecognized state.
    pub async fn get_task_result(&self, task_id: Uuid) -> Result<TaskPollResult, SpawnError> {
        let queue_name = self.queue_name();

        // Query the task table for state, completed_payload, and params
        let query = format!(
            "SELECT state, completed_payload, params FROM durable.t_{queue_name} WHERE task_id = $1"
        );

        let row: Option<(String, Option<JsonValue>, Option<JsonValue>)> =
            sqlx::query_as(sqlx::AssertSqlSafe(query))
                .bind(task_id)
                .fetch_optional(self.pool())
                .await?;

        let (state_str, completed_payload, params) =
            row.ok_or(SpawnError::TaskNotFound(task_id))?;

        let status: TaskStatus = state_str
            .parse()
            .map_err(|_| SpawnError::InvalidState(state_str.clone()))?;

        // For failed tasks, fetch the failure_reason from the latest run
        let error = if status == TaskStatus::Failed {
            let error_query = format!(
                "SELECT failure_reason FROM durable.r_{queue_name} \
                 WHERE task_id = $1 AND failure_reason IS NOT NULL \
                 ORDER BY attempt DESC LIMIT 1"
            );
            let error_row: Option<(Option<JsonValue>,)> =
                sqlx::query_as(sqlx::AssertSqlSafe(error_query))
                    .bind(task_id)
                    .fetch_optional(self.pool())
                    .await?;
            error_row.and_then(|(reason,)| reason)
        } else {
            None
        };

        Ok(TaskPollResult {
            status,
            result: completed_payload,
            error,
            params,
        })
    }

    /// Get the latest checkpoint name for a running task.
    /// Returns `None` if no checkpoints exist yet.
    pub async fn get_task_progress(&self, task_id: Uuid) -> Result<Option<String>, SpawnError> {
        let queue_name = self.queue_name();
        let query = format!(
            "SELECT checkpoint_name FROM durable.c_{queue_name} \
             WHERE task_id = $1 \
             ORDER BY updated_at DESC \
             LIMIT 1"
        );
        let row: Option<(String,)> = sqlx::query_as(sqlx::AssertSqlSafe(query))
            .bind(task_id)
            .fetch_optional(self.pool())
            .await?;
        Ok(row.map(|(name,)| name))
    }

    /// Compute the queue position of a pending/sleeping task: the number of
    /// currently-claimable runs that would be claimed before this task's
    /// latest run (claim order is `(available_at, run_id)`).
    ///
    /// Returns `Ok(None)` if no task with the given ID exists. Returns
    /// `Ok(Some(0))` when the task is next in line (or not pending).
    ///
    /// # Errors
    ///
    /// Returns an error if any database operation fails.
    pub async fn get_queue_position(&self, task_id: Uuid) -> Result<Option<i64>, SpawnError> {
        let queue_name = self.queue_name();

        // Bail out early if the task does not exist at all.
        // Table names are dynamic (`t_<queue>`), so use `QueryBuilder`:
        // `.push()` for the trusted queue name, `.push_bind()` for values.
        let mut exists_query: QueryBuilder<Postgres> =
            QueryBuilder::new("SELECT state FROM durable.t_");
        exists_query.push(queue_name);
        exists_query.push(" WHERE task_id = ");
        exists_query.push_bind(task_id);
        let task_state: Option<(String,)> = exists_query
            .build_query_as()
            .fetch_optional(self.pool())
            .await?;
        if task_state.is_none() {
            return Ok(None);
        }

        let mut query: QueryBuilder<Postgres> =
            QueryBuilder::new("SELECT count(*)::BIGINT FROM durable.r_");
        query.push(queue_name);
        query.push(" r JOIN durable.t_");
        query.push(queue_name);
        query.push(
            " t ON t.task_id = r.task_id \
             WHERE r.state IN ('pending', 'sleeping') \
             AND t.state IN ('pending', 'sleeping', 'running') \
             AND r.available_at <= durable.current_time() \
             AND (r.available_at, r.run_id) < (\
                 SELECT available_at, run_id FROM durable.r_",
        );
        query.push(queue_name);
        query.push(" WHERE task_id = ");
        query.push_bind(task_id);
        query.push(" ORDER BY attempt DESC LIMIT 1)");

        let count: i64 = query.build_query_scalar().fetch_one(self.pool()).await?;
        Ok(Some(count))
    }

    /// Get the enqueue / first-start timing of a task, for computing
    /// `started_at` / `elapsed_ms` of running tasks.
    ///
    /// Returns `Ok(None)` if no task with the given ID exists.
    ///
    /// # Errors
    ///
    /// Returns an error if any database operation fails.
    pub async fn get_task_timing(&self, task_id: Uuid) -> Result<Option<TaskTiming>, SpawnError> {
        let queue_name = self.queue_name();
        let mut query: QueryBuilder<Postgres> =
            QueryBuilder::new("SELECT enqueue_at, first_started_at FROM durable.t_");
        query.push(queue_name);
        query.push(" WHERE task_id = ");
        query.push_bind(task_id);
        let row: Option<(DateTime<Utc>, Option<DateTime<Utc>>)> =
            query.build_query_as().fetch_optional(self.pool()).await?;
        Ok(row.map(|(enqueue_at, first_started_at)| TaskTiming {
            enqueue_at,
            first_started_at,
        }))
    }

    /// Cancel a task by ID. Running tasks will be cancelled at their next
    /// checkpoint or heartbeat.
    ///
    /// # Errors
    ///
    /// Returns an error if the database operation fails.
    pub async fn cancel_task(&self, task_id: Uuid) -> Result<(), SpawnError> {
        self.durable
            .cancel_task(task_id, None)
            .await
            .map_err(Into::into)
    }

    /// Interrupt all durable tasks associated with a session ID.
    ///
    /// This queries for non-terminal tasks where `params->'side_info'->>'session_id'`
    /// matches the provided session ID, then interrupts each one. The durable framework's
    /// `cancel_task` function automatically cascades interruption to child tasks.
    ///
    /// # Returns
    ///
    /// Returns the number of tasks that were interrupted.
    ///
    /// # Errors
    ///
    /// Returns an error if any database operation fails.
    pub async fn interrupt_tasks_by_session_id(&self, session_id: Uuid) -> Result<u64, SpawnError> {
        let queue_name = self.queue_name();

        // Find all non-terminal tasks with this session_id
        // Using QueryBuilder for dynamic table name (queue_name is a trusted internal value)
        let mut query_builder: sqlx::QueryBuilder<Postgres> =
            sqlx::QueryBuilder::new("SELECT task_id FROM durable.t_");
        query_builder.push(queue_name);
        query_builder.push(" WHERE params->'side_info'->>'session_id' = ");
        query_builder.push_bind(session_id.to_string());
        query_builder.push(" AND state NOT IN ('completed', 'failed', 'cancelled')");

        let task_ids: Vec<(Uuid,)> = query_builder
            .build_query_as()
            .fetch_all(self.pool())
            .await?;

        // Interrupt each task (cascade to children handled by durable.cancel_task)
        for (task_id,) in &task_ids {
            sqlx::query("SELECT durable.cancel_task($1, $2)")
                .bind(queue_name)
                .bind(task_id)
                .execute(self.pool())
                .await?;
        }

        Ok(task_ids.len() as u64)
    }
}

/// Builder for creating a [`SpawnClient`].
pub struct SpawnClientBuilder {
    database_url: Option<SecretString>,
    pool: Option<PgPool>,
    queue_name: String,
}

impl SpawnClientBuilder {
    /// Create a new builder with default settings.
    pub fn new() -> Self {
        Self {
            database_url: None,
            pool: None,
            queue_name: "tools".to_string(),
        }
    }

    /// Set the database URL (will create a new connection pool).
    #[must_use]
    pub fn database_url(mut self, url: SecretString) -> Self {
        self.database_url = Some(url);
        self
    }

    /// Use an existing connection pool.
    #[must_use]
    pub fn pool(mut self, pool: PgPool) -> Self {
        self.pool = Some(pool);
        self
    }

    /// Set the queue name (default: "tools").
    #[must_use]
    pub fn queue_name(mut self, name: impl Into<String>) -> Self {
        self.queue_name = name.into();
        self
    }

    /// Build the [`SpawnClient`].
    ///
    /// # Errors
    ///
    /// Returns an error if the database connection fails or the queue name
    /// contains unsafe characters.
    pub async fn build(self) -> Result<SpawnClient, SpawnError> {
        // Validate queue_name since it is interpolated into SQL table names
        if !self
            .queue_name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_')
        {
            return Err(SpawnError::MissingConfig(
                "Queue name must contain only ASCII alphanumeric characters and underscores",
            ));
        }

        let pool = if let Some(pool) = self.pool {
            pool
        } else {
            let url = self.database_url.ok_or(SpawnError::MissingConfig(
                "No database URL configured. Set database_url() or pool()",
            ))?;
            PgPool::connect(url.expose_secret())
                .await
                .map_err(SpawnError::Database)?
        };

        let durable = DurableBuilder::new()
            .pool(pool)
            .queue_name(&self.queue_name)
            .build()
            .await?;

        Ok(SpawnClient { durable })
    }
}

impl Default for SpawnClientBuilder {
    fn default() -> Self {
        Self::new()
    }
}
