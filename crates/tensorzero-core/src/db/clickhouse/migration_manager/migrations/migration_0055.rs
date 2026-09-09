// Modified by Delta-AI under Apache 2.0
use super::check_column_exists;
use crate::db::clickhouse::ClickHouseConnectionInfo;
use crate::db::clickhouse::migration_manager::migration_trait::Migration;
use crate::error::delayed_error::DelayedError;
use async_trait::async_trait;

const MIGRATION_ID: &str = "0055";

/// Adds a Nullable `error` column to `ChatInference`, `JsonInference`, and
/// `ModelInference` so failed inferences can be recorded alongside successful
/// ones. NULL means the inference succeeded (or predates failure recording).
pub struct Migration0055<'a> {
    pub clickhouse: &'a ClickHouseConnectionInfo,
}

#[async_trait]
impl Migration for Migration0055<'_> {
    async fn can_apply(&self) -> Result<(), DelayedError> {
        Ok(())
    }

    async fn should_apply(&self) -> Result<bool, DelayedError> {
        Ok(
            !check_column_exists(self.clickhouse, "ChatInference", "error", MIGRATION_ID).await?
                || !check_column_exists(self.clickhouse, "JsonInference", "error", MIGRATION_ID)
                    .await?
                || !check_column_exists(self.clickhouse, "ModelInference", "error", MIGRATION_ID)
                    .await?,
        )
    }

    async fn apply(&self, _clean_start: bool) -> Result<(), DelayedError> {
        let on_cluster_name = self.clickhouse.get_on_cluster_name();

        for table in ["ChatInference", "JsonInference", "ModelInference"] {
            self.clickhouse
                .run_query_synchronous_no_params_delayed_err(format!(
                    "ALTER TABLE {table}{on_cluster_name} ADD COLUMN IF NOT EXISTS error Nullable(String)"
                ))
                .await?;
        }

        Ok(())
    }

    fn rollback_instructions(&self) -> String {
        let on_cluster_name = self.clickhouse.get_on_cluster_name();
        format!(
            "ALTER TABLE ChatInference{on_cluster_name} DROP COLUMN error;\nALTER TABLE JsonInference{on_cluster_name} DROP COLUMN error;\nALTER TABLE ModelInference{on_cluster_name} DROP COLUMN error;"
        )
    }

    async fn has_succeeded(&self) -> Result<bool, DelayedError> {
        Ok(
            check_column_exists(self.clickhouse, "ChatInference", "error", MIGRATION_ID).await?
                && check_column_exists(self.clickhouse, "JsonInference", "error", MIGRATION_ID)
                    .await?
                && check_column_exists(self.clickhouse, "ModelInference", "error", MIGRATION_ID)
                    .await?,
        )
    }
}
