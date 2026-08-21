// Modified by Delta-AI under Apache 2.0
use super::check_column_exists;
use crate::db::clickhouse::ClickHouseConnectionInfo;
use crate::db::clickhouse::migration_manager::migration_trait::Migration;
use crate::error::delayed_error::DelayedError;
use async_trait::async_trait;

const MIGRATION_ID: &str = "0054";

/// Adds an ISO 4217 `currency` column to `ModelInference`.
/// NULL means the inference predates currency tracking or cost was not configured.
pub struct Migration0054<'a> {
    pub clickhouse: &'a ClickHouseConnectionInfo,
}

#[async_trait]
impl Migration for Migration0054<'_> {
    async fn can_apply(&self) -> Result<(), DelayedError> {
        Ok(())
    }

    async fn should_apply(&self) -> Result<bool, DelayedError> {
        Ok(
            !check_column_exists(self.clickhouse, "ModelInference", "currency", MIGRATION_ID)
                .await?,
        )
    }

    async fn apply(&self, _clean_start: bool) -> Result<(), DelayedError> {
        let on_cluster_name = self.clickhouse.get_on_cluster_name();

        self.clickhouse
            .run_query_synchronous_no_params_delayed_err(format!(
                "ALTER TABLE ModelInference{on_cluster_name} ADD COLUMN IF NOT EXISTS currency Nullable(String)"
            ))
            .await?;

        Ok(())
    }

    fn rollback_instructions(&self) -> String {
        let on_cluster_name = self.clickhouse.get_on_cluster_name();
        format!("ALTER TABLE ModelInference{on_cluster_name} DROP COLUMN currency;")
    }

    async fn has_succeeded(&self) -> Result<bool, DelayedError> {
        Ok(
            check_column_exists(self.clickhouse, "ModelInference", "currency", MIGRATION_ID)
                .await?,
        )
    }
}
