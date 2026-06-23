// Modified by Delta-AI under Apache 2.0
use serde::Serialize;
use std::sync::Arc;

/// A single target within an alias: which provider + model to try.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, ts_rs::TS)]
#[ts(export)]
pub struct ModelAliasTarget {
    pub provider_type: Arc<str>,
    pub model_name: Arc<str>,
}

/// A named alias that maps to one or more (provider, model) targets.
///
/// `task`: If `Some`, this alias only matches lookups with the same task type
///   (e.g. "chat", "embedding", "rerank"). If `None`, it matches any task.
/// `targets`: Ordered list of (provider, model) pairs to try.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, ts_rs::TS)]
#[ts(export, optional_fields)]
pub struct ModelAlias {
    pub name: Arc<str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub task: Option<Arc<str>>,
    pub targets: Vec<ModelAliasTarget>,
}

/// Lookup table for model aliases, shared across chat/embedding/rerank tables.
#[derive(Clone, Debug, Default, Serialize, ts_rs::TS)]
#[ts(export)]
pub struct ModelAliasTable {
    pub aliases: Vec<ModelAlias>,
}

impl ModelAliasTable {
    /// Look up an alias by name, with optional task filtering.
    ///
    /// If `task` is `Some`, only aliases with matching `task` or `None` task
    /// (wildcard) are returned. If `task` is `None`, all aliases match.
    pub fn resolve(&self, name: &str, task: Option<&str>) -> Option<&ModelAlias> {
        self.aliases.iter().find(|a| {
            a.name.as_ref() == name
                && (a.task.is_none() || task.is_none_or(|t| a.task.as_deref() == Some(t)))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use googletest::prelude::*;
    use std::sync::Arc;

    fn make_alias(name: &str, task: Option<&str>, targets: Vec<(&str, &str)>) -> ModelAlias {
        ModelAlias {
            name: Arc::from(name),
            task: task.map(|t| Arc::from(t)),
            targets: targets
                .into_iter()
                .map(|(p, m)| ModelAliasTarget {
                    provider_type: Arc::from(p),
                    model_name: Arc::from(m),
                })
                .collect(),
        }
    }

    #[gtest]
    fn resolve_with_matching_task() {
        let table = ModelAliasTable {
            aliases: vec![make_alias(
                "fast-model",
                Some("chat"),
                vec![("openai", "gpt-4o-mini")],
            )],
        };
        let found = table.resolve("fast-model", Some("chat"));
        expect_that!(found.is_some(), eq(true));
    }

    #[gtest]
    fn resolve_wildcard_task_matches_any() {
        let table = ModelAliasTable {
            aliases: vec![make_alias(
                "fast-model",
                None,
                vec![("openai", "gpt-4o-mini")],
            )],
        };
        expect_that!(table.resolve("fast-model", Some("chat")).is_some(), eq(true));
        expect_that!(
            table.resolve("fast-model", Some("embedding")).is_some(),
            eq(true)
        );
        expect_that!(table.resolve("fast-model", None).is_some(), eq(true));
    }

    #[gtest]
    fn resolve_task_mismatch_returns_none() {
        let table = ModelAliasTable {
            aliases: vec![make_alias(
                "fast-model",
                Some("chat"),
                vec![("openai", "gpt-4o-mini")],
            )],
        };
        expect_that!(table.resolve("fast-model", Some("embedding")).is_none(), eq(true));
    }

    #[gtest]
    fn resolve_unknown_name_returns_none() {
        let table = ModelAliasTable::default();
        expect_that!(
            table.resolve("nonexistent", Some("chat")).is_none(),
            eq(true)
        );
    }

    #[gtest]
    fn default_table_is_empty() {
        let table = ModelAliasTable::default();
        expect_that!(table.aliases.is_empty(), eq(true));
    }
}
