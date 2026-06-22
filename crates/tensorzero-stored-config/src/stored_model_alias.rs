// Modified by Delta-AI under Apache 2.0
use serde::{Deserialize, Serialize};

#[serde_with::skip_serializing_none]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StoredModelAliasTarget {
    pub provider: String,
    pub model: String,
}

#[serde_with::skip_serializing_none]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StoredModelAlias {
    pub name: String,
    pub task: Option<String>,
    pub targets: Vec<StoredModelAliasTarget>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use googletest::prelude::*;

    #[gtest]
    fn roundtrip_with_task() {
        let alias = StoredModelAlias {
            name: "fast-model".into(),
            task: Some("chat".into()),
            targets: vec![StoredModelAliasTarget {
                provider: "openai".into(),
                model: "gpt-4o-mini".into(),
            }],
        };
        let json = serde_json::to_string(&alias).expect("serialize");
        let back: StoredModelAlias = serde_json::from_str(&json).expect("deserialize");
        expect_that!(back.name, eq("fast-model"));
        expect_that!(back.task.as_deref(), eq(Some("chat")));
    }

    #[gtest]
    fn roundtrip_wildcard_task() {
        let alias = StoredModelAlias {
            name: "any-model".into(),
            task: None,
            targets: vec![],
        };
        let json = serde_json::to_string(&alias).expect("serialize");
        let back: StoredModelAlias = serde_json::from_str(&json).expect("deserialize");
        expect_that!(back.task.is_none(), eq(true));
    }
}
