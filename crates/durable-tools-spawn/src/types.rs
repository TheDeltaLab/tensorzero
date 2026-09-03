// Modified by Delta-AI under Apache 2.0
//! Types for task polling.

use std::fmt;
use std::str::FromStr;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;

/// Status of a durable task.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskStatus {
    Pending,
    Running,
    Sleeping,
    Completed,
    Failed,
    Cancelled,
}

/// Error returned when parsing an invalid task status string.
#[derive(Debug, Clone)]
pub struct InvalidTaskStatus(pub String);

impl fmt::Display for InvalidTaskStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "invalid task status: {}", self.0)
    }
}

impl std::error::Error for InvalidTaskStatus {}

impl FromStr for TaskStatus {
    type Err = InvalidTaskStatus;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "pending" => Ok(Self::Pending),
            "running" => Ok(Self::Running),
            "sleeping" => Ok(Self::Sleeping),
            "completed" => Ok(Self::Completed),
            "failed" => Ok(Self::Failed),
            "cancelled" => Ok(Self::Cancelled),
            _ => Err(InvalidTaskStatus(s.to_string())),
        }
    }
}

impl TaskStatus {
    /// Returns true if the task is in a terminal state.
    pub fn is_terminal(self) -> bool {
        matches!(self, Self::Completed | Self::Failed | Self::Cancelled)
    }
}

/// Result of polling a durable task.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskPollResult {
    /// Current status of the task.
    pub status: TaskStatus,
    /// The task's output payload (only present when status is `Completed`).
    pub result: Option<JsonValue>,
    /// Error information from the last failed run (only present when status is `Failed`).
    pub error: Option<JsonValue>,
    /// The task's input params (from the `params` column).
    pub params: Option<JsonValue>,
}

/// Timing information for a durable task, read from the task table.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct TaskTiming {
    /// When the task was enqueued.
    pub enqueue_at: DateTime<Utc>,
    /// When the task first started running (`None` if it has never been claimed).
    pub first_started_at: Option<DateTime<Utc>>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use googletest::prelude::*;

    #[gtest]
    fn task_status_parses_all_states() {
        for (raw, expected) in [
            ("pending", TaskStatus::Pending),
            ("running", TaskStatus::Running),
            ("sleeping", TaskStatus::Sleeping),
            ("completed", TaskStatus::Completed),
            ("failed", TaskStatus::Failed),
            ("cancelled", TaskStatus::Cancelled),
        ] {
            expect_that!(raw.parse::<TaskStatus>(), ok(eq(&expected)));
        }
        expect_that!("unknown".parse::<TaskStatus>(), err(anything()));
    }

    #[gtest]
    fn task_status_is_terminal() {
        for status in [
            TaskStatus::Pending,
            TaskStatus::Running,
            TaskStatus::Sleeping,
        ] {
            expect_that!(status.is_terminal(), eq(false));
        }
        for status in [
            TaskStatus::Completed,
            TaskStatus::Failed,
            TaskStatus::Cancelled,
        ] {
            expect_that!(status.is_terminal(), eq(true));
        }
    }
}
