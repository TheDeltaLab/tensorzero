// Modified by Delta-AI under Apache 2.0
//! Type definitions for OpenAI-compatible API.
//!
//! This module contains all the type definitions used by the OpenAI-compatible endpoints,
//! organized into submodules for chat completions, embeddings, input files, streaming,
//! tools, and usage tracking.

pub mod chat_completions;
pub mod completions;
pub mod embeddings;
pub mod input_files;
pub mod responses;
pub mod streaming;
pub mod tool;
pub mod usage;
