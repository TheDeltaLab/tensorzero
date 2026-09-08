// Modified by Delta-AI under Apache 2.0
// `JSONSchema` moved to `tensorzero-providers`; re-exported here for existing
// callers. The `ResolvedTomlPathData`-based constructor stays in core because
// it depends on core's config path machinery (Delta-AI fork).
pub use tensorzero_providers::jsonschema_util::{JSONSchema, SchemaWithMetadata};

use crate::config::path::ResolvedTomlPathData;
use tensorzero_error::Error;

/// Creates a JSONSchema from a resolved config path.
///
/// Parses the JSON and compiles the schema synchronously.
pub fn from_path(path: ResolvedTomlPathData) -> Result<JSONSchema, Error> {
    JSONSchema::from_str_with_key(&path.get_template_key(), path.data())
}
