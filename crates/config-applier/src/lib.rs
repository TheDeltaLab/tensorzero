// Modified by Delta-AI under Apache 2.0
//! TensorZero Config Applier
//!
//! A crate for applying targeted edits to TensorZero config TOML files while preserving formatting.
//! Supports 4 operations: upsert variant, upsert experimentation config, upsert evaluation, upsert evaluator.

mod edit;
mod error;
mod locator;
mod path_resolver;
mod toml_writer;

pub use edit::{EditPayload, UpsertVariantPayload};
pub use error::ConfigApplierError;

use std::path::{Path, PathBuf};

use locator::LoadedConfigFile;
use path_resolver::FileToWrite;
use tensorzero_core::config::{ConfigFileGlob, UninitializedVariantConfig};
use tensorzero_core::utils::retries::RetryConfig;
use toml_edit::DocumentMut;

/// Validate that an evaluator config won't produce undeserializable TOML after cleaning.
///
/// `strip_empty_tables` removes empty inline tables like `variants = {}`, but some evaluator
/// types (e.g. `llm_judge`) require `variants` to be present. We reject these early so we
/// never write invalid config.
/// Convert subtables to inline tables, strip the given keys, and remove empty tables.
///
/// This must be called after `extract_resolved_paths` (which needs regular tables)
/// and before `upsert_*` (which inserts into the document).
fn clean_serialized_item(item: &mut toml_edit::Item, keys_to_strip: &[&str]) {
    let Some(table) = item.as_table_mut() else {
        return;
    };

    toml_writer::convert_subtables_to_inline(table);
    toml_writer::strip_keys(table, keys_to_strip);
    toml_writer::strip_empty_tables(table);
}

/// ConfigApplier handles applying edits to TensorZero config files.
pub struct ConfigApplier {
    /// The base directory extracted from the glob pattern
    glob_base: PathBuf,
    /// The loaded config files with their parsed TOML documents
    files: Vec<LoadedConfigFile>,
}

impl ConfigApplier {
    /// Create a new ConfigApplier by loading all config files matching the glob pattern.
    pub async fn new(glob_pattern: &str) -> Result<Self, ConfigApplierError> {
        let config_glob = ConfigFileGlob::new(glob_pattern.to_string()).map_err(|e| {
            ConfigApplierError::InvalidGlob {
                pattern: glob_pattern.to_string(),
                message: e.to_string(),
            }
        })?;

        let mut glob_base = config_glob.base_path();
        if glob_base.is_file() {
            glob_base = glob_base
                .parent()
                .unwrap_or_else(|| Path::new("."))
                .to_path_buf();
        }

        // Load all config files using toml_edit for format preservation
        let mut files = Vec::new();
        for path in &config_glob.paths {
            let content = tokio::fs::read_to_string(path)
                .await
                .map_err(|e| ConfigApplierError::io(path, e))?;

            let document: DocumentMut = content
                .parse()
                .map_err(|e: toml_edit::TomlError| ConfigApplierError::toml_parse(path, e))?;

            files.push(LoadedConfigFile::new(path.clone(), document));
        }

        Ok(Self { glob_base, files })
    }

    /// Apply an edit to the config files.
    /// Returns the paths of all files that were written (TOML config + any template/schema files).
    pub async fn apply_edit(
        &mut self,
        edit: &EditPayload,
    ) -> Result<Vec<PathBuf>, ConfigApplierError> {
        let files_to_write = match edit {
            EditPayload::UpsertVariant(payload) => self.apply_upsert_variant(payload)?,
        };

        // Write all files (TOML and any extracted template/schema files)
        let mut written_paths = Vec::with_capacity(files_to_write.len());
        for file_to_write in files_to_write {
            // Create parent directories if needed
            if let Some(parent) = file_to_write.absolute_path.parent() {
                tokio::fs::create_dir_all(parent)
                    .await
                    .map_err(|e| ConfigApplierError::io(&file_to_write.absolute_path, e))?;
            }

            tokio::fs::write(&file_to_write.absolute_path, &file_to_write.content)
                .await
                .map_err(|e| ConfigApplierError::io(&file_to_write.absolute_path, e))?;

            written_paths.push(file_to_write.absolute_path);
        }

        Ok(written_paths)
    }

    fn apply_upsert_variant(
        &mut self,
        payload: &UpsertVariantPayload,
    ) -> Result<Vec<FileToWrite>, ConfigApplierError> {
        path_resolver::validate_path_component(&payload.function_name, "function_name")?;
        path_resolver::validate_path_component(&payload.variant_name, "variant_name")?;

        let location = locator::locate_function(&mut self.files, &payload.function_name)?;
        let toml_file_dir = location
            .file
            .path
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .to_path_buf();

        // Serialize the variant to a TOML item
        let mut variant_item = toml_writer::serialize_to_item(&payload.variant)?;

        // Extract any ResolvedTomlPathData fields and convert to relative paths
        let template_files = path_resolver::extract_resolved_paths(
            &mut variant_item,
            &self.glob_base,
            &toml_file_dir,
            &[
                "functions",
                &payload.function_name,
                "variants",
                &payload.variant_name,
            ],
        )?;

        // Determine which keys have default values and should be stripped
        let mut keys_to_strip = Vec::new();
        let retries = match &payload.variant.inner {
            UninitializedVariantConfig::ChatCompletion(c) => Some(c.retries),
            UninitializedVariantConfig::Dicl(c) => Some(c.retries),
            UninitializedVariantConfig::ChainOfThought(c) => Some(c.inner.retries),
            UninitializedVariantConfig::BestOfNSampling(_)
            | UninitializedVariantConfig::MixtureOfN(_) => None,
        };
        if retries == Some(RetryConfig::default()) {
            keys_to_strip.push("retries");
        }

        // Convert subtables to inline, strip defaults, and remove empty tables
        clean_serialized_item(&mut variant_item, &keys_to_strip);

        // Apply the edit to the document
        toml_writer::upsert_variant(
            &mut location.file.document,
            &payload.function_name,
            &payload.variant_name,
            variant_item,
        )?;

        // Prepare files to write
        let mut files = template_files;
        files.push(FileToWrite {
            absolute_path: location.file.path.clone(),
            content: location.file.document.to_string(),
        });

        Ok(files)
    }

    /// Get the base directory extracted from the glob pattern.
    pub fn glob_base(&self) -> &Path {
        &self.glob_base
    }

    /// Get the paths of all loaded config files.
    pub fn config_paths(&self) -> Vec<&Path> {
        self.files.iter().map(|f| f.path.as_path()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::fs;
    use std::sync::Arc;
    use tempfile::TempDir;

    use tensorzero_core::config::{UninitializedVariantConfig, UninitializedVariantInfo};
    use tensorzero_core::utils::retries::RetryConfig;

    use tensorzero_core::variant::chat_completion::UninitializedChatCompletionConfig;

    fn setup_test_config(dir: &Path) {
        fs::write(
            dir.join("tensorzero.toml"),
            r#"[functions.my_function]
type = "chat"

[functions.my_function.variants.baseline]
type = "chat_completion"
model = "gpt-4"
"#,
        )
        .expect("failed to write test config");
    }

    #[tokio::test]
    async fn test_config_writer_new() {
        let tmp = TempDir::new().expect("failed to create temp dir");
        setup_test_config(tmp.path());

        let glob = format!("{}/**/*.toml", tmp.path().display());
        let writer = ConfigApplier::new(&glob)
            .await
            .expect("failed to create writer");

        assert_eq!(writer.config_paths().len(), 1);
    }

    #[tokio::test]
    async fn test_config_writer_no_files() {
        let tmp = TempDir::new().expect("failed to create temp dir");

        let glob = format!("{}/**/*.toml", tmp.path().display());
        let result = ConfigApplier::new(&glob).await;

        assert!(result.is_err());
        if let Err(ConfigApplierError::InvalidGlob { pattern, .. }) = result {
            assert_eq!(pattern, glob);
        } else {
            panic!("expected InvalidGlob error");
        }
    }

    #[tokio::test]
    async fn test_locate_function() {
        let tmp = TempDir::new().expect("failed to create temp dir");
        setup_test_config(tmp.path());

        let glob = format!("{}/**/*.toml", tmp.path().display());
        let mut writer = ConfigApplier::new(&glob)
            .await
            .expect("failed to create writer");

        // Test that we can find an existing function
        let location = locator::locate_function(&mut writer.files, "my_function");
        assert!(location.is_ok());

        // Test that we get an error for a non-existent function
        let location = locator::locate_function(&mut writer.files, "nonexistent");
        assert!(location.is_err());
    }

    #[tokio::test]
    async fn test_config_writer_base_path_for_single_file_glob() {
        let tmp = TempDir::new().expect("failed to create temp dir");
        setup_test_config(tmp.path());

        let config_path = tmp.path().join("tensorzero.toml");
        let glob = config_path.display().to_string();
        let writer = ConfigApplier::new(&glob)
            .await
            .expect("failed to create writer");

        assert_eq!(
            writer.glob_base(),
            tmp.path(),
            "expected glob_base to be the parent dir for a single-file path"
        );
        assert!(
            writer.glob_base().is_dir(),
            "expected glob_base to be a directory path"
        );
    }

    #[tokio::test]
    async fn test_upsert_variant_inline_tables_and_default_stripping() {
        let tmp = TempDir::new().expect("failed to create temp dir");
        setup_test_config(tmp.path());

        let glob = format!("{}/tensorzero.toml", tmp.path().display());
        let mut writer = ConfigApplier::new(&glob)
            .await
            .expect("failed to create writer");

        // Create a variant with default retries (should be stripped)
        let variant_default_retries = UninitializedVariantInfo {
            inner: UninitializedVariantConfig::ChatCompletion(UninitializedChatCompletionConfig {
                model: Arc::from("gpt-4o"),
                retries: RetryConfig::default(),
                ..Default::default()
            }),
            timeouts: None,
            namespace: None,
        };

        let edit = EditPayload::UpsertVariant(Box::new(UpsertVariantPayload {
            function_name: "my_function".to_string(),
            variant_name: "default_retries".to_string(),
            variant: variant_default_retries,
        }));

        writer
            .apply_edit(&edit)
            .await
            .expect("failed to apply variant edit");

        let toml_contents =
            fs::read_to_string(tmp.path().join("tensorzero.toml")).expect("failed to read config");

        // Default retries should NOT appear in output (check for the key, not the variant name)
        assert!(
            !toml_contents.contains("num_retries"),
            "default retries should be stripped from output, got:\n{toml_contents}"
        );
        assert!(
            !toml_contents.contains("max_delay_s"),
            "default retries should be stripped from output, got:\n{toml_contents}"
        );
    }

    #[tokio::test]
    async fn test_upsert_variant_non_default_retries_preserved() {
        let tmp = TempDir::new().expect("failed to create temp dir");
        setup_test_config(tmp.path());

        let glob = format!("{}/tensorzero.toml", tmp.path().display());
        let mut writer = ConfigApplier::new(&glob)
            .await
            .expect("failed to create writer");

        // Create a variant with non-default retries (should be preserved)
        let variant_custom_retries = UninitializedVariantInfo {
            inner: UninitializedVariantConfig::ChatCompletion(UninitializedChatCompletionConfig {
                model: Arc::from("gpt-4o"),
                retries: RetryConfig {
                    num_retries: 3,
                    max_delay_s: 5.0,
                },
                ..Default::default()
            }),
            timeouts: None,
            namespace: None,
        };

        let edit = EditPayload::UpsertVariant(Box::new(UpsertVariantPayload {
            function_name: "my_function".to_string(),
            variant_name: "custom_retries".to_string(),
            variant: variant_custom_retries,
        }));

        writer
            .apply_edit(&edit)
            .await
            .expect("failed to apply variant edit");

        let toml_contents =
            fs::read_to_string(tmp.path().join("tensorzero.toml")).expect("failed to read config");

        // Non-default retries SHOULD appear as inline table
        assert!(
            toml_contents.contains("retries"),
            "non-default retries should be preserved in output, got:\n{toml_contents}"
        );
        assert!(
            toml_contents.contains("num_retries = 3"),
            "expected num_retries = 3 in output, got:\n{toml_contents}"
        );

        // Retries should be an inline table, not a separate section
        assert!(
            !toml_contents.contains("[functions.my_function.variants.custom_retries.retries]"),
            "retries should be inline table, not a separate section, got:\n{toml_contents}"
        );
    }

    #[tokio::test]
    async fn test_upsert_variant_empty_timeouts_stripped() {
        let tmp = TempDir::new().expect("failed to create temp dir");
        setup_test_config(tmp.path());

        let glob = format!("{}/tensorzero.toml", tmp.path().display());
        let mut writer = ConfigApplier::new(&glob)
            .await
            .expect("failed to create writer");

        // Create a variant with default (empty) timeouts
        let variant = UninitializedVariantInfo {
            inner: UninitializedVariantConfig::ChatCompletion(UninitializedChatCompletionConfig {
                model: Arc::from("gpt-4o"),
                ..Default::default()
            }),
            timeouts: Some(Default::default()),
            namespace: None,
        };

        let edit = EditPayload::UpsertVariant(Box::new(UpsertVariantPayload {
            function_name: "my_function".to_string(),
            variant_name: "empty_timeouts".to_string(),
            variant,
        }));

        writer
            .apply_edit(&edit)
            .await
            .expect("failed to apply variant edit");

        let toml_contents =
            fs::read_to_string(tmp.path().join("tensorzero.toml")).expect("failed to read config");

        // Empty timeouts should be stripped (timeouts = {} or timeouts with empty sub-tables)
        // Check specifically for the timeouts key assignment, not the variant name
        assert!(
            !toml_contents.contains("timeouts ="),
            "empty timeouts should be stripped from output, got:\n{toml_contents}"
        );
    }

    #[tokio::test]
    async fn noop_placeholder() {}
}
