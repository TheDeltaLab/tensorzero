// Modified by Delta-AI under Apache 2.0
use std::collections::HashMap;
use std::sync::Arc;

use futures::future::join_all;
use tokio::sync::Semaphore;

use crate::config::Config;
use crate::error::{Error, ErrorDetails};
use crate::stored_inference::{RenderedSample, StoredSample, render_stored_sample};

const DEFAULT_CONCURRENCY: usize = 100;

pub async fn render_samples<T: StoredSample>(
    config: Arc<Config>,
    stored_samples: Vec<T>,
    variants: HashMap<String, String>,
    concurrency: Option<usize>,
) -> Result<Vec<RenderedSample>, Error> {
    for (function_name, variant_name) in &variants {
        let function_config = config.get_function(function_name)?;
        function_config.variants().get(variant_name).ok_or_else(|| {
            crate::error::Error::new(crate::error::ErrorDetails::InvalidRequest {
                message: format!(
                    "Variant {variant_name} for function {function_name} not found.",
                ),
            })
        })?;
    }

    let concurrency = concurrency.unwrap_or(DEFAULT_CONCURRENCY);
    if concurrency == 0 {
        return Err(ErrorDetails::InvalidRequest {
            message: "concurrency must be at least 1".to_string(),
        }
        .into());
    }
    let semaphore = Arc::new(Semaphore::new(concurrency));

    // Process all samples concurrently with semaphore-limited concurrency.
    // For now, we drop the errors here.
    // They are logged on construction in the task.
    // TODO: make it configurable whether to drop or error on failures.
    let futures = stored_samples.into_iter().map(|sample| {
        let semaphore = semaphore.clone();
        let config = config.clone();
        let variants = variants.clone();
        async move {
            // Acquire semaphore permit for this sample's processing
            let _permit = semaphore.acquire().await.ok()?;

            // Resolve the input.
            // If the input has TTLed (is None), this sample will be skipped.
            let resolved_input = sample.input()?.clone().reresolve(&*config).await.ok()?;

            // Render the sample
            render_stored_sample(sample, resolved_input, &config, &variants)
                .await
                .ok()
        }
    });

    let final_rendered_examples: Vec<RenderedSample> =
        join_all(futures).await.into_iter().flatten().collect();

    Ok(final_rendered_examples)
}
