-- Modified by Delta-AI under Apache 2.0
-- ISO 4217 code for model_inferences.cost. NULL means the inference predates
-- currency tracking or cost was not configured.
ALTER TABLE tensorzero.model_inferences ADD COLUMN IF NOT EXISTS currency TEXT;
