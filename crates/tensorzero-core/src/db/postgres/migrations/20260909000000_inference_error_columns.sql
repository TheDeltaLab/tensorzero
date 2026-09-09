-- Modified by Delta-AI under Apache 2.0
-- Record failed inferences alongside successful ones.
--
-- Adds a nullable `error` column to the inference metadata tables. NULL means
-- the inference succeeded (or predates failure recording). The archive tables
-- get the same column appended last so the positional
-- `INSERT INTO ..._archive SELECT *` in the retention cleanup stays valid.
--
-- Failed rows have no model output, so `output` in the payload (data) tables
-- becomes nullable.

ALTER TABLE tensorzero.chat_inferences ADD COLUMN IF NOT EXISTS error TEXT;
ALTER TABLE tensorzero.json_inferences ADD COLUMN IF NOT EXISTS error TEXT;
ALTER TABLE tensorzero.model_inferences ADD COLUMN IF NOT EXISTS error TEXT;

ALTER TABLE tensorzero.chat_inferences_archive ADD COLUMN IF NOT EXISTS error TEXT;
ALTER TABLE tensorzero.json_inferences_archive ADD COLUMN IF NOT EXISTS error TEXT;

ALTER TABLE tensorzero.chat_inference_data ALTER COLUMN output DROP NOT NULL;
ALTER TABLE tensorzero.json_inference_data ALTER COLUMN output DROP NOT NULL;
ALTER TABLE tensorzero.model_inference_data ALTER COLUMN output DROP NOT NULL;

ALTER TABLE tensorzero.chat_inference_data_archive ALTER COLUMN output DROP NOT NULL;
ALTER TABLE tensorzero.json_inference_data_archive ALTER COLUMN output DROP NOT NULL;
