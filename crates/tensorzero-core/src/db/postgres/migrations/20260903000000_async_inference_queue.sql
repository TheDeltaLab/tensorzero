-- Modified by Delta-AI under Apache 2.0
-- Durable queue for the async inference API (`POST .../async` submit endpoints)
SELECT durable.create_queue('async_inference');
