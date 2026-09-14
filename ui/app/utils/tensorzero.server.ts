import { getConfig } from "./config/index.server";
import { getTensorZeroClient } from "./get-tensorzero-client.server";
import {
  FeedbackRequestSchema,
  TensorZeroServerError,
} from "~/utils/tensorzero";
import type { JsonValue } from "~/types/tensorzero";
import { getFeedbackConfig } from "./config/feedback";

// Re-export for backcompat.
export { getTensorZeroClient };

export async function addHumanFeedback(formData: FormData) {
  const metricName = formData.get("metricName")?.toString();
  if (!metricName) {
    throw new TensorZeroServerError.InvalidMetricName(
      "Metric name is required",
    );
  }
  const config = await getConfig();
  const metricConfig = getFeedbackConfig(metricName, config);
  if (!metricConfig) {
    throw new TensorZeroServerError.UnknownMetric(
      `Metric ${metricName} not found`,
    );
  }
  const metricType = metricConfig.type;
  // Metrics can be of type boolean, float, comment, or demonstration.
  // In this case we need to handle the value differently depending on the metric type.
  const formValue = formData.get("value");
  if (!formValue || typeof formValue !== "string") {
    throw new TensorZeroServerError.InputValidation("Value is required");
  }
  let value: JsonValue;
  if (metricType === "boolean") {
    value = formValue === "true";
  } else if (metricType === "float") {
    value = parseFloat(formValue);
  } else if (metricType === "comment") {
    value = formValue;
  } else if (metricType === "demonstration") {
    value = JSON.parse(formValue);
  } else {
    throw new TensorZeroServerError.InputValidation(
      `Unsupported metric type: ${metricType}`,
    );
  }
  const episodeId = formData.get("episodeId");
  const inferenceId = formData.get("inferenceId");
  const tags: Record<string, string> = {
    "tensorzero::human_feedback": "true",
  };
  if ((episodeId && inferenceId) || (!episodeId && !inferenceId)) {
    throw new TensorZeroServerError.InputValidation(
      "Exactly one of episodeId and inferenceId should be provided",
    );
  }
  const feedbackRequest = FeedbackRequestSchema.safeParse({
    metric_name: metricName,
    value,
    episode_id: episodeId,
    inference_id: inferenceId,
    tags,
    internal: true,
  });
  if (!feedbackRequest.success) {
    throw new TensorZeroServerError.InputValidation(
      feedbackRequest.error.message,
    );
  }
  const response = await getTensorZeroClient().feedback(feedbackRequest.data);
  return response;
}
