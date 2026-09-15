// Modified by Delta-AI under Apache 2.0
import {
  type RouteConfig,
  index,
  prefix,
  route,
} from "@react-router/dev/routes";

export default [
  index("routes/index.tsx"),

  // API routes
  ...prefix("api", [
    route("auth/set_gateway_key", "routes/api/auth/set_gateway_key.route.ts"),

    route(
      "function/:function_name/feedback_counts",
      "routes/api/function/$function_name/feedback_counts.route.ts",
    ),

    ...prefix("tensorzero", [
      route("inference", "routes/api/tensorzero/inference.ts"),
      route("status", "routes/api/tensorzero/status.ts"),
      route(
        "resolve_uuid/:uuid",
        "routes/api/tensorzero/resolve_uuid.route.ts",
      ),
      route(
        "inference_preview/:inference_id",
        "routes/api/tensorzero/inference_preview.route.ts",
      ),
      route(
        "episode_preview/:episode_id",
        "routes/api/tensorzero/episode_preview.route.ts",
      ),
    ]),

    route(
      "inference/:inference_id",
      "routes/api/inference/$inference_id/route.ts",
    ),

    route(
      "inference/:inference_id/protection",
      "routes/api/inference/$inference_id/protection/route.ts",
    ),

    route("feedback", "routes/api/feedback/route.ts"),
  ]),

  // Playground
  route("playground", "routes/playground/route.tsx"),
  route("playground/embeddings", "routes/playground/embeddings.tsx"),
  route("playground/rerank", "routes/playground/rerank.tsx"),

  // Observability
  ...prefix("observability", [
    route("functions", "routes/observability/functions/layout.tsx", [
      index("routes/observability/functions/route.tsx"),
      route(
        ":function_name",
        "routes/observability/functions/$function_name/layout.tsx",
        [
          index("routes/observability/functions/$function_name/route.tsx"),
          route(
            "variants/:variant_name",
            "routes/observability/functions/$function_name/variants/route.tsx",
          ),
        ],
      ),
    ]),
    route("inferences", "routes/observability/inferences/layout.tsx", [
      index("routes/observability/inferences/route.tsx"),
      route(
        ":inference_id",
        "routes/observability/inferences/$inference_id/route.tsx",
      ),
    ]),
    route("episodes", "routes/observability/episodes/layout.tsx", [
      index("routes/observability/episodes/route.tsx"),
      route(
        ":episode_id",
        "routes/observability/episodes/$episode_id/route.tsx",
      ),
    ]),

    route("models", "routes/observability/models/route.tsx"),
    route("analysis", "routes/observability/analysis/route.tsx"),
  ]),

  // API Keys
  route("api-keys", "routes/api-keys/route.tsx"),

  // Inference storage management
  route("storage", "routes/storage/route.tsx"),


  // Async Tasks (Delta-AI fork: restored after the #60 strip)
  route("async-tasks", "routes/async-tasks/route.tsx"),
  route("async-tasks/:taskId", "routes/async-tasks/$taskId/route.tsx"),
  // Dashboard users (Azure allowlist)
  route("users", "routes/users/route.tsx"),

  // Config editor
  route("config", "routes/config/route.tsx"),

  // Health
  route("health", "routes/health/route.tsx"),
] satisfies RouteConfig;
