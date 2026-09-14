// Modified by Delta-AI under Apache 2.0
import * as React from "react";
import { useFetcher, type FetcherFormProps } from "react-router";
import type { SubmitTarget, FetcherSubmitOptions } from "react-router";
import { DEFAULT_FUNCTION } from "~/utils/constants";
import type { StoredInference } from "~/types/tensorzero";
import type { InferenceUsage } from "~/utils/clickhouse/helpers";
import { logger } from "~/utils/logger";
import type {
  ClientInferenceParams,
  Input,
  ContentBlockChatOutput,
  JsonInferenceOutput,
  InferenceResponse,
} from "~/types/tensorzero";

interface InferenceActionError {
  message: string;
  caught: unknown;
}

type InferenceActionResponse = {
  info?: VariantResponseInfo;
  raw: InferenceResponse;
};

type InferenceActionContext =
  | { state: "init"; data: null; error: null }
  | { state: "idle"; data: InferenceActionResponse | null; error: null }
  | {
      state: "submitting";
      data: InferenceActionResponse | null;
      error: null | InferenceActionError;
    }
  | {
      state: "loading";
      data: InferenceActionResponse | null;
      error: null | InferenceActionError;
    }
  | {
      state: "error";
      data: (Pick<InferenceActionResponse, "raw"> & { info?: never }) | null;
      error: InferenceActionError;
    };

const ENDPOINT = "/api/tensorzero/inference";

type ActionFetcher = InferenceActionContext & {
  Form: React.FC<Omit<FetcherFormProps, "method" | "encType" | "action">>;
  submit(
    target: SubmitTarget,
    opts?: Omit<FetcherSubmitOptions, "method" | "encType" | "action">,
  ): Promise<void>;
};

/**
 * A wrapper around the `useFetcher` hook to handle POST requests to the
 * inference endpoint.
 */
export function useInferenceActionFetcher() {
  const fetcher = useFetcher<InferenceResponse>();
  /**
   * The fetcher's state gives us the current status of the request alongside
   * its data, but it does so in a generic interface that we still need to parse
   * and interpret for rendering related UI. Because this fetcher is only be
   * used for submitting POST requests to the given endpoint, we can safely add
   * types and provide more specific context based on the shape of our data.
   * This also gives our fetcher two additional states:
   *  - `init`: before a request is actually made
   *  - `error`: when the request fails, or when a successful response contains
   *    an error instead of inference data
   *
   * All of this is derived from the fetcher's state and data so that we can
   * avoid managing any state or synchronization via effects internally.
   */
  const context = React.useMemo<InferenceActionContext>(() => {
    const inferenceOutput = fetcher.data;
    if (inferenceOutput) {
      try {
        // Check if the response contains an error
        if ("error" in inferenceOutput && inferenceOutput.error) {
          return {
            state: fetcher.state === "idle" ? "error" : fetcher.state,
            data: { raw: inferenceOutput },
            error: {
              caught: inferenceOutput.error,
              message: `Inference Failed: ${typeof inferenceOutput.error === "string" ? inferenceOutput.error : JSON.stringify(inferenceOutput.error)}`,
            },
          } satisfies InferenceActionContext;
        }

        return {
          state: fetcher.state,
          data: {
            raw: inferenceOutput,
            info:
              "content" in inferenceOutput
                ? {
                    type: "chat" as const,
                    output: inferenceOutput.content,
                    usage: inferenceOutput.usage,
                  }
                : {
                    type: "json" as const,
                    output: inferenceOutput.output,
                    usage: inferenceOutput.usage,
                  },
          },
          error: null,
        } satisfies InferenceActionContext;
      } catch (error) {
        return {
          state: "error",
          data: { raw: inferenceOutput },
          error: { message: "Failed to process response data", caught: error },
        } satisfies InferenceActionContext;
      }
    } else if (fetcher.state === "idle") {
      return {
        state: "init",
        data: null,
        error: null,
      } satisfies InferenceActionContext;
    } else {
      return {
        state: fetcher.state,
        data: null,
        error: null,
      } satisfies InferenceActionContext;
    }
  }, [fetcher.state, fetcher.data]);

  const submit = React.useCallback<ActionFetcher["submit"]>(
    (target, opts) => {
      const submit = fetcher.submit;
      return submit(target, {
        ...opts,
        method: "POST",
        action: ENDPOINT,
      });
    },
    [fetcher.submit],
  );

  const Form = React.useMemo<ActionFetcher["Form"]>(
    () => (props) => {
      const Form = fetcher.Form;
      return <Form {...props} method="POST" action={ENDPOINT} />;
    },
    [fetcher.Form],
  );

  React.useEffect(() => {
    if (context.error?.caught) {
      logger.error("Error processing response:", context.error.caught);
    }
  }, [context.error]);

  return {
    ...context,
    Form,
    submit,
  } satisfies ActionFetcher;
}

interface InferenceActionArgs {
  source: "inference";
  resource: StoredInference;
  input: Input;
  variant: string;
}

interface InferenceDefaultFunctionActionArgs {
  source: "inference";
  resource: StoredInference;
  input: Input;
  variant?: undefined;
  model_name: string;
}

type ActionArgs = InferenceActionArgs | InferenceDefaultFunctionActionArgs;

function isDefaultFunctionArgs(
  args: ActionArgs,
): args is InferenceDefaultFunctionActionArgs {
  return (
    args.source === "inference" &&
    args.resource.function_name === DEFAULT_FUNCTION
  );
}

export function prepareInferenceActionRequest(
  args: ActionArgs,
): ClientInferenceParams {
  // Create base ClientInferenceParams with default values
  const baseParams: ClientInferenceParams = {
    input: { system: undefined, messages: [] },
    params: {
      chat_completion: {
        temperature: null,
        max_tokens: null,
        seed: null,
        top_p: null,
        presence_penalty: null,
        frequency_penalty: null,
        json_mode: null,
        stop_sequences: null,
      },
    },
    provider_tools: [],
    internal: true,
    tags: {
      "tensorzero::ui": "true",
    },
    output_schema: null,
    credentials: new Map(),
    cache_options: {
      max_age_s: null,
      enabled: "off",
    },
    include_original_response: false, // deprecated
    include_raw_response: false,
    include_raw_usage: false,
    include_aggregated_response: false,
  };

  // Prepare request based on source and function type
  if (isDefaultFunctionArgs(args)) {
    const defaultRequest = prepareDefaultFunctionRequest(
      args.resource,
      args.input,
      args.model_name,
    );
    return { ...baseParams, ...defaultRequest };
  }

  if (
    args.source === "inference" &&
    args.resource.extra_body &&
    args.resource.extra_body.length > 0
  ) {
    throw new Error("Extra body is not supported for inference in UI.");
  }

  return {
    ...baseParams,
    function_name: args.resource.function_name,
    input: args.input,
    variant_name: args.variant,
  };
}

function prepareDefaultFunctionRequest(
  inference: StoredInference,
  input: Input,
  selectedVariant: string,
): Partial<ClientInferenceParams> {
  if (inference.type === "chat") {
    const tool_choice = inference.tool_choice;
    const parallel_tool_calls = inference.parallel_tool_calls;
    const allowed_tools = inference.allowed_tools;
    return {
      model_name: selectedVariant,
      input,
      tool_choice: tool_choice,
      parallel_tool_calls: parallel_tool_calls,
      allowed_tools,
      // We need to add all tools as additional for the default function
      additional_tools: inference.additional_tools,
    };
  } else if (inference.type === "json") {
    // This should never happen, just in case and for type safety
    const output_schema = inference.output_schema;
    return {
      model_name: selectedVariant,
      input,
      output_schema: output_schema || null,
    };
  }

  // Fallback case
  return {
    model_name: selectedVariant,
    input,
  };
}

export type VariantResponseInfo =
  | {
      type: "chat";
      output?: ContentBlockChatOutput[];
      usage?: InferenceUsage;
    }
  | {
      type: "json";
      output?: JsonInferenceOutput;
      usage?: InferenceUsage;
    };

/**
 * Extracts the demonstration value from inference output.
 * For JSON inferences, returns the parsed output.
 * For chat inferences, returns the raw output array.
 */
export function extractDemonstrationValue(
  output: ContentBlockChatOutput[] | JsonInferenceOutput,
) {
  // JSON output has 'parsed' property, chat output is an array
  if ("parsed" in output) {
    return output.parsed;
  }
  return output;
}

/**
 * Prepares demonstration feedback value from variant output.
 * Returns the parsed output for JSON inferences, or the raw output for chat inferences.
 */
export function prepareDemonstrationFromVariantOutput(
  variantOutput: VariantResponseInfo,
) {
  const output = variantOutput.output;
  if (output === undefined) {
    return undefined;
  }
  return extractDemonstrationValue(output);
}
