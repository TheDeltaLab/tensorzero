// Modified by Delta-AI under Apache 2.0
import type {
  LanguageModelV4Content,
  LanguageModelV4FinishReason,
  LanguageModelV4GenerateResult,
  LanguageModelV4Usage,
} from "@ai-sdk/provider";

interface ChatCompletionMessage {
  content?: string | null;
  reasoning_content?: string | null;
  tool_calls?: Array<{
    id?: string;
    type?: string;
    function?: { name?: string; arguments?: string };
  }>;
}

interface ChatCompletionBody {
  id?: string;
  created?: number;
  model?: string;
  choices?: Array<{
    finish_reason?: string | null;
    message?: ChatCompletionMessage;
  }>;
  usage?: {
    prompt_tokens?: number;
    completion_tokens?: number;
    prompt_tokens_details?: { cached_tokens?: number };
    completion_tokens_details?: { reasoning_tokens?: number };
  };
}

/**
 * Convert an OpenAI chat completions response body (the `response` payload of
 * a completed async task submitted via `POST /v1/chat/completions/async`) into
 * a `LanguageModelV4GenerateResult`.
 */
export function chatCompletionToGenerateResult(
  body: unknown,
): LanguageModelV4GenerateResult {
  const completion = body as ChatCompletionBody;
  const choice = completion.choices?.[0];
  const message = choice?.message ?? {};

  const content: LanguageModelV4Content[] = [];
  if (typeof message.reasoning_content === "string" && message.reasoning_content) {
    content.push({ type: "reasoning", text: message.reasoning_content });
  }
  if (typeof message.content === "string" && message.content) {
    content.push({ type: "text", text: message.content });
  }
  for (const toolCall of message.tool_calls ?? []) {
    content.push({
      type: "tool-call",
      toolCallId: toolCall.id ?? "",
      toolName: toolCall.function?.name ?? "",
      input: toolCall.function?.arguments ?? "",
    });
  }

  return {
    content,
    finishReason: mapChatFinishReason(choice?.finish_reason),
    usage: mapChatUsage(completion.usage),
    response: {
      id: completion.id,
      timestamp:
        typeof completion.created === "number"
          ? new Date(completion.created * 1000)
          : undefined,
      modelId: completion.model,
      body,
    },
    warnings: [],
  };
}

function mapChatFinishReason(raw: string | null | undefined): LanguageModelV4FinishReason {
  const unified = (() => {
    switch (raw) {
      case "stop":
        return "stop" as const;
      case "length":
        return "length" as const;
      case "content_filter":
        return "content-filter" as const;
      case "tool_calls":
      case "function_call":
        return "tool-calls" as const;
      default:
        return "other" as const;
    }
  })();
  return { unified, raw: raw ?? undefined };
}

function mapChatUsage(usage: ChatCompletionBody["usage"]): LanguageModelV4Usage {
  return {
    inputTokens: {
      total: usage?.prompt_tokens,
      noCache:
        usage?.prompt_tokens !== undefined &&
        usage.prompt_tokens_details?.cached_tokens !== undefined
          ? usage.prompt_tokens - usage.prompt_tokens_details.cached_tokens
          : undefined,
      cacheRead: usage?.prompt_tokens_details?.cached_tokens,
      cacheWrite: undefined,
    },
    outputTokens: {
      total: usage?.completion_tokens,
      text:
        usage?.completion_tokens !== undefined &&
        usage.completion_tokens_details?.reasoning_tokens !== undefined
          ? usage.completion_tokens - usage.completion_tokens_details.reasoning_tokens
          : usage?.completion_tokens,
      reasoning: usage?.completion_tokens_details?.reasoning_tokens,
    },
    raw: usage as LanguageModelV4Usage["raw"],
  };
}

interface ResponsesApiBody {
  id?: string;
  created_at?: number;
  model?: string;
  status?: string;
  incomplete_details?: { reason?: string } | null;
  output?: Array<{
    type?: string;
    role?: string;
    status?: string;
    content?: Array<{
      type?: string;
      text?: string;
      annotations?: unknown[];
    }>;
    summary?: Array<{ type?: string; text?: string }>;
    call_id?: string;
    name?: string;
    arguments?: string;
  }>;
  usage?: {
    input_tokens?: number;
    output_tokens?: number;
    total_tokens?: number;
    input_tokens_details?: { cached_tokens?: number };
    output_tokens_details?: { reasoning_tokens?: number };
  };
}

/**
 * Convert an OpenAI Responses API response body (the `response` payload of a
 * completed async task submitted via `POST /v1/responses/async`) into a
 * `LanguageModelV4GenerateResult`.
 */
export function responsesApiToGenerateResult(
  body: unknown,
): LanguageModelV4GenerateResult {
  const response = body as ResponsesApiBody;

  const content: LanguageModelV4Content[] = [];
  for (const item of response.output ?? []) {
    if (item.type === "reasoning") {
      const text = (item.summary ?? [])
        .map((part) => part.text ?? "")
        .join("");
      if (text) content.push({ type: "reasoning", text });
      continue;
    }
    if (item.type === "message") {
      for (const part of item.content ?? []) {
        if (part.type === "output_text" && part.text) {
          content.push({ type: "text", text: part.text });
        }
      }
      continue;
    }
    if (item.type === "function_call") {
      content.push({
        type: "tool-call",
        toolCallId: item.call_id ?? "",
        toolName: item.name ?? "",
        input: item.arguments ?? "",
      });
    }
  }

  return {
    content,
    finishReason: mapResponsesFinishReason(response),
    usage: mapResponsesUsage(response.usage),
    response: {
      id: response.id,
      timestamp:
        typeof response.created_at === "number"
          ? new Date(response.created_at * 1000)
          : undefined,
      modelId: response.model,
      body,
    },
    warnings: [],
  };
}

function mapResponsesFinishReason(body: ResponsesApiBody): LanguageModelV4FinishReason {
  if (body.status === "incomplete") {
    const reason = body.incomplete_details?.reason;
    const unified =
      reason === "max_output_tokens"
        ? ("length" as const)
        : reason === "content_filter"
          ? ("content-filter" as const)
          : ("other" as const);
    return { unified, raw: reason };
  }
  if (body.status === "failed") {
    return { unified: "error", raw: body.incomplete_details?.reason ?? "failed" };
  }
  const hasToolCall = body.output?.some((item) => item.type === "function_call");
  if (hasToolCall) {
    return { unified: "tool-calls", raw: "tool_calls" };
  }
  return { unified: "stop", raw: "stop" };
}

function mapResponsesUsage(usage: ResponsesApiBody["usage"]): LanguageModelV4Usage {
  return {
    inputTokens: {
      total: usage?.input_tokens,
      noCache:
        usage?.input_tokens !== undefined &&
        usage.input_tokens_details?.cached_tokens !== undefined
          ? usage.input_tokens - usage.input_tokens_details.cached_tokens
          : undefined,
      cacheRead: usage?.input_tokens_details?.cached_tokens,
      cacheWrite: undefined,
    },
    outputTokens: {
      total: usage?.output_tokens,
      text:
        usage?.output_tokens !== undefined &&
        usage.output_tokens_details?.reasoning_tokens !== undefined
          ? usage.output_tokens - usage.output_tokens_details.reasoning_tokens
          : usage?.output_tokens,
      reasoning: usage?.output_tokens_details?.reasoning_tokens,
    },
    raw: usage as LanguageModelV4Usage["raw"],
  };
}
