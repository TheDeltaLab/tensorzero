// Modified by Delta-AI under Apache 2.0
import type {
  LanguageModelV4Content,
  LanguageModelV4FinishReason,
  LanguageModelV4GenerateResult,
  LanguageModelV4Usage,
} from "@ai-sdk/provider";

/**
 * Convert an OpenAI chat completions response body (the `response` payload of
 * a completed async task submitted via `POST /v1/chat/completions/async`) into
 * a `LanguageModelV4GenerateResult`.
 */

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

function mapFinishReason(raw: string | null | undefined): LanguageModelV4FinishReason {
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

function mapUsage(usage: ChatCompletionBody["usage"]): LanguageModelV4Usage {
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
    finishReason: mapFinishReason(choice?.finish_reason),
    usage: mapUsage(completion.usage),
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
