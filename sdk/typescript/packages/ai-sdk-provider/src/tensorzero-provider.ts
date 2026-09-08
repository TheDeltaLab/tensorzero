// Modified by Delta-AI under Apache 2.0
import type { LanguageModelV4, ProviderV4 } from "@ai-sdk/provider";
import {
  createOpenAICompatible,
  type OpenAICompatibleProviderSettings,
} from "@ai-sdk/openai-compatible";
import {
  createTensorZeroClient,
  type TensorZeroClient,
  type TensorZeroClientOptions,
} from "@delta-ai/tensorzero-sdk";
import {
  TensorZeroChatLanguageModelImpl,
  type TensorZeroChatLanguageModel,
} from "./batch-model.js";

export interface TensorZeroProviderSettings {
  /**
   * Base URL of the TensorZero gateway's OpenAI-compatible API, including the
   * version prefix, e.g. `https://gateway.example.com/v1` (or
   * `https://gateway.example.com/openai/v1`).
   */
  baseURL: string;
  /** Bearer API key. Omit only if the gateway has auth disabled. */
  apiKey?: string;
  /** Extra headers sent on every request. */
  headers?: Record<string, string>;
  /** Custom fetch implementation. */
  fetch?: OpenAICompatibleProviderSettings["fetch"];
  /**
   * Forwarded to `createOpenAICompatible`, e.g. `supportsStructuredOutputs`,
   * `includeUsage`, `metadataExtractor`, `transformRequestBody`.
   */
  openAICompatible?: Partial<
    Omit<
      OpenAICompatibleProviderSettings,
      "name" | "baseURL" | "apiKey" | "headers" | "fetch"
    >
  >;
}

export interface TensorZeroProvider extends ProviderV4 {
  (modelId: string): TensorZeroChatLanguageModel;
  languageModel(modelId: string): TensorZeroChatLanguageModel;
  chatModel(modelId: string): TensorZeroChatLanguageModel;
  /** Async-task client for the same gateway (submit / poll / stream / wait). */
  readonly asyncClient: TensorZeroClient;
}

/**
 * Strip the trailing `/v1` (and slashes) from the provider base URL to get
 * the gateway origin the async client builds `/v1/...` paths on.
 */
function gatewayBaseURL(baseURL: string): string {
  return baseURL.replace(/\/+$/, "").replace(/\/v1$/, "");
}

export function createTensorZero(
  settings: TensorZeroProviderSettings,
): TensorZeroProvider {
  const compatibleSettings: OpenAICompatibleProviderSettings = {
    name: "tensorzero",
    baseURL: settings.baseURL,
    apiKey: settings.apiKey,
    headers: settings.headers,
    fetch: settings.fetch,
    ...settings.openAICompatible,
  };

  const provider = createOpenAICompatible(compatibleSettings);

  const clientOptions: TensorZeroClientOptions = {
    baseURL: gatewayBaseURL(settings.baseURL),
    apiKey: settings.apiKey,
    headers: settings.headers,
    fetch: settings.fetch as TensorZeroClientOptions["fetch"],
  };
  const asyncClient = createTensorZeroClient(clientOptions);

  const createChatModel = (modelId: string): TensorZeroChatLanguageModel => {
    const inner = provider.chatModel(modelId);
    return new TensorZeroChatLanguageModelImpl(
      inner,
      () => {
        // One-shot model whose fetch records the request body instead of
        // performing the HTTP call (see `captureRequestBody`).
        const captured: { value?: { url: string; body: Record<string, unknown> } } =
          {};
        const captureProvider = createOpenAICompatible({
          ...compatibleSettings,
          fetch: (async (input: string | URL | Request, init?: RequestInit) => {
            captured.value = {
              url: String(input),
              body: init?.body
                ? (JSON.parse(String(init.body)) as Record<string, unknown>)
                : {},
            };
            throw new Error("tensorzero batch request body captured");
          }) as NonNullable<OpenAICompatibleProviderSettings["fetch"]>,
        });
        return { model: captureProvider.chatModel(modelId), captured };
      },
      asyncClient,
    );
  };

  const tensorzeroProvider = (modelId: string): TensorZeroChatLanguageModel =>
    createChatModel(modelId);

  return Object.assign(tensorzeroProvider, {
    specificationVersion: "v4" as const,
    languageModel: (modelId: string) => createChatModel(modelId),
    chatModel: (modelId: string) => createChatModel(modelId),
    embeddingModel: (modelId: string) => provider.embeddingModel(modelId),
    imageModel: (modelId: string) => provider.imageModel(modelId),
    asyncClient,
  });
}

export type { LanguageModelV4 };
