// Modified by Delta-AI under Apache 2.0
import type { LanguageModelV4, ProviderV4 } from "@ai-sdk/provider";
import { createOpenAI, type OpenAIProviderSettings } from "@ai-sdk/openai";
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
  TensorZeroBatchLanguageModelImpl,
  chatCompletionsProtocol,
  responsesProtocol,
  type CaptureModelFactory,
  type TensorZeroChatLanguageModel,
  type TensorZeroResponsesLanguageModel,
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
  /**
   * A model over the OpenAI-compatible Responses API: synchronous
   * `doGenerate`/`doStream` against `POST /v1/responses`, async batch
   * against `POST /v1/responses/async`. Structured output rides in the
   * request's `text.format` (strict json_schema), which providers like
   * DeepSeek only enforce on the Responses API — not on chat completions.
   */
  responsesModel(modelId: string): TensorZeroResponsesLanguageModel;
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

/**
 * One-shot model whose fetch records the request body instead of performing
 * the HTTP call — driving `doGenerate` on it yields exactly the body the
 * sync path would send (see `TensorZeroBatchLanguageModelImpl`).
 */
function captureFetchFactory(): {
  captured: { value?: { url: string; body: Record<string, unknown> } };
  fetch: NonNullable<OpenAICompatibleProviderSettings["fetch"]>;
} {
  const captured: { value?: { url: string; body: Record<string, unknown> } } = {};
  return {
    captured,
    fetch: (async (input: string | URL | Request, init?: RequestInit) => {
      captured.value = {
        url: String(input),
        body: init?.body
          ? (JSON.parse(String(init.body)) as Record<string, unknown>)
          : {},
      };
      throw new Error("tensorzero batch request body captured");
    }) as NonNullable<OpenAICompatibleProviderSettings["fetch"]>,
  };
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

  const responsesSettings: OpenAIProviderSettings = {
    name: "tensorzero",
    baseURL: settings.baseURL,
    apiKey: settings.apiKey,
    headers: settings.headers,
    fetch: settings.fetch as OpenAIProviderSettings["fetch"],
  };

  const clientOptions: TensorZeroClientOptions = {
    baseURL: gatewayBaseURL(settings.baseURL),
    apiKey: settings.apiKey,
    headers: settings.headers,
    fetch: settings.fetch as TensorZeroClientOptions["fetch"],
  };
  const asyncClient = createTensorZeroClient(clientOptions);

  const createChatModel = (modelId: string): TensorZeroChatLanguageModel => {
    const inner = provider.chatModel(modelId);
    const createCaptureModel: CaptureModelFactory = () => {
      const { captured, fetch } = captureFetchFactory();
      const captureProvider = createOpenAICompatible({
        ...compatibleSettings,
        fetch,
      });
      return { model: captureProvider.chatModel(modelId), captured };
    };
    return new TensorZeroBatchLanguageModelImpl(
      inner,
      createCaptureModel,
      asyncClient,
      chatCompletionsProtocol,
    );
  };

  const createResponsesModel = (
    modelId: string,
  ): TensorZeroResponsesLanguageModel => {
    const inner = createOpenAI(responsesSettings).responses(modelId);
    const createCaptureModel: CaptureModelFactory = () => {
      const { captured, fetch } = captureFetchFactory();
      const captureProvider = createOpenAI({
        ...responsesSettings,
        fetch: fetch as OpenAIProviderSettings["fetch"],
      });
      return { model: captureProvider.responses(modelId), captured };
    };
    return new TensorZeroBatchLanguageModelImpl(
      inner,
      createCaptureModel,
      asyncClient,
      responsesProtocol,
    );
  };

  const tensorzeroProvider = (modelId: string): TensorZeroChatLanguageModel =>
    createChatModel(modelId);

  return Object.assign(tensorzeroProvider, {
    specificationVersion: "v4" as const,
    languageModel: (modelId: string) => createChatModel(modelId),
    chatModel: (modelId: string) => createChatModel(modelId),
    responsesModel: (modelId: string) => createResponsesModel(modelId),
    embeddingModel: (modelId: string) => provider.embeddingModel(modelId),
    imageModel: (modelId: string) => provider.imageModel(modelId),
    asyncClient,
  });
}

export type { LanguageModelV4 };
