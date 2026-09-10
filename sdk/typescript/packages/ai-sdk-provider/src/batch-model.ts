// Modified by Delta-AI under Apache 2.0
import type {
  Experimental_BatchModelV4,
  Experimental_BatchV4ItemResult,
  Experimental_BatchV4OperationOptions,
  Experimental_BatchV4StartOptions,
  Experimental_BatchV4StartResult,
  Experimental_BatchV4Status,
  Experimental_LanguageModelV4BatchRequest,
  LanguageModelV4,
  LanguageModelV4GenerateResult,
  SharedV4Warning,
} from "@ai-sdk/provider";
import {
  TensorZeroTimeoutError,
  type AsyncTaskStatus,
  type TensorZeroClient,
} from "@delta-ai/tensorzero-sdk";
import { decodeBatchId, encodeBatchId } from "./batch-reference.js";
import {
  chatCompletionToGenerateResult,
  responsesApiToGenerateResult,
} from "./convert-response.js";

type BatchModel = Experimental_BatchModelV4<
  Experimental_LanguageModelV4BatchRequest,
  LanguageModelV4GenerateResult
>;

/** Options for `TensorZeroAsyncLanguageModel.waitForBatch`. */
export interface TensorZeroBatchWaitOptions {
  /** Initial poll interval in ms. Defaults to 1_000. */
  intervalMs?: number;
  /** Maximum poll interval in ms (exponential backoff cap). Defaults to 10_000. */
  maxIntervalMs?: number;
  /** Give up after this many ms with a `TensorZeroTimeoutError`. Defaults to no timeout. */
  timeoutMs?: number;
  /** Abort the wait loop. */
  signal?: AbortSignal;
}

/**
 * A TensorZero language model: the wrapped protocol's synchronous
 * `doGenerate`/`doStream` plus the async-task batch capability
 * (`experimental_doStartBatch` submits each request as a durable async task
 * over the same protocol), plus a convenience wait that polls the batch
 * status with exponential backoff until it reaches a terminal state.
 */
export type TensorZeroAsyncLanguageModel = LanguageModelV4 &
  BatchModel & {
    waitForBatch(
      batchId: string,
      options?: TensorZeroBatchWaitOptions,
    ): Promise<Experimental_BatchV4Status>;
  };

/** Chat-completions flavor (`chatModel`). */
export type TensorZeroChatLanguageModel = TensorZeroAsyncLanguageModel;

/** OpenAI-Responses flavor (`responsesModel`). */
export type TensorZeroResponsesLanguageModel = TensorZeroAsyncLanguageModel;

interface CapturedRequest {
  url: string;
  body: Record<string, unknown>;
}

/**
 * The wrapped protocol models keep their request-body serialization
 * (`getArgs`) private. To reuse it, the provider factory hands us a function
 * that builds a one-shot model whose fetch records the outgoing request body
 * instead of performing I/O; driving `doGenerate` then yields exactly the
 * body the sync path would send.
 */
export type CaptureModelFactory = () => {
  model: LanguageModelV4;
  captured: { value?: CapturedRequest };
};

/** How the async batch talks to the gateway for one wire protocol. */
export interface AsyncBatchProtocol {
  /** Protocol label for error messages, e.g. `"chat completions"`. */
  label: string;
  /** Submit a captured request body to the protocol's `/async` endpoint. */
  submit(
    client: TensorZeroClient,
    body: Record<string, unknown>,
  ): Promise<{ taskId: string }>;
  /** Convert a completed task's response payload into a generate result. */
  convert(response: unknown): LanguageModelV4GenerateResult;
}

function errorMessage(error: unknown): string {
  if (typeof error === "string") return error;
  if (
    typeof error === "object" &&
    error !== null &&
    "message" in error &&
    typeof (error as { message: unknown }).message === "string"
  ) {
    return (error as { message: string }).message;
  }
  return JSON.stringify(error);
}

export class TensorZeroBatchLanguageModelImpl implements TensorZeroAsyncLanguageModel {
  readonly specificationVersion = "v4" as const;

  constructor(
    private readonly inner: LanguageModelV4,
    private readonly createCaptureModel: CaptureModelFactory,
    private readonly asyncClient: TensorZeroClient,
    private readonly protocol: AsyncBatchProtocol,
  ) {}

  get provider(): string {
    return this.inner.provider;
  }

  get modelId(): string {
    return this.inner.modelId;
  }

  get supportedUrls(): LanguageModelV4["supportedUrls"] {
    return this.inner.supportedUrls;
  }

  doGenerate(options: Parameters<LanguageModelV4["doGenerate"]>[0]) {
    return this.inner.doGenerate(options);
  }

  doStream(options: Parameters<LanguageModelV4["doStream"]>[0]) {
    return this.inner.doStream(options);
  }

  /** Serialize a normalized batch request into a protocol request body. */
  private async captureRequestBody(
    request: Experimental_LanguageModelV4BatchRequest,
    abortSignal?: AbortSignal,
  ): Promise<Record<string, unknown>> {
    const { model, captured } = this.createCaptureModel();
    try {
      await model.doGenerate({
        prompt: request.options.prompt,
        maxOutputTokens: request.options.maxOutputTokens,
        temperature: request.options.temperature,
        stopSequences: request.options.stopSequences,
        topP: request.options.topP,
        topK: request.options.topK,
        presencePenalty: request.options.presencePenalty,
        frequencyPenalty: request.options.frequencyPenalty,
        seed: request.options.seed,
        reasoning: request.options.reasoning,
        responseFormat: request.options.responseFormat,
        toolChoice: request.options.toolChoice,
        tools: request.options.tools,
        providerOptions: request.options.providerOptions,
        abortSignal,
      });
    } catch {
      // The capture fetch always aborts the call once the body is recorded.
    }
    if (!captured.value) {
      throw new Error(
        `Failed to serialize batch request "${request.id}" into a ${this.protocol.label} body`,
      );
    }
    return captured.value.body;
  }

  async experimental_doStartBatch(
    options: Experimental_BatchV4StartOptions<Experimental_LanguageModelV4BatchRequest>,
  ): Promise<Experimental_BatchV4StartResult> {
    const warnings: Array<{ requestId?: string; warning: SharedV4Warning }> = [];
    if (options.webhookUrl != null) {
      warnings.push({
        warning: {
          type: "unsupported",
          feature: "webhookUrl",
          details:
            "The TensorZero async inference API does not support completion webhooks.",
        },
      });
    }

    // Capture is sequential (the capture slot is per-model and each capture
    // does no I/O); task submission fans out in parallel afterwards.
    const bodies: Record<string, unknown>[] = [];
    for (const request of options.requests) {
      bodies.push(await this.captureRequestBody(request, options.abortSignal));
    }
    const launches = await Promise.all(
      bodies.map((body) => this.protocol.submit(this.asyncClient, body)),
    );

    const items = options.requests.map((request, index) => ({
      id: request.id,
      taskId: launches[index]!.taskId,
    }));

    return {
      batchId: encodeBatchId({ v: 1, items }),
      status: "pending",
      rawStatus: "queued",
      requestCounts: {
        total: items.length,
        pending: items.length,
        completed: 0,
        failed: 0,
      },
      warnings,
    };
  }

  async experimental_doGetBatchStatus(
    options: Experimental_BatchV4OperationOptions,
  ): Promise<Experimental_BatchV4Status> {
    const reference = decodeBatchId(options.batchId);
    const statuses = await Promise.all(
      reference.items.map((item) => this.asyncClient.getTask(item.taskId)),
    );

    const counts = { total: statuses.length, pending: 0, completed: 0, failed: 0 };
    let firstError: { message: string } | undefined;
    for (const status of statuses) {
      if (status.status === "completed") {
        counts.completed += 1;
      } else if (status.status === "failed" || status.status === "cancelled") {
        counts.failed += 1;
        if (!firstError) {
          firstError = {
            message:
              status.error !== undefined
                ? errorMessage(status.error)
                : `Async task ${status.taskId} ${status.status}`,
          };
        }
      } else {
        counts.pending += 1;
      }
    }

    const status =
      counts.pending > 0 ? "pending" : counts.failed > 0 ? "failed" : "completed";
    return {
      status,
      rawStatus: `${counts.completed}/${counts.total} completed`,
      requestCounts: counts,
      error: status === "failed" ? firstError : undefined,
    };
  }

  /**
   * Poll `experimental_doGetBatchStatus` with exponential backoff until the
   * batch reaches a terminal state (completed or failed). Mirrors the raw
   * client's `waitForCompletion`, but aggregates every task in the batch.
   */
  async waitForBatch(
    batchId: string,
    options?: TensorZeroBatchWaitOptions,
  ): Promise<Experimental_BatchV4Status> {
    const intervalMs = options?.intervalMs ?? 1_000;
    const maxIntervalMs = options?.maxIntervalMs ?? Math.max(intervalMs, 10_000);
    const deadline =
      options?.timeoutMs !== undefined
        ? Date.now() + options.timeoutMs
        : undefined;
    let attempt = 0;
    for (;;) {
      const status = await this.experimental_doGetBatchStatus({
        batchId,
        abortSignal: options?.signal,
      });
      if (status.status !== "pending") {
        return status;
      }
      const delay = Math.min(intervalMs * 2 ** attempt, maxIntervalMs);
      attempt += 1;
      if (deadline !== undefined && Date.now() + delay > deadline) {
        throw new TensorZeroTimeoutError(
          `Batch ${batchId} did not reach a terminal state within ${options?.timeoutMs}ms`,
        );
      }
      await sleep(delay, options?.signal);
    }
  }

  async experimental_doGetBatchResults(
    options: Experimental_BatchV4OperationOptions,
  ): Promise<
    ReadableStream<Experimental_BatchV4ItemResult<LanguageModelV4GenerateResult>>
  > {
    const reference = decodeBatchId(options.batchId);
    const statuses = await Promise.all(
      reference.items.map((item) => this.asyncClient.getTask(item.taskId)),
    );

    const results = reference.items.map((item, index) =>
      toItemResult(item.id, statuses[index]!, this.protocol),
    );

    return new ReadableStream<
      Experimental_BatchV4ItemResult<LanguageModelV4GenerateResult>
    >({
      start(controller) {
        for (const result of results) {
          controller.enqueue(result);
        }
        controller.close();
      },
    });
  }
}

function toItemResult(
  id: string,
  status: AsyncTaskStatus,
  protocol: AsyncBatchProtocol,
): Experimental_BatchV4ItemResult<LanguageModelV4GenerateResult> {
  if (status.status === "completed") {
    return {
      id,
      status: "succeeded",
      result: protocol.convert(status.response),
    };
  }
  if (status.status === "failed") {
    return {
      id,
      status: "failed",
      error: {
        message:
          status.error !== undefined
            ? errorMessage(status.error)
            : `Async task ${status.taskId} failed`,
      },
    };
  }
  if (status.status === "cancelled") {
    return {
      id,
      status: "cancelled",
      error: {
        message:
          status.error !== undefined
            ? errorMessage(status.error)
            : `Async task ${status.taskId} was cancelled`,
      },
    };
  }
  return {
    id,
    status: "failed",
    error: { message: `Async task ${status.taskId} is still ${status.status}` },
  };
}

/** Async batch over `POST /v1/chat/completions/async`. */
export const chatCompletionsProtocol: AsyncBatchProtocol = {
  label: "chat completions",
  submit: (client, body) => client.submitChatCompletion(body),
  convert: chatCompletionToGenerateResult,
};

/** Async batch over `POST /v1/responses/async`. */
export const responsesProtocol: AsyncBatchProtocol = {
  label: "responses",
  submit: (client, body) => client.submitResponses(body),
  convert: responsesApiToGenerateResult,
};

/** Sleep, rejecting immediately if the signal aborts. */
function sleep(ms: number, signal?: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) {
      reject(signal.reason ?? new DOMException("Aborted", "AbortError"));
      return;
    }
    const timer = setTimeout(() => {
      signal?.removeEventListener("abort", onAbort);
      resolve();
    }, ms);
    const onAbort = () => {
      clearTimeout(timer);
      reject(signal?.reason ?? new DOMException("Aborted", "AbortError"));
    };
    signal?.addEventListener("abort", onAbort, { once: true });
  });
}
