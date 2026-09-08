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
import type { AsyncTaskStatus, TensorZeroClient } from "@thedeltalab/tensorzero-sdk";
import { decodeBatchId, encodeBatchId } from "./batch-reference.js";
import { chatCompletionToGenerateResult } from "./convert-response.js";

type BatchModel = Experimental_BatchModelV4<
  Experimental_LanguageModelV4BatchRequest,
  LanguageModelV4GenerateResult
>;

export type TensorZeroChatLanguageModel = LanguageModelV4 & BatchModel;

interface CapturedRequest {
  url: string;
  body: Record<string, unknown>;
}

/**
 * The openai-compatible chat model keeps its request-body serialization
 * (`getArgs`) private. To reuse it, the provider factory hands us a function
 * that builds a one-shot model whose fetch records the outgoing request body
 * instead of performing I/O; driving `doGenerate` then yields exactly the
 * body the sync path would send.
 */
export type CaptureModelFactory = () => {
  model: LanguageModelV4;
  captured: { value?: CapturedRequest };
};

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

export class TensorZeroChatLanguageModelImpl implements TensorZeroChatLanguageModel {
  readonly specificationVersion = "v4" as const;

  constructor(
    private readonly inner: LanguageModelV4,
    private readonly createCaptureModel: CaptureModelFactory,
    private readonly asyncClient: TensorZeroClient,
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

  /** Serialize a normalized batch request into a chat completions body. */
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
        `Failed to serialize batch request "${request.id}" into a chat completions body`,
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
      bodies.map((body) => this.asyncClient.submitChatCompletion(body)),
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
      toItemResult(item.id, statuses[index]!),
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
): Experimental_BatchV4ItemResult<LanguageModelV4GenerateResult> {
  if (status.status === "completed") {
    return {
      id,
      status: "succeeded",
      result: chatCompletionToGenerateResult(status.response),
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
