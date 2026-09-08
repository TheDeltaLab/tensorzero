// Modified by Delta-AI under Apache 2.0
/**
 * Serializable reference for a TensorZero async batch.
 *
 * A "batch" is not a server-side object: `experimental_doStartBatch` submits
 * each request as an independent durable async task. The batch reference is
 * the list of `(request id, task_id)` pairs, JSON-encoded into the opaque
 * `batchId` string returned to the AI SDK caller.
 */

export interface TensorZeroBatchItem {
  /** Application-provided request id (`LanguageModelV4BatchRequest.id`). */
  id: string;
  /** Durable async task id returned by the submit endpoint. */
  taskId: string;
}

export interface TensorZeroBatchReference {
  v: 1;
  items: TensorZeroBatchItem[];
}

export function encodeBatchId(reference: TensorZeroBatchReference): string {
  return JSON.stringify(reference);
}

export function decodeBatchId(batchId: string): TensorZeroBatchReference {
  let parsed: unknown;
  try {
    parsed = JSON.parse(batchId);
  } catch {
    throw new Error(
      `Invalid TensorZero batch id: expected a JSON-encoded batch reference`,
    );
  }
  const ref = parsed as TensorZeroBatchReference;
  if (
    typeof ref !== "object" ||
    ref === null ||
    ref.v !== 1 ||
    !Array.isArray(ref.items) ||
    ref.items.some(
      (item) =>
        typeof item !== "object" ||
        item === null ||
        typeof item.id !== "string" ||
        typeof item.taskId !== "string",
    )
  ) {
    throw new Error(
      `Invalid TensorZero batch id: malformed batch reference payload`,
    );
  }
  return ref;
}
