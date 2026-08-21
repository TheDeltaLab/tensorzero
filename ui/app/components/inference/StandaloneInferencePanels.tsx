// Modified by Delta-AI under Apache 2.0
import type { ReactNode } from "react";
import type { Input, StoredInference, StoredInput } from "~/types/tensorzero";
import { EmptyMessage } from "~/components/input_output/ContentBlockElement";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
  TableEmptyState,
} from "~/components/ui/table";
import {
  embeddingInputTexts,
  formatRelevanceScore,
  parseStandaloneOutput,
  rerankInputView,
  type ObservabilityInferenceKind,
} from "~/utils/observability/standaloneInference";

type InferenceInput = Input | StoredInput;

function Panel({ children, testId }: { children: ReactNode; testId: string }) {
  return (
    <div
      className="bg-bg-primary border-border flex w-full flex-col gap-4 rounded-lg border p-4"
      data-testid={testId}
    >
      {children}
    </div>
  );
}

function LabeledText({
  label,
  text,
  index,
}: {
  label: string;
  text: string;
  index?: number;
}) {
  return (
    <div className="flex w-full flex-col gap-1">
      <div className="text-fg-tertiary text-xs font-medium">
        {index === undefined ? label : `${label} ${index}`}
      </div>
      <pre className="bg-bg-secondary text-fg-primary max-h-64 overflow-auto rounded-sm p-3 font-mono text-xs whitespace-pre-wrap">
        {text}
      </pre>
    </div>
  );
}

export function StandaloneInputElement({
  kind,
  input,
}: {
  kind: ObservabilityInferenceKind;
  input?: InferenceInput;
}) {
  if (kind === "embedding") {
    const texts = embeddingInputTexts(input);
    if (texts.length === 0) {
      return (
        <Panel testId="embedding-input">
          <EmptyMessage message="No input texts" />
        </Panel>
      );
    }
    return (
      <Panel testId="embedding-input">
        {texts.map((text, index) => (
          <LabeledText key={index} label="Text" index={index + 1} text={text} />
        ))}
      </Panel>
    );
  }

  const { query, documents } = rerankInputView(input);
  return (
    <Panel testId="rerank-input">
      <LabeledText label="Query" text={query || "(empty query)"} />
      {documents.length === 0 ? (
        <EmptyMessage message="No documents" />
      ) : (
        documents.map((document, index) => (
          <LabeledText
            key={index}
            label="Document"
            index={index}
            text={document}
          />
        ))
      )}
    </Panel>
  );
}

export function StandaloneOutputElement({
  kind,
  inference,
}: {
  kind: ObservabilityInferenceKind;
  inference: StoredInference;
}) {
  const parsed = parseStandaloneOutput(inference, kind);
  if (kind === "embedding") {
    const view =
      parsed?.kind === "embedding"
        ? parsed
        : {
            count: 0,
            dimensions: 0,
            summary: "No embedding output",
          };
    return (
      <Panel testId="embedding-output">
        <div className="flex flex-col gap-1">
          <div className="text-fg-primary text-sm font-medium">
            {view.count === 1 ? "1 embedding" : `${view.count} embeddings`}
            {view.dimensions > 0 ? ` · ${view.dimensions} dimensions` : ""}
          </div>
          <p className="text-fg-muted text-sm">
            Vectors are omitted from observability storage. Open a model
            inference for the provider raw request and response.
          </p>
        </div>
      </Panel>
    );
  }

  const view =
    parsed?.kind === "rerank"
      ? parsed
      : { count: 0, results: [], summary: "No rerank output" };
  const { documents } = rerankInputView(inference.input);
  const rows =
    view.results.length > 0
      ? view.results
      : documents.map((_, index) => ({ index, relevanceScore: undefined }));

  return (
    <Panel testId="rerank-output">
      <div className="text-fg-primary text-sm font-medium">
        {view.summary ||
          (view.count === 1
            ? "1 ranked document"
            : `${view.count} ranked documents`)}
      </div>
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>Rank</TableHead>
            <TableHead>Original index</TableHead>
            <TableHead>Score</TableHead>
            <TableHead>Document</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {rows.length === 0 ? (
            <TableEmptyState message="No ranked documents" />
          ) : (
            rows.map((row, rank) => (
              <TableRow key={`${row.index}-${rank}`}>
                <TableCell>{rank + 1}</TableCell>
                <TableCell className="font-mono">{row.index}</TableCell>
                <TableCell className="font-mono">
                  {formatRelevanceScore(row.relevanceScore)}
                </TableCell>
                <TableCell className="max-w-xl">
                  <span className="line-clamp-3 font-mono text-xs whitespace-pre-wrap">
                    {documents[row.index] ?? "—"}
                  </span>
                </TableCell>
              </TableRow>
            ))
          )}
        </TableBody>
      </Table>
    </Panel>
  );
}
