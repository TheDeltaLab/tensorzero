// Modified by Delta-AI under Apache 2.0
import { SectionHeader, SectionLayout } from "~/components/layout/PageLayout";
import {
  SnippetContent,
  SnippetLayout,
} from "~/components/layout/SnippetLayout";
import { CodeEditor } from "~/components/ui/code-editor";
import { Badge } from "~/components/ui/badge";

/**
 * A flattened leaf error from a serialized `ErrorDetails` tree
 * (e.g. `AllVariantsFailed` -> `AllModelProvidersFailed` -> provider error).
 */
interface FlattenedError {
  /** Path of variant / provider names from the root to this error. */
  path: string[];
  /** The `ErrorDetails` variant name, e.g. `InferenceServer`. */
  kind: string;
  message?: string;
  statusCode?: number;
}

/**
 * Recursively walks a serialized `ErrorDetails` tree. Each node is a
 * single-key object (`{ "VariantName": { ...fields } }`); container variants
 * hold nested error maps keyed by variant/provider name.
 */
function flattenErrorTree(
  node: unknown,
  path: string[],
  out: FlattenedError[],
): void {
  if (node === null || typeof node !== "object" || Array.isArray(node)) {
    return;
  }
  const entries = Object.entries(node);
  if (entries.length !== 1) {
    return;
  }
  const [kind, payload] = entries[0];
  if (payload === null || typeof payload !== "object") {
    return;
  }
  const body = payload as Record<string, unknown>;
  for (const key of ["errors", "provider_errors", "candidate_errors"]) {
    const children = body[key];
    if (children !== null && typeof children === "object") {
      if (Array.isArray(children)) {
        // AllRetriesFailed holds a plain array of errors
        for (const child of children) {
          flattenErrorTree(child, path, out);
        }
      } else {
        for (const [name, child] of Object.entries(children)) {
          flattenErrorTree(child, [...path, name], out);
        }
      }
      return;
    }
  }
  out.push({
    path,
    kind,
    message: typeof body.message === "string" ? body.message : undefined,
    statusCode:
      typeof body.status_code === "number" ? body.status_code : undefined,
  });
}

export function parseFlattenedErrors(error: string): FlattenedError[] {
  let parsed: unknown;
  try {
    parsed = JSON.parse(error);
  } catch {
    return [];
  }
  const flattened: FlattenedError[] = [];
  flattenErrorTree(parsed, [], flattened);
  return flattened;
}

/**
 * Renders the contents of a stored `error` column: one card per leaf error
 * (with its variant/provider path), plus the raw JSON for debugging.
 */
export function InferenceErrorDetails({ error }: { error: string }) {
  const flattened = parseFlattenedErrors(error);
  const prettyError = (() => {
    try {
      return JSON.stringify(JSON.parse(error), null, 2);
    } catch {
      return error;
    }
  })();

  return (
    <div className="flex flex-col gap-4">
      {flattened.length === 0 ? (
        <div className="text-fg-muted text-sm">
          This inference failed, but the error details could not be parsed. See
          the raw error below.
        </div>
      ) : (
        flattened.map((entry, index) => (
          <div
            key={index}
            className="border-border flex flex-col gap-2 rounded-md border p-3"
          >
            <div className="flex flex-row flex-wrap items-center gap-1">
              {entry.path.map((segment, segmentIndex) => (
                <Badge key={segmentIndex} variant="outline">
                  {segment}
                </Badge>
              ))}
              <Badge variant="destructive">{entry.kind}</Badge>
              {entry.statusCode !== undefined && (
                <Badge variant="secondary">HTTP {entry.statusCode}</Badge>
              )}
            </div>
            {entry.message !== undefined && (
              <p className="text-sm break-words whitespace-pre-wrap">
                {entry.message}
              </p>
            )}
          </div>
        ))
      )}
      <SnippetLayout>
        <SnippetContent maxHeight={400}>
          <CodeEditor
            allowedLanguages={["json"]}
            value={prettyError}
            readOnly
          />
        </SnippetContent>
      </SnippetLayout>
    </div>
  );
}

export function ErrorSection({ error }: { error: string }) {
  return (
    <SectionLayout>
      <SectionHeader heading="Error" />
      <InferenceErrorDetails error={error} />
    </SectionLayout>
  );
}
