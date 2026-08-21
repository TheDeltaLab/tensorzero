// Modified by Delta-AI under Apache 2.0
export const HEADER_TAG_PREFIX = "tensorzero::header::";
export const REQUEST_HEADERS_TAG = "tensorzero::request_headers";

export type SplitInferenceTags = {
  headers: Record<string, string>;
  userTags: Record<string, string>;
};

export function isUserTagKey(key: string): boolean {
  return !key.startsWith("tensorzero::");
}

export function splitInferenceTags(
  tags: Record<string, string | undefined> | undefined,
): SplitInferenceTags {
  const headers: Record<string, string> = {};
  const userTags: Record<string, string> = {};
  if (!tags) {
    return { headers, userTags };
  }
  for (const [key, value] of Object.entries(tags)) {
    if (value === undefined) {
      continue;
    }
    if (key.startsWith(HEADER_TAG_PREFIX)) {
      headers[key.slice(HEADER_TAG_PREFIX.length)] = value;
    } else if (key === REQUEST_HEADERS_TAG) {
      continue;
    } else if (isUserTagKey(key)) {
      userTags[key] = value;
    }
  }
  return { headers, userTags };
}

/// Parse `x-tensorzero-tags` / the Inferences search `tags` param.
export function parseCsvTags(raw: string): Record<string, string> {
  const tags: Record<string, string> = {};
  for (const piece of raw.split(",")) {
    const trimmed = piece.trim();
    if (!trimmed) {
      continue;
    }
    const eq = trimmed.indexOf("=");
    if (eq === -1) {
      tags[trimmed] = "true";
    } else {
      const key = trimmed.slice(0, eq).trim();
      const value = trimmed.slice(eq + 1).trim();
      if (key) {
        tags[key] = value;
      }
    }
  }
  return tags;
}

export function formatCsvTags(tags: Record<string, string>): string {
  return Object.entries(tags)
    .sort(([left], [right]) => left.localeCompare(right))
    .map(([key, value]) => (value === "true" ? key : `${key}=${value}`))
    .join(",");
}
