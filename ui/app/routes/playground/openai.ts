// Modified by Delta-AI under Apache 2.0

export type ChatMessage = {
  role: "user" | "assistant";
  content: string;
};

export type ChatActionData =
  | { ok: true; reply: string }
  | { ok: false; error: string };

export function extractAssistantReply(result: unknown): string {
  const body = result as {
    choices?: Array<{
      message?: {
        content?: string | Array<{ type?: string; text?: string }>;
        reasoning_content?: string;
      };
    }>;
  };
  const message = body.choices?.[0]?.message;
  if (!message) {
    return "";
  }
  if (typeof message.content === "string" && message.content.length > 0) {
    return message.content;
  }
  if (Array.isArray(message.content)) {
    const text = message.content
      .map((part) => part.text ?? "")
      .join("")
      .trim();
    if (text.length > 0) {
      return text;
    }
  }
  if (
    typeof message.reasoning_content === "string" &&
    message.reasoning_content.length > 0
  ) {
    return message.reasoning_content;
  }
  return "";
}

export type EmbeddingItem = {
  index: number;
  embedding: number[];
};

export function extractEmbeddings(result: unknown): EmbeddingItem[] {
  const body = result as {
    data?: Array<{ index?: number; embedding?: number[] }>;
  };
  return (body.data ?? []).map((item, index) => ({
    index: item.index ?? index,
    embedding: item.embedding ?? [],
  }));
}

export type RerankItem = {
  index: number;
  relevanceScore: number;
  document?: string;
};

export function extractRerankResults(
  result: unknown,
  documents: string[],
): RerankItem[] {
  const body = result as {
    results?: Array<{
      index?: number;
      relevance_score?: number;
      relevanceScore?: number;
      document?: { text?: string } | string;
    }>;
  };
  return (body.results ?? []).map((item, fallbackIndex) => {
    const index = item.index ?? fallbackIndex;
    const documentField = item.document;
    const documentFromResult =
      typeof documentField === "string" ? documentField : documentField?.text;
    return {
      index,
      relevanceScore: item.relevance_score ?? item.relevanceScore ?? 0,
      document: documentFromResult ?? documents[index],
    };
  });
}
