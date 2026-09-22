// Modified by Delta-AI under Apache 2.0

export const JEV_MODELS = ["jev-latest", "jev-1.13", "jev"] as const;

export type JevQuestionType = "noul" | "choice" | "score";

export type JevOptionDraft = {
  key: string;
  description: string;
};

export type JevQuestionDraft = {
  id: string;
  type: JevQuestionType;
  instructions: string;
  trueLabel: string;
  falseLabel: string;
  options: JevOptionDraft[];
  levels: string[];
};

export type SystemOneRequest = {
  model: string;
  state: string | Record<string, unknown> | unknown[];
  questions: Record<string, unknown>;
};

export type JevAnswerView = {
  id: string;
  type: string;
  headline: string;
  lines: string[];
};

export function emptyQuestion(index = 0): JevQuestionDraft {
  return {
    id: index === 0 ? "is_urgent" : "",
    type: "noul",
    instructions: index === 0 ? "Does this message convey urgency?" : "",
    trueLabel: "",
    falseLabel: "",
    options: [
      { key: "", description: "" },
      { key: "", description: "" },
    ],
    levels: ["", ""],
  };
}

export function buildSystemOneRequest(args: {
  model: string;
  state: string;
  questions: JevQuestionDraft[];
}): { ok: true; request: SystemOneRequest } | { ok: false; error: string } {
  const model = args.model.trim();
  if (!model) {
    return { ok: false, error: "Select a Jev model." };
  }
  const stateText = args.state.trim();
  if (!stateText) {
    return { ok: false, error: "Enter a state to evaluate." };
  }
  if (args.questions.length === 0) {
    return { ok: false, error: "Add at least one question." };
  }

  const questions: Record<string, unknown> = {};
  const seen = new Set<string>();
  for (const [index, draft] of args.questions.entries()) {
    const id = draft.id.trim();
    if (!id) {
      return { ok: false, error: `Question ${index + 1} needs an id.` };
    }
    if (seen.has(id)) {
      return { ok: false, error: `Question id "${id}" is duplicated.` };
    }
    seen.add(id);
    const built = buildQuestion(draft, index);
    if (!built.ok) {
      return built;
    }
    questions[id] = built.question;
  }

  return {
    ok: true,
    request: {
      model,
      state: parseState(stateText),
      questions,
    },
  };
}

function buildQuestion(
  draft: JevQuestionDraft,
  index: number,
):
  | { ok: true; question: Record<string, unknown> }
  | { ok: false; error: string } {
  const label = `Question ${index + 1}`;
  const instructions = draft.instructions.trim();
  if (!instructions) {
    return { ok: false, error: `${label} needs instructions.` };
  }
  if (draft.type === "noul") {
    const question: Record<string, unknown> = {
      type: "noul",
      instructions,
    };
    const yes = draft.trueLabel.trim();
    const no = draft.falseLabel.trim();
    if (yes || no) {
      if (!yes || !no) {
        return {
          ok: false,
          error: `${label} needs both yes and no criteria, or neither.`,
        };
      }
      question.criteria = { true: yes, false: no };
    }
    return { ok: true, question };
  }
  if (draft.type === "choice") {
    const criteria: Record<string, string | null> = {};
    for (const option of draft.options) {
      const key = option.key.trim();
      if (!key) {
        continue;
      }
      const description = option.description.trim();
      criteria[key] = description || null;
    }
    if (Object.keys(criteria).length === 0) {
      return { ok: false, error: `${label} needs at least one choice.` };
    }
    return {
      ok: true,
      question: { type: "choice", instructions, criteria },
    };
  }
  const levels = draft.levels.map((level) => level.trim()).filter(Boolean);
  if (levels.length < 2) {
    return { ok: false, error: `${label} needs at least two score levels.` };
  }
  if (levels.length > 10) {
    return { ok: false, error: `${label} has more than 10 score levels.` };
  }
  return {
    ok: true,
    question: { type: "score", instructions, criteria: levels },
  };
}

function parseState(state: string): SystemOneRequest["state"] {
  if (state.startsWith("{") || state.startsWith("[")) {
    try {
      const parsed: unknown = JSON.parse(state);
      if (parsed && typeof parsed === "object") {
        return parsed as SystemOneRequest["state"];
      }
    } catch {
      return state;
    }
  }
  return state;
}

export function answersFromResponse(body: unknown): JevAnswerView[] {
  if (!body || typeof body !== "object" || Array.isArray(body)) {
    return [];
  }
  const answers = (body as { answers?: unknown }).answers;
  if (!answers || typeof answers !== "object" || Array.isArray(answers)) {
    return [];
  }
  return Object.entries(answers as Record<string, unknown>).map(
    ([id, answer]) => summarizeAnswer(id, answer),
  );
}

function summarizeAnswer(id: string, answer: unknown): JevAnswerView {
  if (!answer || typeof answer !== "object" || Array.isArray(answer)) {
    return { id, type: "unknown", headline: "No answer", lines: [] };
  }
  const record = answer as Record<string, unknown>;
  const type = typeof record.type === "string" ? record.type : "unknown";
  if (type === "noul") {
    return {
      id,
      type,
      headline: formatNumber(record.noul),
      lines: ["Probability of yes"],
    };
  }
  if (type === "choice") {
    return {
      id,
      type,
      headline: typeof record.choice === "string" ? record.choice : "—",
      lines: [
        `Confidence ${formatNumber(record.confidence)}`,
        ...distributionLines(record.probabilities),
      ],
    };
  }
  if (type === "score") {
    return {
      id,
      type,
      headline: formatNumber(record.score),
      lines: [
        `Confidence ${formatNumber(record.confidence)}`,
        ...legendLines(record.legend),
      ],
    };
  }
  return { id, type, headline: "Unrecognized answer", lines: [] };
}

function distributionLines(value: unknown): string[] {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    return [];
  }
  return Object.entries(value as Record<string, unknown>).map(
    ([key, probability]) => `${key} ${formatNumber(probability)}`,
  );
}

function legendLines(value: unknown): string[] {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    return [];
  }
  return Object.entries(value as Record<string, unknown>)
    .sort(([left], [right]) => Number(left) - Number(right))
    .map(
      ([index, label]) => `${index}. ${typeof label === "string" ? label : ""}`,
    );
}

function formatNumber(value: unknown): string {
  return typeof value === "number" && Number.isFinite(value)
    ? String(value)
    : "—";
}
