// Modified by Delta-AI under Apache 2.0
import {
  Form,
  useActionData,
  useNavigation,
  type ActionFunctionArgs,
  type RouteHandle,
} from "react-router";
import { useState, type KeyboardEvent } from "react";
import { Plus, X } from "lucide-react";
import { PageHeader, PageLayout } from "~/components/layout/PageLayout";
import { Button } from "~/components/ui/button";
import { Input } from "~/components/ui/input";
import { Label } from "~/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "~/components/ui/select";
import { Textarea } from "~/components/ui/textarea";
import { getTensorZeroClient } from "~/utils/tensorzero.server";
import { logger } from "~/utils/logger";
import { PlaygroundNav } from "./PlaygroundNav";
import {
  JEV_MODELS,
  answersFromResponse,
  buildSystemOneRequest,
  emptyQuestion,
  type JevQuestionDraft,
  type JevQuestionType,
} from "./jev";

export const handle: RouteHandle = {
  crumb: () => ["Playground", "Jev"],
};

export async function action({ request }: ActionFunctionArgs) {
  const form = await request.formData();
  const model = form.get("model")?.toString() ?? "";
  const state = form.get("state")?.toString() ?? "";
  const rawQuestions = form.get("questions")?.toString() ?? "[]";
  let questions: JevQuestionDraft[];
  try {
    const parsed: unknown = JSON.parse(rawQuestions);
    if (!Array.isArray(parsed)) {
      return { ok: false as const, error: "Questions were not a list." };
    }
    questions = parsed as JevQuestionDraft[];
  } catch {
    return { ok: false as const, error: "Could not read the questions." };
  }
  const built = buildSystemOneRequest({ model, state, questions });
  if (!built.ok) {
    return built;
  }
  try {
    const result = await getTensorZeroClient().systemOne(built.request);
    return { ok: true as const, result };
  } catch (error) {
    logger.error(error);
    return {
      ok: false as const,
      error: error instanceof Error ? error.message : String(error),
    };
  }
}

export default function JevPlayground() {
  const [model, setModel] = useState<string>(JEV_MODELS[0]);
  const [questions, setQuestions] = useState<JevQuestionDraft[]>([
    emptyQuestion(0),
  ]);
  const actionData = useActionData<typeof action>();
  const navigation = useNavigation();
  const busy = navigation.state !== "idle";
  const answers =
    actionData?.ok === true ? answersFromResponse(actionData.result) : [];

  const updateQuestion = (index: number, patch: Partial<JevQuestionDraft>) => {
    setQuestions((current) =>
      current.map((question, questionIndex) =>
        questionIndex === index ? { ...question, ...patch } : question,
      ),
    );
  };

  const handleKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === "Enter" && (event.metaKey || event.ctrlKey)) {
      event.preventDefault();
      event.currentTarget.form?.requestSubmit();
    }
  };

  return (
    <PageLayout>
      <PageHeader heading="Playground" />
      <PlaygroundNav current="/playground/jev" />
      <Form method="post" className="flex max-w-180 flex-col gap-4">
        <input type="hidden" name="model" value={model} />
        <input
          type="hidden"
          name="questions"
          value={JSON.stringify(questions)}
        />
        <div className="flex flex-col gap-2">
          <Label>Model</Label>
          <Select value={model} onValueChange={setModel}>
            <SelectTrigger aria-label="Jev model">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {JEV_MODELS.map((item) => (
                <SelectItem key={item} value={item}>
                  {item}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          <p className="text-muted-foreground text-xs">
            Jev answers typed questions about a state. It does not generate chat
            text. Ctrl/Cmd+Enter to run.
          </p>
        </div>
        <div className="flex flex-col gap-2">
          <Label htmlFor="jev-state">State</Label>
          <Textarea
            id="jev-state"
            name="state"
            required
            rows={4}
            className="resize-y"
            placeholder="The text or JSON to judge…"
            onKeyDown={handleKeyDown}
          />
        </div>
        <div className="flex flex-col gap-3">
          <div className="flex items-center justify-between">
            <Label>Questions</Label>
            <Button
              type="button"
              variant="outline"
              size="sm"
              onClick={() =>
                setQuestions((current) => [
                  ...current,
                  emptyQuestion(current.length),
                ])
              }
            >
              <Plus className="h-4 w-4" />
              Add question
            </Button>
          </div>
          {questions.map((question, index) => (
            <QuestionCard
              key={index}
              index={index}
              question={question}
              canRemove={questions.length > 1}
              onChange={(patch) => updateQuestion(index, patch)}
              onRemove={() =>
                setQuestions((current) =>
                  current.filter((_, itemIndex) => itemIndex !== index),
                )
              }
              onKeyDown={handleKeyDown}
            />
          ))}
        </div>
        <Button type="submit" disabled={busy}>
          {busy ? "Running…" : "Run Jev"}
        </Button>
      </Form>
      {actionData?.ok === true ? (
        <div className="flex max-w-180 flex-col gap-3">
          {answers.map((answer) => (
            <div key={answer.id} className="rounded-lg border p-4">
              <div className="flex items-baseline justify-between gap-3">
                <p className="text-sm font-medium">{answer.id}</p>
                <p className="text-muted-foreground text-xs">{answer.type}</p>
              </div>
              <p className="mt-1 font-mono text-sm">{answer.headline}</p>
              {answer.lines.length > 0 ? (
                <ul className="text-muted-foreground mt-2 space-y-1 text-xs">
                  {answer.lines.map((line) => (
                    <li key={line}>{line}</li>
                  ))}
                </ul>
              ) : null}
            </div>
          ))}
          <details className="rounded-lg border p-4">
            <summary className="cursor-pointer text-sm font-medium">
              Raw response
            </summary>
            <pre className="mt-3 overflow-auto font-mono text-xs whitespace-pre-wrap">
              {JSON.stringify(actionData.result, null, 2)}
            </pre>
          </details>
        </div>
      ) : null}
      {actionData && !actionData.ok ? (
        <p className="text-red-600">{actionData.error}</p>
      ) : null}
    </PageLayout>
  );
}

function QuestionCard({
  index,
  question,
  canRemove,
  onChange,
  onRemove,
  onKeyDown,
}: {
  index: number;
  question: JevQuestionDraft;
  canRemove: boolean;
  onChange: (patch: Partial<JevQuestionDraft>) => void;
  onRemove: () => void;
  onKeyDown: (event: KeyboardEvent<HTMLTextAreaElement>) => void;
}) {
  return (
    <div className="flex flex-col gap-3 rounded-lg border p-4">
      <div className="flex items-center justify-between gap-2">
        <p className="text-sm font-medium">Question {index + 1}</p>
        {canRemove ? (
          <Button type="button" variant="ghost" size="sm" onClick={onRemove}>
            <X className="h-4 w-4" />
            Remove
          </Button>
        ) : null}
      </div>
      <div className="grid gap-3 sm:grid-cols-2">
        <div className="flex flex-col gap-2">
          <Label htmlFor={`jev-id-${index}`}>Id</Label>
          <Input
            id={`jev-id-${index}`}
            value={question.id}
            onChange={(event) => onChange({ id: event.target.value })}
            placeholder="refund"
          />
        </div>
        <div className="flex flex-col gap-2">
          <Label>Type</Label>
          <Select
            value={question.type}
            onValueChange={(value) =>
              onChange({ type: value as JevQuestionType })
            }
          >
            <SelectTrigger aria-label={`Question ${index + 1} type`}>
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="noul">noul (yes / no)</SelectItem>
              <SelectItem value="choice">choice</SelectItem>
              <SelectItem value="score">score</SelectItem>
            </SelectContent>
          </Select>
        </div>
      </div>
      <div className="flex flex-col gap-2">
        <Label htmlFor={`jev-instructions-${index}`}>Instructions</Label>
        <Textarea
          id={`jev-instructions-${index}`}
          rows={2}
          className="resize-y"
          value={question.instructions}
          onChange={(event) => onChange({ instructions: event.target.value })}
          onKeyDown={onKeyDown}
        />
      </div>
      {question.type === "noul" ? (
        <div className="grid gap-3 sm:grid-cols-2">
          <div className="flex flex-col gap-2">
            <Label htmlFor={`jev-yes-${index}`}>Yes means</Label>
            <Input
              id={`jev-yes-${index}`}
              value={question.trueLabel}
              onChange={(event) => onChange({ trueLabel: event.target.value })}
              placeholder="Optional"
            />
          </div>
          <div className="flex flex-col gap-2">
            <Label htmlFor={`jev-no-${index}`}>No means</Label>
            <Input
              id={`jev-no-${index}`}
              value={question.falseLabel}
              onChange={(event) => onChange({ falseLabel: event.target.value })}
              placeholder="Optional"
            />
          </div>
        </div>
      ) : null}
      {question.type === "choice" ? (
        <OptionList
          options={question.options}
          onChange={(options) => onChange({ options })}
        />
      ) : null}
      {question.type === "score" ? (
        <LevelList
          levels={question.levels}
          onChange={(levels) => onChange({ levels })}
        />
      ) : null}
    </div>
  );
}

function OptionList({
  options,
  onChange,
}: {
  options: JevQuestionDraft["options"];
  onChange: (options: JevQuestionDraft["options"]) => void;
}) {
  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-center justify-between">
        <Label>Choices</Label>
        <Button
          type="button"
          variant="outline"
          size="sm"
          onClick={() => onChange([...options, { key: "", description: "" }])}
        >
          <Plus className="h-4 w-4" />
          Add choice
        </Button>
      </div>
      {options.map((option, index) => (
        <div key={index} className="grid grid-cols-[1fr_2fr_auto] gap-2">
          <Input
            aria-label={`Choice ${index + 1} key`}
            value={option.key}
            placeholder="billing"
            onChange={(event) =>
              onChange(
                options.map((item, itemIndex) =>
                  itemIndex === index
                    ? { ...item, key: event.target.value }
                    : item,
                ),
              )
            }
          />
          <Input
            aria-label={`Choice ${index + 1} description`}
            value={option.description}
            placeholder="Payments and refunds"
            onChange={(event) =>
              onChange(
                options.map((item, itemIndex) =>
                  itemIndex === index
                    ? { ...item, description: event.target.value }
                    : item,
                ),
              )
            }
          />
          <Button
            type="button"
            variant="ghost"
            size="sm"
            aria-label={`Remove choice ${index + 1}`}
            onClick={() =>
              onChange(
                options.length <= 1
                  ? [{ key: "", description: "" }]
                  : options.filter((_, itemIndex) => itemIndex !== index),
              )
            }
          >
            <X className="h-4 w-4" />
          </Button>
        </div>
      ))}
    </div>
  );
}

function LevelList({
  levels,
  onChange,
}: {
  levels: string[];
  onChange: (levels: string[]) => void;
}) {
  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-center justify-between">
        <Label>Levels, low to high</Label>
        <Button
          type="button"
          variant="outline"
          size="sm"
          onClick={() => onChange([...levels, ""])}
        >
          <Plus className="h-4 w-4" />
          Add level
        </Button>
      </div>
      {levels.map((level, index) => (
        <div key={index} className="grid grid-cols-[auto_1fr_auto] gap-2">
          <span className="text-muted-foreground flex h-9 items-center text-xs">
            {index}
          </span>
          <Input
            aria-label={`Level ${index}`}
            value={level}
            placeholder={index === 0 ? "Calm" : "Very angry"}
            onChange={(event) =>
              onChange(
                levels.map((item, itemIndex) =>
                  itemIndex === index ? event.target.value : item,
                ),
              )
            }
          />
          <Button
            type="button"
            variant="ghost"
            size="sm"
            aria-label={`Remove level ${index}`}
            onClick={() =>
              onChange(
                levels.length <= 2
                  ? levels
                  : levels.filter((_, itemIndex) => itemIndex !== index),
              )
            }
          >
            <X className="h-4 w-4" />
          </Button>
        </div>
      ))}
    </div>
  );
}
