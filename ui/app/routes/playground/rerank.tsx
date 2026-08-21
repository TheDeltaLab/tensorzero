// Modified by Delta-AI under Apache 2.0
import {
  Form,
  useActionData,
  useNavigation,
  type ActionFunctionArgs,
  type RouteHandle,
} from "react-router";
import { useRef, useState, type KeyboardEvent } from "react";
import { PageHeader, PageLayout } from "~/components/layout/PageLayout";
import { Button } from "~/components/ui/button";
import { Input } from "~/components/ui/input";
import { Label } from "~/components/ui/label";
import { Textarea } from "~/components/ui/textarea";
import { getTensorZeroClient } from "~/utils/tensorzero.server";
import { logger } from "~/utils/logger";
import {
  ProviderModelSelect,
  useProviderModelSelection,
} from "./ProviderModelSelect";
import { PlaygroundNav } from "./PlaygroundNav";
import { TextItemList } from "./TextItemList";
import { nonEmptyItems } from "./models";
import { extractRerankResults } from "./openai";

export const handle: RouteHandle = {
  crumb: () => ["Playground", "Rerank"],
};

export async function action({ request }: ActionFunctionArgs) {
  const form = await request.formData();
  const model = form.get("model")?.toString() ?? "";
  const query = form.get("query")?.toString() ?? "";
  const documents = nonEmptyItems(
    form.getAll("document").map((value) => value.toString()),
  );
  const topNRaw = form.get("top_n")?.toString();
  const top_n = topNRaw ? Number(topNRaw) : undefined;
  try {
    const result = await getTensorZeroClient().rerank({
      model,
      query,
      documents,
      top_n: Number.isFinite(top_n) ? top_n : undefined,
    });
    return { ok: true as const, result, documents };
  } catch (error) {
    logger.error(error);
    return {
      ok: false as const,
      error: error instanceof Error ? error.message : String(error),
    };
  }
}

export default function RerankPlayground() {
  const {
    providers,
    models,
    provider,
    model,
    requestModel,
    setProvider,
    setModel,
  } = useProviderModelSelection("rerank");
  const [documents, setDocuments] = useState<string[]>([""]);
  const actionData = useActionData<typeof action>();
  const navigation = useNavigation();
  const formRef = useRef<HTMLFormElement>(null);
  const busy = navigation.state !== "idle";
  const ranked =
    actionData?.ok === true
      ? extractRerankResults(actionData.result, actionData.documents)
      : [];

  const handleKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === "Enter" && (event.metaKey || event.ctrlKey)) {
      event.preventDefault();
      formRef.current?.requestSubmit();
    }
  };

  return (
    <PageLayout>
      <PageHeader heading="Playground" />
      <PlaygroundNav current="/playground/rerank" />
      <Form
        method="post"
        ref={formRef}
        className="flex max-w-180 flex-col gap-4"
      >
        <input type="hidden" name="model" value={requestModel} />
        <ProviderModelSelect
          providers={providers}
          models={models}
          provider={provider}
          model={model}
          onProviderChange={setProvider}
          onModelChange={setModel}
        />
        <div className="flex flex-col gap-2">
          <Label htmlFor="query">Query</Label>
          <Textarea
            id="query"
            name="query"
            required
            rows={2}
            className="resize-y"
            onKeyDown={handleKeyDown}
            placeholder="Enter your search query…"
          />
        </div>
        <div className="flex flex-col gap-2">
          <div className="flex items-center justify-between">
            <Label>Documents</Label>
            <span className="text-muted-foreground text-xs">
              {nonEmptyItems(documents).length} / {documents.length} filled
            </span>
          </div>
          <TextItemList
            name="document"
            items={documents}
            onChange={setDocuments}
            placeholder={(index) => `Document ${index + 1}`}
            onKeyDown={handleKeyDown}
          />
          <p className="text-muted-foreground text-xs">
            Each box is one document, so newlines stay inside that item. Press
            Ctrl/Cmd+Enter to rerank.
          </p>
        </div>
        <div className="flex flex-col gap-2">
          <Label htmlFor="top_n">top_n (optional)</Label>
          <Input id="top_n" name="top_n" type="number" min={1} />
        </div>
        <Button type="submit" disabled={busy || !provider || !model}>
          Rerank
        </Button>
      </Form>
      {actionData?.ok ? (
        <div className="space-y-3">
          {ranked.map((item, rank) => (
            <div
              key={`${item.index}-${rank}`}
              className="rounded-lg border p-4"
            >
              <div className="flex items-center justify-between gap-2">
                <p className="text-sm font-medium">
                  #{rank + 1} · original #{item.index + 1}
                </p>
                <p className="font-mono text-sm">
                  {(item.relevanceScore * 100).toFixed(1)}%
                </p>
              </div>
              {item.document ? (
                <p className="mt-2 whitespace-pre-wrap text-sm">
                  {item.document}
                </p>
              ) : null}
            </div>
          ))}
        </div>
      ) : null}
      {actionData && !actionData.ok ? (
        <p className="text-red-600">{actionData.error}</p>
      ) : null}
    </PageLayout>
  );
}
