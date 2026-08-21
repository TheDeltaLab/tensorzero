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
import { Label } from "~/components/ui/label";
import { getTensorZeroClient } from "~/utils/tensorzero.server";
import { logger } from "~/utils/logger";
import {
  ProviderModelSelect,
  useProviderModelSelection,
} from "./ProviderModelSelect";
import { PlaygroundNav } from "./PlaygroundNav";
import { TextItemList } from "./TextItemList";
import { nonEmptyItems } from "./models";
import { extractEmbeddings } from "./openai";

const VECTOR_PREVIEW_COUNT = 8;

export const handle: RouteHandle = {
  crumb: () => ["Playground", "Embeddings"],
};

export async function action({ request }: ActionFunctionArgs) {
  const form = await request.formData();
  const model = form.get("model")?.toString() ?? "";
  const input = nonEmptyItems(
    form.getAll("input").map((value) => value.toString()),
  );
  try {
    const result = await getTensorZeroClient().embeddings({
      model,
      input,
    });
    return { ok: true as const, result };
  } catch (error) {
    logger.error(error);
    return {
      ok: false as const,
      error: error instanceof Error ? error.message : String(error),
    };
  }
}

export default function EmbeddingsPlayground() {
  const {
    providers,
    models,
    provider,
    model,
    requestModel,
    setProvider,
    setModel,
  } = useProviderModelSelection("embedding");
  const [texts, setTexts] = useState<string[]>([""]);
  const actionData = useActionData<typeof action>();
  const navigation = useNavigation();
  const formRef = useRef<HTMLFormElement>(null);
  const busy = navigation.state !== "idle";
  const embeddings =
    actionData?.ok === true ? extractEmbeddings(actionData.result) : [];

  const handleKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === "Enter" && (event.metaKey || event.ctrlKey)) {
      event.preventDefault();
      formRef.current?.requestSubmit();
    }
  };

  return (
    <PageLayout>
      <PageHeader heading="Playground" />
      <PlaygroundNav current="/playground/embeddings" />
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
          <div className="flex items-center justify-between">
            <Label>Texts</Label>
            <span className="text-muted-foreground text-xs">
              {nonEmptyItems(texts).length} / {texts.length} filled
            </span>
          </div>
          <TextItemList
            name="input"
            items={texts}
            onChange={setTexts}
            placeholder={(index) => `Text ${index + 1}`}
            onKeyDown={handleKeyDown}
          />
          <p className="text-muted-foreground text-xs">
            Each box is one input, so newlines stay inside that item. Press
            Ctrl/Cmd+Enter to embed.
          </p>
        </div>
        <Button type="submit" disabled={busy || !provider || !model}>
          Embed
        </Button>
      </Form>
      {actionData?.ok ? (
        <div className="space-y-3">
          {embeddings.map((item) => (
            <div key={item.index} className="rounded-lg border p-4">
              <p className="text-sm font-medium">
                #{item.index + 1} · {item.embedding.length} dimensions
              </p>
              <p className="text-muted-foreground mt-2 font-mono text-xs break-all">
                [{item.embedding.slice(0, VECTOR_PREVIEW_COUNT).join(", ")}
                {item.embedding.length > VECTOR_PREVIEW_COUNT ? ", …" : ""}]
              </p>
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
