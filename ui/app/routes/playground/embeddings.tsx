// Modified by Delta-AI under Apache 2.0
import {
  Form,
  useActionData,
  useNavigation,
  type ActionFunctionArgs,
  type RouteHandle,
} from "react-router";
import { PageHeader, PageLayout } from "~/components/layout/PageLayout";
import { Button } from "~/components/ui/button";
import { Input } from "~/components/ui/input";
import { Label } from "~/components/ui/label";
import { Textarea } from "~/components/ui/textarea";
import { getTensorZeroClient } from "~/utils/tensorzero.server";
import { logger } from "~/utils/logger";
import { PlaygroundNav } from "./PlaygroundNav";

export const handle: RouteHandle = {
  crumb: () => ["Playground", "Embeddings"],
};

export async function action({ request }: ActionFunctionArgs) {
  const form = await request.formData();
  const model = form.get("model")?.toString() ?? "";
  const input = form.get("input")?.toString() ?? "";
  try {
    const result = await getTensorZeroClient().embeddings({
      model,
      input: input
        .split("\n")
        .map((line) => line.trim())
        .filter(Boolean),
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
  const actionData = useActionData<typeof action>();
  const navigation = useNavigation();
  const busy = navigation.state !== "idle";
  return (
    <PageLayout>
      <PageHeader heading="Playground" />
      <PlaygroundNav current="/playground/embeddings" />
      <Form method="post" className="flex max-w-180 flex-col gap-4">
        <div className="flex flex-col gap-2">
          <Label htmlFor="model">Model alias</Label>
          <Input
            id="model"
            name="model"
            required
            placeholder="qwen3-embedding-4b"
          />
        </div>
        <div className="flex flex-col gap-2">
          <Label htmlFor="input">Texts (one per line)</Label>
          <Textarea id="input" name="input" required rows={6} />
        </div>
        <Button type="submit" disabled={busy}>
          Embed
        </Button>
      </Form>
      {actionData?.ok ? (
        <pre className="bg-bg-hover overflow-auto rounded-md p-4 text-sm">
          {JSON.stringify(actionData.result, null, 2)}
        </pre>
      ) : null}
      {actionData && !actionData.ok ? (
        <p className="text-red-600">{actionData.error}</p>
      ) : null}
    </PageLayout>
  );
}
