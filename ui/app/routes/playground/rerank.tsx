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
  crumb: () => ["Playground", "Rerank"],
};

export async function action({ request }: ActionFunctionArgs) {
  const form = await request.formData();
  const model = form.get("model")?.toString() ?? "";
  const query = form.get("query")?.toString() ?? "";
  const documents = (form.get("documents")?.toString() ?? "")
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean);
  const topNRaw = form.get("top_n")?.toString();
  const top_n = topNRaw ? Number(topNRaw) : undefined;
  try {
    const result = await getTensorZeroClient().rerank({
      model,
      query,
      documents,
      top_n: Number.isFinite(top_n) ? top_n : undefined,
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

export default function RerankPlayground() {
  const actionData = useActionData<typeof action>();
  const navigation = useNavigation();
  const busy = navigation.state !== "idle";
  return (
    <PageLayout>
      <PageHeader heading="Playground" />
      <PlaygroundNav current="/playground/rerank" />
      <Form method="post" className="flex max-w-180 flex-col gap-4">
        <div className="flex flex-col gap-2">
          <Label htmlFor="model">Model alias</Label>
          <Input id="model" name="model" required placeholder="qwen3-rerank" />
        </div>
        <div className="flex flex-col gap-2">
          <Label htmlFor="query">Query</Label>
          <Input id="query" name="query" required />
        </div>
        <div className="flex flex-col gap-2">
          <Label htmlFor="documents">Documents (one per line)</Label>
          <Textarea id="documents" name="documents" required rows={8} />
        </div>
        <div className="flex flex-col gap-2">
          <Label htmlFor="top_n">top_n (optional)</Label>
          <Input id="top_n" name="top_n" type="number" min={1} />
        </div>
        <Button type="submit" disabled={busy}>
          Rerank
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
