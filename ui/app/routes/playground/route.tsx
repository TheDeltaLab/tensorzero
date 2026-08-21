// Modified by Delta-AI under Apache 2.0
import {
  redirect,
  type ActionFunctionArgs,
  type LoaderFunctionArgs,
  type RouteHandle,
} from "react-router";
import { PageHeader, PageLayout } from "~/components/layout/PageLayout";
import { getTensorZeroClient } from "~/utils/tensorzero.server";
import { logger } from "~/utils/logger";
import { ChatPlayground } from "./ChatPlayground";
import { PlaygroundNav } from "./PlaygroundNav";
import {
  extractAssistantReply,
  type ChatActionData,
  type ChatMessage,
} from "./openai";

export const handle: RouteHandle = {
  crumb: () => ["Playground"],
};

export async function loader({ request }: LoaderFunctionArgs) {
  const url = new URL(request.url);
  if (
    url.searchParams.has("functionName") ||
    url.searchParams.has("datasetName") ||
    url.searchParams.has("variants")
  ) {
    throw redirect(`/playground/functions${url.search}`);
  }
  return null;
}

export async function action({
  request,
}: ActionFunctionArgs): Promise<ChatActionData> {
  const body = (await request.json()) as {
    model?: string;
    messages?: ChatMessage[];
    temperature?: number;
    max_tokens?: number;
  };
  const model = body.model?.trim() ?? "";
  const messages = body.messages ?? [];
  if (!model) {
    return { ok: false, error: "Select a model first." };
  }
  if (messages.length === 0) {
    return { ok: false, error: "Enter a message first." };
  }
  try {
    const result = await getTensorZeroClient().chatCompletions({
      model,
      messages,
      temperature: body.temperature,
      max_tokens: body.max_tokens,
    });
    const reply = extractAssistantReply(result);
    if (!reply) {
      return {
        ok: false,
        error: "The model returned an empty response.",
      };
    }
    return { ok: true, reply };
  } catch (error) {
    logger.error(error);
    return {
      ok: false,
      error: error instanceof Error ? error.message : String(error),
    };
  }
}

export default function ChatPlaygroundRoute() {
  return (
    <PageLayout className="gap-4">
      <PageHeader heading="Playground" />
      <PlaygroundNav current="/playground" />
      <ChatPlayground />
    </PageLayout>
  );
}
