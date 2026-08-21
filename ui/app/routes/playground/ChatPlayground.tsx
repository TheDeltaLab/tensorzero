// Modified by Delta-AI under Apache 2.0
import { Bot, Send, Trash2, User } from "lucide-react";
import { useEffect, useRef, useState, type KeyboardEvent } from "react";
import { useFetcher } from "react-router";
import { Button } from "~/components/ui/button";
import { Input } from "~/components/ui/input";
import { Label } from "~/components/ui/label";
import { Textarea } from "~/components/ui/textarea";
import { cn } from "~/utils/common";
import {
  ProviderModelSelect,
  useProviderModelSelection,
} from "./ProviderModelSelect";
import type { ChatActionData, ChatMessage } from "./openai";

export function ChatPlayground() {
  const {
    providers,
    models: modelOptions,
    provider,
    model,
    requestModel,
    setProvider,
    setModel,
  } = useProviderModelSelection("chat");
  const [temperature, setTemperature] = useState(0.7);
  const [maxTokens, setMaxTokens] = useState(4096);
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const fetcher = useFetcher<ChatActionData>();
  const appliedDataRef = useRef<ChatActionData | undefined>(undefined);
  const scrollRef = useRef<HTMLDivElement>(null);
  const busy = fetcher.state !== "idle";
  const hasModel = provider.length > 0 && model.length > 0;

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages, busy]);

  useEffect(() => {
    if (fetcher.state !== "idle" || !fetcher.data) {
      return;
    }
    if (appliedDataRef.current === fetcher.data) {
      return;
    }
    appliedDataRef.current = fetcher.data;
    if (fetcher.data.ok) {
      const reply = fetcher.data.reply;
      setMessages((current) => [
        ...current,
        { role: "assistant", content: reply },
      ]);
    }
  }, [fetcher.state, fetcher.data]);

  const send = (content: string) => {
    const trimmed = content.trim();
    if (!trimmed || !hasModel || busy) {
      return;
    }
    const nextMessages: ChatMessage[] = [
      ...messages,
      { role: "user", content: trimmed },
    ];
    setMessages(nextMessages);
    setInput("");
    void fetcher.submit(
      {
        model: requestModel,
        messages: nextMessages,
        temperature,
        max_tokens: maxTokens,
      },
      { method: "POST", encType: "application/json" },
    );
  };

  const handleKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      send(input);
    }
  };

  return (
    <div className="flex min-h-[70vh] overflow-hidden rounded-lg border">
      <aside className="w-80 shrink-0 space-y-6 overflow-y-auto border-r p-4">
        <div>
          <h2 className="text-lg font-semibold">Settings</h2>
          <p className="text-muted-foreground text-sm">
            Pick a provider and model, then send a message. No TensorZero
            function required.
          </p>
        </div>
        <ProviderModelSelect
          providers={providers}
          models={modelOptions}
          provider={provider}
          model={model}
          onProviderChange={setProvider}
          onModelChange={setModel}
        />
        <div className="space-y-2">
          <Label htmlFor="chat-temperature">Temperature</Label>
          <Input
            id="chat-temperature"
            type="number"
            min={0}
            max={2}
            step={0.1}
            value={temperature}
            onChange={(event) =>
              setTemperature(Number.parseFloat(event.target.value) || 0)
            }
          />
        </div>
        <div className="space-y-2">
          <Label htmlFor="chat-max-tokens">Max tokens</Label>
          <Input
            id="chat-max-tokens"
            type="number"
            min={1}
            step={1}
            value={maxTokens}
            onChange={(event) =>
              setMaxTokens(Number.parseInt(event.target.value, 10) || 1)
            }
          />
        </div>
        <Button
          type="button"
          variant="outline"
          className="w-full"
          disabled={messages.length === 0}
          onClick={() => {
            setMessages([]);
            appliedDataRef.current = undefined;
          }}
        >
          <Trash2 className="h-4 w-4" />
          Clear conversation
        </Button>
      </aside>
      <div className="flex min-w-0 flex-1 flex-col">
        <div ref={scrollRef} className="flex-1 overflow-y-auto p-4">
          {messages.length === 0 && !busy ? (
            <div className="flex h-full flex-col items-center justify-center">
              <Bot className="text-muted-foreground/50 h-12 w-12" />
              <p className="text-muted-foreground mt-4 text-lg font-medium">
                Start a conversation
              </p>
              <p className="text-muted-foreground mt-1 text-sm">
                Select a provider and model, then press Enter to send.
                Shift+Enter inserts a newline.
              </p>
            </div>
          ) : (
            <div className="space-y-4">
              {messages.map((message, index) => (
                <div
                  key={`${message.role}-${index}`}
                  className={cn(
                    "flex gap-3",
                    message.role === "user" ? "justify-end" : "justify-start",
                  )}
                >
                  {message.role === "assistant" ? (
                    <div className="bg-primary text-primary-foreground flex h-8 w-8 shrink-0 items-center justify-center rounded-full">
                      <Bot className="h-4 w-4" />
                    </div>
                  ) : null}
                  <div
                    className={cn(
                      "max-w-[70%] rounded-lg px-4 py-2 whitespace-pre-wrap",
                      message.role === "user"
                        ? "bg-primary text-primary-foreground"
                        : "bg-bg-hover",
                    )}
                  >
                    {message.content}
                  </div>
                  {message.role === "user" ? (
                    <div className="bg-bg-hover flex h-8 w-8 shrink-0 items-center justify-center rounded-full">
                      <User className="h-4 w-4" />
                    </div>
                  ) : null}
                </div>
              ))}
              {busy ? (
                <p className="text-muted-foreground text-sm">Thinking…</p>
              ) : null}
            </div>
          )}
        </div>
        {fetcher.data && !fetcher.data.ok ? (
          <p className="text-red-600 px-4 text-sm">{fetcher.data.error}</p>
        ) : null}
        <div className="border-t p-4">
          <div className="flex gap-2">
            <Textarea
              value={input}
              onChange={(event) => setInput(event.target.value)}
              onKeyDown={handleKeyDown}
              placeholder={
                hasModel
                  ? "Message the model…"
                  : "Select a provider and model first"
              }
              disabled={!hasModel || busy}
              className="min-h-[60px] resize-none"
              rows={2}
            />
            <Button
              type="button"
              size="icon"
              className="h-[60px] w-[60px]"
              disabled={!hasModel || busy || !input.trim()}
              onClick={() => send(input)}
            >
              <Send className="h-5 w-5" />
            </Button>
          </div>
          <p className="text-muted-foreground mt-2 text-xs">
            Press Enter to send, Shift+Enter for a new line
          </p>
        </div>
      </div>
    </div>
  );
}
