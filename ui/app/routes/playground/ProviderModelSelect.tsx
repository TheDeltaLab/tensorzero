// Modified by Delta-AI under Apache 2.0
import { useEffect, useMemo, useState } from "react";
import { Label } from "~/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "~/components/ui/select";
import { useConfig } from "~/context/config";
import {
  defaultChoice,
  modelsForProvider,
  playgroundChoices,
  playgroundRequestModel,
  providersFromChoices,
  type PlaygroundTask,
} from "./models";

export function useProviderModelSelection(task: PlaygroundTask) {
  const config = useConfig();
  const configured =
    task === "embedding"
      ? config.embedding_model_providers
      : config.model_providers;
  const choices = useMemo(
    () => playgroundChoices(config.model_aliases, task, configured),
    [config.model_aliases, configured, task],
  );
  const providers = useMemo(() => providersFromChoices(choices), [choices]);
  const initial = useMemo(
    () => defaultChoice(config.model_aliases, task, choices),
    [config.model_aliases, choices, task],
  );
  const [provider, setProvider] = useState(initial?.provider ?? "");
  const [model, setModel] = useState(initial?.model ?? "");

  useEffect(() => {
    if (provider || !initial) {
      return;
    }
    setProvider(initial.provider);
    setModel(initial.model);
  }, [initial, provider]);
  const models = modelsForProvider(choices, provider);
  const requestModel = playgroundRequestModel(provider, model, configured);

  return {
    providers,
    models,
    provider,
    model,
    requestModel,
    setProvider: (nextProvider: string) => {
      setProvider(nextProvider);
      const nextModels = modelsForProvider(choices, nextProvider);
      setModel(nextModels.includes(model) ? model : (nextModels[0] ?? ""));
    },
    setModel,
  };
}

export function ProviderModelSelect({
  providers,
  models,
  provider,
  model,
  onProviderChange,
  onModelChange,
}: {
  providers: string[];
  models: string[];
  provider: string;
  model: string;
  onProviderChange: (provider: string) => void;
  onModelChange: (model: string) => void;
}) {
  const hasProviders = providers.length > 0;
  return (
    <div className="space-y-4">
      <div className="space-y-2">
        <Label>Provider</Label>
        <Select
          value={provider || undefined}
          onValueChange={onProviderChange}
          disabled={!hasProviders}
        >
          <SelectTrigger aria-label="Provider">
            <SelectValue
              placeholder={
                hasProviders ? "Select a provider" : "No providers configured"
              }
            />
          </SelectTrigger>
          <SelectContent>
            {providers.map((name) => (
              <SelectItem key={name} value={name}>
                {name}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>
      <div className="space-y-2">
        <Label>Model</Label>
        <Select
          value={model || undefined}
          onValueChange={onModelChange}
          disabled={!provider || models.length === 0}
        >
          <SelectTrigger aria-label="Model">
            <SelectValue
              placeholder={
                provider ? "Select a model" : "Select a provider first"
              }
            />
          </SelectTrigger>
          <SelectContent>
            {models.map((name) => (
              <SelectItem key={name} value={name}>
                {name}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>
    </div>
  );
}
