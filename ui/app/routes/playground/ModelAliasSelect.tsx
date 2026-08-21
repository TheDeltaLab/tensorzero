// Modified by Delta-AI under Apache 2.0
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectLabel,
  SelectTrigger,
  SelectValue,
} from "~/components/ui/select";

export function ModelAliasSelect({
  aliasNames,
  modelNames,
  value,
  onValueChange,
  placeholder,
  disabled = false,
}: {
  aliasNames: string[];
  modelNames: string[];
  value: string;
  onValueChange: (value: string) => void;
  placeholder: string;
  disabled?: boolean;
}) {
  const aliasSet = new Set(aliasNames);
  const extraModels = modelNames.filter((name) => !aliasSet.has(name));
  const hasOptions = aliasNames.length > 0 || extraModels.length > 0;

  if (!hasOptions) {
    return (
      <Select disabled>
        <SelectTrigger>
          <SelectValue placeholder="No models configured" />
        </SelectTrigger>
      </Select>
    );
  }

  return (
    <Select
      value={value || undefined}
      onValueChange={onValueChange}
      disabled={disabled}
    >
      <SelectTrigger>
        <SelectValue placeholder={placeholder} />
      </SelectTrigger>
      <SelectContent>
        {aliasNames.length > 0 ? (
          <SelectGroup>
            <SelectLabel>Model aliases</SelectLabel>
            {aliasNames.map((name) => (
              <SelectItem key={`alias:${name}`} value={name}>
                {name}
              </SelectItem>
            ))}
          </SelectGroup>
        ) : null}
        {extraModels.length > 0 ? (
          <SelectGroup>
            <SelectLabel>Models</SelectLabel>
            {extraModels.map((name) => (
              <SelectItem key={`model:${name}`} value={name}>
                {name}
              </SelectItem>
            ))}
          </SelectGroup>
        ) : null}
      </SelectContent>
    </Select>
  );
}
