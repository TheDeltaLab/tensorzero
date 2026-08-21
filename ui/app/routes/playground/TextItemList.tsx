// Modified by Delta-AI under Apache 2.0
import { Plus, X } from "lucide-react";
import type { KeyboardEvent } from "react";
import { Button } from "~/components/ui/button";
import { Textarea } from "~/components/ui/textarea";

export function TextItemList({
  name,
  items,
  onChange,
  placeholder,
  onKeyDown,
}: {
  name: string;
  items: string[];
  onChange: (items: string[]) => void;
  placeholder?: (index: number) => string;
  onKeyDown?: (event: KeyboardEvent<HTMLTextAreaElement>) => void;
}) {
  const updateItem = (index: number, value: string) => {
    onChange(
      items.map((item, itemIndex) => (itemIndex === index ? value : item)),
    );
  };

  const addAfter = (index: number) => {
    onChange([...items.slice(0, index + 1), "", ...items.slice(index + 1)]);
  };

  const removeAt = (index: number) => {
    onChange(
      items.length <= 1
        ? [""]
        : items.filter((_, itemIndex) => itemIndex !== index),
    );
  };

  return (
    <div className="space-y-2">
      {items.map((item, index) => (
        <div key={`${name}-${index}`} className="flex items-start gap-2">
          <div className="text-muted-foreground flex w-8 shrink-0 items-center justify-center pt-2 font-mono text-xs">
            #{index + 1}
          </div>
          <Textarea
            name={name}
            value={item}
            onChange={(event) => updateItem(index, event.target.value)}
            onKeyDown={onKeyDown}
            placeholder={placeholder?.(index)}
            rows={2}
            className="flex-1 resize-y"
          />
          <div className="flex flex-col gap-1">
            <Button
              type="button"
              variant="outline"
              size="iconSm"
              onClick={() => addAfter(index)}
              title="Add item below"
            >
              <Plus className="h-4 w-4" />
            </Button>
            <Button
              type="button"
              variant="ghost"
              size="iconSm"
              onClick={() => removeAt(index)}
              disabled={items.length === 1 && item.length === 0}
              title="Remove this item"
            >
              <X className="h-4 w-4" />
            </Button>
          </div>
        </div>
      ))}
    </div>
  );
}
