// Modified by Delta-AI under Apache 2.0
import * as React from "react";

import { Button } from "~/components/ui/button";
import { DateTimePicker } from "~/components/ui/date-time-picker";
import { Label } from "~/components/ui/label";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "~/components/ui/popover";

export function CustomRangePicker({
  active,
  from,
  to,
  onApply,
}: {
  active: boolean;
  from: string;
  to: string;
  onApply: (fromIso: string, toIso: string) => void;
}) {
  const [open, setOpen] = React.useState(false);
  const [draftFrom, setDraftFrom] = React.useState<Date | undefined>();
  const [draftTo, setDraftTo] = React.useState<Date | undefined>();

  // Seed the drafts each time the popover opens: the active custom range if
  // there is one, otherwise a trailing 24h window ending now.
  React.useEffect(() => {
    if (!open) {
      return;
    }
    setDraftFrom(
      from ? new Date(from) : new Date(Date.now() - 24 * 60 * 60 * 1000),
    );
    setDraftTo(to ? new Date(to) : new Date());
  }, [open, from, to]);

  const valid =
    draftFrom !== undefined &&
    draftTo !== undefined &&
    draftFrom.getTime() < draftTo.getTime();

  const apply = () => {
    if (!valid) {
      return;
    }
    onApply(draftFrom.toISOString(), draftTo.toISOString());
    setOpen(false);
  };

  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger asChild>
        <Button
          type="button"
          size="sm"
          variant={active ? "default" : "outline"}
          aria-pressed={active}
        >
          Custom
        </Button>
      </PopoverTrigger>
      <PopoverContent className="w-auto p-4" align="start">
        <div className="flex flex-col gap-4 sm:flex-row sm:items-end">
          <div className="space-y-2">
            <Label htmlFor="analysis-custom-from">From</Label>
            <DateTimePicker
              id="analysis-custom-from"
              value={draftFrom}
              onChange={setDraftFrom}
              placeholder="Window start"
            />
          </div>
          <div className="space-y-2">
            <Label htmlFor="analysis-custom-to">To</Label>
            <DateTimePicker
              id="analysis-custom-to"
              value={draftTo}
              onChange={setDraftTo}
              placeholder="Window end"
              minDate={draftFrom}
              aria-invalid={!valid}
            />
          </div>
        </div>
        <p className="text-muted-foreground mt-3 text-xs">
          {valid
            ? "Bucket granularity adapts to the window length."
            : "Pick a start and end time; the end must be after the start."}
        </p>
        <div className="mt-3 flex justify-end gap-2">
          <Button
            type="button"
            size="sm"
            variant="outline"
            onClick={() => setOpen(false)}
          >
            Cancel
          </Button>
          <Button type="button" size="sm" disabled={!valid} onClick={apply}>
            Apply
          </Button>
        </div>
      </PopoverContent>
    </Popover>
  );
}
