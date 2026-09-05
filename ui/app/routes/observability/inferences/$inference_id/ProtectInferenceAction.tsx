// Modified by Delta-AI under Apache 2.0
import { useEffect, useState } from "react";
import { ShieldCheck, ShieldOff } from "lucide-react";
import type { InferenceProtectionEntry } from "~/types/tensorzero";
import { useFetcherWithReset } from "~/hooks/use-fetcher-with-reset";
import { Button, ButtonIcon } from "~/components/ui/button";
import { Badge } from "~/components/ui/badge";
import { ReadOnlyGuard } from "~/components/utils/read-only-guard";
import { useToast } from "~/hooks/use-toast";
import { formatDate } from "~/utils/date";
import type { ProtectionActionData } from "~/routes/api/inference/$inference_id/protection/route";

interface ProtectInferenceActionProps {
  inferenceId: string;
  protection?: InferenceProtectionEntry;
}

export function ProtectInferenceAction({
  inferenceId,
  protection,
}: ProtectInferenceActionProps) {
  const fetcher = useFetcherWithReset<ProtectionActionData>();
  const { data: fetcherData, state: fetcherState, reset } = fetcher;
  const { toast } = useToast();

  // The page does not revalidate on protection submissions, so track the
  // state locally once a submission succeeds.
  const [protectedAt, setProtectedAt] = useState(protection?.protected_at);
  useEffect(() => {
    setProtectedAt(protection?.protected_at);
  }, [protection]);

  useEffect(() => {
    if (fetcherState !== "idle" || !fetcherData) {
      return;
    }
    if ("error" in fetcherData && fetcherData.error) {
      toast.error({ title: fetcherData.error });
      reset();
      return;
    }
    if ("success" in fetcherData && fetcherData.success) {
      setProtectedAt(
        fetcherData.protected
          ? (fetcherData.protected_at ?? new Date().toISOString())
          : undefined,
      );
      reset();
    }
  }, [fetcherData, fetcherState, reset, toast]);

  const isProtected = protectedAt !== undefined;
  const busy = fetcherState !== "idle";

  return (
    <fetcher.Form
      method="post"
      action={`/api/inference/${inferenceId}/protection`}
      className="flex items-center gap-2"
    >
      <input
        type="hidden"
        name="protected"
        value={isProtected ? "false" : "true"}
      />
      <ReadOnlyGuard asChild>
        <Button variant="outline" size="sm" type="submit" disabled={busy}>
          <ButtonIcon
            as={isProtected ? ShieldOff : ShieldCheck}
            variant="tertiary"
          />
          {isProtected ? "Remove protection" : "Protect from cleanup"}
        </Button>
      </ReadOnlyGuard>
      {isProtected && (
        <Badge variant="secondary">
          Protected {formatDate(new Date(protectedAt))}
        </Badge>
      )}
    </fetcher.Form>
  );
}
