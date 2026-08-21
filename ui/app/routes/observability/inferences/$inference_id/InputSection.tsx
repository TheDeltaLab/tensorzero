// Modified by Delta-AI under Apache 2.0
import { Suspense } from "react";
import { Await } from "react-router";
import { Skeleton } from "~/components/ui/skeleton";
import { SectionHeader, SectionLayout } from "~/components/layout/PageLayout";
import { SectionAsyncErrorState } from "~/components/ui/error/ErrorContentPrimitives";
import { InputElement } from "~/components/input_output/InputElement";
import { EmptyMessage } from "~/components/input_output/ContentBlockElement";
import { StandaloneInputElement } from "~/components/inference/StandaloneInferencePanels";
import type { Input } from "~/types/tensorzero";
import {
  isStandaloneInferenceKind,
  type ObservabilityInferenceKind,
} from "~/utils/observability/standaloneInference";

interface InputSectionProps {
  promise: Promise<Input | undefined>;
  locationKey: string;
  kind: ObservabilityInferenceKind;
}

export function InputSection({
  promise,
  locationKey,
  kind,
}: InputSectionProps) {
  const standalone = isStandaloneInferenceKind(kind);
  return (
    <SectionLayout>
      <SectionHeader heading="Input" />
      <Suspense
        key={`input-${locationKey}`}
        fallback={<Skeleton className="h-32 w-full" />}
      >
        <Await
          resolve={promise}
          errorElement={
            <SectionAsyncErrorState defaultMessage="Failed to load input" />
          }
        >
          {(input) =>
            standalone ? (
              <StandaloneInputElement kind={kind} input={input} />
            ) : input ? (
              <InputElement input={input} />
            ) : (
              <div className="bg-bg-primary border-border flex w-full flex-col gap-1 rounded-lg border p-4">
                <EmptyMessage message="No input" />
              </div>
            )
          }
        </Await>
      </Suspense>
    </SectionLayout>
  );
}
