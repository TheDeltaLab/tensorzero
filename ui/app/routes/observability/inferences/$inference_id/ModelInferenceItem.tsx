// Modified by Delta-AI under Apache 2.0
import type { ParsedModelInferenceRow } from "~/utils/clickhouse/inference";
import { InputElement } from "~/components/input_output/InputElement";
import {
  BasicInfoLayout,
  BasicInfoItem,
  BasicInfoItemTitle,
  BasicInfoItemContent,
} from "~/components/layout/BasicInfoLayout";
import {
  PageLayout,
  PageHeader,
  SectionLayout,
  SectionHeader,
  SectionsGroup,
} from "~/components/layout/PageLayout";
import {
  Timer,
  InputIcon,
  Output,
  Calendar,
  Cached,
  Cost,
} from "~/components/icons/Icons";
import Chip from "~/components/ui/Chip";
import { formatCost } from "~/utils/cost";
import {
  formatOutputTps,
  outputTpsExcludingTtft,
  toFiniteMs,
} from "~/utils/observability/usageDetails";
import { formatDateWithSeconds } from "~/utils/date";
import { TimestampTooltip } from "~/components/ui/TimestampTooltip";
import {
  SnippetLayout,
  SnippetContent,
} from "~/components/layout/SnippetLayout";
import { CodeEditor } from "~/components/ui/code-editor";
import ModelInferenceOutput from "~/components/input_output/ModelInferenceOutput";
import { ModelInferenceUsageDetails } from "~/components/inference/UsageDetails";

interface ModelInferenceItemProps {
  inference: ParsedModelInferenceRow;
}

export function ModelInferenceItem({ inference }: ModelInferenceItemProps) {
  const ttftMs = toFiniteMs(inference.ttft_ms);
  const outputTps = outputTpsExcludingTtft({
    outputTokens: inference.output_tokens,
    durationMs: inference.response_time_ms,
    ttftMs,
  });
  return (
    <PageLayout>
      <PageHeader eyebrow="Model Inference" name={inference.id}>
        <BasicInfoLayout>
          <BasicInfoItem>
            <BasicInfoItemTitle>Model</BasicInfoItemTitle>
            <BasicInfoItemContent>
              <Chip label={inference.model_name} font="mono" />
            </BasicInfoItemContent>
          </BasicInfoItem>

          <BasicInfoItem>
            <BasicInfoItemTitle>Model Provider</BasicInfoItemTitle>
            <BasicInfoItemContent>
              <Chip label={inference.model_provider_name} />
            </BasicInfoItemContent>
          </BasicInfoItem>

          <BasicInfoItem>
            <BasicInfoItemTitle>Usage</BasicInfoItemTitle>
            <BasicInfoItemContent wrap>
              <div className="flex flex-row flex-wrap gap-1">
                <Chip
                  icon={<InputIcon className="text-fg-tertiary" />}
                  label={`${inference.input_tokens === undefined ? "—" : inference.input_tokens} in`}
                  tooltip="Input tokens"
                />
                <Chip
                  icon={<Output className="text-fg-tertiary" />}
                  label={`${inference.output_tokens === undefined ? "—" : inference.output_tokens} out`}
                  tooltip="Output tokens"
                />
                {inference.provider_cache_read_input_tokens != null &&
                  inference.provider_cache_read_input_tokens > 0 && (
                    <Chip
                      icon={<Cached className="text-fg-tertiary" />}
                      label={`${inference.provider_cache_read_input_tokens} cache read`}
                      tooltip="Provider cache read tokens"
                    />
                  )}
                {inference.provider_cache_write_input_tokens != null &&
                  inference.provider_cache_write_input_tokens > 0 && (
                    <Chip
                      icon={<Cached className="text-fg-tertiary" />}
                      label={`${inference.provider_cache_write_input_tokens} cache write`}
                      tooltip="Provider cache write tokens"
                    />
                  )}
                {inference.cost !== undefined && (
                  <Chip
                    icon={<Cost className="text-fg-tertiary" />}
                    label={formatCost(inference.cost, inference.currency)}
                    tooltip="Cost"
                  />
                )}
                {inference.response_time_ms != null && (
                  <Chip
                    icon={<Timer className="text-fg-tertiary" />}
                    label={`${inference.response_time_ms} ms`}
                    tooltip="Response time"
                  />
                )}
                {ttftMs != null && (
                  <Chip
                    icon={<Timer className="text-fg-tertiary" />}
                    label={`${ttftMs} ms TTFT`}
                    tooltip="Time to first token"
                  />
                )}
                {outputTps != null && (
                  <Chip
                    icon={<Output className="text-fg-tertiary" />}
                    label={formatOutputTps(outputTps)}
                    tooltip="Output tokens per second, excluding TTFT"
                  />
                )}
                {inference.cached && (
                  <Chip
                    icon={<Cached className="text-fg-tertiary" />}
                    label="Cached"
                    tooltip="Model Inference was cached by TensorZero"
                  />
                )}
              </div>
            </BasicInfoItemContent>
          </BasicInfoItem>

          <BasicInfoItem>
            <BasicInfoItemTitle>Timestamp</BasicInfoItemTitle>
            <BasicInfoItemContent>
              <Chip
                icon={<Calendar className="text-fg-tertiary" />}
                label={formatDateWithSeconds(new Date(inference.timestamp))}
                tooltip={<TimestampTooltip timestamp={inference.timestamp} />}
              />
            </BasicInfoItemContent>
          </BasicInfoItem>
        </BasicInfoLayout>
      </PageHeader>

      <SectionsGroup>
        <SectionLayout>
          <SectionHeader heading="Usage details" />
          <ModelInferenceUsageDetails inference={inference} />
        </SectionLayout>

        <SectionLayout>
          <SectionHeader heading="Input" />
          <InputElement
            input={{
              system: inference.system ?? undefined,
              messages: inference.input_messages,
            }}
          />
        </SectionLayout>

        <SectionLayout>
          <SectionHeader heading="Output" />
          <ModelInferenceOutput output={inference.output} />
        </SectionLayout>

        {inference.raw_request != null && (
          <SectionLayout>
            <SectionHeader heading="Raw Request" />
            <SnippetLayout>
              <SnippetContent maxHeight={400}>
                <CodeEditor
                  allowedLanguages={["json"]}
                  value={(() => {
                    try {
                      return JSON.stringify(
                        JSON.parse(inference.raw_request),
                        null,
                        2,
                      );
                    } catch {
                      return inference.raw_request;
                    }
                  })()}
                  readOnly
                />
              </SnippetContent>
            </SnippetLayout>
          </SectionLayout>
        )}

        {inference.raw_response != null && (
          <SectionLayout>
            <SectionHeader heading="Raw Response" />
            <SnippetLayout>
              <SnippetContent maxHeight={400}>
                <CodeEditor
                  allowedLanguages={["json"]}
                  value={(() => {
                    try {
                      return JSON.stringify(
                        JSON.parse(inference.raw_response),
                        null,
                        2,
                      );
                    } catch {
                      return inference.raw_response;
                    }
                  })()}
                  readOnly
                />
              </SnippetContent>
            </SnippetLayout>
          </SectionLayout>
        )}
      </SectionsGroup>
    </PageLayout>
  );
}
