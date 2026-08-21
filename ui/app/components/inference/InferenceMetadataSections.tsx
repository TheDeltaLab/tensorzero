// Modified by Delta-AI under Apache 2.0
import { TagsTable } from "~/components/tags/TagsTable";
import { SectionHeader, SectionLayout } from "~/components/layout/PageLayout";
import { splitInferenceTags } from "~/utils/observability/inferenceTags";

export function InferenceMetadataSections({
  tags,
}: {
  tags: Record<string, string | undefined>;
}) {
  const defined = Object.fromEntries(
    Object.entries(tags).filter(
      (entry): entry is [string, string] => entry[1] !== undefined,
    ),
  );
  const { headers, userTags } = splitInferenceTags(defined);
  return (
    <>
      <SectionLayout>
        <SectionHeader heading="Headers" />
        <TagsTable
          tags={headers}
          isEditing={false}
          emptyMessage="No request headers"
        />
      </SectionLayout>
      <SectionLayout>
        <SectionHeader heading="Tags" />
        <TagsTable tags={userTags} isEditing={false} emptyMessage="No tags" />
      </SectionLayout>
    </>
  );
}
