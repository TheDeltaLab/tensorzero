// Modified by Delta-AI under Apache 2.0
import type { Meta, StoryObj } from "@storybook/react-vite";
import { Breadcrumbs } from "./Breadcrumbs";

const meta = {
  title: "Layout/Breadcrumbs",
  component: Breadcrumbs,
  decorators: [
    (Story) => (
      <div className="p-4">
        <Story />
      </div>
    ),
  ],
} satisfies Meta<typeof Breadcrumbs>;

export default meta;
type Story = StoryObj<typeof meta>;

export const SingleSegment: Story = {
  args: {
    segments: [{ label: "Functions", href: "/observability/functions" }],
  },
};

export const TwoSegments: Story = {
  args: {
    segments: [
      { label: "Functions", href: "/observability/functions" },
      {
        label: "extract_user_info",
        href: "/observability/functions/extract_user_info",
        isIdentifier: true,
      },
    ],
  },
};

export const WithNonClickableSegment: Story = {
  args: {
    segments: [
      { label: "Functions", href: "/observability/functions" },
      {
        label: "extract_user_info",
        href: "/observability/functions/extract_user_info",
        isIdentifier: true,
      },
      { label: "Variants" },
    ],
  },
};

export const FunctionVariant: Story = {
  args: {
    segments: [
      { label: "Functions", href: "/observability/functions" },
      {
        label: "extract_user_info",
        href: "/observability/functions/extract_user_info",
        isIdentifier: true,
      },
      { label: "Variants" },
    ],
  },
};

export const EpisodeDetail: Story = {
  args: {
    segments: [
      { label: "Episodes", href: "/observability/episodes" },
      {
        label: "01926d96-72e1-7000-8be0-8990c7e878f8",
        href: "/observability/episodes/01926d96-72e1-7000-8be0-8990c7e878f8",
        isIdentifier: true,
      },
    ],
  },
};
