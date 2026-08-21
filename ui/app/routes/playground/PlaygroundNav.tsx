// Modified by Delta-AI under Apache 2.0
import { Link } from "react-router";

const items = [
  { href: "/playground", label: "Chat" },
  { href: "/playground/embeddings", label: "Embeddings" },
  { href: "/playground/rerank", label: "Rerank" },
  { href: "/playground/functions", label: "Functions" },
] as const;

export function PlaygroundNav({ current }: { current: string }) {
  return (
    <nav className="flex flex-wrap gap-2">
      {items.map((item) => (
        <Link
          key={item.href}
          to={item.href}
          className={
            current === item.href
              ? "bg-bg-hover rounded-md px-3 py-1 text-sm font-medium"
              : "hover:bg-bg-hover rounded-md px-3 py-1 text-sm"
          }
        >
          {item.label}
        </Link>
      ))}
    </nav>
  );
}
