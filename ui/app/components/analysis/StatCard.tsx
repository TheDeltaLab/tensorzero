// Modified by Delta-AI under Apache 2.0
import type { LucideIcon } from "lucide-react";
import { Card, CardContent } from "~/components/ui/card";

export function StatCard({
  title,
  value,
  description,
  icon: Icon,
}: {
  title: string;
  value: string;
  description?: string;
  icon: LucideIcon;
}) {
  return (
    <Card>
      <CardContent className="pt-6">
        <div className="flex items-center justify-between">
          <p className="text-muted-foreground text-sm font-medium">{title}</p>
          <Icon className="text-muted-foreground h-4 w-4" />
        </div>
        <p className="mt-2 text-3xl font-bold tracking-tight">{value}</p>
        {description ? (
          <p className="text-muted-foreground mt-1 text-xs">{description}</p>
        ) : null}
      </CardContent>
    </Card>
  );
}
