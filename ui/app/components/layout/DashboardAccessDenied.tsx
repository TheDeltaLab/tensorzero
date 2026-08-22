// Modified by Delta-AI under Apache 2.0
import { LogOut } from "lucide-react";
import { Button } from "~/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "~/components/ui/card";

export function DashboardAccessDenied({
  email,
  logoutUrl,
}: {
  email: string | null;
  logoutUrl: string;
}) {
  return (
    <div className="bg-background flex min-h-screen items-center justify-center p-6">
      <Card className="max-w-lg">
        <CardHeader>
          <CardTitle>Access denied</CardTitle>
        </CardHeader>
        <CardContent className="flex flex-col gap-4">
          <p className="text-muted-foreground text-sm">
            {email ? (
              <>
                Azure account{" "}
                <code className="bg-muted rounded px-1 py-0.5 font-mono text-xs">
                  {email}
                </code>{" "}
                is not on the dashboard allowlist. Ask an admin to add this
                email on the Users page.
              </>
            ) : (
              <>
                This dashboard requires an Azure login email, but none was
                forwarded. Confirm oauth2-proxy is setting{" "}
                <code className="bg-muted rounded px-1 py-0.5 font-mono text-xs">
                  X-Auth-Request-Email
                </code>
                .
              </>
            )}
          </p>
          <Button asChild variant="outline" className="w-fit">
            <a href={logoutUrl}>
              <LogOut className="h-4 w-4" />
              Sign out
            </a>
          </Button>
        </CardContent>
      </Card>
    </div>
  );
}
