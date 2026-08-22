// Modified by Delta-AI under Apache 2.0
import type { Route } from "./+types/route";
import { useEffect, useState } from "react";
import { data, useFetcher, type RouteHandle } from "react-router";
import {
  PageHeader,
  PageLayout,
  SectionLayout,
} from "~/components/layout/PageLayout";
import { ActionBar } from "~/components/layout/ActionBar";
import { Button } from "~/components/ui/button";
import { Input } from "~/components/ui/input";
import { Switch, SwitchSize } from "~/components/ui/switch";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
  TableEmptyState,
} from "~/components/ui/table";
import { Badge } from "~/components/ui/badge";
import { LayoutErrorBoundary } from "~/components/ui/error";
import { ReadOnlyGuard } from "~/components/utils/read-only-guard";
import { useReadOnly } from "~/context/read-only";
import { formatDate } from "~/utils/date";
import { logger } from "~/utils/logger";
import {
  getAzureLoginEmail,
  isAzureAuthEnabled,
} from "~/utils/azure-auth.server";
import { getTensorZeroClient } from "~/utils/get-tensorzero-client.server";
import type { DashboardUser } from "~/utils/tensorzero/tensorzero";
import { Trash } from "lucide-react";
import { useDashboardSession } from "~/context/dashboard-session";
import { TensorZeroServerError } from "~/utils/tensorzero/errors";

export const handle: RouteHandle = {
  crumb: () => ["Users"],
};

export async function loader({ request }: Route.LoaderArgs) {
  if (!isAzureAuthEnabled()) {
    throw data("Not found", { status: 404 });
  }

  const email = getAzureLoginEmail(request);
  if (!email) {
    throw data("Azure login email is required", { status: 403 });
  }

  const client = getTensorZeroClient();
  const session = await client.getDashboardSession(email);
  if (!session.is_admin) {
    throw data("Only dashboard admins can manage users", { status: 403 });
  }

  const users = await client.listDashboardUsers(email);
  return { users, actorEmail: email };
}

export async function action({ request }: Route.ActionArgs) {
  if (!isAzureAuthEnabled()) {
    throw data("Not found", { status: 404 });
  }

  const actorEmail = getAzureLoginEmail(request);
  if (!actorEmail) {
    return data({ error: "Azure login email is required" }, { status: 403 });
  }

  const formData = await request.formData();
  const actionType = formData.get("action");
  const client = getTensorZeroClient();

  try {
    if (actionType === "create") {
      const emailRaw = formData.get("email");
      if (typeof emailRaw !== "string" || !emailRaw.trim()) {
        return data({ error: "Email is required" }, { status: 400 });
      }
      const isAdmin = formData.get("is_admin") === "true";
      await client.createDashboardUser(actorEmail, {
        email: emailRaw.trim(),
        is_admin: isAdmin,
      });
      return { success: true };
    }

    if (actionType === "set_admin") {
      const emailRaw = formData.get("email");
      const isAdminRaw = formData.get("is_admin");
      if (typeof emailRaw !== "string" || !emailRaw.trim()) {
        return data({ error: "Email is required" }, { status: 400 });
      }
      if (typeof isAdminRaw !== "string") {
        return data({ error: "is_admin is required" }, { status: 400 });
      }
      await client.updateDashboardUser(actorEmail, {
        email: emailRaw.trim(),
        is_admin: isAdminRaw === "true",
      });
      return { success: true };
    }

    if (actionType === "delete") {
      const emailRaw = formData.get("email");
      if (typeof emailRaw !== "string" || !emailRaw.trim()) {
        return data({ error: "Email is required" }, { status: 400 });
      }
      await client.deleteDashboardUser(actorEmail, emailRaw.trim());
      return { success: true };
    }

    return data({ error: "Invalid action" }, { status: 400 });
  } catch (error) {
    logger.error("Dashboard user action failed", error);
    const message =
      error instanceof TensorZeroServerError
        ? error.message
        : "Failed to update dashboard users. Please try again.";
    return data({ error: message }, { status: 400 });
  }
}

function AddUserForm({ error }: { error?: string }) {
  const fetcher = useFetcher<typeof action>();
  const isReadOnly = useReadOnly();
  const busy = fetcher.state !== "idle";
  const [isAdmin, setIsAdmin] = useState(false);
  const [formKey, setFormKey] = useState(0);

  useEffect(() => {
    if (fetcher.state === "idle" && fetcher.data && "success" in fetcher.data) {
      setIsAdmin(false);
      setFormKey((key) => key + 1);
    }
  }, [fetcher.state, fetcher.data]);

  const actionError =
    error ??
    (fetcher.data && "error" in fetcher.data ? fetcher.data.error : undefined);

  return (
    <fetcher.Form
      key={formKey}
      method="post"
      className="flex flex-wrap items-center gap-3"
    >
      <input type="hidden" name="action" value="create" />
      <input type="hidden" name="is_admin" value={isAdmin ? "true" : "false"} />
      <Input
        name="email"
        type="email"
        required
        placeholder="user@example.com"
        className="w-72"
        disabled={busy}
      />
      <label className="flex items-center gap-2 text-sm">
        <Switch
          size={SwitchSize.Small}
          checked={isAdmin}
          onCheckedChange={setIsAdmin}
          disabled={busy || isReadOnly}
        />
        Admin
      </label>
      <ReadOnlyGuard>
        <Button type="submit" disabled={busy}>
          Add user
        </Button>
      </ReadOnlyGuard>
      {actionError && (
        <span className="text-destructive text-sm">{actionError}</span>
      )}
    </fetcher.Form>
  );
}

function UserRow({
  user,
  actorEmail,
  adminCount,
}: {
  user: DashboardUser;
  actorEmail: string;
  adminCount: number;
}) {
  const setAdminFetcher = useFetcher<typeof action>();
  const deleteFetcher = useFetcher<typeof action>();
  const isReadOnly = useReadOnly();
  const lastAdmin = user.is_admin && adminCount <= 1;
  const busy =
    setAdminFetcher.state !== "idle" || deleteFetcher.state !== "idle";
  const isYou = user.email === actorEmail.toLowerCase();

  return (
    <TableRow>
      <TableCell className="font-mono text-sm">
        {user.email}
        {isYou && (
          <Badge variant="secondary" className="ml-2">
            You
          </Badge>
        )}
      </TableCell>
      <TableCell>
        <Switch
          size={SwitchSize.Small}
          checked={user.is_admin}
          disabled={busy || lastAdmin || isReadOnly}
          onCheckedChange={(checked) => {
            setAdminFetcher.submit(
              {
                action: "set_admin",
                email: user.email,
                is_admin: checked ? "true" : "false",
              },
              { method: "post" },
            );
          }}
        />
      </TableCell>
      <TableCell className="text-muted-foreground whitespace-nowrap">
        {formatDate(new Date(user.created_at))}
      </TableCell>
      <TableCell>
        <deleteFetcher.Form method="post">
          <input type="hidden" name="action" value="delete" />
          <input type="hidden" name="email" value={user.email} />
          <ReadOnlyGuard>
            <Button
              type="submit"
              variant="ghost"
              size="iconSm"
              disabled={busy || lastAdmin}
              aria-label={`Remove ${user.email}`}
            >
              <Trash className="h-4 w-4" />
            </Button>
          </ReadOnlyGuard>
        </deleteFetcher.Form>
      </TableCell>
    </TableRow>
  );
}

export default function UsersPage({ loaderData }: Route.ComponentProps) {
  const { users, actorEmail } = loaderData;
  const session = useDashboardSession();
  const adminCount = users.filter((user) => user.is_admin).length;

  return (
    <PageLayout>
      <PageHeader heading="Users" count={users.length}>
        <p className="text-muted-foreground text-sm">
          Azure accounts that can open this dashboard. Admins can add emails and
          grant admin access.
        </p>
      </PageHeader>
      <SectionLayout>
        <ActionBar>
          <AddUserForm />
        </ActionBar>
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Email</TableHead>
              <TableHead className="w-0 whitespace-nowrap">Admin</TableHead>
              <TableHead className="w-0 whitespace-nowrap">Added</TableHead>
              <TableHead className="w-0"></TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {users.length === 0 ? (
              <TableEmptyState message="No dashboard users yet. Add an email to grant access." />
            ) : (
              users.map((user) => (
                <UserRow
                  key={user.email}
                  user={user}
                  actorEmail={session.email ?? actorEmail}
                  adminCount={adminCount}
                />
              ))
            )}
          </TableBody>
        </Table>
      </SectionLayout>
    </PageLayout>
  );
}

export function ErrorBoundary({ error }: Route.ErrorBoundaryProps) {
  return <LayoutErrorBoundary error={error} />;
}
