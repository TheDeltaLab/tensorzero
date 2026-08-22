// Modified by Delta-AI under Apache 2.0
import { getEnv } from "./env.server";

const EMAIL_HEADERS = [
  "x-auth-request-email",
  "x-forwarded-email",
  "x-auth-request-preferred-username",
];

export function isAzureAuthEnabled(): boolean {
  return getEnv().TENSORZERO_UI_AZURE_AUTH;
}

export function getAzureLogoutUrl(): string {
  return getEnv().TENSORZERO_UI_AZURE_LOGOUT_URL;
}

export function getAzureLoginEmail(request: Request): string | null {
  for (const name of EMAIL_HEADERS) {
    const value = request.headers.get(name);
    if (value?.trim()) {
      return value.trim();
    }
  }
  return null;
}

export function dashboardEmailHeaders(
  email: string | null,
): Record<string, string> | undefined {
  if (!email) return undefined;
  return { "X-Auth-Request-Email": email };
}
