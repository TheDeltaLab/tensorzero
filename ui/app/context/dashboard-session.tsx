"use client";
// Modified by Delta-AI under Apache 2.0

import { createContext, useContext } from "react";

export interface DashboardSession {
  enabled: boolean;
  allowed: boolean;
  email: string | null;
  is_admin: boolean;
  logoutUrl: string;
}

export const DISABLED_DASHBOARD_SESSION: DashboardSession = {
  enabled: false,
  allowed: true,
  email: null,
  is_admin: false,
  logoutUrl: "/oauth2/sign_out",
};

const DashboardSessionContext = createContext<DashboardSession>(
  DISABLED_DASHBOARD_SESSION,
);
DashboardSessionContext.displayName = "DashboardSessionContext";

export function useDashboardSession(): DashboardSession {
  return useContext(DashboardSessionContext);
}

export function DashboardSessionProvider({
  children,
  value,
}: {
  children: React.ReactNode;
  value: DashboardSession;
}) {
  return (
    <DashboardSessionContext.Provider value={value}>
      {children}
    </DashboardSessionContext.Provider>
  );
}
