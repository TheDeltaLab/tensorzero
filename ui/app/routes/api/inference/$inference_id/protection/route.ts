// Modified by Delta-AI under Apache 2.0
import { data, type ActionFunctionArgs } from "react-router";
import { getTensorZeroClient } from "~/utils/tensorzero.server";
import { isTensorZeroServerError } from "~/utils/tensorzero";
import { isReadOnlyMode } from "~/utils/read-only.server";
import { logger } from "~/utils/logger";

export type ProtectionActionData =
  | {
      success: true;
      protected: boolean;
      protected_at?: string;
      error?: never;
    }
  | { success?: never; error: string };

/**
 * Dedicated API route for protecting an inference from retention cleanup.
 *
 * Expected form data:
 * - protected: "true" | "false"
 */
export async function action({ request, params }: ActionFunctionArgs) {
  if (isReadOnlyMode()) {
    return data<ProtectionActionData>(
      { error: "Inference protection cannot be changed in read-only mode." },
      { status: 403 },
    );
  }

  const { inference_id } = params;
  if (!inference_id) {
    return data<ProtectionActionData>(
      { error: "Inference ID is required" },
      { status: 400 },
    );
  }

  const formData = await request.formData();
  const protectedRaw = formData.get("protected");
  if (protectedRaw !== "true" && protectedRaw !== "false") {
    return data<ProtectionActionData>(
      { error: "`protected` must be `true` or `false`" },
      { status: 400 },
    );
  }
  const protect = protectedRaw === "true";

  try {
    const response = await getTensorZeroClient().setInferenceProtection(
      inference_id,
      protect,
    );
    return data<ProtectionActionData>({
      success: true,
      protected: protect,
      protected_at: response.protected_at,
    });
  } catch (error) {
    if (isTensorZeroServerError(error)) {
      return data<ProtectionActionData>(
        { error: error.message },
        { status: error.status },
      );
    }
    logger.error("Failed to set inference protection:", error);
    return data<ProtectionActionData>(
      { error: "Unknown server error. Try again." },
      { status: 500 },
    );
  }
}
