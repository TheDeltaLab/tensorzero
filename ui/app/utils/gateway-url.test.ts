import { describe, expect, it } from "vitest";
import { buildGatewayUrl } from "./gateway-url";

describe("buildGatewayUrl", () => {
  it("preserves a configured base path without a trailing slash", () => {
    expect(
      buildGatewayUrl(
        "http://tensorzero-gateway:3000/tensorzero/api/v1",
        "/internal/models/usage",
      ).toString(),
    ).toBe(
      "http://tensorzero-gateway:3000/tensorzero/api/v1/internal/models/usage",
    );
  });

  it("preserves a configured base path with a trailing slash", () => {
    expect(
      buildGatewayUrl(
        "http://tensorzero-gateway:3000/tensorzero/api/v1/",
        "/internal/models/usage",
      ).toString(),
    ).toBe(
      "http://tensorzero-gateway:3000/tensorzero/api/v1/internal/models/usage",
    );
  });

  it("works when the provided path omits the leading slash", () => {
    expect(
      buildGatewayUrl(
        "http://tensorzero-gateway:3000/tensorzero/api/v1/",
        "internal/ui_config",
      ).toString(),
    ).toBe(
      "http://tensorzero-gateway:3000/tensorzero/api/v1/internal/ui_config",
    );
  });

  it("preserves query strings on the provided path", () => {
    expect(
      buildGatewayUrl(
        "http://tensorzero-gateway:3000/tensorzero/api/v1/",
        "/internal/models/usage?time_window=day&max_periods=30",
      ).toString(),
    ).toBe(
      "http://tensorzero-gateway:3000/tensorzero/api/v1/internal/models/usage?time_window=day&max_periods=30",
    );
  });

  it("still works for a gateway configured at the origin root", () => {
    expect(
      buildGatewayUrl(
        "http://tensorzero-gateway:3000/",
        "/internal/ui_config",
      ).toString(),
    ).toBe("http://tensorzero-gateway:3000/internal/ui_config");
  });
});
