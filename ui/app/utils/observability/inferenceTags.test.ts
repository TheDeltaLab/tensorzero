// Modified by Delta-AI under Apache 2.0
import { describe, expect, test } from "vitest";
import {
  formatCsvTags,
  parseCsvTags,
  splitInferenceTags,
} from "./inferenceTags";

describe("inferenceTags", () => {
  test("splits headers from user tags and hides system tags", () => {
    const split = splitInferenceTags({
      env: "prod",
      "tensorzero::header::x-tensorzero-request-id": "req-1",
      "tensorzero::request_headers": '{"x-tensorzero-request-id":"req-1"}',
      "tensorzero::provider": "dummy",
      "tensorzero::cost": "0.01",
    });
    expect(split.headers).toEqual({ "x-tensorzero-request-id": "req-1" });
    expect(split.userTags).toEqual({ env: "prod" });
  });

  test("parses comma-separated tags like the request header", () => {
    expect(parseCsvTags("env=prod, team=ml, canary")).toEqual({
      env: "prod",
      team: "ml",
      canary: "true",
    });
    expect(formatCsvTags({ env: "prod", canary: "true" })).toBe(
      "canary,env=prod",
    );
  });
});
