// Modified by Delta-AI under Apache 2.0
import { describe, it, expect } from "vitest";
import {
  toFunctionUrl,
  toVariantUrl,
  toInferenceUrl,
  toInferencesListUrl,
  toEpisodeUrl,
} from "./urls";

describe("URL helper functions", () => {
  describe("toFunctionUrl", () => {
    it("should encode function names with special characters", () => {
      expect(toFunctionUrl("my_function")).toBe(
        "/observability/functions/my_function",
      );
      expect(toFunctionUrl("my_function/abc")).toBe(
        "/observability/functions/my_function%2Fabc",
      );
      expect(toFunctionUrl("func#test")).toBe(
        "/observability/functions/func%23test",
      );
      expect(toFunctionUrl("func?query")).toBe(
        "/observability/functions/func%3Fquery",
      );
    });

    it("should append snapshot_hash when provided", () => {
      expect(toFunctionUrl("my_function", "abc123")).toBe(
        "/observability/functions/my_function?snapshot_hash=abc123",
      );
    });
  });

  describe("toVariantUrl", () => {
    it("should encode both function and variant names", () => {
      expect(toVariantUrl("my_function", "variant_1")).toBe(
        "/observability/functions/my_function/variants/variant_1",
      );
      expect(toVariantUrl("func/abc", "var/xyz")).toBe(
        "/observability/functions/func%2Fabc/variants/var%2Fxyz",
      );
    });

    it("should append snapshot_hash when provided", () => {
      expect(toVariantUrl("my_function", "variant_1", "abc123")).toBe(
        "/observability/functions/my_function/variants/variant_1?snapshot_hash=abc123",
      );
    });

    it("should encode snapshot_hash with special characters", () => {
      expect(
        toVariantUrl("my_function", "variant_1", "hash/with#special"),
      ).toBe(
        "/observability/functions/my_function/variants/variant_1?snapshot_hash=hash%2Fwith%23special",
      );
    });
  });

  describe("toInferenceUrl", () => {
    it("should encode inference IDs", () => {
      expect(toInferenceUrl("123")).toBe("/observability/inferences/123");
      expect(toInferenceUrl("id/with/slashes")).toBe(
        "/observability/inferences/id%2Fwith%2Fslashes",
      );
    });
  });

  describe("toInferencesListUrl", () => {
    it("should add api_key when provided", () => {
      expect(toInferencesListUrl()).toBe("/observability/inferences");
      expect(toInferencesListUrl({ api_key: "synabcdefghi" })).toBe(
        "/observability/inferences?api_key=synabcdefghi",
      );
    });
  });

  describe("toEpisodeUrl", () => {
    it("should encode episode IDs", () => {
      expect(toEpisodeUrl("456")).toBe("/observability/episodes/456");
      expect(toEpisodeUrl("ep#123")).toBe("/observability/episodes/ep%23123");
    });
  });
});
