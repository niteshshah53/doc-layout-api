import { describe, expect, it } from "vitest";
import { MAX_FILE_SIZE_BYTES, validateImageFile } from "./fileValidation";

describe("validateImageFile", () => {
  it("accepts supported images under the size limit", () => {
    expect(validateImageFile(new File(["image"], "page.png", { type: "image/png" }))).toBeNull();
  });
  it("rejects unsupported types and oversized files", () => {
    expect(validateImageFile(new File(["text"], "page.pdf", { type: "application/pdf" }))).toContain("PNG");
    const large = new File(["x"], "large.png", { type: "image/png" });
    Object.defineProperty(large, "size", { value: MAX_FILE_SIZE_BYTES + 1 });
    expect(validateImageFile(large)).toContain("10 MB");
  });
});
