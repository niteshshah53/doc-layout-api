import { afterEach, describe, expect, it, vi } from "vitest";
import { getHealth, predictLayout } from "./layoutApi";

afterEach(() => vi.restoreAllMocks());

describe("predictLayout", () => {
  it("posts the selected file and returns the API response", async () => {
    const response = { num_blocks: 0, image_size: { width: 10, height: 20 }, inference_time_ms: 4, blocks: [] };
    const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(JSON.stringify(response), { status: 200 }),
    );
    const result = await predictLayout(new File(["x"], "page.png", { type: "image/png" }));
    expect(result).toEqual(response);
    expect(fetchMock).toHaveBeenCalledWith(
      "http://localhost:8000/api/v1/predict",
      expect.objectContaining({ method: "POST", body: expect.any(FormData) }),
    );
  });
  it("turns backend errors into useful messages", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(JSON.stringify({ detail: "Model is loading" }), { status: 503 }),
    );
    await expect(predictLayout(new File(["x"], "page.png", { type: "image/png" })))
      .rejects.toThrow("unavailable or still loading");
  });
  it("reads the genuine backend model status", async () => {
    const health = {
      status: "ok",
      model_loaded: true,
      model_info: { type: "docling-heron", error: null },
      version: "1.0.0",
    };
    const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(JSON.stringify(health), { status: 200 }),
    );
    await expect(getHealth()).resolves.toEqual(health);
    expect(fetchMock).toHaveBeenCalledWith("http://localhost:8000/api/v1/health");
  });
});
