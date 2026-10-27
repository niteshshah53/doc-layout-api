import { describe, expect, it } from "vitest";
import { scaleBoundingBox } from "./imageGeometry";

describe("scaleBoundingBox", () => {
  it("maps original pixel coordinates to a resized display", () => {
    expect(scaleBoundingBox(
      { x1: 100, y1: 200, x2: 500, y2: 800 },
      { width: 1000, height: 1600 },
      { width: 500, height: 800 },
    )).toEqual({ left: 50, top: 100, width: 200, height: 300 });
  });

  it("preserves geometry for portrait and landscape displays", () => {
    expect(scaleBoundingBox(
      { x1: 50, y1: 100, x2: 450, y2: 700 },
      { width: 500, height: 1000 },
      { width: 250, height: 500 },
    )).toEqual({ left: 25, top: 50, width: 200, height: 300 });
    expect(scaleBoundingBox(
      { x1: 100, y1: 50, x2: 900, y2: 450 },
      { width: 1000, height: 500 },
      { width: 1500, height: 750 },
    )).toEqual({ left: 150, top: 75, width: 1200, height: 600 });
  });
});
