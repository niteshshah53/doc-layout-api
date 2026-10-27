import type { BoundingBox } from "../types";

export interface DisplayedBox {
  left: number;
  top: number;
  width: number;
  height: number;
}

export function scaleBoundingBox(
  bbox: BoundingBox,
  original: { width: number; height: number },
  displayed: { width: number; height: number },
): DisplayedBox {
  const scaleX = displayed.width / original.width;
  const scaleY = displayed.height / original.height;
  return {
    left: bbox.x1 * scaleX,
    top: bbox.y1 * scaleY,
    width: (bbox.x2 - bbox.x1) * scaleX,
    height: (bbox.y2 - bbox.y1) * scaleY,
  };
}
