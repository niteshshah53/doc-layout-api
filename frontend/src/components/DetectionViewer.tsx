import { useEffect, useMemo, useRef, useState } from "react";
import type { LayoutBlock, PredictionResponse } from "../types";
import { scaleBoundingBox } from "../utils/imageGeometry";

interface DetectionViewerProps {
  imageUrl: string;
  prediction: PredictionResponse;
  activeIndex: number | null;
  onActiveChange: (index: number | null) => void;
  selectedIndex: number | null;
  onSelect: (index: number) => void;
  visibleBlocks: { block: LayoutBlock; index: number }[];
  showBoxes: boolean;
  showLabels: boolean;
}

export function DetectionViewer({
  imageUrl,
  prediction,
  activeIndex,
  onActiveChange,
  selectedIndex,
  onSelect,
  visibleBlocks,
  showBoxes,
  showLabels,
}: DetectionViewerProps) {
  const imageRef = useRef<HTMLImageElement>(null);
  const shellRef = useRef<HTMLDivElement>(null);
  const [naturalSize, setNaturalSize] = useState({ width: 0, height: 0 });
  const [containerSize, setContainerSize] = useState({ width: 0, height: 0 });
  const [zoom, setZoom] = useState(1);

  useEffect(() => {
    const shell = shellRef.current;
    if (!shell) return;
    const updateSize = () => setContainerSize({
      width: shell.clientWidth,
      height: shell.clientHeight,
    });
    updateSize();
    const observer = new ResizeObserver(updateSize);
    observer.observe(shell);
    return () => observer.disconnect();
  }, [imageUrl]);

  const fitScale = useMemo(() => {
    if (!naturalSize.width || !naturalSize.height || !containerSize.width) return 1;
    const widthScale = containerSize.width / naturalSize.width;
    const heightScale = containerSize.height
      ? containerSize.height / naturalSize.height
      : widthScale;
    return Math.min(widthScale, heightScale, 1);
  }, [containerSize, naturalSize]);
  const renderScale = Math.max(fitScale, 0.1) * zoom;
  const renderedSize = {
    width: naturalSize.width * renderScale,
    height: naturalSize.height * renderScale,
  };
  const labelsForAll = visibleBlocks.length <= 10;

  return (
    <div className="viewer-shell" ref={shellRef}>
      <div className="viewer-controls" aria-label="Document zoom controls">
        <button type="button" onClick={() => setZoom((value) => Math.max(0.75, value - 0.25))} disabled={zoom <= 0.75} aria-label="Zoom out">−</button>
        <span>{Math.round(zoom * 100)}%</span>
        <button type="button" onClick={() => setZoom((value) => Math.min(3, value + 0.25))} disabled={zoom >= 3} aria-label="Zoom in">+</button>
        <button type="button" onClick={() => setZoom(1)} className={zoom === 1 ? "is-fit" : ""}>Fit</button>
      </div>
      <div className="image-stage" style={{ width: renderedSize.width || undefined, height: renderedSize.height || undefined }}>
        <img
          ref={imageRef}
          src={imageUrl}
          style={{ width: renderedSize.width || undefined, height: renderedSize.height || undefined }}
          alt="Uploaded document with detected layout regions"
          onLoad={() =>
            setNaturalSize({
              width: imageRef.current?.naturalWidth ?? 0,
              height: imageRef.current?.naturalHeight ?? 0,
            })
          }
        />
        {showBoxes && renderedSize.width > 0 &&
          visibleBlocks.map(({ block, index }) => (
            <DetectionBox
              key={`${block.label}-${index}`}
              block={block}
              index={index}
              prediction={prediction}
              displayedSize={renderedSize}
              active={activeIndex === index || selectedIndex === index}
              onActiveChange={onActiveChange}
              showLabel={showLabels && (labelsForAll || activeIndex === index || selectedIndex === index)}
              onSelect={() => onSelect(index)}
            />
          ))}
      </div>
    </div>
  );
}

function DetectionBox({
  block,
  index,
  prediction,
  displayedSize,
  active,
  onActiveChange,
  showLabel,
  onSelect,
}: {
  block: LayoutBlock;
  index: number;
  prediction: PredictionResponse;
  displayedSize: { width: number; height: number };
  active: boolean;
  onActiveChange: (index: number | null) => void;
  showLabel: boolean;
  onSelect: () => void;
}) {
  const box = scaleBoundingBox(block.bbox, prediction.image_size, displayedSize);
  return (
    <button
      className={`detection-box ${active ? "is-active" : ""}`}
      style={{ left: box.left, top: box.top, width: box.width, height: box.height }}
      aria-label={`${block.label}, ${Math.round(block.score * 100)} percent confidence`}
      onMouseEnter={() => onActiveChange(index)}
      onMouseLeave={() => onActiveChange(null)}
      onFocus={() => onActiveChange(index)}
      onBlur={() => onActiveChange(null)}
      onClick={onSelect}
    >
      {showLabel && <span>{block.label}</span>}
      {showLabel && <strong>{Math.round(block.score * 100)}%</strong>}
    </button>
  );
}
