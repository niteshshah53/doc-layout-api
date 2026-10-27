import type { LayoutBlock } from "../types";

export function DetectionList({
  blocks,
  visibleIndices,
  activeIndex,
  onActiveChange,
  onSelect,
}: {
  blocks: LayoutBlock[];
  visibleIndices: number[];
  activeIndex: number | null;
  onActiveChange: (index: number | null) => void;
  onSelect: (index: number) => void;
}) {
  return (
    <section className="detection-panel" aria-labelledby="detections-heading">
      <div className="panel-heading">
        <div>
          <p className="eyebrow">Model output</p>
          <h2 id="detections-heading">Detected regions</h2>
        </div>
        <span className="count-pill">{visibleIndices.length}</span>
      </div>
      <div className="detection-list">
        {visibleIndices.length === 0 && (
          <p className="empty-result">No regions match the selected labels.</p>
        )}
        {visibleIndices.map((index) => {
          const block = blocks[index];
          return (
          <button
            className={`detection-row ${activeIndex === index ? "is-active" : ""}`}
            key={`${block.label}-${index}`}
            onMouseEnter={() => onActiveChange(index)}
            onMouseLeave={() => onActiveChange(null)}
            onFocus={() => onActiveChange(index)}
            onBlur={() => onActiveChange(null)}
            onClick={() => onSelect(index)}
          >
            <span className="region-index">{String(index + 1).padStart(2, "0")}</span>
            <span className="region-copy">
              <strong>{block.label}</strong>
              <small>
                x {Math.round(block.bbox.x1)}–{Math.round(block.bbox.x2)} · y{" "}
                {Math.round(block.bbox.y1)}–{Math.round(block.bbox.y2)}
              </small>
            </span>
            <span className="confidence">{Math.round(block.score * 100)}%</span>
          </button>
          );
        })}
      </div>
    </section>
  );
}
