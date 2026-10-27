import { type DragEvent, type RefObject, useEffect, useMemo, useRef, useState } from "react";
import { getHealth, predictLayout } from "./api/layoutApi";
import { DetectionList } from "./components/DetectionList";
import { DetectionViewer } from "./components/DetectionViewer";
import type { HealthResponse, PredictionResponse } from "./types";
import { validateImageFile } from "./utils/fileValidation";

const accept = "image/jpeg,image/png,image/tiff,image/bmp,image/webp";

export default function App() {
  const inputRef = useRef<HTMLInputElement>(null);
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState("");
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [health, setHealth] = useState<HealthResponse | null | undefined>(undefined);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [dragging, setDragging] = useState(false);
  const [activeIndex, setActiveIndex] = useState<number | null>(null);
  const [selectedIndex, setSelectedIndex] = useState<number | null>(null);
  const [selectedLabels, setSelectedLabels] = useState<string[]>([]);
  const [showBoxes, setShowBoxes] = useState(true);
  const [showLabels, setShowLabels] = useState(true);
  const [sidebarOpen, setSidebarOpen] = useState(true);

  useEffect(() => {
    getHealth().then(setHealth).catch(() => setHealth(null));
  }, []);
  useEffect(() => () => { if (previewUrl) URL.revokeObjectURL(previewUrl); }, [previewUrl]);

  const labels = useMemo(
    () => prediction ? [...new Set(prediction.blocks.map((block) => block.label))].sort() : [],
    [prediction],
  );
  const visibleIndices = useMemo(
    () => prediction
      ? prediction.blocks.map((block, index) => selectedLabels.length === 0 || selectedLabels.includes(block.label) ? index : -1).filter((index) => index >= 0)
      : [],
    [prediction, selectedLabels],
  );
  const visibleBlocks = useMemo(
    () => visibleIndices.map((index) => ({ block: prediction!.blocks[index], index })),
    [prediction, visibleIndices],
  );

  function chooseFile(nextFile: File | undefined) {
    if (!nextFile) return;
    const validationError = validateImageFile(nextFile);
    setError(validationError ?? "");
    setPrediction(null);
    setActiveIndex(null);
    setSelectedIndex(null);
    if (validationError) return;
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setFile(nextFile);
    setPreviewUrl(URL.createObjectURL(nextFile));
  }
  function onDrop(event: DragEvent<HTMLDivElement>) {
    event.preventDefault();
    setDragging(false);
    chooseFile(event.dataTransfer.files[0]);
  }
  async function submit() {
    if (!file) return;
    setLoading(true);
    setError("");
    try {
      const result = await predictLayout(file);
      setPrediction(result);
      setSelectedLabels([]);
      setSelectedIndex(null);
      getHealth().then(setHealth).catch(() => undefined);
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "The request failed.");
    } finally { setLoading(false); }
  }
  function reset() {
    setFile(null); setPrediction(null); setError(""); setActiveIndex(null); setSelectedIndex(null);
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setPreviewUrl("");
    if (inputRef.current) inputRef.current.value = "";
  }
  function toggleLabel(label: string) {
    setSelectedLabels((current) => current.includes(label) ? current.filter((item) => item !== label) : [...current, label]);
  }

  const statusText = health === undefined ? "Checking backend…" : health?.model_loaded ? "Heron model online" : "Backend unavailable";

  return (
    <main className="app-shell">
      <header className="topbar">
        <div className="brand-mark">H</div>
        <div><p className="brand-kicker">Docling Heron · RT-DETR v2</p><h1>Layout Vision</h1></div>
        <div className={`status-chip ${health === undefined ? "checking" : health?.model_loaded ? "online" : "offline"}`} role="status">
          <i /> {statusText}
        </div>
      </header>

      {!file && (
        <section className="landing">
          <div className="landing-copy">
            <p className="eyebrow">Document intelligence, made visible</p>
            <h2>See the structure<br /><em>inside every page.</em></h2>
            <p>Explore genuine document-layout predictions from Docling Heron. Upload a research paper, report, or scanned page to begin.</p>
          </div>
          <div className="upload-card landing-upload">
            <UploadZone inputRef={inputRef} dragging={dragging} onDrop={onDrop} setDragging={setDragging} chooseFile={chooseFile} />
            {error && <div className="alert error" role="alert">{error}</div>}
          </div>
        </section>
      )}

      {file && (
        <>
          <section className="workspace-header">
            <div><p className="eyebrow">Document workspace</p><h2>{file.name}</h2><span className="file-meta">{(file.size / 1024 / 1024).toFixed(2)} MB · Heron layout analysis</span></div>
            <div className="workspace-actions"><button className="secondary-button" onClick={reset}>New document</button><button className="primary-button" disabled={loading || Boolean(error)} onClick={submit}>{loading ? <><span className="spinner" /> Analyzing</> : prediction ? "Run again" : "Analyze document"} <span>→</span></button></div>
          </section>
          {error && <div className="alert error workspace-alert" role="alert">{error}</div>}
          {!prediction && <div className="preflight"><img src={previewUrl} alt="Selected document preview" /><div><p className="eyebrow">Ready for inference</p><h3>Analyze this document page</h3><p>Heron will return text, heading, table, figure, caption, and other layout regions.</p><button className="primary-button" onClick={submit} disabled={loading}>{loading ? "Running…" : "Start analysis →"}</button></div></div>}
          {prediction && (
            <section className={`analysis-workspace ${sidebarOpen ? "" : "sidebar-collapsed"}`}>
              <div className="canvas-panel">
                <div className="canvas-toolbar"><div><p className="eyebrow">03 / Layout map</p><span className="canvas-note">{prediction.image_size.width} × {prediction.image_size.height}px original</span></div><div className="toolbar-actions"><button className={showBoxes ? "tool-button active" : "tool-button"} onClick={() => setShowBoxes((value) => !value)} aria-pressed={showBoxes}>▣ Boxes</button><button className={showLabels ? "tool-button active" : "tool-button"} onClick={() => setShowLabels((value) => !value)} aria-pressed={showLabels}>Aa Labels</button><button className="tool-button" onClick={() => setSidebarOpen((value) => !value)}>{sidebarOpen ? "◧ Hide panel" : "◨ Show panel"}</button></div></div>
                <DetectionViewer imageUrl={previewUrl} prediction={prediction} activeIndex={activeIndex} onActiveChange={setActiveIndex} selectedIndex={selectedIndex} onSelect={setSelectedIndex} visibleBlocks={visibleBlocks} showBoxes={showBoxes} showLabels={showLabels} />
                {!showBoxes && <div className="canvas-hint">Boxes hidden · source image view</div>}
              </div>
              {sidebarOpen && <aside className="inspector">
                <div className="metrics"><Metric value={String(prediction.num_blocks)} label="regions" /><Metric value={String(labels.length)} label="labels" /><Metric value={`${prediction.inference_time_ms.toFixed(0)}ms`} label="inference" /></div>
                <div className="inspector-section"><div className="section-title"><span>Filter labels</span><button onClick={() => setSelectedLabels([])} disabled={selectedLabels.length === 0}>Clear</button></div><div className="legend">{labels.map((label) => <button key={label} className={`legend-item ${selectedLabels.length === 0 || selectedLabels.includes(label) ? "selected" : ""}`} onClick={() => toggleLabel(label)} aria-pressed={selectedLabels.length === 0 || selectedLabels.includes(label)}><i />{label}<b>{prediction.blocks.filter((block) => block.label === label).length}</b></button>)}</div></div>
                <DetectionList blocks={prediction.blocks} visibleIndices={visibleIndices} activeIndex={activeIndex ?? selectedIndex} onActiveChange={setActiveIndex} onSelect={setSelectedIndex} />
              </aside>}
            </section>
          )}
        </>
      )}
      {!file && <footer><span>Powered by <strong>Docling Heron</strong></span><span>17 document region classes · Original-image coordinates</span></footer>}
    </main>
  );
}

function Metric({ value, label }: { value: string; label: string }) {
  return <div className="metric"><strong>{value}</strong><span>{label}</span></div>;
}

function UploadZone({ inputRef, dragging, onDrop, setDragging, chooseFile }: { inputRef: RefObject<HTMLInputElement | null>; dragging: boolean; onDrop: (event: DragEvent<HTMLDivElement>) => void; setDragging: (value: boolean) => void; chooseFile: (file: File | undefined) => void }) {
  return <div className={`drop-zone ${dragging ? "is-dragging" : ""}`} onDragOver={(event) => { event.preventDefault(); setDragging(true); }} onDragLeave={() => setDragging(false)} onDrop={onDrop}>
    <input ref={inputRef} type="file" accept={accept} hidden onChange={(event) => chooseFile(event.target.files?.[0])} />
    <div className="upload-illustration"><span>↑</span><div className="paper-line one" /><div className="paper-line two" /><div className="paper-line three" /></div>
    <strong>Drop a document image here</strong><small>PNG, JPEG, TIFF, BMP, or WebP · 10 MB max</small>
    <button className="browse-button" onClick={() => inputRef.current?.click()}>Browse files <span>↗</span></button>
  </div>;
}
