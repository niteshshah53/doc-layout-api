"""Acquire text-rich public-domain pages and evaluate Docling Heron.

Run from the repository root with:
    python scripts/evaluate_layout.py

The evaluator downloads only the selected pages' source PDFs from NASA NTRS,
renders one representative page from each report, and never downloads a
complete document-layout dataset.
"""

from __future__ import annotations

import io
import json
import sys
import time
from pathlib import Path
from typing import Any

import fitz
import requests
from PIL import Image, ImageDraw, ImageFont, ImageStat

ROOT = Path(__file__).resolve().parents[1]
SAMPLES_DIR = ROOT / "data" / "evaluation_samples"
ARTIFACTS_DIR = ROOT / "artifacts" / "evaluation"
README_IMAGES_DIR = ROOT / "docs" / "images" / "evaluation"

MODEL_ID = "docling-project/docling-layout-heron"
THRESHOLD = 0.5
NASA_BASE = "https://ntrs.nasa.gov"
NASA_HEADERS = {"User-Agent": "doc-layout-api-evaluation/1.0"}

SAMPLES = [
    {
        "id": "nasa-exploration-summary",
        "report_id": "19890007335",
        "page_number": 8,
        "source_url": "https://ntrs.nasa.gov/citations/19890007335",
        "license": "U.S. government work; public domain in the United States",
        "description": "NASA exploration technical summary with a title, section heading, paragraphs, and structured report content.",
        "inspection": "Useful page: the title, two-column body text, section structure, and footer were visible; boxes aligned with the title and text columns. No figure, table, or formula is present.",
    },
    {
        "id": "nasa-microelectronics-report",
        "report_id": "19970014917",
        "page_number": 10,
        "source_url": "https://ntrs.nasa.gov/citations/19970014917",
        "license": "U.S. government work; public domain in the United States",
        "description": "NASA/JPL research report page with several structured technical list entries and dense text.",
        "inspection": "Useful page: seven distinct technical entries were visible and boxed as list items. The model over-generalized the entries as List-item and did not identify a separate heading; no table or figure is present.",
    },
    {
        "id": "nasa-impj-report",
        "report_id": "19740009451",
        "page_number": 7,
        "source_url": "https://ntrs.nasa.gov/citations/19740009451",
        "license": "U.S. government work; public domain in the United States",
        "description": "NASA spacecraft report page with numbered recommendations, paragraphs, and structured technical text.",
        "inspection": "Useful page: a section heading, several paragraph blocks, and a conclusion heading were visible. Text boxes followed the printed paragraphs; the page contains no table or figure.",
    },
    {
        "id": "nasa-range-safety-report",
        "report_id": "20140012574",
        "page_number": 13,
        "source_url": "https://ntrs.nasa.gov/citations/20140012574",
        "license": "U.S. government work; public domain in the United States",
        "description": "NASA annual report page with a section heading, lists, body text, and report-style layout.",
        "inspection": "Useful page: two headings, body paragraphs, and four bulleted audience entries were visible. Section-header, Text, and List-item boxes aligned well; no table or figure is present.",
    },
    {
        "id": "nasa-exploration-table",
        "report_id": "19890007335",
        "page_number": 36,
        "source_url": "https://ntrs.nasa.gov/citations/19890007335",
        "license": "U.S. government work; public domain in the United States",
        "description": "NASA exploration technical summary page containing Table 2, a genuine multi-column comparison table with labeled rows and explanatory values.",
        "inspection": "The original page visibly contains a large ruled table with a title, column headers, row labels, and populated cells. The table occupies most of the page and is not a text reference to a table.",
    },
    {
        "id": "nasa-exploration-figure",
        "report_id": "19890007335",
        "page_number": 41,
        "source_url": "https://ntrs.nasa.gov/citations/19890007335",
        "license": "U.S. government work; public domain in the United States",
        "description": "NASA exploration technical summary page containing Figure 14, a bar chart with a visible caption and surrounding explanatory text.",
        "inspection": "The original page visibly contains a labeled bar chart, legend, axes, Figure 14 caption, and two-column prose below and beside the figure. It is a genuine figure page rather than a page that only mentions a figure.",
    },
]


def request_with_retries(url: str, *, timeout: int) -> requests.Response:
    for attempt in range(3):
        response = requests.get(url, headers=NASA_HEADERS, timeout=timeout)
        if response.status_code not in {429, 500, 502, 503, 504}:
            response.raise_for_status()
            return response
        if attempt < 2:
            time.sleep(2**attempt)
    response.raise_for_status()
    return response


def fetch_report(sample: dict[str, Any]) -> tuple[Path, str, str]:
    """Fetch one report PDF through the NTRS citation API."""
    pdf_path = SAMPLES_DIR / f"{sample['id']}.pdf"
    metadata_url = f"{NASA_BASE}/api/citations/{sample['report_id']}"
    metadata_response = request_with_retries(metadata_url, timeout=60)
    metadata = metadata_response.json()
    download = next(
        (
            item
            for item in metadata.get("downloads", [])
            if item.get("mimetype") == "application/pdf"
        ),
        None,
    )
    if download is None:
        raise RuntimeError(f"NTRS report {sample['report_id']} has no PDF download")
    pdf_url = NASA_BASE + download["links"]["pdf"]
    if not pdf_path.exists():
        response = request_with_retries(pdf_url, timeout=180)
        if not response.content.startswith(b"%PDF-"):
            raise ValueError(f"NTRS returned a non-PDF response for {sample['id']}")
        pdf_path.write_bytes(response.content)
    with fitz.open(pdf_path) as document:
        if sample["page_number"] > document.page_count:
            raise ValueError(
                f"page {sample['page_number']} is outside {document.page_count}-page PDF"
            )
    return pdf_path, metadata_url, pdf_url


def render_page(pdf_path: Path, sample: dict[str, Any]) -> Path:
    """Render the selected PDF page and reject blank or implausibly small pages."""
    output = SAMPLES_DIR / f"{sample['id']}.png"
    with fitz.open(pdf_path) as document:
        page = document.load_page(sample["page_number"] - 1)
        pixmap = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
        image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
    if image.width < 1000 or image.height < 1000:
        raise ValueError(f"rendered page is unexpectedly small: {image.size}")
    if sum(ImageStat.Stat(image).stddev) < 20:
        raise ValueError("rendered page appears blank or nearly blank")
    image.save(output, format="PNG")
    return output


def annotate(image: Image.Image, blocks: list[dict[str, Any]], output: Path) -> None:
    canvas = image.convert("RGB")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    colors: dict[str, tuple[int, int, int]] = {}
    for block in blocks:
        label = block["label"]
        color = colors.setdefault(
            label,
            (
                60 + (len(colors) * 67) % 180,
                70 + (len(colors) * 97) % 160,
                80 + (len(colors) * 127) % 150,
            ),
        )
        box = block["bbox"]
        coordinates = (box["x1"], box["y1"], box["x2"], box["y2"])
        draw.rectangle(coordinates, outline=color, width=max(2, canvas.width // 500))
        caption = f'{label} {block["score"]:.2f}'
        left, top, right, bottom = draw.textbbox(
            (box["x1"], box["y1"]), caption, font=font
        )
        draw.rectangle((left, top, right + 3, bottom + 2), fill=color)
        draw.text((box["x1"] + 2, box["y1"] + 1), caption, fill="white", font=font)
    canvas.save(output, format="PNG")


def contact_sheet(paths: list[Path], output: Path) -> None:
    columns = 2
    rows = (len(paths) + columns - 1) // columns
    thumbnails: list[Image.Image] = []
    for path in paths:
        with Image.open(path) as image:
            thumbnail = image.convert("RGB")
            thumbnail.thumbnail((500, 650))
            thumbnails.append(thumbnail.copy())
    sheet = Image.new("RGB", (1000, rows * 650), "white")
    for index, thumbnail in enumerate(thumbnails):
        x = (index % 2) * 500
        y = (index // 2) * 650
        sheet.paste(thumbnail, (x, y))
    sheet.save(output, format="PNG")


def run() -> int:
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    README_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(ROOT))
    from app.inference import run_full_pipeline
    from app.model import init_model

    init_model()
    failures: list[dict[str, str]] = []
    results: list[dict[str, Any]] = []
    annotated_paths: list[Path] = []

    for sample in SAMPLES:
        print(f"Processing {sample['id']}...")
        try:
            pdf_path, metadata_url, pdf_url = fetch_report(sample)
            image_path = render_page(pdf_path, sample)
            image_bytes = image_path.read_bytes()
            result = run_full_pipeline(image_bytes)
            payload = {
                "sample": sample,
                "source": {
                    "metadata_url": metadata_url,
                    "download_url": pdf_url,
                    "license": sample["license"],
                    "retrieved_pdf": str(pdf_path),
                    "rendered_page": sample["page_number"],
                },
                "model": {"id": MODEL_ID, "score_threshold": THRESHOLD},
                "image_size": result.image_size,
                "inference_time_ms": result.inference_time_ms,
                "num_blocks": result.num_blocks,
                "blocks": [block.model_dump() for block in result.blocks],
                "error": None,
            }
            annotated_path = ARTIFACTS_DIR / f"{sample['id']}.png"
            with Image.open(io.BytesIO(image_bytes)) as image:
                annotate(image, payload["blocks"], annotated_path)
                annotate(image, payload["blocks"], README_IMAGES_DIR / annotated_path.name)
            json_path = ARTIFACTS_DIR / f"{sample['id']}.json"
            json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            results.append(payload)
            annotated_paths.append(annotated_path)
            labels = sorted({block["label"] for block in payload["blocks"]})
            print(f"  {result.num_blocks} detections: {', '.join(labels) or 'none'}")
        except Exception as exc:
            failure = {"sample": sample["id"], "error": f"{type(exc).__name__}: {exc}"}
            failures.append(failure)
            print(f"  FAILED: {failure['error']}", file=sys.stderr)

    if annotated_paths:
        contact_sheet(annotated_paths, ARTIFACTS_DIR / "contact-sheet.png")
        contact_sheet(annotated_paths, README_IMAGES_DIR / "contact-sheet.png")

    summary = {
        "model": MODEL_ID,
        "score_threshold": THRESHOLD,
        "samples_requested": len(SAMPLES),
        "samples_processed": len(results),
        "minimum_required": len(SAMPLES),
        "complete": len(results) == len(SAMPLES) and not failures,
        "failures": failures,
        "results": [
            {
                "sample": item["sample"]["id"],
                "description": item["sample"]["description"],
                "page_number": item["source"]["rendered_page"],
                "num_blocks": item["num_blocks"],
                "labels": sorted({block["label"] for block in item["blocks"]}),
                "json_path": str(ARTIFACTS_DIR / f"{item['sample']['id']}.json"),
                "image_path": str(ARTIFACTS_DIR / f"{item['sample']['id']}.png"),
            }
            for item in results
        ],
    }
    (ARTIFACTS_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    report_lines = [
        "# Docling Heron Evaluation",
        "",
        f"- Model: `{MODEL_ID}`",
        f"- Threshold: `{THRESHOLD}`",
        f"- Processed: {len(results)}/{len(SAMPLES)}",
        f"- Complete target ({len(SAMPLES)}): `{summary['complete']}`",
        "",
        "## Results",
        "",
    ]
    for item in summary["results"]:
        scores = [
            block["score"]
            for result in results
            if result["sample"]["id"] == item["sample"]
            for block in result["blocks"]
        ]
        report_lines.append(
            f"- `{item['sample']}` page {item['page_number']}: "
            f"{item['num_blocks']} detections; labels: "
            f"{', '.join(item['labels']) or 'none'}; "
            f"score range: {min(scores):.2f}-{max(scores):.2f}"
        )
        sample = next(sample for sample in SAMPLES if sample["id"] == item["sample"])
        report_lines.append(f"  - Inspection: {sample['inspection']}")
    if failures:
        report_lines.extend(["", "## Failures", ""])
        report_lines.extend(f"- `{item['sample']}`: {item['error']}" for item in failures)
    (ARTIFACTS_DIR / "REPORT.md").write_text(
        "\n".join(report_lines) + "\n", encoding="utf-8"
    )
    return 0 if summary["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(run())
