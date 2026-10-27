# Layout Vision — Project Profile and CV-Tailoring Reference

**Repository:** https://github.com/niteshshah53/doc-layout-api  
**Project name:** Layout Vision (repository: `doc-layout-api`)  
**Purpose of this document:** A source-of-truth brief for CV/resume tailoring, cover letters, application forms, recruiter conversations, and interview preparation. Use the verified facts below; do not invent achievements or imply work that is not implemented.

## 1. Short description

Layout Vision is a full-stack document-layout detection application. A user uploads a single document image through a React/TypeScript interface; a FastAPI backend runs inference with the pretrained Docling Heron document-layout detection model (RT-DETR v2) and returns layout labels, confidence scores, and bounding boxes in the original image's pixel coordinate system. The frontend draws interactive overlays so users can inspect, filter, and select predicted regions.

The project demonstrates integration of a pretrained computer-vision model into an API and user-facing application, input validation, image preprocessing/postprocessing, coordinate geometry, automated testing, containerization, and qualitative inspection on real technical-report pages.

## 2. Problem and intended use

Documents contain structural regions—titles, section headings, body text, lists, tables, figures, captions, headers, and footers. Identifying where these regions occur can be a useful preprocessing step for document digitization, OCR pipelines, search, and downstream document understanding.

Layout Vision predicts **where** layout regions are and assigns a layout class. It is not an OCR engine and does not extract all document text. It does not currently reconstruct table cells, parse document semantics, or provide a multi-page PDF workflow.

## 3. Implemented user workflow

1. Choose or drag-and-drop a supported image into the React interface.
2. Client-side validation checks the selected file before submission.
3. The browser submits the image as multipart form data to `POST /api/v1/predict`.
4. FastAPI validates the content type and file-size limit, checks model readiness, and invokes the inference pipeline.
5. The backend decodes the image, converts it to RGB where needed, resizes oversized images while preserving aspect ratio, runs model inference, and maps predicted boxes to the original image dimensions.
6. The API returns the number of detected blocks, original image dimensions, inference time, labels, confidence scores, and bounding boxes.
7. The frontend renders the uploaded image with aligned box/label overlays. Users can filter by class, select a result or box, synchronize the selected highlight, zoom, fit the image, and independently show/hide boxes and labels.
8. The interface handles loading, validation, empty-result, backend-health, and inference-error states.

Supported image formats in the backend are JPEG, PNG, TIFF, BMP, and WEBP. The default maximum upload size is 10 MB and the default longest-edge preprocessing limit is 1024 pixels; settings can be overridden through environment configuration.

## 4. Architecture

```text
React + TypeScript + Vite
    | multipart/form-data: file
    v
FastAPI routes (/api/v1/predict, /api/v1/health)
    |
    v
Image decoding and validation
    -> aspect-ratio-preserving resize
    -> Docling Heron RT-DETR v2 inference
    -> confidence/label conversion
    -> bounding-box coordinate mapping
    v
Typed JSON response (Pydantic schemas)
    |
    v
Interactive frontend overlay + filterable detection list
```

### Backend responsibilities

- `app/main.py`: FastAPI application factory, lifespan/startup model initialization, CORS, and route registration.
- `app/config.py`: typed environment-backed settings, including device, confidence threshold, maximum image size, and upload limit.
- `app/model.py`: loads the pretrained `docling-project/docling-layout-heron` processor/model, selects CPU or CUDA when available, runs detection, and maps model class IDs to human-readable labels.
- `app/inference.py`: image decoding, preprocessing, model invocation, timing, and postprocessing into API response objects.
- `app/routes.py`: HTTP request validation, health endpoint, prediction endpoint, and error/status handling.
- `app/schemas.py`: Pydantic response structures for boxes, layout blocks, predictions, health, and errors.

### Frontend responsibilities

- `frontend/src/App.tsx`: upload flow, prediction state, health status, filtering, selection, and coordination between the viewer and results list.
- `frontend/src/api/layoutApi.ts`: HTTP calls and readable error messages.
- `frontend/src/components/DetectionViewer.tsx`: image display, box/label overlays, zoom, fit-to-view, and selection interactions.
- `frontend/src/components/DetectionList.tsx`: detection list and interaction with the visualization.
- `frontend/src/utils/imageGeometry.ts`: scales original-image bounding boxes into displayed-image coordinates.
- `frontend/src/utils/fileValidation.ts`: frontend file checks.

## 5. Model and API contract

**Model:** `docling-project/docling-layout-heron`, an RT-DETR v2 document-layout detector loaded through Hugging Face Transformers. The application integrates pretrained weights; it does not train or fine-tune this model.

The API maps 17 classes: Caption, Footnote, Formula, List-item, Page-footer, Page-header, Picture, Section-header, Table, Text, Title, Document Index, Code, Checkbox-Selected, Checkbox-Unselected, Form, and Key-Value Region.

### Prediction endpoint

`POST /api/v1/predict`  
Multipart field: `file`

The successful response contains:
- `num_blocks`: number of predicted regions;
- `image_size.width` and `image_size.height`: original upload dimensions;
- `inference_time_ms`: model inference duration (not total end-to-end request latency);
- `blocks[]`: each predicted `label`, `score` in the 0–1 range, and `bbox` with `x1, y1, x2, y2` in original-image pixel coordinates.

The other endpoint is `GET /api/v1/health`, which reports whether the model is loaded and the API version. Interactive OpenAPI documentation is exposed at `/docs`.

## 6. Technology inventory

**Backend / ML:** Python 3.11, PyTorch, Hugging Face Transformers, Docling Heron RT-DETR v2, FastAPI, Uvicorn, Pydantic, Pillow, NumPy, OpenCV-headless, Loguru.

**Frontend:** React 19, TypeScript, Vite, Vitest, jsdom, CSS.

**Quality / delivery:** pytest, mocked unit and HTTP tests, frontend unit tests, Docker, Docker Compose, health checks, non-root container user, environment-based configuration, a real-model opt-in test, and a qualitative evaluation script.

Use the exact dependency pins in `requirements.txt` and `frontend/package.json` if exact versions are required. Do not claim cloud deployment, CI/CD, authentication, database persistence, or production monitoring; those are not established by the current repository.

## 7. Testing and evaluation evidence

At the last reported local verification for the committed project:
- Backend suite: **29 passed, 1 skipped**.
- Frontend suite: **7 passed**.
- Frontend production build: **passed**.
- `git diff --check`: **passed** before commit.
- A real API smoke test was reported as successful: health returned HTTP 200 and a figure-page prediction returned HTTP 200 with 11 detections. Treat this as a single smoke-test observation, not a general performance or accuracy claim.

Backend tests cover image decoding/format validation, preprocessing and coordinate handling, route status/error behavior, and model behavior with mocks. The real-model regression test is opt-in because it downloads/loads model weights. Frontend tests cover API handling, file validation, and image geometry.

### Qualitative document inspection

The repository contains a gallery of six NASA technical-report page examples, including text-heavy pages, a table page, and a figure page with a caption. The evaluation script downloads selected reports, renders specified pages, invokes the real model, and writes annotated images and a report into ignored generated-artifact paths.

This is **qualitative inspection only**. There is no ground-truth-labelled benchmark, reported precision/recall/mAP, or validated accuracy claim. Do not describe the project as achieving a particular accuracy. A confidence score is a model output, not proof that a prediction is correct.

## 8. Deployment and operational details

- Local backend uses Uvicorn and loads model weights on startup; first startup needs network access to download the pretrained model.
- Local frontend runs separately through Vite and calls the backend URL configured by `VITE_API_BASE_URL` (default `http://localhost:8000`).
- Docker Compose provides a CPU-based API container. The Dockerfile uses official CPU PyTorch wheels, runs as a non-root user, and defines a health check.
- CPU inference is the verified deployment path described by the README. CUDA is selectable in host-native configuration when available, but GPU deployment/performance is not claimed as verified.
- Weights and evaluation outputs are generated/cached locally rather than committed.

## 9. Resume-ready project bullets

Choose bullets that match the target role. Keep them accurate and adapt wording to the job description; do not use every bullet in one CV.

### General software / ML engineering
- Built **Layout Vision**, a full-stack document-layout detection application combining a FastAPI inference service with a React/TypeScript interface for uploading document images and inspecting predicted regions.
- Integrated the pretrained Docling Heron RT-DETR v2 model through Hugging Face Transformers and exposed typed predictions—including layout classes, confidence scores, and original-image bounding boxes—through a REST API.
- Implemented image validation, RGB conversion, aspect-ratio-preserving resizing, prediction postprocessing, and bounding-box coordinate mapping to keep frontend overlays aligned with the source image.
- Added backend and frontend automated tests and packaged a CPU inference path using Docker Compose and health checks.

### Computer vision / applied ML
- Integrated a pretrained RT-DETR v2 document-layout detector for 17 region classes, including titles, text, tables, figures, captions, headers, and footers.
- Built an inference pipeline that preprocesses document images, measures model inference time, and maps detections back to original-image pixel coordinates for visualization.
- Conducted qualitative inspection on six NASA technical-report page examples, including text-rich, table, and figure pages; explicitly documented that this is not a ground-truth accuracy benchmark.

### Full-stack / backend
- Designed a FastAPI REST contract with Pydantic response schemas, file type/size validation, model-health reporting, and structured HTTP error handling.
- Developed a React/TypeScript document viewer with class filtering, synchronized selection between results and overlays, zoom/fit controls, and independent visibility controls for boxes and labels.
- Added a Dockerized CPU service with a non-root runtime user and health checks, plus automated tests for API behavior and frontend image geometry.

## 10. ATS keywords supported by the implementation

Python; PyTorch; Hugging Face Transformers; RT-DETR v2; computer vision; object detection; document layout analysis; image preprocessing; bounding-box geometry; inference pipeline; FastAPI; REST API; Pydantic; React; TypeScript; Vite; API integration; automated testing; pytest; Vitest; Docker; Docker Compose; CPU inference; model integration; qualitative evaluation.

Only include terms that are relevant to the vacancy and that you can explain in an interview.

## 11. Honest scope boundaries — do not claim these

Unless the repository is extended and the new work verified, do **not** claim:
- trained or fine-tuned an RT-DETR model, or developed the underlying Heron model;
- achieved a measured accuracy, mAP, precision, or recall score;
- OCR, full text extraction, semantic document understanding, or table-cell extraction;
- multi-page PDF processing in the interactive application;
- production/cloud deployment, Kubernetes, CI/CD, authentication, a database, or user accounts;
- verified GPU throughput, production-scale latency, or a performance improvement percentage;
- that the six qualitative examples establish general model accuracy.

A safe ownership description is: **implemented an application and inference integration around a pretrained document-layout model**. Be ready to explain the code, design choices, tests, limitations, and the distinction between model integration and model training.

## 12. Interview discussion prompts

Prepare to explain:
1. Why layout detection is distinct from OCR.
2. Why the model is loaded at startup rather than on every request.
3. How input validation and file-size limits protect the inference endpoint.
4. Why large images are resized and how aspect ratio is preserved.
5. How detections are mapped back to original-image coordinates.
6. What the Pydantic schemas contribute to the API contract and OpenAPI docs.
7. How mocked tests differ from a real-model smoke test.
8. Why qualitative examples are useful but insufficient for a benchmark.
9. What would be required to add multi-page PDF support, OCR, table structure extraction, and a ground-truth evaluation set.
10. What is verified for CPU versus what remains unverified for GPU.

## 13. Tailoring instructions for an LLM

When using this document to tailor a CV:
- First extract the vacancy's required skills, responsibilities, and seniority.
- Select only the project facts and bullets that directly match those requirements.
- Preserve the distinction between integrating a pretrained model and training one.
- Never invent metrics, deployment, scale, users, business impact, or responsibilities.
- Do not turn a qualitative evaluation into a numerical accuracy claim.
- Prefer two or three targeted bullets over a long technology inventory.
- Optimize for clear, defensible evidence; the user must be able to explain every statement in an interview.
