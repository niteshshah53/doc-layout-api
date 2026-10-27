import type { HealthResponse, PredictionResponse } from "../types";

export const apiBaseUrl =
  import.meta.env.VITE_API_BASE_URL?.replace(/\/+$/, "") ||
  "http://localhost:8000";

function errorMessage(status: number, payload: unknown): string {
  if (status === 400) return "This file could not be processed as a supported image.";
  if (status === 413) return "This file is larger than the 10 MB upload limit.";
  if (status === 503) return "The layout model is unavailable or still loading.";
  if (status >= 500) return "Inference failed on the server. Please try another image.";

  if (typeof payload === "object" && payload !== null) {
    const body = payload as { detail?: unknown; error?: unknown };
    if (typeof body.detail === "string") return body.detail;
    if (typeof body.error === "string") return body.error;
  }
  return `Request failed with status ${status}.`;
}

export async function predictLayout(
  file: File,
  signal?: AbortSignal,
): Promise<PredictionResponse> {
  const formData = new FormData();
  formData.append("file", file);
  const response = await fetch(`${apiBaseUrl}/api/v1/predict`, {
    method: "POST",
    body: formData,
    signal,
  });

  const payload: unknown = await response.json().catch(() => null);
  if (!response.ok) {
    throw new Error(errorMessage(response.status, payload));
  }
  return payload as PredictionResponse;
}

export async function getHealth(): Promise<HealthResponse> {
  const response = await fetch(`${apiBaseUrl}/api/v1/health`);
  const payload = (await response.json().catch(() => null)) as HealthResponse | null;
  if (!response.ok || !payload) {
    throw new Error("Backend health check failed.");
  }
  return payload;
}
