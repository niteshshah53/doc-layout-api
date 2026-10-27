export interface BoundingBox {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

export interface LayoutBlock {
  label: string;
  score: number;
  bbox: BoundingBox;
}

export interface PredictionResponse {
  num_blocks: number;
  image_size: {
    width: number;
    height: number;
  };
  inference_time_ms: number;
  blocks: LayoutBlock[];
}

export interface HealthResponse {
  status: string;
  model_loaded: boolean;
  model_info: { type?: string; error?: string | null } | null;
  version: string;
}
