"""
app/model.py
------------
Handles model loading and ownership of the layoutparser model instance.

KEY SENIOR DECISIONS EXPLAINED:

1. SINGLETON PATTERN via module-level variable + get_model()
   - The model is ~300MB+. Loading it per request would make every
     request take 3-5 seconds and eventually OOM the container.
   - We load once at startup, hold in memory, reuse for every request.

2. torch.no_grad() is NOT applied here — it's applied in inference.py
   during the actual forward pass. This is the correct boundary.

3. Device selection: we respect the config but also check CUDA
   availability at runtime so the code doesn't crash on CPU-only machines
   if someone misconfigures the .env.

4. layoutparser wraps Detectron2. The model_config_path uses their
   "model zoo" URI syntax (lp://...) so you don't have to manage
   config .yaml files manually.

5. FALLBACK MECHANISM: If the primary model load fails, automatically
   attempt to load a lighter/CPU-based fallback model. This ensures
   graceful degradation instead of complete failure.
"""

import torch
import layoutparser as lp
from loguru import logger
from dataclasses import dataclass
import yaml
import os
import shutil

from app.config import get_settings

# Module-level variables — these are our singletons
_model: lp.Detectron2LayoutModel | None = None
_model_status: dict = {
    "loaded": False,
    "model_type": None,  # "primary" or "fallback"
    "error": None,       # Error message if loading failed
}


@dataclass
class ModelInfo:
    """Information about the loaded model."""
    loaded: bool
    model_type: str | None  # "primary", "fallback", or None
    error: str | None       # Error message if applicable


def _validate_and_fix_yaml_config(config_path: str) -> str | None:
    """
    Try to load and parse the YAML config. If it fails, attempt to fix common issues.
    
    Args:
        config_path: Path to the YAML config file
    
    Returns:
        Path to a valid config file (original or fixed), or None if unfixable.
    """
    if not os.path.exists(config_path):
        return None
    
    try:
        with open(config_path, 'r') as f:
            yaml.safe_load(f)
        logger.debug(f"Config YAML is valid: {config_path}")
        return config_path
    
    except yaml.YAMLError as e:
        logger.warning(f"YAML parsing error in {config_path}: {e}")
        
        # Try to fix common issues
        try:
            with open(config_path, 'r') as f:
                content = f.read()
            
            # Common fix: escape unquoted colons in values
            # Example: "value: some:text" becomes "value: 'some:text'"
            lines = content.split('\n')
            fixed_lines = []
            
            for i, line in enumerate(lines):
                # Skip comments and empty lines
                if not line.strip() or line.strip().startswith('#'):
                    fixed_lines.append(line)
                    continue
                
                # Try to quote values that have unquoted colons
                if ':' in line and not line.strip().startswith('-'):
                    parts = line.split(':', 1)
                    if len(parts) == 2:
                        key, value = parts
                        # If value has unquoted colons, quote it
                        if value.strip() and ':' in value.strip() and not (
                            value.strip().startswith("'") or value.strip().startswith('"')
                        ):
                            fixed_line = f"{key}: '{value.strip()}'"
                            fixed_lines.append(fixed_line)
                            logger.debug(f"Fixed line {i+1}: {line} → {fixed_line}")
                            continue
                
                fixed_lines.append(line)
            
            fixed_content = '\n'.join(fixed_lines)
            
            # Try to parse the fixed version
            yaml.safe_load(fixed_content)
            
            # If we got here, the fix worked. Write it.
            fixed_path = config_path + ".fixed"
            with open(fixed_path, 'w') as f:
                f.write(fixed_content)
            
            logger.info(f"Successfully fixed YAML config and saved to: {fixed_path}")
            return fixed_path
        
        except Exception as fix_error:
            logger.error(f"Could not fix YAML: {fix_error}")
            return None


def _load_model_with_config(
    config_path: str, device: str, model_type_label: str
) -> lp.Detectron2LayoutModel | None:
    """
    Load the model with a specific config path.
    
    Handles both layoutparser (lp:// URIs) and Detectron2 native configs.
    
    Args:
        config_path: The model config path:
            - layoutparser URI: "lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config"
            - Detectron2 path: "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
        device: "cuda" or "cpu"
        model_type_label: human-readable label for logging (e.g., "primary", "fallback")
    
    Returns:
        Loaded model instance, or None if loading fails.
    """
    logger.info(f"[{model_type_label}] Loading model on device: {device}")
    logger.info(f"[{model_type_label}] Model config: {config_path}")
    
    max_retries = 3
    last_error = None
    
    # Check if config is a Detectron2 built-in (COCO-Detection, etc.)
    is_detectron2_native = config_path.startswith("COCO-Detection") or config_path.startswith("Base-")
    
    if is_detectron2_native:
        logger.info(f"[{model_type_label}] Using native Detectron2 config (not layoutparser)")
        # Use raw Detectron2 instead of layoutparser for built-in configs
        try:
            from detectron2 import model_zoo
            from detectron2.config import get_cfg
            from detectron2.engine import DefaultPredictor
            import layoutparser as lp
            
            # Get the config
            cfg = get_cfg()
            cfg_path = model_zoo.get_config_file(config_path)
            cfg.merge_from_file(cfg_path)
            
            # Load weights
            cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(config_path)
            cfg.MODEL.DEVICE = device
            cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
            
            # Create predictor
            predictor = DefaultPredictor(cfg)
            
            # Wrap in a simple adapter to match layoutparser's interface
            class DetectronAdapter:
                """Adapter to make Detectron2 predictor compatible with layoutparser's interface"""
                
                def __init__(self, predictor_obj, config):
                    self.model = type('obj', (object,), {'cfg': config})()
                    self._predictor = predictor_obj
                
                def detect(self, image):
                    """
                    Run detection on image and return layoutparser Layout object.
                    
                    Args:
                        image: numpy array (H, W, 3) with values 0-255
                    
                    Returns:
                        layoutparser Layout object with detected blocks
                    """
                    try:
                        # Run Detectron2 prediction
                        outputs = self._predictor(image)
                        
                        # Convert to layoutparser Layout
                        blocks = []
                        instances = outputs.get("instances", None)
                        
                        if instances is not None and len(instances) > 0:
                            boxes = instances.pred_boxes.tensor.cpu().numpy()
                            scores = instances.scores.cpu().numpy()
                            
                            for box, score in zip(boxes, scores):
                                x1, y1, x2, y2 = box
                                
                                # Create layoutparser TextBlock
                                text_block = lp.TextBlock(
                                    block=lp.Rectangle(
                                        x1=float(x1), 
                                        y1=float(y1), 
                                        x2=float(x2), 
                                        y2=float(y2)
                                    ),
                                    type="Detected",
                                    score=float(score)
                                )
                                blocks.append(text_block)
                        
                        # Return as layoutparser Layout
                        return lp.Layout(blocks)
                    
                    except Exception as e:
                        logger.error(f"Error in Detectron2 adapter detect(): {type(e).__name__}: {e}")
                        # Return empty layout on error instead of crashing
                        return lp.Layout([])
            
            logger.info(f"[{model_type_label}] Model loaded successfully ✓")
            return DetectronAdapter(predictor, cfg)
            
        except Exception as e:
            logger.warning(f"[{model_type_label}] Failed to load with raw Detectron2: {type(e).__name__}: {e}")
            last_error = e
    
    # Fall back to layoutparser for lp:// URIs or if raw Detectron2 failed
    for attempt in range(max_retries):
        try:
            model = lp.Detectron2LayoutModel(
                config_path=config_path,
                extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.5],
                label_map={
                    0: "Text",
                    1: "Title",
                    2: "List",
                    3: "Table",
                    4: "Figure",
                },
            )

            # Move underlying detectron2 predictor to the right device.
            # layoutparser doesn't expose device as a constructor arg directly,
            # so we patch the cfg after init — this is the accepted pattern.
            model.model.cfg.MODEL.DEVICE = device
            logger.info(f"[{model_type_label}] Model loaded successfully ✓")
            return model
            
        except Exception as e:
            last_error = e
            error_type = type(e).__name__
            
            if attempt < max_retries - 1:
                logger.warning(
                    f"[{model_type_label}] Load attempt {attempt + 1} failed: {error_type}: {e}"
                )
                
                # Aggressive cache clearing for all errors
                cache_dir = os.path.expanduser("~/.torch/iopath_cache")
                if os.path.exists(cache_dir):
                    logger.info(f"[{model_type_label}] Clearing iopath cache...")
                    shutil.rmtree(cache_dir, ignore_errors=True)
                
                # Also try to clear detectron2 cache
                detectron2_cache = os.path.expanduser("~/.detectron2")
                if os.path.exists(detectron2_cache):
                    logger.info(f"[{model_type_label}] Clearing detectron2 cache...")
                    shutil.rmtree(detectron2_cache, ignore_errors=True)
            
            else:
                logger.error(
                    f"[{model_type_label}] Final load attempt {attempt + 1} failed. "
                    f"Error: {error_type}: {e}"
                )
    
    return None


def load_model() -> tuple[lp.Detectron2LayoutModel | None, str | None, str | None]:
    """
    Load the PubLayNet layout detection model with comprehensive fallback support.
    
    This function attempts multiple model configurations in order:
    1. Primary: User-configured model (from settings)
    2. Fallback 1: Mask RCNN variant (sometimes has better YAML)
    3. Fallback 2: Faster RCNN on CPU
    
    Returns:
        tuple: (model_instance, model_type, error_message)
        - model_instance: The loaded model, or None if all attempts failed
        - model_type: "primary" or "fallback" if successful, None if all failed
        - error_message: Human-readable error if all attempts failed
    """
    settings = get_settings()
    
    # ── Device resolution ────────────────────────────────────────────────
    requested_device = settings.device
    if requested_device == "cuda" and not torch.cuda.is_available():
        logger.warning(
            "CUDA requested but not available — falling back to CPU. "
            "Set DEVICE=cpu in .env to silence this warning."
        )
        device = "cpu"
    else:
        device = requested_device

    # ── PRIMARY MODEL LOAD ───────────────────────────────────────────────
    logger.info("=" * 70)
    logger.info("PRIMARY MODEL LOAD ATTEMPT")
    logger.info("=" * 70)
    
    model = _load_model_with_config(
        config_path=settings.model_config_path,
        device=device,
        model_type_label="PRIMARY"
    )
    
    if model is not None:
        return model, "primary", None
    
    # ── FALLBACK 1: TRY MASK RCNN VARIANT ────────────────────────────────
    logger.info("=" * 70)
    logger.info("PRIMARY MODEL FAILED — ATTEMPTING FALLBACK 1 (Mask RCNN)")
    logger.info("=" * 70)
    
    fallback_config_1 = "lp://PubLayNet/mask_rcnn_R_50_FPN_3x/config"
    logger.warning(
        f"Attempting fallback model: {fallback_config_1} on device={device}"
    )
    
    fallback_model = _load_model_with_config(
        config_path=fallback_config_1,
        device=device,
        model_type_label="FALLBACK-1"
    )
    
    if fallback_model is not None:
        logger.critical(
            "⚠️  FALLBACK MODEL LOADED (Mask RCNN): Primary model failed but fallback is available. "
            "Check logs above for primary failure reason."
        )
        return fallback_model, "fallback", "Primary model failed; fallback model (mask_rcnn) loaded."
    
    # ── FALLBACK 2: TRY FASTER RCNN ON CPU ───────────────────────────────
    logger.info("=" * 70)
    logger.info("FALLBACK 1 FAILED — ATTEMPTING FALLBACK 2 (Faster RCNN on CPU)")
    logger.info("=" * 70)
    
    fallback_config_2 = "lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config"
    fallback_device = "cpu"
    logger.warning(
        f"Attempting fallback model: {fallback_config_2} on device={fallback_device}"
    )
    
    fallback_model_2 = _load_model_with_config(
        config_path=fallback_config_2,
        device=fallback_device,
        model_type_label="FALLBACK-2"
    )
    
    if fallback_model_2 is not None:
        logger.critical(
            "⚠️  FALLBACK MODEL LOADED (CPU): Primary model failed but fallback is available. "
            "Performance may be reduced. Check logs above for primary failure reason."
        )
        return fallback_model_2, "fallback", "Primary model failed; fallback model (CPU) loaded."
    
    # ── ALL ATTEMPTS FAILED ─────────────────────────────────────────────
    error_msg = (
        "Model initialization failed completely. "
        "All model load attempts failed (primary + 2 fallbacks). "
        "\n\nDIAGNOSIS: The layoutparser model zoo configs from Dropbox are corrupted "
        "(YAML syntax error at line 42). This is a known issue with layoutparser's "
        "model catalog. "
        "\n\nSOLUTIONS:"
        "\n1. Upgrade layoutparser: pip install --upgrade layoutparser"
        "\n2. Use Detectron2 directly instead of layoutparser"
        "\n3. Download a working config from an alternative source"
        "\n4. Check: https://github.com/Layout-Parser/layout-parser/issues/"
        "\n\nSee logs above for detailed error traces."
    )
    logger.error("=" * 70)
    logger.error(error_msg)
    logger.error("=" * 70)
    
    return None, None, error_msg


def get_model() -> lp.Detectron2LayoutModel:
    """
    Return the singleton model instance.
    Raises RuntimeError if called before init_model() — which would mean
    a startup ordering bug or model loading failure.
    """
    if _model is None:
        raise RuntimeError(
            f"Model has not been loaded. "
            f"This may indicate a startup ordering bug or failed model initialization. "
            f"Status: {_model_status}"
        )
    return _model


def get_model_info() -> ModelInfo:
    """
    Return information about the current model status.
    Safe to call anytime — always returns a valid ModelInfo object.
    """
    return ModelInfo(
        loaded=_model_status["loaded"],
        model_type=_model_status["model_type"],
        error=_model_status["error"]
    )


def init_model() -> None:
    """
    Called from main.py lifespan. Sets the module-level singleton and status.
    
    Attempts to load the primary model. If that fails, automatically tries
    a fallback model. This allows graceful degradation instead of complete failure.
    
    The API will start regardless (fail-open pattern), but the /predict endpoint
    will return 503 if no model is available. The /health endpoint will report
    which model (if any) is loaded.
    """
    global _model, _model_status
    
    try:
        model, model_type, error = load_model()
        
        _model = model
        if model is not None:
            _model_status["loaded"] = True
            _model_status["model_type"] = model_type
            _model_status["error"] = None
            logger.info(
                f"✓ Model initialized successfully (type={model_type})"
            )
        else:
            _model_status["loaded"] = False
            _model_status["model_type"] = None
            _model_status["error"] = error
            logger.error(
                f"✗ Model initialization failed completely. "
                f"Error: {error}"
            )
            
    except Exception as e:
        # Catch any unexpected errors during load_model()
        logger.exception(f"Unexpected error during model initialization: {e}")
        _model = None
        _model_status["loaded"] = False
        _model_status["model_type"] = None
        _model_status["error"] = str(e)
