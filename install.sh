#!/bin/bash
# install.sh — Run this instead of "pip install -r requirements.txt"
#
# WHY THIS SCRIPT EXISTS:
#   detectron2 imports torch in its setup.py to determine the CUDA version
#   before it can build its wheel. This means torch MUST be installed first.
#   pip doesn't know about this ordering constraint, so installing everything
#   at once always fails.
#
# USAGE:
#   chmod +x install.sh
#   ./install.sh

set -euo pipefail  # exit immediately on any error

DETECTRON2_WHEEL_URL="https://dl.fbaipublicfiles.com/detectron2/wheels/cu121/torch2.3/index.html"

echo "=== Step 1: Installing PyTorch (CUDA 12.1) ==="
pip install torch==2.3.0 torchvision==0.18.0 --index-url https://download.pytorch.org/whl/cu121

echo ""
echo "=== Step 2: Verifying CUDA is accessible ==="
python -c "import torch; print(f'PyTorch {torch.__version__} | CUDA available: {torch.cuda.is_available()}')"

echo ""
echo "=== Step 3: Installing detectron2 ==="
# We try the pre-built wheel first. If the current Python version has no
# matching wheel, fall back to a source build so setup still completes.
# See: https://detectron2.readthedocs.io/en/latest/tutorials/install.html
if ! pip install --no-cache-dir detectron2 -f "$DETECTRON2_WHEEL_URL"; then
	echo ""
	echo "No compatible Detectron2 wheel was found for this Python/CUDA combo."
	echo "Falling back to a source install."

	if command -v nvcc >/dev/null 2>&1; then
		export FORCE_CUDA=1
		echo "CUDA compiler detected: building Detectron2 with CUDA support."
	else
		export FORCE_CUDA=0
		echo "nvcc not found: building Detectron2 without CUDA extensions."
		echo "If you want GPU acceleration, install a CUDA toolkit / nvcc-enabled environment."
	fi

	pip install --no-cache-dir ninja
	pip install --no-cache-dir --no-build-isolation "git+https://github.com/facebookresearch/detectron2.git"
fi

echo ""
echo "=== Step 4: Installing remaining dependencies ==="
pip install -r requirements.txt

echo ""
echo "=== Step 5: Verifying layoutparser can import detectron2 ==="
python -c "import layoutparser; print('layoutparser OK')"
python -c "import detectron2; print('detectron2 OK')"

echo ""
echo "=== All dependencies installed successfully ==="
