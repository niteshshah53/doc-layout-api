#!/bin/bash
# Install the pinned, supported runtime. The API uses torchvision's COCO
# detector, so Detectron2 source builds and CUDA toolkit detection are not
# required.
set -euo pipefail

pip install --upgrade pip
pip install -r requirements.txt
python -c "import torch, torchvision; print(f'PyTorch {torch.__version__} | torchvision {torchvision.__version__} | CUDA: {torch.cuda.is_available()}')"
echo "Dependencies installed successfully."
