#!/usr/bin/env bash
set -euo pipefail
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
# Download pretrained weights at build time when possible, instead of during
# the first live API request. This requires internet access during the build.
python -c 'import open_clip; open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k", device="cpu")'
