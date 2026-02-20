# OpenShot-ComfyUI

OpenShot-focused ComfyUI nodes for robust video workflows.

This V1 intentionally stays close to standard ComfyUI primitives and types while fixing long-video SAM2 pain points.

## What this includes (V1)

- `OpenShotDownloadAndLoadSAM2Model`
- `OpenShotSam2Segmentation` (single-image)
- `OpenShotSam2VideoSegmentationAddPoints`
- `OpenShotSam2VideoSegmentationChunked` (meta-batch/chunk friendly)

## Why this exists

Some existing SAM2 video nodes accumulate entire-video masks in memory and can OOM on longer clips. This pack provides chunk-safe behavior while keeping Comfy-native node patterns.

## Attribution

This project is inspired by and partially based on ideas and APIs from:

- `kijai/ComfyUI-segment-anything-2`
- Meta SAM2 research/code

Please see upstream projects for full original implementations and credits.

## Requirements

- ComfyUI
- PyTorch (as used by your Comfy install)
- `decord` (required by many SAM2 video predictor builds)
- `sam2` Python package/module available in your Comfy runtime

Install this node pack into `ComfyUI/custom_nodes/OpenShot-ComfyUI` and restart ComfyUI.

### Install dependencies

```bash
/home/jonathan/miniforge3/envs/comfyui/bin/python -m pip install -r /home/jonathan/apps/ComfyUI/custom_nodes/OpenShot-ComfyUI/requirements.txt
```

### Important note about SAM2 installs

`requirements.txt` intentionally does **not** install SAM2 from Git. This avoids repeated large temporary downloads and long install times during routine updates.

Ensure your Comfy environment already has a compatible `sam2` installed.

## Models

`OpenShotDownloadAndLoadSAM2Model` auto-downloads supported SAM2 checkpoints into `ComfyUI/models/sam2` if missing.

## Notes

- `OpenShotSam2VideoSegmentationChunked` returns only the requested chunk range (bounded memory) instead of collecting whole-video masks.
- For very long videos, pair chunked outputs with batch-safe downstream nodes (VHS meta-batch, staged processing, or on-disk intermediates).
