# OpenShot-ComfyUI

OpenShot-ComfyUI provides production-focused ComfyUI nodes built for OpenShot integration, with a strong focus on reliable SAM2 workflows for longer videos.

The goal is simple: make advanced segmentation and video analysis features feel native inside OpenShot's UI, while keeping the underlying Comfy graphs stable, predictable, and memory-safe.

## Why this exists

OpenShot needs SAM2 pipelines that can handle real-world clips, not just short demos.

Many SAM2 custom-node workflows process or retain full-video state in ways that become fragile or memory-heavy as clip length grows. In practice, that can lead to slowdowns, failures, or OOM behavior on longer timelines.

This project addresses that gap with chunk-oriented processing designed specifically for OpenShot's planned UI integration path.

## How this works

- Keep node interfaces close to standard ComfyUI types and patterns.
- Process video segmentation in bounded chunks instead of retaining full-video mask history.
- Return outputs that are easier for OpenShot to consume and orchestrate in larger editing workflows.
- Include practical companion nodes (GroundingDINO + TransNetV2) that support automated, timeline-aware tooling.

## What this includes (V1)

- `OpenShotDownloadAndLoadSAM2Model`
- `OpenShotSam2Segmentation` (single-image)
- `OpenShotSam2VideoSegmentationAddPoints`
- `OpenShotSam2VideoSegmentationChunked` (meta-batch/chunk friendly)
- `OpenShotGroundingDinoDetect` (text-prompted object detection -> mask + JSON)
- `OpenShotTransNetSceneDetect` (direct TransNetV2 inference -> IN/OUT JSON ranges)

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
- GroundingDINO runtime via Hugging Face Transformers (installed from `requirements.txt`)
- `transnetv2-pytorch` (installed from `requirements.txt`)

Install this node pack into `ComfyUI/custom_nodes/OpenShot-ComfyUI` and restart ComfyUI.

## Quick install (copy/paste)

Install this node's Python dependencies:

```bash
python -m pip install -r requirements.txt
```

Install SAM2 manually (required, not in `requirements.txt`):

```bash
python -m pip install git+https://github.com/facebookresearch/sam2.git
```

Optional: verify the SAM2 import works in your Comfy environment:

```bash
python -c "import sam2; print('sam2 import OK')"
```

Restart ComfyUI after install.

### Why SAM2 is manual

`requirements.txt` intentionally does **not** install SAM2 from Git. This avoids repeated large temporary downloads and long install times during routine updates.

### GroundingDINO model downloads

`OpenShotGroundingDinoDetect` will download selected model weights from Hugging Face on first use and cache them in your HF cache directory.

### TransNet scene detection behavior

`OpenShotTransNetSceneDetect` runs TransNetV2 inference directly and returns scene range JSON (`start_seconds`, `end_seconds`), so no external TransNet Comfy node pack is required.

## Models

`OpenShotDownloadAndLoadSAM2Model` auto-downloads supported SAM2 checkpoints into `ComfyUI/models/sam2` if missing.

## Notes

- `OpenShotSam2VideoSegmentationChunked` returns only the requested chunk range (bounded memory) instead of collecting whole-video masks.
- For very long videos, pair chunked outputs with batch-safe downstream nodes (VHS meta-batch, staged processing, or on-disk intermediates).

---

Copyright (C) 2026 OpenShot Studios, LLC

Licensed under the GNU General Public License v3.0 (GPLv3).
See [LICENSE.md](LICENSE.md) for the full license text.
