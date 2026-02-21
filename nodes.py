import json
import os
import hashlib
import subprocess
import shutil
import time
import tempfile
from contextlib import nullcontext
from urllib.parse import urlparse
from fractions import Fraction

import numpy as np
import torch
import torch.nn.functional as F
from torch.hub import download_url_to_file
from PIL import Image

import comfy.model_management as mm
from comfy.utils import ProgressBar, common_upscale
import folder_paths
from hydra import initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

try:
    import sam2.build_sam as sam2_build
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
except Exception as ex:  # pragma: no cover - runtime env specific
    sam2_build = None
    build_sam2 = None
    SAM2ImagePredictor = None
    _sam2_import_error = ex
else:
    _sam2_import_error = None

try:
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
except Exception as ex:  # pragma: no cover - runtime env specific
    AutoModelForZeroShotObjectDetection = None
    AutoProcessor = None
    _groundingdino_import_error = ex
else:
    _groundingdino_import_error = None

try:
    from transnetv2_pytorch import TransNetV2 as _TransNetV2
except Exception as ex:  # pragma: no cover - runtime env specific
    _TransNetV2 = None
    _transnet_import_error = ex
else:
    _transnet_import_error = None


SAM2_MODEL_DIR = "sam2"
OPENSHOT_NODEPACK_VERSION = "v1.1.2-track-object-keyframes"
GROUNDING_DINO_MODEL_IDS = (
    "IDEA-Research/grounding-dino-tiny",
    "IDEA-Research/grounding-dino-base",
)
GROUNDING_DINO_CACHE = {}
def _sam2_debug_enabled():
    # Temporary: always-on debug while we diagnose chunk/carry drift.
    return True


def _sam2_debug(*parts):
    if _sam2_debug_enabled():
        try:
            print("[OpenShot-SAM2-DEBUG]", *parts)
        except Exception:
            pass


SAM2_MODELS = {
    "sam2.1_hiera_tiny.safetensors": {
        "url": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt",
        "config": "sam2.1_hiera_t.yaml",
    },
    "sam2.1_hiera_small.safetensors": {
        "url": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt",
        "config": "sam2.1_hiera_s.yaml",
    },
    "sam2.1_hiera_base_plus.safetensors": {
        "url": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt",
        "config": "sam2.1_hiera_b+.yaml",
    },
    "sam2.1_hiera_large.safetensors": {
        "url": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt",
        "config": "sam2.1_hiera_l.yaml",
    },
}


def _require_sam2():
    if build_sam2 is None or SAM2ImagePredictor is None:
        raise RuntimeError(
            "SAM2 imports failed. Ensure `sam2` is available in Comfy runtime. Error: {}".format(_sam2_import_error)
        )


def _require_groundingdino():
    if AutoModelForZeroShotObjectDetection is None or AutoProcessor is None:
        raise RuntimeError(
            "GroundingDINO imports failed. Install requirements and restart ComfyUI. Error: {}".format(
                _groundingdino_import_error
            )
        )


def _require_transnet():
    if _TransNetV2 is None:
        raise RuntimeError(
            "TransNetV2 imports failed. Install `transnetv2-pytorch` and restart ComfyUI. Error: {}".format(
                _transnet_import_error
            )
        )


def _model_storage_dir():
    path = os.path.join(folder_paths.models_dir, SAM2_MODEL_DIR)
    os.makedirs(path, exist_ok=True)
    return path


def _safe_get_filename_list(model_dir_name):
    try:
        return list(folder_paths.get_filename_list(model_dir_name) or [])
    except Exception:
        # Folder key may not be registered in some Comfy installs.
        path = os.path.join(folder_paths.models_dir, model_dir_name)
        if not os.path.isdir(path):
            return []
        return sorted(
            name
            for name in os.listdir(path)
            if os.path.isfile(os.path.join(path, name))
        )


def _safe_get_full_path(model_dir_name, name):
    try:
        full = folder_paths.get_full_path(model_dir_name, name)
        if full:
            return full
    except Exception:
        pass
    fallback = os.path.join(folder_paths.models_dir, model_dir_name, name)
    if os.path.exists(fallback):
        return fallback
    return ""


def _model_options():
    available = set(_safe_get_filename_list(SAM2_MODEL_DIR))
    merged = list(SAM2_MODELS.keys())
    for name in sorted(available):
        if name not in merged:
            merged.append(name)
    return merged


def _download_if_needed(model_name):
    model_name = str(model_name or "").strip()
    if not model_name:
        raise ValueError("Model name is required")

    full_path = _safe_get_full_path(SAM2_MODEL_DIR, model_name)
    if full_path and os.path.exists(full_path):
        return full_path

    if model_name not in SAM2_MODELS:
        raise ValueError("Model not found locally and no download mapping for '{}'".format(model_name))

    url = SAM2_MODELS[model_name]["url"]
    parsed = urlparse(url)
    src_name = os.path.basename(parsed.path)
    target = os.path.join(_model_storage_dir(), src_name)
    if not os.path.exists(target):
        download_url_to_file(url, target)
    return target


def _resolve_config_candidates(model_name, checkpoint_path):
    candidates = []

    info = SAM2_MODELS.get(model_name)
    if info and info.get("config"):
        candidates.append(str(info["config"]))

    base = os.path.basename(checkpoint_path).replace(".pt", "")
    variants = {
        base,
        base.replace("2.1", "2_1"),
        base.replace("2.1", "2"),
        base.replace("sam2.1", "sam2"),
        base.replace("sam2_1", "sam2"),
    }
    for variant in sorted(variants):
        candidates.append("{}.yaml".format(variant))

    # De-duplicate while preserving order.
    seen = set()
    ordered = []
    for name in candidates:
        if name in seen:
            continue
        seen.add(name)
        ordered.append(name)
    return ordered


def _pack_config_dir():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "sam2_configs")


def _init_hydra_for_local_configs():
    cfg_dir = _pack_config_dir()
    if not os.path.isdir(cfg_dir):
        raise RuntimeError("OpenShot SAM2 config directory not found: {}".format(cfg_dir))
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    initialize_config_dir(config_dir=cfg_dir, version_base=None)


def _to_device_dtype(device_name, precision):
    device_name = str(device_name or "").strip().lower()
    if device_name in ("", "auto"):
        device = mm.get_torch_device()
    elif device_name == "cpu":
        device = torch.device("cpu")
    elif device_name == "cuda":
        device = torch.device("cuda")
    elif device_name == "mps":
        device = torch.device("mps")
    else:
        device = mm.get_torch_device()

    precision = str(precision or "fp16").strip().lower()
    if precision == "bf16":
        dtype = torch.bfloat16
    elif precision == "fp32":
        dtype = torch.float32
    else:
        dtype = torch.float16
    return device, dtype


def _parse_points(text):
    text = str(text or "").strip()
    if not text:
        return []
    try:
        parsed = json.loads(text.replace("'", '"'))
    except Exception:
        return []
    if not isinstance(parsed, list):
        return []
    pts = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        try:
            pts.append((float(item["x"]), float(item["y"])))
        except Exception:
            continue
    return pts


def _parse_rects(text):
    text = str(text or "").strip()
    if not text:
        return []
    try:
        parsed = json.loads(text.replace("'", '"'))
    except Exception:
        return []
    if not isinstance(parsed, list):
        return []
    out = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        if all(k in item for k in ("x1", "y1", "x2", "y2")):
            try:
                x1 = float(item["x1"])
                y1 = float(item["y1"])
                x2 = float(item["x2"])
                y2 = float(item["y2"])
            except Exception:
                continue
        elif all(k in item for k in ("x", "y", "w", "h")):
            try:
                x1 = float(item["x"])
                y1 = float(item["y"])
                x2 = x1 + float(item["w"])
                y2 = y1 + float(item["h"])
            except Exception:
                continue
        else:
            continue
        out.append((x1, y1, x2, y2))
    return out


def _parse_tracking_selection(text):
    text = str(text or "").strip()
    if not text:
        return {"seed_frame_idx": 0, "schedule": {}}
    try:
        parsed = json.loads(text.replace("'", '"'))
    except Exception:
        return {"seed_frame_idx": 0, "schedule": {}}
    if not isinstance(parsed, dict):
        return {"seed_frame_idx": 0, "schedule": {}}

    try:
        seed_frame_idx = max(0, int(parsed.get("seed_frame", 1)) - 1)
    except Exception:
        seed_frame_idx = 0

    frames = parsed.get("frames", {})
    if not isinstance(frames, dict):
        frames = {}

    schedule = {}
    for frame_key, frame_data in frames.items():
        if not isinstance(frame_data, dict):
            continue
        try:
            frame_idx = int(frame_key)
        except Exception:
            continue
        frame_idx = max(0, frame_idx - 1)

        pos = []
        neg = []
        for item in frame_data.get("positive_points", []) or []:
            if not isinstance(item, dict):
                continue
            try:
                pos.append((float(item["x"]), float(item["y"])))
            except Exception:
                continue
        for item in frame_data.get("negative_points", []) or []:
            if not isinstance(item, dict):
                continue
            try:
                neg.append((float(item["x"]), float(item["y"])))
            except Exception:
                continue

        pos_rects = []
        neg_rects = []
        for item in frame_data.get("positive_rects", []) or []:
            if not isinstance(item, dict):
                continue
            try:
                pos_rects.append(
                    (
                        float(item["x1"]),
                        float(item["y1"]),
                        float(item["x2"]),
                        float(item["y2"]),
                    )
                )
            except Exception:
                continue
        for item in frame_data.get("negative_rects", []) or []:
            if not isinstance(item, dict):
                continue
            try:
                neg_rects.append(
                    (
                        float(item["x1"]),
                        float(item["y1"]),
                        float(item["x2"]),
                        float(item["y2"]),
                    )
                )
            except Exception:
                continue

        points = []
        labels = []
        object_prompts = []
        for idx, (x, y) in enumerate(pos):
            obj_id = int(idx)
            points.append((x, y))
            labels.append(1)
            object_prompts.append(
                {
                    "obj_id": obj_id,
                    "points": [(x, y)],
                    "labels": [1],
                    "positive_rects": [],
                }
            )
        for x, y in neg:
            points.append((x, y))
            labels.append(0)
        for extra_idx, rect in enumerate(pos_rects):
            obj_id = int(len(object_prompts) + extra_idx)
            object_prompts.append(
                {
                    "obj_id": obj_id,
                    "points": [],
                    "labels": [],
                    "positive_rects": [rect],
                }
            )

        if points or pos_rects or neg_rects:
            schedule[int(frame_idx)] = {
                "points": points,
                "labels": labels,
                "positive_rects": pos_rects,
                "negative_rects": neg_rects,
                "object_prompts": object_prompts,
            }

    return {"seed_frame_idx": int(seed_frame_idx), "schedule": schedule}


def _clip_rect(rect, width, height):
    x1, y1, x2, y2 = [float(v) for v in rect]
    left = max(0, min(int(np.floor(min(x1, x2))), int(width)))
    top = max(0, min(int(np.floor(min(y1, y2))), int(height)))
    right = max(0, min(int(np.ceil(max(x1, x2))), int(width)))
    bottom = max(0, min(int(np.ceil(max(y1, y2))), int(height)))
    if right <= left or bottom <= top:
        return None
    return (left, top, right, bottom)


def _rect_center_points(rects):
    out = []
    for x1, y1, x2, y2 in rects:
        out.append(((float(x1) + float(x2)) * 0.5, (float(y1) + float(y2)) * 0.5))
    return out


def _mask_stack_like(base_mask, image):
    if base_mask is None:
        return None
    mask = base_mask.float()
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    if mask.ndim == 4:
        mask = mask.squeeze(-1)
    if mask.ndim != 3:
        return None
    b = int(image.shape[0])
    h = int(image.shape[1])
    w = int(image.shape[2])
    if int(mask.shape[0]) == 1 and b > 1:
        mask = mask.repeat(b, 1, 1)
    if int(mask.shape[0]) != b:
        return None
    if int(mask.shape[1]) != h or int(mask.shape[2]) != w:
        mask = F.interpolate(mask.unsqueeze(1), size=(h, w), mode="nearest").squeeze(1)
    return torch.clamp(mask, 0.0, 1.0)


def _apply_negative_rects(mask_tensor, negative_rects):
    if mask_tensor is None or not negative_rects:
        return mask_tensor
    if mask_tensor.ndim != 3:
        return mask_tensor
    h = int(mask_tensor.shape[1])
    w = int(mask_tensor.shape[2])
    out = mask_tensor.clone()
    for rect in negative_rects:
        clipped = _clip_rect(rect, w, h)
        if not clipped:
            continue
        left, top, right, bottom = clipped
        out[:, top:bottom, left:right] = 0.0
    return out


def _tensor_to_pil_image(img):
    arr = torch.clamp(img, 0.0, 1.0).mul(255.0).byte().cpu().numpy()
    return Image.fromarray(arr)


def _resolve_dino_device(device_name):
    device_name = str(device_name or "auto").strip().lower()
    if device_name == "auto":
        return mm.get_torch_device()
    return torch.device(device_name)


def _get_groundingdino_model_and_processor(model_id, device):
    key = "{}::{}".format(str(model_id), str(device))
    if key in GROUNDING_DINO_CACHE:
        return GROUNDING_DINO_CACHE[key]
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)
    model.to(device)
    model.eval()
    GROUNDING_DINO_CACHE[key] = (processor, model)
    return processor, model


def _detect_groundingdino_boxes(image_tensor, prompt, model_id, box_threshold, text_threshold, device_name):
    prompt = str(prompt or "").strip()
    if not prompt:
        return []
    _require_groundingdino()
    if not prompt.endswith("."):
        prompt = "{}.".format(prompt)
    if image_tensor is None or int(image_tensor.shape[0]) <= 0:
        return []

    device = _resolve_dino_device(device_name)
    processor, model = _get_groundingdino_model_and_processor(model_id, device)
    pil = _tensor_to_pil_image(image_tensor[0])
    h = int(image_tensor.shape[1])
    w = int(image_tensor.shape[2])
    with torch.inference_mode():
        inputs = processor(images=pil, text=prompt, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model(**inputs)
        post_kwargs = {
            "target_sizes": [(h, w)],
            "text_threshold": float(text_threshold),
        }
        try:
            result = processor.post_process_grounded_object_detection(
                outputs,
                inputs["input_ids"],
                box_threshold=float(box_threshold),
                **post_kwargs,
            )[0]
        except TypeError:
            try:
                result = processor.post_process_grounded_object_detection(
                    outputs,
                    inputs["input_ids"],
                    threshold=float(box_threshold),
                    **post_kwargs,
                )[0]
            except TypeError:
                result = processor.post_process_grounded_object_detection(
                    outputs,
                    inputs["input_ids"],
                    threshold=float(box_threshold),
                    target_sizes=[(h, w)],
                )[0]
        boxes = result.get("boxes")
        if boxes is None or boxes.numel() == 0:
            return []
        boxes_cpu = boxes.detach().cpu()
        return [tuple(float(v) for v in boxes_cpu[i].tolist()) for i in range(int(boxes_cpu.shape[0]))]


def _sam2_add_prompts(model, state, frame_idx, obj_id, coords, labels, positive_rects):
    errors = []
    if coords is not None and labels is not None and len(coords) > 0 and len(labels) > 0:
        for call in (
            lambda: model.add_new_points(
                inference_state=state,
                frame_idx=int(frame_idx),
                obj_id=int(obj_id),
                points=coords,
                labels=labels,
            ),
            lambda: model.add_new_points_or_box(
                inference_state=state,
                frame_idx=int(frame_idx),
                obj_id=int(obj_id),
                points=coords,
                labels=labels,
            ),
        ):
            try:
                call()
                break
            except Exception as ex:
                errors.append(str(ex))
        else:
            raise RuntimeError("Failed SAM2 add points across API variants: {}".format(errors))

    for rect in positive_rects or []:
        box = np.array([float(rect[0]), float(rect[1]), float(rect[2]), float(rect[3])], dtype=np.float32)
        rect_errors = []
        for call in (
            lambda: model.add_new_points_or_box(
                inference_state=state,
                frame_idx=int(frame_idx),
                obj_id=int(obj_id),
                box=box,
            ),
            lambda: model.add_new_points_or_box(
                inference_state=state,
                frame_idx=int(frame_idx),
                obj_id=int(obj_id),
                points=np.empty((0, 2), dtype=np.float32),
                labels=np.empty((0,), dtype=np.int32),
                box=box,
            ),
        ):
            try:
                call()
                rect_errors = []
                break
            except Exception as ex:
                rect_errors.append(str(ex))
        if rect_errors:
            errors.extend(rect_errors)
    return errors


def _resolve_video_path_for_sam2(path_text):
    """Resolve Comfy-style path text to an absolute local file path for SAM2 video predictor."""
    path_text = str(path_text or "").strip()
    if not path_text:
        return ""
    # Strip Comfy annotation suffixes if present.
    if path_text.endswith("]") and " [" in path_text:
        path_text = path_text.rsplit(" [", 1)[0].strip()

    if os.path.isabs(path_text) and os.path.exists(path_text):
        return path_text

    # Handles plain names and annotated names like "clip.mp4 [input]".
    try:
        resolved = folder_paths.get_annotated_filepath(path_text)
        if resolved and os.path.exists(resolved):
            return resolved
    except Exception:
        pass

    # Fallback to Comfy input directory.
    try:
        candidate = os.path.join(folder_paths.get_input_directory(), path_text)
        if os.path.exists(candidate):
            return candidate
        # fallback to basename if caller passed nested/odd relative path tokens
        candidate2 = os.path.join(folder_paths.get_input_directory(), os.path.basename(path_text))
        if os.path.exists(candidate2):
            return candidate2
    except Exception:
        pass

    return path_text


def _ensure_mp4_for_sam2(video_path):
    """Convert non-MP4 input videos to MP4 for SAM2VideoPredictor compatibility."""
    video_path = str(video_path or "").strip()
    if not video_path:
        return video_path
    ext = os.path.splitext(video_path)[1].lower()
    if ext == ".mp4":
        return video_path
    if not os.path.isfile(video_path):
        return video_path

    cache_dir = os.path.join(folder_paths.get_temp_directory(), "openshot_sam2_mp4_cache")
    os.makedirs(cache_dir, exist_ok=True)

    st = os.stat(video_path)
    key = "{}|{}|{}".format(video_path, int(st.st_mtime_ns), int(st.st_size))
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
    out_path = os.path.join(cache_dir, "{}.mp4".format(digest))
    if os.path.exists(out_path):
        return out_path

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        video_path,
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        "18",
        out_path,
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    except FileNotFoundError:
        raise RuntimeError("ffmpeg not found; required to convert '{}' to MP4".format(video_path))
    except subprocess.CalledProcessError as ex:
        err = (ex.stderr or "").strip()
        if len(err) > 500:
            err = err[:500] + "...(truncated)"
        raise RuntimeError("ffmpeg conversion to MP4 failed: {}".format(err))
    return out_path


def _load_video_frame_tensor_for_dino(video_path, frame_index=0):
    """Load one RGB frame from video as IMAGE tensor shape [1,H,W,C] in 0..1."""
    vp = _resolve_video_path_for_sam2(video_path)
    vp = _ensure_mp4_for_sam2(vp)
    if not vp or (not os.path.isfile(vp)):
        return None

    try:
        frame_index = int(max(0, frame_index))
    except Exception:
        frame_index = 0

    tmp_dir = tempfile.mkdtemp(prefix="openshot_dino_frame_", dir=folder_paths.get_temp_directory())
    out_png = os.path.join(tmp_dir, "seed.png")
    filter_expr = r"select=eq(n\,{})".format(frame_index)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        vp,
        "-vf",
        filter_expr,
        "-vframes",
        "1",
        out_png,
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        if not os.path.isfile(out_png):
            return None
        pil = Image.open(out_png).convert("RGB")
        arr = np.asarray(pil, dtype=np.float32) / 255.0
        return torch.from_numpy(arr).unsqueeze(0)
    except Exception:
        return None
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _build_sam2_video_predictor(config_name, checkpoint, torch_device):
    """Build a SAM2 video predictor across package variants."""
    if sam2_build is None:
        raise RuntimeError("sam2.build_sam module unavailable")

    candidate_names = (
        "build_sam2_video_predictor",
        "build_video_predictor",
        "build_sam_video_predictor",
    )
    found = []
    last_error = None
    for name in candidate_names:
        fn = getattr(sam2_build, name, None)
        if not callable(fn):
            continue
        found.append(name)
        for kwargs in (
            {"device": torch_device},
            {},
        ):
            try:
                return fn(config_name, checkpoint, **kwargs)
            except TypeError:
                continue
            except Exception as ex:
                last_error = ex
                continue
    raise RuntimeError(
        "Could not build SAM2 video predictor. Found builders={} last_error={}".format(found, last_error)
    )


class OpenShotTransNetSceneDetect:
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return ""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source_video_path": ("STRING", {"default": ""}),
                "threshold": ("FLOAT", {"default": 0.50, "min": 0.01, "max": 0.99, "step": 0.01}),
                "min_scene_length_frames": ("INT", {"default": 30, "min": 1, "max": 10000}),
                "device": (["auto", "cuda", "cpu", "mps"], {"default": "auto"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("scene_ranges_json",)
    FUNCTION = "detect"
    CATEGORY = "OpenShot/Video"

    def _resolve_device_name(self, device_name):
        value = str(device_name or "auto").strip().lower()
        if value != "auto":
            return value
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _build_model(self, device_name):
        errors = []
        for kwargs in (
            {"device": device_name},
            {},
        ):
            try:
                return _TransNetV2(**kwargs)
            except Exception as ex:
                errors.append(str(ex))
        raise RuntimeError("Failed to initialize TransNetV2 model: {}".format(errors[:2]))

    def _extract_scenes(self, raw):
        fps = None
        scenes = None

        if isinstance(raw, dict):
            scenes = raw.get("scenes")
            fps_value = raw.get("fps")
            try:
                if fps_value is not None:
                    fps = float(fps_value)
            except Exception:
                fps = None
        else:
            scenes = raw

        normalized = []
        if isinstance(scenes, np.ndarray):
            scenes = scenes.tolist()

        if isinstance(scenes, list):
            for entry in scenes:
                start = end = None
                if isinstance(entry, dict):
                    start = entry.get("start_seconds", entry.get("start_time", entry.get("start")))
                    end = entry.get("end_seconds", entry.get("end_time", entry.get("end")))
                elif isinstance(entry, (list, tuple)) and len(entry) >= 2:
                    start, end = entry[0], entry[1]
                try:
                    start_f = float(start)
                    end_f = float(end)
                except Exception:
                    continue
                if end_f <= start_f:
                    continue
                normalized.append((start_f, end_f))
        return normalized, fps

    def _run_inference(self, model, video_path, threshold):
        errors = []
        for fn_name in ("detect_scenes", "analyze_video", "predict_video"):
            fn = getattr(model, fn_name, None)
            if not callable(fn):
                continue
            for kwargs in (
                {"threshold": float(threshold)},
                {},
            ):
                try:
                    return fn(video_path, **kwargs)
                except TypeError:
                    continue
                except Exception as ex:
                    errors.append("{}: {}".format(fn_name, ex))
                    break
        raise RuntimeError("TransNetV2 inference failed: {}".format(errors[:2]))

    def _apply_min_scene_length(self, scenes, fps, min_scene_length_frames):
        if not scenes:
            return []
        if not fps or fps <= 0:
            return scenes
        min_seconds = float(min_scene_length_frames) / float(fps)
        if min_seconds <= 0:
            return scenes

        out = []
        for start_sec, end_sec in scenes:
            if not out:
                out.append([start_sec, end_sec])
                continue
            duration = end_sec - start_sec
            if duration < min_seconds:
                out[-1][1] = max(out[-1][1], end_sec)
                continue
            out.append([start_sec, end_sec])
        return [(float(s), float(e)) for s, e in out if e > s]

    def detect(self, source_video_path, threshold, min_scene_length_frames, device):
        _require_transnet()
        video_path = _resolve_video_path_for_sam2(source_video_path)
        if not video_path or not os.path.exists(video_path):
            raise ValueError("Video path not found: {}".format(source_video_path))

        device_name = self._resolve_device_name(device)
        model = self._build_model(device_name)
        raw = self._run_inference(model, video_path, threshold)
        scenes, fps = self._extract_scenes(raw)
        scenes = sorted(scenes, key=lambda item: (item[0], item[1]))
        scenes = self._apply_min_scene_length(scenes, fps, int(min_scene_length_frames))

        payload = {
            "version": 1,
            "detector": "openshot-transnetv2",
            "source_video_path": str(video_path),
            "fps": float(fps) if fps else None,
            "segments": [
                {
                    "index": idx,
                    "start_seconds": round(float(start_sec), 6),
                    "end_seconds": round(float(end_sec), 6),
                }
                for idx, (start_sec, end_sec) in enumerate(scenes, start=1)
            ],
        }
        return (json.dumps(payload),)


def _probe_video_info(path_text):
    """Probe basic video metadata via ffprobe."""
    path_text = str(path_text or "").strip()
    if not path_text:
        return {}
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=avg_frame_rate,r_frame_rate:format=duration",
        "-of",
        "json",
        path_text,
    ]
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    except Exception:
        return {}
    try:
        payload = json.loads(result.stdout or "{}")
    except Exception:
        return {}

    stream = {}
    streams = payload.get("streams")
    if isinstance(streams, list) and streams:
        stream = streams[0] if isinstance(streams[0], dict) else {}
    fmt = payload.get("format") if isinstance(payload.get("format"), dict) else {}

    def _parse_rate(text_value):
        text_value = str(text_value or "").strip()
        if not text_value or text_value in ("0/0", "N/A"):
            return None
        if "/" in text_value:
            try:
                frac = Fraction(text_value)
                if frac > 0:
                    return frac
            except Exception:
                return None
        try:
            value = float(text_value)
            if value > 0:
                return Fraction(value).limit_denominator(1000000)
        except Exception:
            return None
        return None

    fps = _parse_rate(stream.get("avg_frame_rate")) or _parse_rate(stream.get("r_frame_rate"))
    duration = None
    try:
        duration = float(fmt.get("duration"))
    except Exception:
        duration = None

    return {
        "fps": fps,
        "duration": duration,
    }


class OpenShotSceneRangesFromSegments:
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return ""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "segment_paths": ("*",),
                "source_video_path": ("STRING", {"default": ""}),
            },
            "optional": {
                "fallback_fps": ("FLOAT", {"default": 30.0, "min": 1.0, "max": 240.0, "step": 0.001}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("scene_ranges_json",)
    FUNCTION = "build"
    CATEGORY = "OpenShot/Video"

    def _as_path_list(self, segment_paths):
        if isinstance(segment_paths, (list, tuple)):
            return [str(p).strip() for p in segment_paths if str(p or "").strip()]
        if isinstance(segment_paths, str):
            text = segment_paths.strip()
            if not text:
                return []
            try:
                parsed = json.loads(text)
                if isinstance(parsed, list):
                    return [str(p).strip() for p in parsed if str(p or "").strip()]
            except Exception:
                pass
            return [text]
        return []

    def _timecode(self, seconds_value, fps_fraction):
        fps_fraction = fps_fraction if isinstance(fps_fraction, Fraction) and fps_fraction > 0 else Fraction(30, 1)
        fps_float = float(fps_fraction)
        total_seconds = max(0.0, float(seconds_value or 0.0))
        hours = int(total_seconds // 3600)
        minutes = int((total_seconds % 3600) // 60)
        secs = int(total_seconds % 60)
        frames = int(round((total_seconds - int(total_seconds)) * fps_float))
        fps_ceiling = int(round(fps_float)) or 1
        if frames >= fps_ceiling:
            frames = 0
            secs += 1
            if secs >= 60:
                secs = 0
                minutes += 1
                if minutes >= 60:
                    minutes = 0
                    hours += 1
        if hours > 0:
            return "{:02d}:{:02d}:{:02d};{:02d}".format(hours, minutes, secs, frames)
        if minutes > 0:
            return "{:02d}:{:02d};{:02d}".format(minutes, secs, frames)
        return "{:02d};{:02d}".format(secs, frames)

    def build(self, segment_paths, source_video_path, fallback_fps=30.0):
        paths = self._as_path_list(segment_paths)
        if not paths:
            return (json.dumps({"segments": []}),)

        source_info = _probe_video_info(source_video_path)
        fps_fraction = source_info.get("fps")
        if fps_fraction is None or fps_fraction <= 0:
            try:
                fps_fraction = Fraction(float(fallback_fps)).limit_denominator(1000000)
            except Exception:
                fps_fraction = Fraction(30, 1)
        fps_float = float(fps_fraction)

        source_duration = source_info.get("duration")
        running_start = 0.0
        segments = []

        for idx, segment_path in enumerate(paths, start=1):
            info = _probe_video_info(segment_path)
            duration = info.get("duration")
            if duration is None:
                continue
            duration = max(0.0, float(duration))
            if duration <= 0.0:
                continue
            start_seconds = running_start
            end_seconds = running_start + duration
            if source_duration is not None:
                end_seconds = min(end_seconds, float(source_duration))
            if end_seconds <= start_seconds:
                continue

            start_frame = int(round(start_seconds * fps_float)) + 1
            end_frame = int(round(end_seconds * fps_float))
            if end_frame < start_frame:
                end_frame = start_frame

            segments.append(
                {
                    "index": idx,
                    "path": str(segment_path),
                    "start_seconds": round(start_seconds, 6),
                    "end_seconds": round(end_seconds, 6),
                    "duration_seconds": round(end_seconds - start_seconds, 6),
                    "start_frame": int(start_frame),
                    "end_frame": int(end_frame),
                    "start_timecode": self._timecode(start_seconds, fps_fraction),
                    "end_timecode": self._timecode(end_seconds, fps_fraction),
                }
            )
            running_start = end_seconds

        payload = {
            "version": 1,
            "source_video_path": str(source_video_path or ""),
            "fps": {
                "num": int(fps_fraction.numerator),
                "den": int(fps_fraction.denominator),
                "float": fps_float,
            },
            "segments": segments,
        }
        return (json.dumps(payload),)


class OpenShotDownloadAndLoadSAM2Model:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (_model_options(),),
                "segmentor": (["video", "single_image"], {"default": "video"}),
                "device": (["auto", "cuda", "cpu", "mps"], {"default": "auto"}),
                "precision": (["fp16", "bf16", "fp32"], {"default": "fp16"}),
            }
        }

    RETURN_TYPES = ("SAM2MODEL",)
    RETURN_NAMES = ("sam2_model",)
    FUNCTION = "load"
    CATEGORY = "OpenShot/SAM2"

    def load(self, model, segmentor, device, precision):
        _require_sam2()

        checkpoint = _download_if_needed(model)
        config_candidates = _resolve_config_candidates(model, checkpoint)
        torch_device, dtype = _to_device_dtype(device, precision)

        _init_hydra_for_local_configs()
        print(
            "[OpenShot-ComfyUI:{}] Loading SAM2 model='{}' checkpoint='{}' configs={}".format(
                OPENSHOT_NODEPACK_VERSION, model, checkpoint, config_candidates
            )
        )

        sam_model = None
        last_error = None
        for config_name in config_candidates:
            try:
                if str(segmentor or "video") == "video":
                    sam_model = _build_sam2_video_predictor(config_name, checkpoint, torch_device)
                else:
                    sam_model = build_sam2(config_name, checkpoint, device=torch_device)
                break
            except Exception as ex:
                last_error = ex
                # Missing config names are expected across SAM2 package variants.
                if "Cannot find primary config" in str(ex):
                    continue
                raise
        if sam_model is None:
            raise RuntimeError(
                "Failed loading SAM2 model. Tried configs {}. Last error: {}".format(config_candidates, last_error)
            )
        return ({
            "model": sam_model,
            "device": torch_device,
            "dtype": dtype,
            "segmentor": str(segmentor or "video"),
            "model_name": str(model),
            "checkpoint": str(checkpoint),
        },)


class OpenShotSam2Segmentation:
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return ""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sam2_model": ("SAM2MODEL",),
                "image": ("IMAGE",),
                "auto_mode": ("BOOLEAN", {"default": False}),
                "keep_model_loaded": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "positive_points_json": ("STRING", {"default": ""}),
                "negative_points_json": ("STRING", {"default": ""}),
                "positive_rects_json": ("STRING", {"default": ""}),
                "negative_rects_json": ("STRING", {"default": ""}),
                "base_mask": ("MASK",),
                "meta_batch": ("VHS_BatchManager",),
            },
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "segment"
    CATEGORY = "OpenShot/SAM2"

    def segment(
        self,
        sam2_model,
        image,
        auto_mode,
        keep_model_loaded,
        positive_points_json="",
        negative_points_json="",
        positive_rects_json="",
        negative_rects_json="",
        base_mask=None,
    ):
        _require_sam2()

        model = sam2_model["model"]
        device = sam2_model["device"]
        dtype = sam2_model["dtype"]

        positive = _parse_points(positive_points_json)
        negative = _parse_points(negative_points_json)
        positive_rects = _parse_rects(positive_rects_json)
        negative_rects = _parse_rects(negative_rects_json)

        predictor = SAM2ImagePredictor(model)
        base_mask_stack = _mask_stack_like(base_mask, image)

        out_masks = []
        autocast_device = mm.get_autocast_device(device)
        autocast_ok = not mm.is_device_mps(device)
        with torch.autocast(autocast_device, dtype=dtype) if autocast_ok else nullcontext():
            for frame_idx, frame in enumerate(image):
                frame_np = np.clip((frame.cpu().numpy() * 255.0), 0, 255).astype(np.uint8)
                predictor.set_image(frame_np[..., :3])
                h, w = frame_np.shape[0], frame_np.shape[1]

                final_mask = torch.zeros((h, w), dtype=torch.float32)
                if base_mask_stack is not None:
                    final_mask = torch.maximum(final_mask, (base_mask_stack[frame_idx].cpu() > 0.5).float())

                seed_points = list(positive)
                if bool(auto_mode) and not seed_points and not positive_rects and base_mask_stack is None:
                    seed_points = [(float(w) * 0.5, float(h) * 0.5)]

                if seed_points or negative:
                    pos_arr = np.array(seed_points, dtype=np.float32) if seed_points else np.empty((0, 2), dtype=np.float32)
                    neg_arr = np.array(negative, dtype=np.float32) if negative else np.empty((0, 2), dtype=np.float32)
                    coords = np.concatenate((pos_arr, neg_arr), axis=0)
                    labels = np.concatenate(
                        (
                            np.ones((len(pos_arr),), dtype=np.int32),
                            np.zeros((len(neg_arr),), dtype=np.int32),
                        ),
                        axis=0,
                    )
                    masks, _scores, _logits = predictor.predict(
                        point_coords=coords,
                        point_labels=labels,
                        multimask_output=False,
                    )
                    final_mask = torch.maximum(final_mask, torch.from_numpy(masks[0]).float())

                for rect in positive_rects:
                    clipped = _clip_rect(rect, w, h)
                    if not clipped:
                        continue
                    left, top, right, bottom = clipped
                    box = np.array([float(left), float(top), float(right), float(bottom)], dtype=np.float32)
                    try:
                        masks, _scores, _logits = predictor.predict(box=box, multimask_output=False)
                    except TypeError:
                        masks, _scores, _logits = predictor.predict(box=box, point_coords=None, point_labels=None, multimask_output=False)
                    final_mask = torch.maximum(final_mask, torch.from_numpy(masks[0]).float())

                final_mask = _apply_negative_rects(final_mask.unsqueeze(0), negative_rects).squeeze(0)
                out_masks.append(torch.clamp(final_mask, 0.0, 1.0))

        if not keep_model_loaded:
            model.to(mm.unet_offload_device())
            mm.soft_empty_cache()

        return (torch.stack(out_masks, dim=0),)


class OpenShotSam2VideoSegmentationAddPoints:
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return ""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sam2_model": ("SAM2MODEL",),
                "frame_index": ("INT", {"default": 0, "min": 0}),
                "object_index": ("INT", {"default": 0, "min": 0}),
                "windowed_mode": ("BOOLEAN", {"default": True}),
                "offload_video_to_cpu": ("BOOLEAN", {"default": False}),
                "offload_state_to_cpu": ("BOOLEAN", {"default": False}),
                "auto_mode": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "image": ("IMAGE",),
                "video_path": ("STRING", {"default": ""}),
                "positive_points_json": ("STRING", {"default": ""}),
                "negative_points_json": ("STRING", {"default": ""}),
                "positive_rects_json": ("STRING", {"default": ""}),
                "negative_rects_json": ("STRING", {"default": ""}),
                "tracking_selection_json": ("STRING", {"default": "{}"}),
                "dino_prompt": ("STRING", {"default": ""}),
                "dino_model_id": (GROUNDING_DINO_MODEL_IDS,),
                "dino_box_threshold": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.01}),
                "dino_text_threshold": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01}),
                "dino_device": (("auto", "cpu", "cuda", "mps"),),
                "prev_inference_state": ("SAM2INFERENCESTATE",),
                "base_mask": ("MASK",),
            },
        }

    RETURN_TYPES = ("SAM2MODEL", "SAM2INFERENCESTATE")
    RETURN_NAMES = ("sam2_model", "inference_state")
    FUNCTION = "add_points"
    CATEGORY = "OpenShot/SAM2"

    def add_points(
        self,
        sam2_model,
        frame_index,
        object_index,
        windowed_mode,
        offload_video_to_cpu,
        offload_state_to_cpu,
        auto_mode,
        image=None,
        video_path="",
        positive_points_json="",
        negative_points_json="",
        positive_rects_json="",
        negative_rects_json="",
        tracking_selection_json="{}",
        dino_prompt="",
        dino_model_id="IDEA-Research/grounding-dino-tiny",
        dino_box_threshold=0.35,
        dino_text_threshold=0.25,
        dino_device="auto",
        prev_inference_state=None,
        base_mask=None,
        meta_batch=None,
    ):
        model = sam2_model["model"]
        device = sam2_model["device"]
        dtype = sam2_model["dtype"]
        segmentor = sam2_model.get("segmentor", "video")
        if segmentor != "video":
            raise ValueError("Loaded SAM2 model is not configured for video")

        pos = _parse_points(positive_points_json)
        neg = _parse_points(negative_points_json)
        pos_rects = _parse_rects(positive_rects_json)
        neg_rects = _parse_rects(negative_rects_json)
        tracking_selection = _parse_tracking_selection(tracking_selection_json)
        prompt_schedule = dict(tracking_selection.get("schedule") or {})
        seed_frame_idx = int(max(0, tracking_selection.get("seed_frame_idx", int(max(0, frame_index)))))

        # In windowed meta-batch mode, reuse evolving state between chunks so
        # prompts are not re-seeded from frame 1 on each batch.
        if bool(windowed_mode) and prev_inference_state is None and meta_batch is not None:
            try:
                cached_state = getattr(meta_batch, "_openshot_sam2_window_state", None)
                if isinstance(cached_state, dict) and cached_state.get("windowed_mode", False):
                    return (sam2_model, cached_state)
            except Exception:
                pass

        if base_mask is not None:
            mask_stack = _mask_stack_like(base_mask, image) if image is not None else None
            if mask_stack is not None and int(mask_stack.shape[0]) > 0:
                ys, xs = torch.where(mask_stack[0] > 0.5)
                if xs.numel() > 0:
                    pos.append((float(xs.float().mean().item()), float(ys.float().mean().item())))

        if bool(auto_mode) and (not pos) and (not pos_rects):
            if image is not None:
                h = int(image.shape[1])
                w = int(image.shape[2])
                pos = [(float(w) * 0.5, float(h) * 0.5)]

        dino_prompt = str(dino_prompt or "").strip()
        if dino_prompt:
            dino_image = image
            if dino_image is None and str(video_path or "").strip():
                dino_image = _load_video_frame_tensor_for_dino(video_path, seed_frame_idx)

            if dino_image is not None:
                try:
                    dino_boxes = _detect_groundingdino_boxes(
                        dino_image,
                        dino_prompt,
                        dino_model_id,
                        float(dino_box_threshold),
                        float(dino_text_threshold),
                        dino_device,
                    )
                    print(
                        "[OpenShot-ComfyUI:{}] DINO prompt='{}' boxes={} model='{}' box_th={} text_th={}".format(
                            OPENSHOT_NODEPACK_VERSION,
                            dino_prompt,
                            len(dino_boxes),
                            dino_model_id,
                            float(dino_box_threshold),
                            float(dino_text_threshold),
                        )
                    )
                except Exception as ex:
                    raise RuntimeError(
                        "GroundingDINO detection failed for prompt '{}': {}".format(dino_prompt, ex)
                    )
            else:
                dino_boxes = []
                print(
                    "[OpenShot-ComfyUI:{}] DINO prompt='{}' skipped (no image or video_path frame available)".format(
                        OPENSHOT_NODEPACK_VERSION,
                        dino_prompt,
                    )
                )

            if dino_boxes:
                _sam2_debug(
                    "dino-seed",
                    "prompt=", dino_prompt,
                    "seed_frame_idx=", int(seed_frame_idx),
                    "boxes=", len(dino_boxes),
                )
                seed_entry = dict(prompt_schedule.get(seed_frame_idx, {}) or {})
                seed_object_prompts = list(seed_entry.get("object_prompts") or [])
                next_obj_id = 0
                for op in seed_object_prompts:
                    try:
                        next_obj_id = max(next_obj_id, int(op.get("obj_id", 0)) + 1)
                    except Exception:
                        continue
                for box in dino_boxes:
                    seed_object_prompts.append(
                        {
                            "obj_id": int(next_obj_id),
                            "points": [],
                            "labels": [],
                            "positive_rects": [tuple(box)],
                        }
                    )
                    next_obj_id += 1
                seed_entry["object_prompts"] = seed_object_prompts
                seed_rects = list(seed_entry.get("positive_rects") or [])
                seed_rects.extend([tuple(b) for b in dino_boxes])
                seed_entry["positive_rects"] = seed_rects
                prompt_schedule[int(seed_frame_idx)] = seed_entry

        # Backward-compatible seed injection if no explicit keyframed payload exists.
        if seed_frame_idx not in prompt_schedule and (pos or neg or pos_rects or neg_rects):
            points = []
            labels = []
            for x, y in pos:
                points.append((float(x), float(y)))
                labels.append(1)
            for x, y in neg:
                points.append((float(x), float(y)))
                labels.append(0)
            prompt_schedule[int(seed_frame_idx)] = {
                "points": points,
                "labels": labels,
                "positive_rects": list(pos_rects),
                "negative_rects": list(neg_rects),
            }

        has_any_positive = False
        for entry in prompt_schedule.values():
            labels = list(entry.get("labels") or [])
            object_prompts = list(entry.get("object_prompts") or [])
            has_object_positive = any(bool((op or {}).get("positive_rects") or []) for op in object_prompts if isinstance(op, dict))
            if any(int(v) == 1 for v in labels) or bool(entry.get("positive_rects") or []) or has_object_positive:
                has_any_positive = True
                break
        allow_empty_schedule = bool(dino_prompt) or bool(auto_mode)
        if not has_any_positive and not allow_empty_schedule:
            raise ValueError("No positive points/rectangles provided")

        _sam2_debug(
            "add_points",
            "seed_frame_idx=", int(seed_frame_idx),
            "schedule_frames=", sorted([int(k) for k in prompt_schedule.keys()]),
            "windowed=", bool(windowed_mode),
            "has_prev_state=", bool(prev_inference_state is not None),
        )

        serial_schedule = []
        for fidx in sorted(prompt_schedule.keys()):
            entry = prompt_schedule.get(fidx, {}) or {}
            serial_schedule.append(
                {
                    "frame_idx": int(fidx),
                    "points": [[float(p[0]), float(p[1])] for p in (entry.get("points") or [])],
                    "labels": [int(v) for v in (entry.get("labels") or [])],
                    "positive_rects": [[float(r[0]), float(r[1]), float(r[2]), float(r[3])] for r in (entry.get("positive_rects") or [])],
                    "negative_rects": [[float(r[0]), float(r[1]), float(r[2]), float(r[3])] for r in (entry.get("negative_rects") or [])],
                    "object_prompts": [
                        {
                            "obj_id": int(op.get("obj_id", 0)),
                            "points": [[float(p[0]), float(p[1])] for p in (op.get("points") or [])],
                            "labels": [int(v) for v in (op.get("labels") or [])],
                            "positive_rects": [
                                [float(r[0]), float(r[1]), float(r[2]), float(r[3])]
                                for r in (op.get("positive_rects") or [])
                            ],
                        }
                        for op in (entry.get("object_prompts") or [])
                        if isinstance(op, dict)
                    ],
                }
            )

        # Keep these for backward compatibility / fallback behavior.
        if prompt_schedule:
            first_frame = int(sorted(prompt_schedule.keys())[0])
            first_entry = prompt_schedule.get(first_frame, {}) or {}
        else:
            first_frame = int(seed_frame_idx)
            first_entry = {}
        pos_seed = [tuple(p) for p, lbl in zip(first_entry.get("points") or [], first_entry.get("labels") or []) if int(lbl) == 1]
        neg_seed = [tuple(p) for p, lbl in zip(first_entry.get("points") or [], first_entry.get("labels") or []) if int(lbl) == 0]
        pos_arr = np.atleast_2d(np.array(pos_seed, dtype=np.float32)) if pos_seed else np.empty((0, 2), dtype=np.float32)
        neg_arr = np.atleast_2d(np.array(neg_seed, dtype=np.float32)) if neg_seed else np.empty((0, 2), dtype=np.float32)
        coords = np.concatenate((pos_arr, neg_arr), axis=0) if (len(pos_arr) or len(neg_arr)) else np.empty((0, 2), dtype=np.float32)
        labels = np.concatenate((np.ones((len(pos_arr),), dtype=np.int32), np.zeros((len(neg_arr),), dtype=np.int32)), axis=0) if (len(pos_arr) or len(neg_arr)) else np.empty((0,), dtype=np.int32)
        first_pos_rects = [tuple(r) for r in (first_entry.get("positive_rects") or [])]
        first_neg_rects = [tuple(r) for r in (first_entry.get("negative_rects") or [])]

        # Windowed mode does not hold full-video SAM2 state in memory.
        if bool(windowed_mode):
            state = dict(prev_inference_state or {})
            state["windowed_mode"] = True
            state["seed_points"] = coords.tolist()
            state["seed_labels"] = labels.tolist()
            state["last_points"] = coords.tolist()
            state["last_labels"] = labels.tolist()
            state["seed_rects"] = [[float(a), float(b), float(c), float(d)] for (a, b, c, d) in first_pos_rects]
            state["negative_rects"] = [[float(a), float(b), float(c), float(d)] for (a, b, c, d) in first_neg_rects]
            state["active_negative_rects"] = [[float(a), float(b), float(c), float(d)] for (a, b, c, d) in first_neg_rects]
            state["prompt_schedule"] = serial_schedule
            state["object_index"] = int(object_index)
            state["next_frame_idx"] = int(max(0, state.get("next_frame_idx", 0) or 0))
            state["num_frames"] = int(state.get("num_frames", 0) or 0)
            state["offload_video_to_cpu"] = bool(offload_video_to_cpu)
            state["offload_state_to_cpu"] = bool(offload_state_to_cpu)
            state["object_carries"] = dict(state.get("object_carries", {}) or {})
            state["prompt_frames_applied"] = list(state.get("prompt_frames_applied", []) or [])
            state["boundary_reseed_frames"] = int(max(1, state.get("boundary_reseed_frames", 4) or 4))
            if meta_batch is not None:
                try:
                    setattr(meta_batch, "_openshot_sam2_window_state", state)
                except Exception:
                    pass
            return (sam2_model, state)

        if (image is None and not str(video_path or "").strip()) and prev_inference_state is None:
            raise ValueError("Image or video_path input is required for initial inference state")

        model.to(device)
        if prev_inference_state is None:
            # Support SAM2 API variants for init_state signature.
            init_errors = []
            state = None
            num_frames = 0

            # Preferred path for newer SAM2 video predictors: initialize from source video path.
            if str(video_path or "").strip():
                vp = _resolve_video_path_for_sam2(video_path)
                vp = _ensure_mp4_for_sam2(vp)
                print(
                    "[OpenShot-ComfyUI:{}] SAM2 init_state path='{}' exists={} ext='{}'".format(
                        OPENSHOT_NODEPACK_VERSION,
                        vp,
                        os.path.exists(vp),
                        os.path.splitext(vp)[1].lower(),
                    )
                )
                # Prefer CPU-offloaded inference state to avoid huge VRAM spikes on long videos.
                for call in (
                    lambda: model.init_state(
                        vp,
                        offload_video_to_cpu=bool(offload_video_to_cpu),
                        offload_state_to_cpu=bool(offload_state_to_cpu),
                    ),
                    lambda: model.init_state(vp, offload_video_to_cpu=bool(offload_video_to_cpu)),
                    lambda: model.init_state(vp, offload_state_to_cpu=bool(offload_state_to_cpu)),
                    lambda: model.init_state(vp),
                    lambda: model.init_state(vp, device=device),
                ):
                    try:
                        state = call()
                        break
                    except Exception as ex:
                        init_errors.append(str(ex))

            # Fallback for tensor-accepting SAM2 variants.
            if state is None and image is not None:
                b, h, w, _c = image.shape
                if hasattr(model, "image_size"):
                    size = int(model.image_size)
                    image = common_upscale(image.movedim(-1, 1), size, size, "bilinear", "disabled").movedim(1, -1)
                video_tensor = image.permute(0, 3, 1, 2).contiguous()
                for call in (
                    lambda: model.init_state(video_tensor, h, w, device=device),
                    lambda: model.init_state(video_tensor, h, w),
                    lambda: model.init_state(video_tensor, device=device),
                    lambda: model.init_state(video_tensor),
                ):
                    try:
                        state = call()
                        num_frames = int(b)
                        break
                    except Exception as ex:
                        init_errors.append(str(ex))
            if state is None:
                short_errors = init_errors[:2]
                raise RuntimeError(
                    "SAM2 init_state failed; path='{}' exists={} ext='{}' errors={}".format(
                        vp if str(video_path or "").strip() else "",
                        (os.path.exists(vp) if str(video_path or "").strip() else False),
                        (os.path.splitext(vp)[1].lower() if str(video_path or "").strip() else ""),
                        short_errors,
                    )
                )
        else:
            state = prev_inference_state["inference_state"]
            num_frames = int(prev_inference_state.get("num_frames", 0) or 0)

        autocast_device = mm.get_autocast_device(device)
        autocast_ok = not mm.is_device_mps(device)
        with torch.inference_mode():
            with torch.autocast(autocast_device, dtype=dtype) if autocast_ok else nullcontext():
                add_errors = []
                if len(coords) or len(first_pos_rects):
                    add_errors = _sam2_add_prompts(
                        model,
                        state,
                        int(first_frame),
                        int(object_index),
                        coords,
                        labels,
                        first_pos_rects,
                    )
                if add_errors:
                    raise RuntimeError("Failed applying one or more SAM2 rectangle prompts: {}".format(add_errors[:3]))

        if num_frames <= 0:
            try:
                num_frames = int(state.get("num_frames", 0) or 0)
            except Exception:
                try:
                    num_frames = int(getattr(state, "num_frames", 0) or 0)
                except Exception:
                    num_frames = 0

        return (
            sam2_model,
            {
                "inference_state": state,
                "num_frames": num_frames,
                "next_frame_idx": 0,
                "negative_rects": [[float(a), float(b), float(c), float(d)] for (a, b, c, d) in first_neg_rects],
                "active_negative_rects": [[float(a), float(b), float(c), float(d)] for (a, b, c, d) in first_neg_rects],
                "seed_rects": [[float(a), float(b), float(c), float(d)] for (a, b, c, d) in first_pos_rects],
                "prompt_schedule": serial_schedule,
                "prompt_frames_applied": [int(first_frame)] if (len(coords) or len(first_pos_rects)) else [],
                "object_carries": {},
            },
        )


class OpenShotSam2VideoSegmentationChunked:
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return ""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sam2_model": ("SAM2MODEL",),
                "inference_state": ("SAM2INFERENCESTATE",),
                "image": ("IMAGE",),
                "start_frame": ("INT", {"default": 0, "min": 0}),
                "chunk_size_frames": ("INT", {"default": 32, "min": 1, "max": 4096}),
                "keep_model_loaded": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "meta_batch": ("VHS_BatchManager",),
            },
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "segment_chunk"
    CATEGORY = "OpenShot/SAM2"

    def _get_frames_per_batch(self, meta_batch, fallback):
        if meta_batch is None:
            return int(fallback)
        if isinstance(meta_batch, dict):
            for key in ("frames_per_batch", "batch_size", "frames"):
                try:
                    if key in meta_batch and int(meta_batch[key]) > 0:
                        return int(meta_batch[key])
                except Exception:
                    pass
        for name in ("frames_per_batch", "batch_size", "frames"):
            try:
                value = getattr(meta_batch, name)
                value = int(value)
                if value > 0:
                    return value
            except Exception:
                pass
        return int(fallback)

    def _write_window_jpegs(self, image):
        image_np = np.clip((image.detach().cpu().numpy() * 255.0), 0, 255).astype(np.uint8)
        root = os.path.join(folder_paths.get_temp_directory(), "openshot_sam2_windows")
        os.makedirs(root, exist_ok=True)
        name = "w{}_{}".format(int(time.time() * 1000), hashlib.sha256(os.urandom(16)).hexdigest()[:8])
        window_dir = os.path.join(root, name)
        os.makedirs(window_dir, exist_ok=True)
        for i, frame in enumerate(image_np):
            Image.fromarray(frame[..., :3], mode="RGB").save(
                os.path.join(window_dir, "{:05d}.jpg".format(i)),
                format="JPEG",
                quality=95,
            )
        return window_dir, int(image_np.shape[0]), int(image_np.shape[1]), int(image_np.shape[2])

    def _init_window_state(self, model, window_dir, device, inference_state):
        errs = []
        offload_video_to_cpu = bool(inference_state.get("offload_video_to_cpu", False))
        offload_state_to_cpu = bool(inference_state.get("offload_state_to_cpu", False))
        for call in (
            lambda: model.init_state(
                window_dir,
                offload_video_to_cpu=offload_video_to_cpu,
                offload_state_to_cpu=offload_state_to_cpu,
            ),
            lambda: model.init_state(window_dir, offload_video_to_cpu=offload_video_to_cpu),
            lambda: model.init_state(window_dir, offload_state_to_cpu=offload_state_to_cpu),
            lambda: model.init_state(window_dir),
            lambda: model.init_state(window_dir, device=device),
        ):
            try:
                return call()
            except Exception as ex:
                errs.append(str(ex))
        raise RuntimeError("SAM2 window init_state failed: {}".format(errs[:3]))

    def _prompt_schedule(self, inference_state):
        raw = inference_state.get("prompt_schedule") or []
        out = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            try:
                frame_idx = int(item.get("frame_idx", 0))
            except Exception:
                frame_idx = 0
            points = []
            for p in (item.get("points") or []):
                if not isinstance(p, (list, tuple)) or len(p) != 2:
                    continue
                try:
                    points.append((float(p[0]), float(p[1])))
                except Exception:
                    continue
            labels = []
            for v in (item.get("labels") or []):
                try:
                    labels.append(int(v))
                except Exception:
                    labels.append(0)
            pos_rects = []
            for r in (item.get("positive_rects") or []):
                if not isinstance(r, (list, tuple)) or len(r) != 4:
                    continue
                try:
                    pos_rects.append((float(r[0]), float(r[1]), float(r[2]), float(r[3])))
                except Exception:
                    continue
            neg_rects = []
            for r in (item.get("negative_rects") or []):
                if not isinstance(r, (list, tuple)) or len(r) != 4:
                    continue
                try:
                    neg_rects.append((float(r[0]), float(r[1]), float(r[2]), float(r[3])))
                except Exception:
                    continue
            object_prompts = []
            for op in (item.get("object_prompts") or []):
                if not isinstance(op, dict):
                    continue
                try:
                    obj_id = int(op.get("obj_id", 0))
                except Exception:
                    obj_id = 0
                op_points = []
                for p in (op.get("points") or []):
                    if not isinstance(p, (list, tuple)) or len(p) != 2:
                        continue
                    try:
                        op_points.append((float(p[0]), float(p[1])))
                    except Exception:
                        continue
                op_labels = []
                for v in (op.get("labels") or []):
                    try:
                        op_labels.append(int(v))
                    except Exception:
                        op_labels.append(0)
                op_rects = []
                for r in (op.get("positive_rects") or []):
                    if not isinstance(r, (list, tuple)) or len(r) != 4:
                        continue
                    try:
                        op_rects.append((float(r[0]), float(r[1]), float(r[2]), float(r[3])))
                    except Exception:
                        continue
                object_prompts.append(
                    {
                        "obj_id": int(max(0, obj_id)),
                        "points": op_points,
                        "labels": op_labels,
                        "positive_rects": op_rects,
                    }
                )

            out.append(
                {
                    "frame_idx": int(max(0, frame_idx)),
                    "points": points,
                    "labels": labels,
                    "positive_rects": pos_rects,
                    "negative_rects": neg_rects,
                    "object_prompts": object_prompts,
                }
            )
        out.sort(key=lambda x: int(x.get("frame_idx", 0)))
        return out

    def _apply_prompt_entry(self, model, state_obj, inference_state, frame_idx, entry):
        points_list = list(entry.get("points") or [])
        labels_list = [int(v) for v in (entry.get("labels") or [])]
        rects = [tuple(r) for r in (entry.get("positive_rects") or [])]
        neg_rects = [tuple(r) for r in (entry.get("negative_rects") or [])]
        global_negative_points = [p for p, lbl in zip(points_list, labels_list) if int(lbl) == 0]

        points = np.array(points_list, dtype=np.float32) if points_list else np.empty((0, 2), dtype=np.float32)
        labels = np.array(labels_list, dtype=np.int32) if labels_list else np.empty((0,), dtype=np.int32)
        if points.ndim == 1 and points.size > 0:
            points = points.reshape(1, 2)
        if labels.ndim == 0 and labels.size > 0:
            labels = labels.reshape(1)
        if (points.size == 0 or labels.size == 0) and rects:
            centers = _rect_center_points(rects)
            points = np.array(centers, dtype=np.float32)
            labels = np.ones((len(centers),), dtype=np.int32)

        object_prompts = list(entry.get("object_prompts") or [])
        _sam2_debug(
            "apply_prompt_entry",
            "frame_idx=", int(frame_idx),
            "points=", len(points_list),
            "rects=", len(rects),
            "object_prompts=", len(object_prompts),
            "neg_points=", len(global_negative_points),
        )
        if object_prompts:
            for op in object_prompts:
                if not isinstance(op, dict):
                    continue
                obj_id = int(max(0, int(op.get("obj_id", 0))))
                op_points_list = list(op.get("points") or [])
                op_labels_list = [int(v) for v in (op.get("labels") or [])]
                op_rects = [tuple(r) for r in (op.get("positive_rects") or [])]
                if global_negative_points:
                    op_points_list = list(op_points_list) + list(global_negative_points)
                    op_labels_list = list(op_labels_list) + [0 for _ in global_negative_points]
                op_points = np.array(op_points_list, dtype=np.float32) if op_points_list else np.empty((0, 2), dtype=np.float32)
                op_labels = np.array(op_labels_list, dtype=np.int32) if op_labels_list else np.empty((0,), dtype=np.int32)
                if op_points.ndim == 1 and op_points.size > 0:
                    op_points = op_points.reshape(1, 2)
                if op_labels.ndim == 0 and op_labels.size > 0:
                    op_labels = op_labels.reshape(1)
                if (op_points.size == 0 or op_labels.size == 0) and op_rects:
                    centers = _rect_center_points(op_rects)
                    op_points = np.array(centers, dtype=np.float32)
                    op_labels = np.ones((len(centers),), dtype=np.int32)
                _sam2_add_prompts(model, state_obj, int(frame_idx), obj_id, op_points, op_labels, op_rects)
                if op_points.size > 0:
                    carries = dict(inference_state.get("object_carries", {}) or {})
                    px = float(op_points[0][0])
                    py = float(op_points[0][1])
                    if op_rects:
                        b = tuple(op_rects[0])
                        bx = [float(b[0]), float(b[1]), float(b[2]), float(b[3])]
                    else:
                        bx = [px - 1.0, py - 1.0, px + 1.0, py + 1.0]
                    carries[str(int(obj_id))] = {"point": [px, py], "bbox": bx}
                    inference_state["object_carries"] = carries
        else:
            obj_id = int(inference_state.get("object_index", 0))
            _sam2_add_prompts(model, state_obj, int(frame_idx), obj_id, points, labels, rects)
        inference_state["active_negative_rects"] = [[float(a), float(b), float(c), float(d)] for (a, b, c, d) in neg_rects]
        if points.size > 0:
            inference_state["last_points"] = points.tolist()
            inference_state["last_labels"] = labels.tolist()

    def _seed_window_prompt(self, model, local_state, inference_state):
        carries = dict(inference_state.get("object_carries", {}) or {})
        _sam2_debug(
            "seed_window_prompt",
            "next_frame_idx=", int(max(0, inference_state.get("next_frame_idx", 0) or 0)),
            "carry_count=", len(carries),
            "seed_rects=", len(list(inference_state.get("seed_rects") or [])),
        )
        if carries:
            for raw_obj_id, payload in carries.items():
                try:
                    obj_id = int(raw_obj_id)
                except Exception:
                    continue

                point = payload
                bbox = None
                if isinstance(payload, dict):
                    point = payload.get("point")
                    bbox = payload.get("bbox")

                if not isinstance(point, (list, tuple)) or len(point) != 2:
                    continue
                try:
                    x = float(point[0])
                    y = float(point[1])
                except Exception:
                    continue

                rects = []
                if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                    try:
                        rects = [(
                            float(bbox[0]),
                            float(bbox[1]),
                            float(bbox[2]),
                            float(bbox[3]),
                        )]
                    except Exception:
                        rects = []

                pts = np.array([[x, y]], dtype=np.float32)
                lbs = np.array([1], dtype=np.int32)
                _sam2_add_prompts(model, local_state, 0, obj_id, pts, lbs, rects)
            return

        # Only apply original seed prompts on the very first chunk.
        # Re-using frame-1 seeds on later chunks can cause background drift
        # after objects leave frame.
        next_frame_idx = int(max(0, inference_state.get("next_frame_idx", 0) or 0))
        use_initial_seed = (next_frame_idx <= 0)

        # If an explicit frame-0 prompt exists in prompt_schedule (e.g. DINO boxes),
        # do not also inject fallback seed prompts here, to avoid duplicate/competing seeds.
        has_initial_prompt = False
        if use_initial_seed:
            for entry in list(inference_state.get("prompt_schedule") or []):
                if not isinstance(entry, dict):
                    continue
                try:
                    ef = int(entry.get("frame_idx", 0))
                except Exception:
                    ef = 0
                if ef != 0:
                    continue
                labels0 = [int(v) for v in (entry.get("labels") or [])]
                if any(v == 1 for v in labels0) or bool(entry.get("positive_rects") or []):
                    has_initial_prompt = True
                    break
                for op in list(entry.get("object_prompts") or []):
                    if not isinstance(op, dict):
                        continue
                    op_labels0 = [int(v) for v in (op.get("labels") or [])]
                    if any(v == 1 for v in op_labels0) or bool(op.get("positive_rects") or []):
                        has_initial_prompt = True
                        break
                if has_initial_prompt:
                    break

        seed_points = inference_state.get("seed_points") if (use_initial_seed and not has_initial_prompt) else []
        seed_labels = inference_state.get("seed_labels") if (use_initial_seed and not has_initial_prompt) else []
        seed_rects = inference_state.get("seed_rects") if (use_initial_seed and not has_initial_prompt) else []
        if use_initial_seed and has_initial_prompt:
            _sam2_debug("seed_window_prompt-skip-fallback", "reason=frame0_prompt_schedule")

        points = np.array(inference_state.get("last_points") or seed_points or [], dtype=np.float32)
        labels = np.array(inference_state.get("last_labels") or seed_labels or [], dtype=np.int32)
        rects = [tuple(r) for r in (seed_rects or []) if isinstance(r, (list, tuple)) and len(r) == 4]
        if points.ndim == 1 and points.size > 0:
            points = points.reshape(1, 2)
        if labels.ndim == 0 and labels.size > 0:
            labels = labels.reshape(1)
        if (points.size == 0 or labels.size == 0) and rects:
            centers = _rect_center_points(rects)
            points = np.array(centers, dtype=np.float32)
            labels = np.ones((len(centers),), dtype=np.int32)
        if points.size == 0 and not rects:
            return
        obj_id = int(inference_state.get("object_index", 0))
        _sam2_add_prompts(model, local_state, 0, obj_id, points, labels, rects)
        _sam2_debug(
            "seed_window_prompt-applied",
            "obj_id=", int(obj_id),
            "points=", int(points.shape[0]) if hasattr(points, "shape") else 0,
            "rects=", len(rects),
        )

    def _collect_range_masks(self, model, state_obj, frame_start, frame_count):
        frame_start = int(max(0, frame_start))
        frame_count = int(max(0, frame_count))
        if frame_count <= 0:
            return []
        try:
            iterator = model.propagate_in_video(
                state_obj,
                start_frame_idx=frame_start,
                max_frame_num_to_track=frame_count,
            )
        except TypeError:
            iterator = model.propagate_in_video(state_obj)

        by_idx = {}
        frame_end = frame_start + frame_count
        for out_frame_idx, out_obj_ids, out_mask_logits in iterator:
            idx = int(out_frame_idx)
            if idx < frame_start:
                continue
            if idx >= frame_end:
                break
            combined = None
            for i, _obj_id in enumerate(out_obj_ids):
                current = out_mask_logits[i, 0] > 0.0
                combined = current if combined is None else torch.logical_or(combined, current)
            if combined is None:
                _n, _c, h, w = out_mask_logits.shape
                combined = torch.zeros((h, w), dtype=torch.bool, device=out_mask_logits.device)
            by_idx[idx] = combined.float().cpu()
            del out_mask_logits
        if not by_idx:
            return []
        h = int(next(iter(by_idx.values())).shape[0])
        w = int(next(iter(by_idx.values())).shape[1])
        return [by_idx.get(i, torch.zeros((h, w), dtype=torch.float32)) for i in range(frame_start, frame_end)]

    def _update_prompt_from_last_mask(self, inference_state, masks):
        last = None
        for m in reversed(masks):
            if torch.any(m > 0.0):
                last = m
                break
        if last is None:
            return
        ys, xs = torch.where(last > 0.0)
        if xs.numel() == 0:
            return
        cx = float(xs.float().mean().item())
        cy = float(ys.float().mean().item())
        inference_state["last_points"] = [[cx, cy]]
        inference_state["last_labels"] = [1]

    def _segment_windowed(self, sam2_model, inference_state, image, keep_model_loaded, meta_batch=None):
        model = sam2_model["model"]
        device = sam2_model["device"]
        dtype = sam2_model["dtype"]
        model.to(device)
        autocast_device = mm.get_autocast_device(device)
        autocast_ok = not mm.is_device_mps(device)

        window_dir = None
        local_state = None
        out_chunks = []
        try:
            window_dir, bsz, h, w = self._write_window_jpegs(image)
            progress = ProgressBar(bsz)
            with torch.inference_mode():
                with torch.autocast(autocast_device, dtype=dtype) if autocast_ok else nullcontext():
                    local_state = self._init_window_state(model, window_dir, device, inference_state)
                    # Seed from carried prompt so chunk-to-chunk tracking continues.
                    self._seed_window_prompt(model, local_state, inference_state)

                    global_start = int(max(0, inference_state.get("next_frame_idx", 0) or 0))
                    applied_frames = set(int(v) for v in (inference_state.get("prompt_frames_applied") or []))
                    _sam2_debug(
                        "segment_windowed-start",
                        "global_start=", int(global_start),
                        "bsz=", int(bsz),
                        "applied_frames=", sorted(list(applied_frames))[:8],
                        "carry_count=", len(dict(inference_state.get("object_carries", {}) or {})),
                    )
                    for entry in self._prompt_schedule(inference_state):
                        gidx = int(entry.get("frame_idx", 0))
                        if gidx < global_start or gidx >= (global_start + bsz):
                            continue
                        if gidx in applied_frames:
                            continue
                        self._apply_prompt_entry(model, local_state, inference_state, int(gidx - global_start), entry)
                        _sam2_debug("segment_windowed-prompt-applied", "global_frame=", int(gidx), "local_frame=", int(gidx - global_start))
                        applied_frames.add(gidx)
                    inference_state["prompt_frames_applied"] = sorted(list(applied_frames))

                    # Boundary replay: reinforce carried prompts for first N local frames.
                    boundary_reseed_frames = int(max(1, inference_state.get("boundary_reseed_frames", 4) or 4))
                    carries_for_reseed = dict(inference_state.get("object_carries", {}) or {})
                    if carries_for_reseed and boundary_reseed_frames > 1:
                        max_local = int(min(bsz, boundary_reseed_frames))
                        for local_f in range(1, max_local):
                            for raw_obj_id, payload in carries_for_reseed.items():
                                try:
                                    obj_id = int(raw_obj_id)
                                except Exception:
                                    continue
                                point = payload
                                bbox = None
                                if isinstance(payload, dict):
                                    point = payload.get("point")
                                    bbox = payload.get("bbox")
                                if not isinstance(point, (list, tuple)) or len(point) != 2:
                                    continue
                                try:
                                    x = float(point[0])
                                    y = float(point[1])
                                except Exception:
                                    continue
                                rects = []
                                if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                                    try:
                                        rects = [(
                                            float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
                                        )]
                                    except Exception:
                                        rects = []
                                pts = np.array([[x, y]], dtype=np.float32)
                                lbs = np.array([1], dtype=np.int32)
                                _sam2_add_prompts(model, local_state, int(local_f), obj_id, pts, lbs, rects)
                        _sam2_debug("boundary_reseed", "frames=", int(max_local), "objects=", len(carries_for_reseed))

                    def _entry_has_positive_seed(entry):
                        labels = list((entry or {}).get("labels") or [])
                        if any(int(v) == 1 for v in labels):
                            return True
                        if bool((entry or {}).get("positive_rects") or []):
                            return True
                        for op in list((entry or {}).get("object_prompts") or []):
                            if not isinstance(op, dict):
                                continue
                            op_labels = list(op.get("labels") or [])
                            if any(int(v) == 1 for v in op_labels) or bool(op.get("positive_rects") or []):
                                return True
                        return False

                    by_idx = {}
                    carries = dict(inference_state.get("object_carries", {}) or {})
                    has_window_prompt = False
                    for entry in self._prompt_schedule(inference_state):
                        gidx = int(entry.get("frame_idx", 0))
                        if gidx < global_start or gidx >= (global_start + bsz):
                            continue
                        if _entry_has_positive_seed(entry):
                            has_window_prompt = True
                            break

                    # Gracefully handle prompt-only runs when detector finds no boxes.
                    # If nothing is currently seeded, emit empty masks for this chunk.
                    if (not carries) and (not has_window_prompt):
                        for _ in range(bsz):
                            progress.update(1)
                    else:
                        try:
                            iterator = model.propagate_in_video(
                                local_state,
                                start_frame_idx=0,
                                max_frame_num_to_track=bsz,
                            )
                        except TypeError:
                            iterator = model.propagate_in_video(local_state)

                        seen_obj_ids = set()
                        for out_frame_idx, out_obj_ids, out_mask_logits in iterator:
                            idx = int(out_frame_idx)
                            if idx < 0 or idx >= bsz:
                                continue
                            combined = None
                            for i, _obj_id in enumerate(out_obj_ids):
                                current = out_mask_logits[i, 0] > 0.0
                                combined = current if combined is None else torch.logical_or(combined, current)
                                if torch.any(current):
                                    ys, xs = torch.where(current)
                                    if xs.numel() > 0:
                                        obj_id_int = int(_obj_id)
                                        min_x = int(xs.min().item())
                                        max_x = int(xs.max().item())
                                        min_y = int(ys.min().item())
                                        max_y = int(ys.max().item())
                                        seen_obj_ids.add(obj_id_int)
                                        carries[str(obj_id_int)] = {
                                            "point": [
                                                float(xs.float().mean().item()),
                                                float(ys.float().mean().item()),
                                            ],
                                            "bbox": [
                                                float(min_x), float(min_y), float(max_x), float(max_y),
                                            ],
                                        }
                            if combined is None:
                                combined = torch.zeros((h, w), dtype=torch.bool, device=out_mask_logits.device)
                            by_idx[idx] = combined.float().cpu()
                            progress.update(1)
                            del out_mask_logits
                        inference_state["object_carries"] = {
                            str(obj_id): carries.get(str(obj_id))
                            for obj_id in sorted(seen_obj_ids)
                            if str(obj_id) in carries
                        }
                        _sam2_debug(
                            "segment_windowed-end",
                            "kept_carries=", sorted([int(v) for v in seen_obj_ids]),
                            "next_frame_idx=", int(inference_state.get("next_frame_idx", 0) or 0),
                        )

            for i in range(bsz):
                out_chunks.append(by_idx.get(i, torch.zeros((h, w), dtype=torch.float32)))
            # Do not overwrite user/keyframe prompts with a single centroid carry point.
            # For multi-target prompts (e.g. several cars), centroid carry causes drift/fizzle.
            inference_state["next_frame_idx"] = int(inference_state.get("next_frame_idx", 0) or 0) + bsz
            inference_state["num_frames"] = int(inference_state.get("num_frames", 0) or 0) + bsz
            if meta_batch is not None:
                try:
                    setattr(meta_batch, "_openshot_sam2_window_state", inference_state)
                except Exception:
                    pass
        finally:
            if local_state is not None and hasattr(model, "reset_state"):
                try:
                    model.reset_state(local_state)
                except Exception:
                    pass
            if window_dir and os.path.isdir(window_dir):
                shutil.rmtree(window_dir, ignore_errors=True)
            if not keep_model_loaded:
                model.to(mm.unet_offload_device())
                mm.soft_empty_cache()

        stacked = torch.stack(out_chunks, dim=0)
        return (stacked,)

    def segment_chunk(self, sam2_model, inference_state, image, start_frame, chunk_size_frames, keep_model_loaded, meta_batch=None):
        model = sam2_model["model"]
        device = sam2_model["device"]
        dtype = sam2_model["dtype"]
        segmentor = sam2_model.get("segmentor", "video")
        if segmentor != "video":
            raise ValueError("Loaded SAM2 model is not configured for video")

        if bool(inference_state.get("windowed_mode", False)):
            return self._segment_windowed(sam2_model, inference_state, image, keep_model_loaded, meta_batch=meta_batch)

        state = inference_state["inference_state"]
        chunk_size_frames = int(max(1, chunk_size_frames))
        effective_chunk = self._get_frames_per_batch(meta_batch, chunk_size_frames)
        # Force this node to track VHS chunking cadence exactly.
        try:
            effective_chunk = min(effective_chunk, int(image.shape[0]))
        except Exception:
            pass

        # Persist frame cursor inside the shared inference_state object so each
        # meta-batch call continues from the prior chunk without recomputing frame 0.
        if "next_frame_idx" not in inference_state:
            inference_state["next_frame_idx"] = int(max(0, start_frame))
        current_start = int(max(0, inference_state.get("next_frame_idx", start_frame)))

        total_frames = int(inference_state.get("num_frames", 0) or 0)
        if total_frames > 0:
            remaining = max(0, total_frames - current_start)
            effective_chunk = min(effective_chunk, remaining) if remaining > 0 else 0

        if effective_chunk <= 0:
            raise RuntimeError("No remaining SAM2 frames to process (cursor at end of video)")

        model.to(device)
        autocast_device = mm.get_autocast_device(device)
        autocast_ok = not mm.is_device_mps(device)

        out_chunks = []
        progress = ProgressBar(effective_chunk)
        with torch.inference_mode():
            with torch.autocast(autocast_device, dtype=dtype) if autocast_ok else nullcontext():
                end_frame = current_start + effective_chunk
                schedule_by_frame = {
                    int(entry.get("frame_idx", 0)): entry
                    for entry in self._prompt_schedule(inference_state)
                }
                applied_frames = set(int(v) for v in (inference_state.get("prompt_frames_applied") or []))
                for frame_idx in sorted(schedule_by_frame.keys()):
                    if frame_idx < current_start or frame_idx >= end_frame:
                        continue
                    if frame_idx in applied_frames:
                        continue
                    self._apply_prompt_entry(model, state, inference_state, frame_idx, schedule_by_frame[frame_idx])
                    applied_frames.add(frame_idx)
                inference_state["prompt_frames_applied"] = sorted(list(applied_frames))

                try:
                    iterator = model.propagate_in_video(
                        state,
                        start_frame_idx=current_start,
                        max_frame_num_to_track=effective_chunk,
                    )
                except TypeError:
                    iterator = model.propagate_in_video(state)

                carries = dict(inference_state.get("object_carries", {}) or {})
                seen_obj_ids = set()
                for out_frame_idx, out_obj_ids, out_mask_logits in iterator:
                    idx = int(out_frame_idx)
                    if idx < current_start:
                        continue
                    if idx >= end_frame:
                        break

                    combined = None
                    for i, _obj_id in enumerate(out_obj_ids):
                        current = out_mask_logits[i, 0] > 0.0
                        combined = current if combined is None else torch.logical_or(combined, current)
                        if torch.any(current):
                            ys, xs = torch.where(current)
                            if xs.numel() > 0:
                                h_cur = int(current.shape[0])
                                w_cur = int(current.shape[1])
                                obj_id_int = int(_obj_id)
                                area = int(xs.numel())
                                area_ratio = float(area) / float(max(1, h_cur * w_cur))
                                min_x = int(xs.min().item())
                                max_x = int(xs.max().item())
                                min_y = int(ys.min().item())
                                max_y = int(ys.max().item())
                                bbox_w = int(max_x - min_x + 1)
                                bbox_h = int(max_y - min_y + 1)
                                bbox_area = max(1, bbox_w * bbox_h)
                                fill_ratio = float(area) / float(bbox_area)
                                touches_edge = (min_x <= 1) or (min_y <= 1) or (max_x >= (w_cur - 2)) or (max_y >= (h_cur - 2))

                                seen_obj_ids.add(obj_id_int)
                                carries[str(obj_id_int)] = {
                                    "point": [
                                        float(xs.float().mean().item()),
                                        float(ys.float().mean().item()),
                                    ],
                                    "bbox": [
                                        float(min_x), float(min_y), float(max_x), float(max_y),
                                    ],
                                }

                    if combined is None:
                        _n, _c, h, w = out_mask_logits.shape
                        combined = torch.zeros((h, w), dtype=torch.bool, device=out_mask_logits.device)

                    out_chunks.append(combined.float().cpu())
                    progress.update(1)
                    del out_mask_logits
                inference_state["object_carries"] = {
                    str(obj_id): carries.get(str(obj_id))
                    for obj_id in sorted(seen_obj_ids)
                    if str(obj_id) in carries
                }

        if not out_chunks:
            raise RuntimeError(
                "SAM2 chunk produced no frames. Check cursor/chunk size and inference state. "
                "cursor={} chunk={} total={}".format(current_start, effective_chunk, total_frames)
            )

        inference_state["next_frame_idx"] = current_start + effective_chunk
        if total_frames > 0 and inference_state["next_frame_idx"] >= total_frames:
            if hasattr(model, "reset_state"):
                try:
                    model.reset_state(state)
                except Exception:
                    pass

        if not keep_model_loaded:
            model.to(mm.unet_offload_device())
            mm.soft_empty_cache()

        stacked = torch.stack(out_chunks, dim=0)
        return (stacked,)


def _gaussian_kernel(kernel_size, sigma, device, dtype):
    axis = torch.linspace(-1, 1, kernel_size, device=device, dtype=dtype)
    x, y = torch.meshgrid(axis, axis, indexing="ij")
    d = torch.sqrt(x * x + y * y)
    g = torch.exp(-(d * d) / (2.0 * sigma * sigma))
    return g / g.sum()


def _parse_color_rgba(color_text, default=(1.0, 1.0, 0.0, 1.0)):
    text = str(color_text or "").strip().lower()
    if not text:
        return default
    if text == "transparent":
        return (0.0, 0.0, 0.0, 0.0)
    if text.startswith("#"):
        raw = text[1:]
        try:
            if len(raw) == 6:
                r = int(raw[0:2], 16) / 255.0
                g = int(raw[2:4], 16) / 255.0
                b = int(raw[4:6], 16) / 255.0
                return (r, g, b, 1.0)
            if len(raw) == 8:
                r = int(raw[0:2], 16) / 255.0
                g = int(raw[2:4], 16) / 255.0
                b = int(raw[4:6], 16) / 255.0
                a = int(raw[6:8], 16) / 255.0
                return (r, g, b, a)
        except Exception:
            return default
    return default


class OpenShotImageBlurMasked:
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return ""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "blur_radius": ("INT", {"default": 12, "min": 0, "max": 64, "step": 1}),
                "sigma": ("FLOAT", {"default": 4.0, "min": 0.1, "max": 20.0, "step": 0.1}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "blur_masked"
    CATEGORY = "OpenShot/Video"

    def blur_masked(self, image, mask, blur_radius, sigma):
        blur_radius = int(max(0, blur_radius))
        if blur_radius == 0:
            return (image,)

        device = mm.get_torch_device()
        img = image.to(device)
        m = mask.to(device).float()
        if m.ndim == 3:
            m = m.unsqueeze(-1)
        m = torch.clamp(m, 0.0, 1.0)

        has_mask = (m.view(m.shape[0], -1).max(dim=1).values > 0)
        if not bool(has_mask.any()):
            return (image,)

        out = img.clone()
        idx = torch.nonzero(has_mask, as_tuple=False).squeeze(1)
        work = img[idx]
        work_mask = m[idx]

        kernel_size = blur_radius * 2 + 1
        kernel = _gaussian_kernel(kernel_size, float(sigma), device=work.device, dtype=work.dtype)
        kernel = kernel.repeat(work.shape[-1], 1, 1).unsqueeze(1)

        work_nchw = work.permute(0, 3, 1, 2)
        padded = F.pad(work_nchw, (blur_radius, blur_radius, blur_radius, blur_radius), "reflect")
        blurred = F.conv2d(padded, kernel, padding=kernel_size // 2, groups=work.shape[-1])[
            :, :, blur_radius:-blur_radius, blur_radius:-blur_radius
        ]
        blurred = blurred.permute(0, 2, 3, 1)

        composited = work * (1.0 - work_mask) + blurred * work_mask
        out[idx] = composited
        return (out.to(mm.intermediate_device()),)


class OpenShotImageHighlightMasked:
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return ""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "highlight_color": ("STRING", {"default": "#F5D742"}),
                "highlight_opacity": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.01}),
                "border_color": ("STRING", {"default": "transparent"}),
                "border_width": ("INT", {"default": 0, "min": 0, "max": 64, "step": 1}),
                "mask_brightness": ("FLOAT", {"default": 1.15, "min": 0.0, "max": 3.0, "step": 0.01}),
                "background_brightness": ("FLOAT", {"default": 0.75, "min": 0.0, "max": 3.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "highlight_masked"
    CATEGORY = "OpenShot/Video"

    def highlight_masked(
        self,
        image,
        mask,
        highlight_color,
        highlight_opacity,
        border_color,
        border_width,
        mask_brightness,
        background_brightness,
    ):
        hi_r, hi_g, hi_b, hi_a = _parse_color_rgba(highlight_color, default=(0.96, 0.84, 0.26, 1.0))
        bo_r, bo_g, bo_b, bo_a = _parse_color_rgba(border_color, default=(0.0, 0.0, 0.0, 0.0))
        hi_alpha = float(max(0.0, min(1.0, float(highlight_opacity)))) * float(hi_a)
        border_width = int(max(0, border_width))
        mask_brightness = float(max(0.0, min(3.0, float(mask_brightness))))
        background_brightness = float(max(0.0, min(3.0, float(background_brightness))))

        if hi_alpha <= 0.0 and (border_width <= 0 or bo_a <= 0.0):
            return (image,)

        device = mm.get_torch_device()
        img = image.to(device)
        m = mask.to(device).float()
        if m.ndim == 2:
            m = m.unsqueeze(0)
        if m.ndim == 4:
            m = m.squeeze(-1)
        if m.ndim != 3:
            return (image,)
        m = torch.clamp(m, 0.0, 1.0)
        if int(m.shape[0]) == 1 and int(img.shape[0]) > 1:
            m = m.repeat(int(img.shape[0]), 1, 1)
        if int(m.shape[0]) != int(img.shape[0]):
            return (image,)

        has_mask = (m.view(m.shape[0], -1).max(dim=1).values > 0)
        if not bool(has_mask.any()):
            return (image,)

        out = img.clone()
        idx = torch.nonzero(has_mask, as_tuple=False).squeeze(1)
        work = img[idx]
        work_mask = m[idx].unsqueeze(-1)
        work_bg = torch.clamp(work * background_brightness, 0.0, 1.0)
        work_fg = torch.clamp(work * mask_brightness, 0.0, 1.0)
        work = work_bg * (1.0 - work_mask) + work_fg * work_mask

        if hi_alpha > 0.0:
            hi_color = torch.tensor([hi_r, hi_g, hi_b], device=work.device, dtype=work.dtype).view(1, 1, 1, 3)
            fill_alpha = torch.clamp(work_mask * hi_alpha, 0.0, 1.0)
            work = work * (1.0 - fill_alpha) + hi_color * fill_alpha

        if border_width > 0 and bo_a > 0.0:
            k = border_width * 2 + 1
            base = work_mask.permute(0, 3, 1, 2)
            dilated = F.max_pool2d(base, kernel_size=k, stride=1, padding=border_width)
            border = torch.clamp(dilated - base, 0.0, 1.0).permute(0, 2, 3, 1)
            if torch.any(border > 0.0):
                bo_color = torch.tensor([bo_r, bo_g, bo_b], device=work.device, dtype=work.dtype).view(1, 1, 1, 3)
                border_alpha = torch.clamp(border * bo_a, 0.0, 1.0)
                work = work * (1.0 - border_alpha) + bo_color * border_alpha

        out[idx] = work
        return (out.to(mm.intermediate_device()),)


class OpenShotGroundingDinoDetect:
    _model_cache = {}

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return ""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "prompt": ("STRING", {"default": "person.", "multiline": False}),
                "model_id": (GROUNDING_DINO_MODEL_IDS,),
                "box_threshold": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.01}),
                "text_threshold": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01}),
                "device": (("auto", "cpu", "cuda", "mps"),),
                "keep_model_loaded": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("MASK", "STRING")
    RETURN_NAMES = ("mask", "detections_json")
    FUNCTION = "detect"
    CATEGORY = "OpenShot/GroundingDINO"

    def _resolve_device(self, device_name):
        device_name = str(device_name or "auto").strip().lower()
        if device_name == "auto":
            return mm.get_torch_device()
        return torch.device(device_name)

    def _cache_key(self, model_id, device):
        return "{}::{}".format(model_id, str(device))

    def _get_model_and_processor(self, model_id, device):
        key = self._cache_key(model_id, device)
        if key in self._model_cache:
            return self._model_cache[key]

        processor = AutoProcessor.from_pretrained(model_id)
        model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)
        model.to(device)
        model.eval()
        self._model_cache[key] = (processor, model)
        return processor, model

    def _tensor_to_pil(self, img):
        arr = torch.clamp(img, 0.0, 1.0).mul(255.0).byte().cpu().numpy()
        return Image.fromarray(arr)

    def _boxes_to_mask(self, boxes, height, width):
        frame_mask = torch.zeros((height, width), dtype=torch.float32)
        for box in boxes:
            x0, y0, x1, y1 = [float(v) for v in box]
            left = int(max(0, min(width, np.floor(x0))))
            top = int(max(0, min(height, np.floor(y0))))
            right = int(max(0, min(width, np.ceil(x1))))
            bottom = int(max(0, min(height, np.ceil(y1))))
            if right <= left or bottom <= top:
                continue
            frame_mask[top:bottom, left:right] = 1.0
        return frame_mask

    def detect(self, image, prompt, model_id, box_threshold, text_threshold, device, keep_model_loaded):
        _require_groundingdino()

        prompt = str(prompt or "").strip()
        if not prompt:
            raise ValueError("GroundingDINO prompt must not be empty")
        if not prompt.endswith("."):
            prompt = "{}.".format(prompt)

        device = self._resolve_device(device)
        processor, model = self._get_model_and_processor(model_id, device)
        model.to(device)

        batch = int(image.shape[0])
        height = int(image.shape[1])
        width = int(image.shape[2])
        all_masks = []
        all_detections = []

        with torch.inference_mode():
            for i in range(batch):
                pil = self._tensor_to_pil(image[i])
                inputs = processor(images=pil, text=prompt, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}
                outputs = model(**inputs)
                post_kwargs = {
                    "target_sizes": [(height, width)],
                    "text_threshold": float(text_threshold),
                }
                try:
                    result = processor.post_process_grounded_object_detection(
                        outputs,
                        inputs["input_ids"],
                        box_threshold=float(box_threshold),
                        **post_kwargs,
                    )[0]
                except TypeError:
                    try:
                        result = processor.post_process_grounded_object_detection(
                            outputs,
                            inputs["input_ids"],
                            threshold=float(box_threshold),
                            **post_kwargs,
                        )[0]
                    except TypeError:
                        result = processor.post_process_grounded_object_detection(
                            outputs,
                            inputs["input_ids"],
                            threshold=float(box_threshold),
                            target_sizes=[(height, width)],
                        )[0]

                boxes = result.get("boxes")
                labels = result.get("labels")
                scores = result.get("scores")
                if boxes is None or boxes.numel() == 0:
                    all_masks.append(torch.zeros((height, width), dtype=torch.float32))
                    all_detections.append({"frame_index": i, "detections": []})
                    continue

                boxes_cpu = boxes.detach().cpu()
                mask = self._boxes_to_mask(boxes_cpu, height, width)
                all_masks.append(mask)

                frame_items = []
                for idx in range(boxes_cpu.shape[0]):
                    frame_items.append(
                        {
                            "label": str(labels[idx]),
                            "score": float(scores[idx].item()),
                            "box_xyxy": [float(v) for v in boxes_cpu[idx].tolist()],
                        }
                    )
                all_detections.append({"frame_index": i, "detections": frame_items})

        if not keep_model_loaded:
            model.to(mm.unet_offload_device())
            mm.soft_empty_cache()

        mask_tensor = torch.stack(all_masks, dim=0).to(mm.intermediate_device())
        return (mask_tensor, json.dumps(all_detections))


NODE_CLASS_MAPPINGS = {
    "OpenShotTransNetSceneDetect": OpenShotTransNetSceneDetect,
    "OpenShotDownloadAndLoadSAM2Model": OpenShotDownloadAndLoadSAM2Model,
    "OpenShotSam2Segmentation": OpenShotSam2Segmentation,
    "OpenShotSam2VideoSegmentationAddPoints": OpenShotSam2VideoSegmentationAddPoints,
    "OpenShotSam2VideoSegmentationChunked": OpenShotSam2VideoSegmentationChunked,
    "OpenShotImageBlurMasked": OpenShotImageBlurMasked,
    "OpenShotImageHighlightMasked": OpenShotImageHighlightMasked,
    "OpenShotGroundingDinoDetect": OpenShotGroundingDinoDetect,
    "OpenShotSceneRangesFromSegments": OpenShotSceneRangesFromSegments,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OpenShotTransNetSceneDetect": "OpenShot TransNet Scene Detect",
    "OpenShotDownloadAndLoadSAM2Model": "OpenShot Download+Load SAM2",
    "OpenShotSam2Segmentation": "OpenShot SAM2 Segmentation (Image)",
    "OpenShotSam2VideoSegmentationAddPoints": "OpenShot SAM2 Add Video Points",
    "OpenShotSam2VideoSegmentationChunked": "OpenShot SAM2 Video Segmentation (Chunked)",
    "OpenShotImageBlurMasked": "OpenShot Blur Masked (Skip Empty)",
    "OpenShotImageHighlightMasked": "OpenShot Highlight Masked",
    "OpenShotGroundingDinoDetect": "OpenShot GroundingDINO Detect",
    "OpenShotSceneRangesFromSegments": "OpenShot Scene Ranges From Segments",
}
