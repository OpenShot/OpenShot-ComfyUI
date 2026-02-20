import json
import os
from contextlib import nullcontext
from urllib.parse import urlparse

import numpy as np
import torch
from torch.hub import download_url_to_file

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


SAM2_MODEL_DIR = "sam2"
OPENSHOT_NODEPACK_VERSION = "v1.0.1-hydra-local-config"
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
                "coordinates_positive": ("STRING", {"forceInput": True}),
                "keep_model_loaded": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "coordinates_negative": ("STRING", {"forceInput": True}),
            },
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "segment"
    CATEGORY = "OpenShot/SAM2"

    def segment(self, sam2_model, image, coordinates_positive, keep_model_loaded, coordinates_negative=None):
        _require_sam2()

        model = sam2_model["model"]
        device = sam2_model["device"]
        dtype = sam2_model["dtype"]

        positive = _parse_points(coordinates_positive)
        if not positive:
            raise ValueError("No positive points provided")
        negative = _parse_points(coordinates_negative)

        pos_arr = np.array(positive, dtype=np.float32)
        neg_arr = np.array(negative, dtype=np.float32) if negative else np.empty((0, 2), dtype=np.float32)

        coords = np.concatenate((pos_arr, neg_arr), axis=0)
        labels = np.concatenate((np.ones((len(pos_arr),), dtype=np.int32), np.zeros((len(neg_arr),), dtype=np.int32)), axis=0)

        predictor = SAM2ImagePredictor(model)

        out_masks = []
        autocast_device = mm.get_autocast_device(device)
        autocast_ok = not mm.is_device_mps(device)
        with torch.autocast(autocast_device, dtype=dtype) if autocast_ok else nullcontext():
            for frame in image:
                frame_np = np.clip((frame.cpu().numpy() * 255.0), 0, 255).astype(np.uint8)
                predictor.set_image(frame_np[..., :3])
                masks, _scores, _logits = predictor.predict(
                    point_coords=coords,
                    point_labels=labels,
                    multimask_output=False,
                )
                mask = masks[0]
                out_masks.append(torch.from_numpy(mask).float())

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
                "coordinates_positive": ("STRING", {"forceInput": True}),
                "frame_index": ("INT", {"default": 0, "min": 0}),
                "object_index": ("INT", {"default": 0, "min": 0}),
            },
            "optional": {
                "image": ("IMAGE",),
                "video_path": ("STRING", {"default": ""}),
                "coordinates_negative": ("STRING", {"forceInput": True}),
                "prev_inference_state": ("SAM2INFERENCESTATE",),
            },
        }

    RETURN_TYPES = ("SAM2MODEL", "SAM2INFERENCESTATE")
    RETURN_NAMES = ("sam2_model", "inference_state")
    FUNCTION = "add_points"
    CATEGORY = "OpenShot/SAM2"

    def add_points(
        self,
        sam2_model,
        coordinates_positive,
        frame_index,
        object_index,
        image=None,
        video_path="",
        coordinates_negative=None,
        prev_inference_state=None,
    ):
        model = sam2_model["model"]
        device = sam2_model["device"]
        dtype = sam2_model["dtype"]
        segmentor = sam2_model.get("segmentor", "video")
        if segmentor != "video":
            raise ValueError("Loaded SAM2 model is not configured for video")
        if (image is None and not str(video_path or "").strip()) and prev_inference_state is None:
            raise ValueError("Image or video_path input is required for initial inference state")

        pos = _parse_points(coordinates_positive)
        if not pos:
            raise ValueError("No positive points provided")
        neg = _parse_points(coordinates_negative)

        pos_arr = np.atleast_2d(np.array(pos, dtype=np.float32))
        neg_arr = np.atleast_2d(np.array(neg, dtype=np.float32)) if neg else np.empty((0, 2), dtype=np.float32)

        coords = np.concatenate((pos_arr, neg_arr), axis=0)
        labels = np.concatenate((np.ones((len(pos_arr),), dtype=np.int32), np.zeros((len(neg_arr),), dtype=np.int32)), axis=0)

        model.to(device)
        if prev_inference_state is None:
            # Support SAM2 API variants for init_state signature.
            init_errors = []
            state = None
            num_frames = 0

            # Preferred path for newer SAM2 video predictors: initialize from source video path.
            if str(video_path or "").strip():
                vp = _resolve_video_path_for_sam2(video_path)
                for call in (
                    lambda: model.init_state(vp, device=device),
                    lambda: model.init_state(vp),
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
                raise RuntimeError(
                    "Failed SAM2 init_state across signature variants: {} | resolved_video_path='{}' exists={} ext='{}'".format(
                        init_errors,
                        vp if str(video_path or "").strip() else "",
                        (os.path.exists(vp) if str(video_path or "").strip() else False),
                        (os.path.splitext(vp)[1].lower() if str(video_path or "").strip() else ""),
                    )
                )
        else:
            state = prev_inference_state["inference_state"]
            num_frames = int(prev_inference_state.get("num_frames", 0) or 0)

        autocast_device = mm.get_autocast_device(device)
        autocast_ok = not mm.is_device_mps(device)
        with torch.autocast(autocast_device, dtype=dtype) if autocast_ok else nullcontext():
            add_errors = []
            added = False
            for call in (
                lambda: model.add_new_points(
                    inference_state=state,
                    frame_idx=int(frame_index),
                    obj_id=int(object_index),
                    points=coords,
                    labels=labels,
                ),
                lambda: model.add_new_points_or_box(
                    inference_state=state,
                    frame_idx=int(frame_index),
                    obj_id=int(object_index),
                    points=coords,
                    labels=labels,
                ),
            ):
                try:
                    call()
                    added = True
                    break
                except Exception as ex:
                    add_errors.append(str(ex))
            if not added:
                raise RuntimeError("Failed SAM2 add points across API variants: {}".format(add_errors))

        return (
            sam2_model,
            {
                "inference_state": state,
                "num_frames": num_frames,
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
                "start_frame": ("INT", {"default": 0, "min": 0}),
                "chunk_size_frames": ("INT", {"default": 32, "min": 1, "max": 4096}),
                "keep_model_loaded": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "segment_chunk"
    CATEGORY = "OpenShot/SAM2"

    def segment_chunk(self, sam2_model, inference_state, start_frame, chunk_size_frames, keep_model_loaded):
        model = sam2_model["model"]
        device = sam2_model["device"]
        dtype = sam2_model["dtype"]
        segmentor = sam2_model.get("segmentor", "video")
        if segmentor != "video":
            raise ValueError("Loaded SAM2 model is not configured for video")

        state = inference_state["inference_state"]
        start_frame = int(max(0, start_frame))
        chunk_size_frames = int(max(1, chunk_size_frames))

        model.to(device)
        autocast_device = mm.get_autocast_device(device)
        autocast_ok = not mm.is_device_mps(device)

        out_chunks = []
        progress = ProgressBar(chunk_size_frames)
        with torch.autocast(autocast_device, dtype=dtype) if autocast_ok else nullcontext():
            # Preferred path if SAM2 API supports bounded propagation args.
            try:
                iterator = model.propagate_in_video(
                    state,
                    start_frame_idx=start_frame,
                    max_frame_num_to_track=chunk_size_frames,
                )
            except TypeError:
                iterator = model.propagate_in_video(state)

            for out_frame_idx, out_obj_ids, out_mask_logits in iterator:
                idx = int(out_frame_idx)
                if idx < start_frame:
                    continue
                if idx >= (start_frame + chunk_size_frames):
                    break

                combined = None
                for i, _obj_id in enumerate(out_obj_ids):
                    current = out_mask_logits[i, 0] > 0.0
                    combined = current if combined is None else torch.logical_or(combined, current)

                if combined is None:
                    _n, _c, h, w = out_mask_logits.shape
                    combined = torch.zeros((h, w), dtype=torch.bool, device=out_mask_logits.device)

                out_chunks.append(combined.float().cpu())
                progress.update(1)
                del out_mask_logits

        if not out_chunks:
            raise RuntimeError("SAM2 chunk produced no frames. Check start_frame/chunk_size and inference state.")

        if not keep_model_loaded:
            model.to(mm.unet_offload_device())
            mm.soft_empty_cache()

        return (torch.stack(out_chunks, dim=0),)


NODE_CLASS_MAPPINGS = {
    "OpenShotDownloadAndLoadSAM2Model": OpenShotDownloadAndLoadSAM2Model,
    "OpenShotSam2Segmentation": OpenShotSam2Segmentation,
    "OpenShotSam2VideoSegmentationAddPoints": OpenShotSam2VideoSegmentationAddPoints,
    "OpenShotSam2VideoSegmentationChunked": OpenShotSam2VideoSegmentationChunked,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OpenShotDownloadAndLoadSAM2Model": "OpenShot Download+Load SAM2",
    "OpenShotSam2Segmentation": "OpenShot SAM2 Segmentation (Image)",
    "OpenShotSam2VideoSegmentationAddPoints": "OpenShot SAM2 Add Video Points",
    "OpenShotSam2VideoSegmentationChunked": "OpenShot SAM2 Video Segmentation (Chunked)",
}
