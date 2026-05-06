import atexit
import base64
import logging
import os
import re
import shutil
import sys
import tempfile
from io import BytesIO
from pathlib import Path

import cv2
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

try:
    import decord
except ImportError:
    decord = None

from project_path import PROJECT_PATH

if PROJECT_PATH not in sys.path:
    sys.path.append(PROJECT_PATH)

from pipeline.toolbox.utils.air_frame_selector import AIRFrameSelector
from llm_config import (
    get_max_tokens_param,
    get_reasoning_params,
    get_temperature,
    is_gpt5_model,
    make_client,
)
from pipeline.toolbox.utils.vision_backend import (
    get_vision_api_model_name,
    get_vision_backend,
    use_api_vision_backend,
)

logger = logging.getLogger(__name__)

_IMAGE_EXTENSIONS = {"jpg", "jpeg", "png", "bmp", "gif", "tiff", "webp"}
_VIDEO_EXTENSIONS = {"mp4", "mkv", "avi", "mov", "flv", "wmv", "webm"}
_USE_AIR_FRAME_SELECTION = os.getenv("VLM_USE_AIR_FRAME_SELECTION", "1") != "0"
_AIR_SCORE_MAX_NEW_TOKENS = int(os.getenv("VLM_AIR_SCORE_MAX_NEW_TOKENS", "512"))
_VLM_MAX_NEW_TOKENS = int(os.getenv("VLM_MAX_NEW_TOKENS", "15000"))
_VLM_ENABLE_THINKING = os.getenv("VLM_ENABLE_THINKING", "0") != "0"
_VLM_VIDEO_SAMPLE_STRIDE = max(1, int(os.getenv("VLM_VIDEO_SAMPLE_STRIDE", "30")))
_VLM_VIDEO_MAX_FRAMES = max(1, int(os.getenv("VLM_VIDEO_MAX_FRAMES", "8")))
_VLM_API_VIDEO_MIN_FRAMES = max(1, int(os.getenv("VLM_API_VIDEO_MIN_FRAMES", "4")))
_VLM_API_VIDEO_MAX_FRAMES = max(1, int(os.getenv("VLM_API_VIDEO_MAX_FRAMES", str(_VLM_VIDEO_MAX_FRAMES))))
_VLM_API_MAX_IMAGES = min(500, max(1, int(os.getenv("VLM_API_MAX_IMAGES", "500"))))
_VLM_VIDEO_FPS_RAW = os.getenv("VLM_VIDEO_FPS", "").strip()
_VLM_VIDEO_FPS = float(_VLM_VIDEO_FPS_RAW) if _VLM_VIDEO_FPS_RAW else None
_VLM_QWEN_DO_SAMPLE = os.getenv("VLM_QWEN_DO_SAMPLE", "0") == "1"
_VLM_QWEN_TEMPERATURE = float(os.getenv("VLM_QWEN_TEMPERATURE", "0.7"))
_VLM_QWEN_TOP_P = float(os.getenv("VLM_QWEN_TOP_P", "0.95"))

VISION_BACKEND = get_vision_backend()
_USE_API_BACKEND = use_api_vision_backend()
_VISION_API_MODEL = get_vision_api_model_name()

_SYSTEM_PROMPT = (
    "You are a professional soccer (football) expert with deep knowledge of "
    "tactics, rules, players, teams, leagues, and match analysis. "
    "Carefully analyze the provided content and answer with precision."
)

_AIR_SELECTOR = AIRFrameSelector()

# Per-process cache for A.I.R.-selected frames.
# Key: tuple of sorted absolute video paths.  Value: list of frame file paths in a
# persistent temp dir.  Lets multiple VLM calls on the same materials reuse frame
# selection instead of re-running A.I.R. each time.
_air_frame_cache: dict[tuple, list[str]] = {}
_air_cache_tmp_roots: list[str] = []


def _cleanup_air_cache() -> None:
    for root in _air_cache_tmp_roots:
        shutil.rmtree(root, ignore_errors=True)


atexit.register(_cleanup_air_cache)

# Lazy-loaded qwen runtime globals
_torch = None
_process_vision_info = None
_vlm_model = None
_vlm_processor = None
_VLM_MODEL_NAME = ""
_USE_NATIVE_PIPELINE = False

if not _USE_API_BACKEND:
    # Eager-load qwen when qwen is the selected backend for normal runtime behavior.
    try:
        import torch as _torch_mod
        from qwen_vl_utils import process_vision_info as _pvi
        from pipeline.toolbox.utils.vlm_distribution import (
            vlm_model as _model,
            vlm_processor as _processor,
            VLM_MODEL_NAME as _model_name,
            _USE_NATIVE_PIPELINE as _native,
        )
        _torch = _torch_mod
        _process_vision_info = _pvi
        _vlm_model = _model
        _vlm_processor = _processor
        _VLM_MODEL_NAME = _model_name
        _USE_NATIVE_PIPELINE = _native
    except Exception:
        # Let actual calls surface the real error.
        pass


def _ensure_qwen_runtime():
    global _torch, _process_vision_info, _vlm_model, _vlm_processor, _VLM_MODEL_NAME, _USE_NATIVE_PIPELINE
    if _vlm_model is not None and _vlm_processor is not None and _torch is not None and _process_vision_info is not None:
        return _torch, _process_vision_info, _vlm_model, _vlm_processor

    import torch as _torch_mod
    from qwen_vl_utils import process_vision_info as _pvi
    from pipeline.toolbox.utils.vlm_distribution import (
        vlm_model as _model,
        vlm_processor as _processor,
        VLM_MODEL_NAME as _model_name,
        _USE_NATIVE_PIPELINE as _native,
    )

    _torch = _torch_mod
    _process_vision_info = _pvi
    _vlm_model = _model
    _vlm_processor = _processor
    _VLM_MODEL_NAME = _model_name
    _USE_NATIVE_PIPELINE = _native
    return _torch, _process_vision_info, _vlm_model, _vlm_processor


def _get_qwen_generate_kwargs(max_new_tokens: int) -> dict:
    kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": _VLM_QWEN_DO_SAMPLE,
    }
    if _VLM_QWEN_DO_SAMPLE:
        kwargs["temperature"] = _VLM_QWEN_TEMPERATURE
        kwargs["top_p"] = _VLM_QWEN_TOP_P
    return kwargs


def _ext(path: str) -> str:
    return path.rsplit(".", 1)[-1].lower() if "." in path else ""


def _sorted_media_files(directory: str) -> list[str]:
    all_files = []
    for name in os.listdir(directory):
        full_path = os.path.join(directory, name)
        if not os.path.isfile(full_path):
            continue
        ext = _ext(name)
        if ext in _IMAGE_EXTENSIONS or ext in _VIDEO_EXTENSIONS:
            all_files.append(full_path)

    def get_number(filename: str) -> int:
        match = re.search(r"(\d+)", os.path.basename(filename))
        return int(match.group(1)) if match else 0

    all_files.sort(key=get_number)
    return all_files


def _collect_material_paths(material) -> list[str]:
    if not material:
        return []
    if isinstance(material, str):
        raw_items = [material]
    else:
        raw_items = list(material)

    material_paths = []
    for item in raw_items:
        if not item:
            continue
        if os.path.isdir(item):
            material_paths.extend(_sorted_media_files(item))
        elif os.path.isfile(item):
            material_paths.append(item)
    return material_paths


def _encode_image(path: str) -> str:
    # Re-encode through Pillow so truncated-but-readable images can still be uploaded.
    try:
        with Image.open(path) as img:
            img.load()
            buffer = BytesIO()
            img.save(buffer, format="PNG")
            return base64.b64encode(buffer.getvalue()).decode("utf-8")
    except Exception:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")


def _image_content(path: str) -> dict:
    return {
        "type": "image_url",
        "image_url": {
            "url": f"data:image/png;base64,{_encode_image(path)}",
            "detail": "auto",
        },
    }


def _build_messages_qwen(query: str, media_items: list[dict] | None = None):
    system_content = [{"type": "text", "text": _SYSTEM_PROMPT}] if _USE_NATIVE_PIPELINE else _SYSTEM_PROMPT
    return [
        {"role": "system", "content": system_content},
        {
            "role": "user",
            "content": list(media_items or []) + [{"type": "text", "text": query}],
        },
    ]


def _move_inputs_to_model_device(inputs, model):
    torch_mod, _, _, _ = _ensure_qwen_runtime()
    device_map = getattr(model, "hf_device_map", None)
    if device_map:
        for device in device_map.values():
            if isinstance(device, int):
                return inputs.to(torch_mod.device(f"cuda:{device}"))
            if isinstance(device, str) and device.startswith("cuda:"):
                return inputs.to(torch_mod.device(device))
    return inputs.to(model.device)


def _generate_from_messages_qwen(messages, *, model=None, processor=None, max_new_tokens=_VLM_MAX_NEW_TOKENS):
    torch_mod, process_vision_info, default_model, default_processor = _ensure_qwen_runtime()
    model = model or default_model
    processor = processor or default_processor

    with torch_mod.no_grad():
        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=_VLM_ENABLE_THINKING,
        )
        previous_bridge = None
        if decord is not None:
            try:
                previous_bridge = decord.bridge.get_bridge()
                if previous_bridge != "numpy":
                    decord.bridge.set_bridge("numpy")
            except Exception:
                previous_bridge = None

        image_inputs, video_inputs = process_vision_info(messages)

        if decord is not None and previous_bridge:
            try:
                decord.bridge.set_bridge(previous_bridge)
            except Exception:
                pass

        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = _move_inputs_to_model_device(inputs, model)

        generated_ids = model.generate(**inputs, **_get_qwen_generate_kwargs(max_new_tokens))
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return output_text[0] if output_text else ""


def _generate_text_only_qwen(query: str, *, model=None, processor=None, max_new_tokens=_VLM_MAX_NEW_TOKENS):
    _, _, default_model, default_processor = _ensure_qwen_runtime()
    model = model or default_model
    processor = processor or default_processor
    messages = _build_messages_qwen(query)

    torch_mod, _, _, _ = _ensure_qwen_runtime()
    with torch_mod.no_grad():
        if _USE_NATIVE_PIPELINE:
            inputs = processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=_VLM_ENABLE_THINKING,
                return_dict=True,
                return_tensors="pt",
            )
            inputs = _move_inputs_to_model_device(inputs, model)
        else:
            text = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=_VLM_ENABLE_THINKING,
            )
            inputs = processor(
                text=[text],
                images=None,
                videos=None,
                padding=True,
                return_tensors="pt",
            )
            inputs = _move_inputs_to_model_device(inputs, model)

        generated_ids = model.generate(**inputs, **_get_qwen_generate_kwargs(max_new_tokens))
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return output_text[0] if output_text else ""


def _run_on_images_qwen(query: str, image_paths: list[str], *, model=None, processor=None, max_new_tokens=_VLM_MAX_NEW_TOKENS):
    media_items = [{"type": "image", "image": path} for path in image_paths]
    return _generate_from_messages_qwen(
        _build_messages_qwen(query, media_items),
        model=model,
        processor=processor,
        max_new_tokens=max_new_tokens,
    )


def _run_on_videos_direct_qwen(query: str, video_paths: list[str], *, model=None, processor=None, max_new_tokens=_VLM_MAX_NEW_TOKENS):
    media_items = []
    for path in video_paths:
        item = {"type": "video", "video": path}
        if _VLM_VIDEO_FPS is not None:
            item["fps"] = _VLM_VIDEO_FPS
        media_items.append(item)
    return _generate_from_messages_qwen(
        _build_messages_qwen(query, media_items),
        model=model,
        processor=processor,
        max_new_tokens=max_new_tokens,
    )


def _call_vision_api(query: str, image_paths: list[str] | None = None, *, max_new_tokens: int = _VLM_MAX_NEW_TOKENS) -> str:
    client = make_client(_VISION_API_MODEL)
    selected_paths = list(image_paths or [])
    if len(selected_paths) > _VLM_API_MAX_IMAGES:
        selected_paths = _uniform_sample_image_paths(selected_paths, _VLM_API_MAX_IMAGES)
        logger.warning(
            "[VLM+API] image count exceeded limit; uniformly downsampled to %d image(s).",
            len(selected_paths),
        )

    content = []
    for path in selected_paths:
        content.append(_image_content(path))
    content.append({"type": "text", "text": query})

    extra = {}
    if not is_gpt5_model(_VISION_API_MODEL):
        extra["temperature"] = get_temperature("agent")

    completion = client.chat.completions.create(
        model=_VISION_API_MODEL,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ],
        **get_reasoning_params(_VISION_API_MODEL, role="vlm"),
        **get_max_tokens_param(max_new_tokens, _VISION_API_MODEL),
        **extra,
    )
    return completion.choices[0].message.content or ""


def _run_on_images_api(query: str, image_paths: list[str], *, max_new_tokens: int = _VLM_MAX_NEW_TOKENS) -> str:
    return _call_vision_api(query, image_paths, max_new_tokens=max_new_tokens)


def _generate_text_only_api(query: str, *, max_new_tokens: int = _VLM_MAX_NEW_TOKENS) -> str:
    return _call_vision_api(query, [], max_new_tokens=max_new_tokens)


def _uniform_frame_indices(total_frames: int, nframes: int) -> list[int]:
    if total_frames <= 0 or nframes <= 0:
        return []
    if nframes == 1:
        return [0]
    step = (total_frames - 1) / (nframes - 1)
    indices = [int(round(i * step)) for i in range(nframes)]
    deduped = []
    seen = set()
    for idx in indices:
        idx = max(0, min(total_frames - 1, idx))
        if idx not in seen:
            seen.add(idx)
            deduped.append(idx)
    return deduped


def _uniform_sample_image_paths(paths: list[str], max_images: int) -> list[str]:
    if len(paths) <= max_images:
        return list(paths)
    indices = _uniform_frame_indices(len(paths), max_images)
    return [paths[i] for i in indices]


def _api_target_nframes(total_frames: int, video_fps: float, max_frames: int) -> int:
    if total_frames <= 0:
        return 0
    max_frames = max(1, min(max_frames, total_frames))
    if _VLM_VIDEO_FPS is None or video_fps <= 0:
        return 0

    target_nframes = total_frames / video_fps * _VLM_VIDEO_FPS
    min_frames = min(total_frames, _VLM_API_VIDEO_MIN_FRAMES)
    target_nframes = max(target_nframes, min_frames)
    target_nframes = min(target_nframes, max_frames)
    return max(1, int(round(target_nframes)))


def _sample_video_frames(video_path: str, *, max_frames: int = _VLM_API_VIDEO_MAX_FRAMES, stride: int = _VLM_VIDEO_SAMPLE_STRIDE) -> list[str]:
    temp_dir = tempfile.mkdtemp(prefix="vlm_api_video_")
    frame_paths = []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise ValueError(f"Cannot open video: {video_path}")

    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        video_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)

        # Match the Qwen path more closely when VLM_VIDEO_FPS is set:
        # compute target frame count from requested FPS, then sample uniformly
        # across the whole video under an API-specific max frame budget.
        target_nframes = _api_target_nframes(total_frames, video_fps, max_frames)
        if target_nframes > 0:
            indices = _uniform_frame_indices(total_frames, target_nframes)
            for saved_idx, frame_idx in enumerate(indices):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ok, frame = cap.read()
                if not ok:
                    continue
                frame_path = os.path.join(temp_dir, f"frame_{saved_idx:03d}.jpg")
                cv2.imwrite(frame_path, frame)
                frame_paths.append(frame_path)
        else:
            frame_idx = 0
            saved_idx = 0
            while saved_idx < max_frames:
                ok, frame = cap.read()
                if not ok:
                    break
                if frame_idx % stride == 0:
                    frame_path = os.path.join(temp_dir, f"frame_{saved_idx:03d}.jpg")
                    cv2.imwrite(frame_path, frame)
                    frame_paths.append(frame_path)
                    saved_idx += 1
                frame_idx += 1
    finally:
        cap.release()

    if not frame_paths:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise ValueError(f"No frames extracted from video: {video_path}")
    return frame_paths


def _run_on_videos_sampled_api(query: str, video_paths: list[str], *, max_new_tokens: int = _VLM_MAX_NEW_TOKENS) -> str:
    temp_dirs = []
    selected_frame_paths = []
    try:
        for video_path in video_paths:
            frame_paths = _sample_video_frames(video_path)
            temp_dirs.append(str(Path(frame_paths[0]).parent))
            selected_frame_paths.extend(frame_paths)
        return _run_on_images_api(query, selected_frame_paths, max_new_tokens=max_new_tokens)
    finally:
        for temp_dir in temp_dirs:
            shutil.rmtree(temp_dir, ignore_errors=True)


def _run_vlm(query, material=None, *, backend_override=None, vlm_model=None, vlm_processor=None):
    use_api = _USE_API_BACKEND if backend_override is None else (backend_override == "api")
    material_paths = _collect_material_paths(material)

    if not material_paths:
        if use_api:
            return _generate_text_only_api(query)
        return _generate_text_only_qwen(query, model=vlm_model, processor=vlm_processor)

    first_ext = _ext(material_paths[0])
    if first_ext in _IMAGE_EXTENSIONS:
        if use_api:
            return _run_on_images_api(query, material_paths)
        return _run_on_images_qwen(query, material_paths, model=vlm_model, processor=vlm_processor)

    if first_ext not in _VIDEO_EXTENSIONS:
        raise ValueError(f"Unsupported file extension: {first_ext}")

    if not _USE_AIR_FRAME_SELECTION:
        if use_api:
            return _run_on_videos_sampled_api(query, material_paths)
        return _run_on_videos_direct_qwen(query, material_paths, model=vlm_model, processor=vlm_processor)

    # Check per-process frame cache first: same video paths → reuse selected frames.
    cache_key = tuple(sorted(os.path.abspath(p) for p in material_paths))
    cached_frames = _air_frame_cache.get(cache_key)
    if cached_frames is not None:
        print(f"[VLM+A.I.R.] cache hit: reused {len(cached_frames)} frame(s) from {len(material_paths)} video(s)")
        if use_api:
            return _run_on_images_api(query, cached_frames)
        return _run_on_images_qwen(query, cached_frames, model=vlm_model, processor=vlm_processor)

    # Cache miss — run A.I.R. frame selection.
    temp_roots = []
    try:
        raw_frames: list[str] = []

        def _score_with_vlm(prompt: str, frame_paths: list[str]) -> str:
            if use_api:
                return _run_on_images_api(prompt, frame_paths, max_new_tokens=_AIR_SCORE_MAX_NEW_TOKENS)
            return _run_on_images_qwen(
                prompt,
                frame_paths,
                model=vlm_model,
                processor=vlm_processor,
                max_new_tokens=_AIR_SCORE_MAX_NEW_TOKENS,
            )

        for video_path in material_paths:
            tmp_dir = tempfile.mkdtemp(prefix="vlm_air_")
            temp_roots.append(tmp_dir)
            frame_paths = _AIR_SELECTOR.select_frames(
                video_path=video_path,
                query=query,
                vlm_fn=_score_with_vlm,
                tmp_dir=tmp_dir,
            )
            raw_frames.extend(frame_paths)

        if raw_frames:
            # Copy frames to a persistent dir so they survive temp cleanup, then cache.
            persistent_dir = tempfile.mkdtemp(prefix="vlm_air_cache_")
            _air_cache_tmp_roots.append(persistent_dir)
            selected_frame_paths: list[str] = []
            for i, fp in enumerate(raw_frames):
                dest = os.path.join(persistent_dir, f"{i:04d}_{os.path.basename(fp)}")
                shutil.copy2(fp, dest)
                selected_frame_paths.append(dest)
            _air_frame_cache[cache_key] = selected_frame_paths

            if use_api:
                print(f"[VLM+A.I.R.+API] model={_VISION_API_MODEL} selected {len(selected_frame_paths)} frame(s) from {len(material_paths)} video(s)")
                return _run_on_images_api(query, selected_frame_paths)
            print(f"[VLM+A.I.R.] selected {len(selected_frame_paths)} frame(s) from {len(material_paths)} video(s)")
            return _run_on_images_qwen(query, selected_frame_paths, model=vlm_model, processor=vlm_processor)

        if use_api:
            print(f"[VLM+A.I.R.+API] model={_VISION_API_MODEL} no frames selected -> fallback to sampled frames")
            return _run_on_videos_sampled_api(query, material_paths)
        print("[VLM+A.I.R.] no frames selected -> fallback to direct video inference")
        return _run_on_videos_direct_qwen(query, material_paths, model=vlm_model, processor=vlm_processor)
    except Exception as exc:
        if use_api:
            logger.warning("[VLM+A.I.R.+API] fallback to sampled frames due to: %s", exc)
            return _run_on_videos_sampled_api(query, material_paths)
        logger.warning("[VLM+A.I.R.] fallback to direct video inference due to: %s", exc)
        return _run_on_videos_direct_qwen(query, material_paths, model=vlm_model, processor=vlm_processor)
    finally:
        for root in temp_roots:
            shutil.rmtree(root, ignore_errors=True)


def VLM(query, material=None, vlm_model=None, vlm_processor=None):
    return _run_vlm(query, material=material, backend_override=None, vlm_model=vlm_model, vlm_processor=vlm_processor)


def VLM_QWEN(query, material=None, vlm_model=None, vlm_processor=None):
    return _run_vlm(query, material=material, backend_override="qwen", vlm_model=vlm_model, vlm_processor=vlm_processor)


def VLM_API(query, material=None):
    return _run_vlm(query, material=material, backend_override="api")
