import os
import cv2
import torch
import numpy as np
from PIL import Image
from datetime import datetime
import torch.nn.functional as F
from pathlib import Path
from transformers import CLIPProcessor, CLIPModel, SiglipProcessor
import ffmpeg
import random
from project_path import PROJECT_PATH

# =====================================================================
# [Config] Number of frames to extract per second from the video (FPS)
# Increasing this number allows more thorough frame inspection but increases processing time.
TARGET_FPS = 2
# =====================================================================

# Number of Top-K extractions and minimum NMS interval (seconds)
TOP_K = 3
NMS_MIN_SEC = 3.0

# Number of context frames before and after the storyboard center (total 2*CONTEXT+1 frames including center)
CONTEXT_FRAMES = 0

_CLIP_MODEL = None
_CLIP_PROCESSOR = None
_CLIP_DEVICE = None
_CLIP_MODEL_NAME = None

_MATCHVISION_SENTINEL = "unisoccer/contrastive"
_MATCHVISION_SIGLIP = "google/siglip-base-patch16-224"
_MATCHVISION_CKPT = Path(PROJECT_PATH) / "pipeline/toolbox/unisoccer/checkpoints/pretrained_contrastive.pth"
_MATCHVISION_MODEL_DIR = Path(PROJECT_PATH) / "pipeline/toolbox/unisoccer"
_FALLBACK_CLIP_MODEL = "openai/clip-vit-large-patch14"
_DEFAULT_CLIP_MODEL = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"

# ── Query Refinement (uses VLM, no separate model loading) ────────────────────
_REFINE_PROMPT = (
    "You are a soccer video retrieval assistant. "
    "Given a question about a soccer video, rewrite it as a short declarative phrase "
    "(max 8 words) describing ONLY the key visual action for image-text matching. "
    "STRICT RULES: "
    "1. Do NOT invent or add any colors, jersey colors, team names, or player attributes "
    "that are NOT explicitly stated in the original question. "
    "2. If the question asks 'which color jersey does X', focus on the ACTION X only — "
    "drop the jersey-color framing entirely. "
    "3. Use neutral terms like 'player' or 'team', never assume or hallucinate attributes. "
    "Output only the phrase, nothing else.\n\nQuestion: {query}"
)


def _refine_query(query: str) -> str:
    """Refine the query using the configured VLM (API or local). No separate model loading.
    Returns the original query on failure."""
    try:
        from pipeline.toolbox.vlm import VLM
        prompt = _REFINE_PROMPT.format(query=query)
        refined = VLM(prompt, None).strip().strip('"').strip("'").strip()
        if refined:
            print(f"[FRAME_SELECTION] query refined: '{query}' → '{refined}'")
            return refined
        return query
    except Exception as exc:
        print(f"[FRAME_SELECTION] query refinement failed: {exc}")
        return query


def _select_clip_device():
    env_device = os.getenv("FRAME_SELECTION_DEVICE")
    if env_device:
        return env_device
    use_gpu = os.getenv("FRAME_SELECTION_USE_GPU", "0") == "1"
    if use_gpu and torch.cuda.is_available():
        try:
            from utils.all_devices import next_device
            return next_device()
        except Exception:
            return "cuda:0"
    return "cpu"


def _load_matchvision(device: str):
    import sys as _sys

    mv_dir = str(_MATCHVISION_MODEL_DIR)
    if mv_dir not in _sys.path:
        _sys.path.insert(0, mv_dir)

    from model.MatchVision_contrastive import MatchVision_contrastive_model

    model = MatchVision_contrastive_model()
    ckpt = torch.load(str(_MATCHVISION_CKPT), map_location="cpu")
    state = ckpt.get("state_dict", ckpt.get("model_state_dict", ckpt))
    state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    model = model.to(device).eval()
    processor = SiglipProcessor.from_pretrained(_MATCHVISION_SIGLIP, use_fast=False)
    return model, processor


def _get_clip(model_name: str, device: str):
    global _CLIP_MODEL, _CLIP_PROCESSOR, _CLIP_DEVICE, _CLIP_MODEL_NAME
    if (
        _CLIP_MODEL is None
        or _CLIP_PROCESSOR is None
        or _CLIP_DEVICE != device
        or _CLIP_MODEL_NAME != model_name
    ):
        if model_name == _MATCHVISION_SENTINEL:
            _CLIP_MODEL, _CLIP_PROCESSOR = _load_matchvision(device)
        else:
            _CLIP_MODEL = CLIPModel.from_pretrained(model_name).to(device)
            _CLIP_MODEL.eval()
            _CLIP_PROCESSOR = CLIPProcessor.from_pretrained(model_name)
        _CLIP_DEVICE = device
        _CLIP_MODEL_NAME = model_name
    return _CLIP_MODEL, _CLIP_PROCESSOR, _CLIP_DEVICE


def _probe_video_size(video_path):
    try:
        probe = ffmpeg.probe(video_path)
        video_stream = next((s for s in probe["streams"] if s["codec_type"] == "video"), None)
        coded_width = int(video_stream["width"])
        coded_height = int(video_stream["height"])

        sar = video_stream.get("sample_aspect_ratio", "1:1") or "1:1"
        if sar not in ("1:1", "0:1", "N/A"):
            sar_n, sar_d = map(int, sar.split(":"))
            display_width = round(coded_width * sar_n / sar_d)
        else:
            display_width = coded_width
        return display_width, coded_height
    except Exception:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise
        try:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if width <= 0 or height <= 0:
                raise ValueError(f"Cannot determine video size for {video_path}")
            return width, height
        finally:
            cap.release()


def _extract_frames_cv2(video_path, target_fps):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"The video file cannot be opened: {video_path}")

    try:
        src_fps = cap.get(cv2.CAP_PROP_FPS)
        if not src_fps or src_fps <= 0:
            src_fps = target_fps
        step = max(1, int(round(src_fps / max(target_fps, 1e-6))))

        frames = []
        frame_idx = 0
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break
            if frame_idx % step == 0:
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(frame_rgb))
            frame_idx += 1
        return frames
    finally:
        cap.release()


def _extract_frames(video_path, target_fps):
    try:
        display_width, display_height = _probe_video_size(video_path)
        process = (
            ffmpeg.input(video_path)
            .filter("scale", display_width, display_height)
            .output("pipe:", format="rawvideo", pix_fmt="rgb24", r=target_fps)
            .run_async(pipe_stdout=True, quiet=True)
        )

        frames = []
        while True:
            in_bytes = process.stdout.read(display_width * display_height * 3)
            if not in_bytes:
                break
            frame = Image.frombytes("RGB", (display_width, display_height), in_bytes)
            frames.append(frame.copy())

        process.wait()
        return frames
    except Exception as exc:
        print(f"[FRAME_SELECTION] ffmpeg extraction failed, using cv2 fallback: {exc}")
        return _extract_frames_cv2(video_path, target_fps)


def _score_frames_clip(query, frames, model, processor, device):
    scored_frames = []
    for idx, frame in enumerate(frames):
        inputs = processor(
            text=[query],
            images=frame,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(device)
        with torch.no_grad():
            similarity = model(**inputs).logits_per_image.item()
        scored_frames.append((idx, similarity, frame))
    return scored_frames


def _score_frames_matchvision(query, frames, model, processor, device):
    if not frames:
        return []

    all_pv = [processor(images=frame, return_tensors="pt")["pixel_values"][0] for frame in frames]

    with torch.no_grad():
        tok = model.text_encoder.tokenizer(
            [query], padding="max_length", return_tensors="pt", truncation=True
        )
        tok = {k: v.to(device) for k, v in tok.items()}
        text_embed = model.text_encoder.model(**tok).pooler_output
        text_embed = F.normalize(text_embed, dim=-1)

    chunk = 30
    all_scores = []
    for start in range(0, len(frames), chunk):
        chunk_pv = torch.stack(all_pv[start:start + chunk])
        t_actual = chunk_pv.shape[0]

        if t_actual < chunk:
            pad = chunk_pv[-1:].expand(chunk - t_actual, -1, -1, -1)
            chunk_pv = torch.cat([chunk_pv, pad], dim=0)

        video_tensor = chunk_pv.permute(1, 0, 2, 3).unsqueeze(0).to(device)

        with torch.no_grad():
            vis = model.visual_encoder(video_tensor)
            vis = F.normalize(vis[0], dim=-1)
            sims = (vis @ text_embed[0]).detach().cpu().tolist()

        all_scores.extend(sims[:t_actual])

    return [(idx, score, frames[idx]) for idx, score in enumerate(all_scores)]


def select_rand_frame(video_path):
    output_dir = os.path.join(PROJECT_PATH, "log/cache")

    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"FRAME_SELECTION_{timestamp}.jpg")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"The video file cannot be opened: {video_path}")

    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames <= 0:
            raise ValueError("No available frame in the video")

        random_frame = random.randint(0, total_frames - 1)

        cap.set(cv2.CAP_PROP_POS_FRAMES, random_frame)

        ret, frame = cap.read()
        if not ret:
            raise ValueError("Random frames cannot be read")

        cv2.imwrite(output_path, frame)

        return output_path

    finally:
        cap.release()


def _apply_nms(scored_frames, nms_min_frames):
    """
    NMS: Select Top-K in descending score order.
    Suppress candidates that are within nms_min_frames of an already-selected frame.
    scored_frames: [(frame_idx, score, pil_image), ...]
    Returns: [(frame_idx, score, pil_image), ...] (up to Top-K)
    """
    sorted_frames = sorted(scored_frames, key=lambda x: x[1], reverse=True)
    selected = []
    for candidate in sorted_frames:
        idx = candidate[0]
        too_close = any(abs(idx - s[0]) < nms_min_frames for s in selected)
        if not too_close:
            selected.append(candidate)
        if len(selected) >= TOP_K:
            break
    return selected


def _make_storyboard(frames_list, output_path):
    """
    frames_list: list of PIL.Image (concatenated horizontally in order)
    Save as a single Storyboard JPEG and return the path.
    """
    if not frames_list:
        return None

    w = frames_list[0].width
    h = frames_list[0].height
    total_w = w * len(frames_list)

    board = Image.new("RGB", (total_w, h))
    for i, img in enumerate(frames_list):
        if img.size != (w, h):
            img = img.resize((w, h), Image.LANCZOS)
        board.paste(img, (i * w, 0))

    board.save(output_path, quality=95, subsampling=0)
    return output_path


def FRAME_SELECTION(query, material, output_dir=None):
    """
    Improved FRAME_SELECTION:
    - Extract Top-3 frames + NMS (3-second interval)
    - Collect CONTEXT_FRAMES before and after each Top frame center → generate Storyboard image
    - Return value: "...paths: /path1.jpg, /path2.jpg, /path3.jpg."
      (first path is the best frame's storyboard, for backward compatibility)
    """
    if output_dir is None:
        output_dir = os.path.join(PROJECT_PATH, "log/cache")

    os.makedirs(output_dir, exist_ok=True)

    # ── Query refinement ──────────────────────────────────────────────────────
    use_refine = os.getenv("FRAME_SELECTION_QUERY_REFINE", "1") == "1"
    effective_query = _refine_query(query) if use_refine else query

    model_name = os.getenv("FRAME_SELECTION_CLIP_MODEL", _DEFAULT_CLIP_MODEL)
    DEVICE = _select_clip_device()
    backend = "clip"
    try:
        model, processor, DEVICE = _get_clip(model_name, DEVICE)
        backend = "matchvision" if model_name == _MATCHVISION_SENTINEL else "clip"
    except Exception as exc:
        print(f"[FRAME_SELECTION] Failed to load {model_name}: {exc}")
        if model_name != _FALLBACK_CLIP_MODEL:
            model_name = _FALLBACK_CLIP_MODEL
            model, processor, DEVICE = _get_clip(model_name, DEVICE)
            backend = "clip"
            print(f"[FRAME_SELECTION] Falling back to {_FALLBACK_CLIP_MODEL}")
        else:
            raise

    video_path = material[0]
    try:
        raw_frames = _extract_frames(video_path, TARGET_FPS)
        if backend == "matchvision":
            all_frames = _score_frames_matchvision(effective_query, raw_frames, model, processor, DEVICE)
        else:
            all_frames = _score_frames_clip(effective_query, raw_frames, model, processor, DEVICE)
    except Exception as exc:
        if backend == "matchvision":
            print(f"[FRAME_SELECTION] MatchVision scoring failed: {exc}")
            model_name = _FALLBACK_CLIP_MODEL
            model, processor, DEVICE = _get_clip(model_name, DEVICE)
            raw_frames = _extract_frames(video_path, TARGET_FPS)
            all_frames = _score_frames_clip(effective_query, raw_frames, model, processor, DEVICE)
        else:
            raise

    if not all_frames:
        try:
            output_path = select_rand_frame(material[0])
            return f"Cannot match the exact frame, so random selected a frame and saved in {output_path}."
        except Exception:
            return "Failed in selecting frame!"

    # ── NMS: Select Top-3 ────────────────────────────────────────────────
    nms_min_frames = max(1, int(NMS_MIN_SEC * TARGET_FPS))
    top_candidates = _apply_nms(all_frames, nms_min_frames)
    # Sort in chronological order (for readability when generating storyboard)
    top_candidates.sort(key=lambda x: x[0])

    # ── Pass 2: Collect context around each Top frame → Save Storyboard ─────
    total_frames_count = len(all_frames)
    timestamp_base = datetime.now().strftime("%Y%m%d_%H%M%S%f")
    saved_paths = []

    for rank, (center_idx, score, _) in enumerate(top_candidates):
        # Index range before and after center (boundary clamping)
        ctx_indices = list(range(
            max(0, center_idx - CONTEXT_FRAMES),
            min(total_frames_count - 1, center_idx + CONTEXT_FRAMES) + 1
        ))

        ctx_frames = [all_frames[i][2] for i in ctx_indices]

        storyboard_path = os.path.join(
            output_dir,
            f"FRAME_SELECTION_{timestamp_base}_top{rank+1}.jpg"
        )
        _make_storyboard(ctx_frames, storyboard_path)
        saved_paths.append(storyboard_path)

    if saved_paths:
        paths_str = ", ".join(saved_paths)
        return (
            f"The selected frame sequences (Top-{len(saved_paths)}, NMS applied) "
            f"are saved in {paths_str}."
        )

    # fallback
    try:
        output_path = select_rand_frame(material[0])
        return f"Cannot match the exact frame, so random selected a frame and saved in {output_path}."
    except Exception:
        return "Failed in selecting frame!"
