import os
import sys
import logging
from pathlib import Path
from typing import List, Optional

import torch

from project_path import PROJECT_PATH
from pipeline.toolbox.utils.all_devices import get_replay_grounding_device
from pipeline.toolbox.utils.material_path import resolve_material_path

logger = logging.getLogger(__name__)

EMBED_BACKEND = os.getenv("REPLAY_GROUNDING_EMBED_BACKEND", "qwen").strip().lower()

DEFAULT_INSTRUCTION = "Represent this football clip for similarity matching."
DEFAULT_EMBED_MODEL = str(Path(PROJECT_PATH) / "models" / "Qwen3-VL-Embedding-8B")
DEFAULT_EMBED_DTYPE = "bfloat16"
DEFAULT_EMBED_FPS = 10.0
DEFAULT_EMBED_MAX_FRAMES = 150
DEFAULT_EMBED_MAX_LENGTH = 8192*10

_EMBEDDER = None
_EMBEDDER_CONFIG = None


def _select_embed_device() -> str:
    env_device = os.getenv("QWEN3_VL_EMBED_DEVICE") or os.getenv("EMBED_DEVICE") or os.getenv("REPLAY_GROUNDING_DEVICE")
    if env_device:
        return env_device
    return get_replay_grounding_device()


def _set_cuda_device(device_str: str) -> None:
    if not device_str or not torch.cuda.is_available():
        return
    if device_str.startswith("cuda:"):
        idx = device_str.split(":", 1)[1]
    else:
        idx = device_str
    try:
        torch.cuda.set_device(int(idx))
    except Exception:
        return



def _get_env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if not value:
        return default
    try:
        return float(value)
    except Exception:
        return default


def _get_env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if not value:
        return default
    try:
        return int(float(value))
    except Exception:
        return default


def _default_embed_root() -> Optional[Path]:
    env_root = os.getenv("QWEN3_VL_EMBED_ROOT") or os.getenv("EMBED_ROOT")
    if env_root:
        return Path(env_root)
    candidate = Path(PROJECT_PATH) / "third_party" / "Qwen3-VL-Embedding"
    return candidate if candidate.exists() else None


def _default_embed_model(embed_root: Optional[Path]) -> str:
    env_model = os.getenv("QWEN3_VL_EMBED_MODEL") or os.getenv("EMBED_MODEL")
    if env_model:
        return env_model
    if embed_root:
        for name in ["Qwen3-VL-Embedding-8B", "Qwen3-VL-Embedding-2B"]:
            candidate = embed_root / "models" / name
            if candidate.exists():
                return str(candidate)
    return DEFAULT_EMBED_MODEL


def _default_materials_root() -> Optional[Path]:
    env_root = os.getenv("REPLAY_GROUNDING_MATERIALS_ROOT") or os.getenv("MATERIALS_ROOT")
    if env_root:
        return Path(env_root)
    candidate = Path(PROJECT_PATH) / "challenge" / "test" / "materials"
    return candidate if candidate.exists() else None


def _add_embed_root_to_path(embed_root: Optional[Path]) -> None:
    if not embed_root:
        return
    embed_root = embed_root.resolve()
    if str(embed_root) not in sys.path:
        sys.path.insert(0, str(embed_root))


def _select_torch_dtype(dtype_name: Optional[str]):
    if not dtype_name:
        return None
    name = dtype_name.strip().lower()
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp16", "float16", "half"}:
        return torch.float16
    if name in {"fp32", "float32"}:
        return torch.float32
    return None


def _get_embedder():
    global _EMBEDDER, _EMBEDDER_CONFIG

    embed_root = _default_embed_root()
    embed_model = _default_embed_model(embed_root)
    dtype_name = os.getenv("QWEN3_VL_EMBED_DTYPE", DEFAULT_EMBED_DTYPE)
    fps = _get_env_float("QWEN3_VL_EMBED_FPS", DEFAULT_EMBED_FPS)
    max_frames = _get_env_int("QWEN3_VL_EMBED_MAX_FRAMES", DEFAULT_EMBED_MAX_FRAMES)
    max_length = _get_env_int("QWEN3_VL_EMBED_MAX_LENGTH", DEFAULT_EMBED_MAX_LENGTH)

    config = (
        str(embed_root) if embed_root else None,
        embed_model,
        dtype_name,
        fps,
        max_frames,
        max_length,
    )

    if _EMBEDDER is not None and _EMBEDDER_CONFIG == config:
        return _EMBEDDER

    device = _select_embed_device()
    _set_cuda_device(device)

    if embed_root:
        _add_embed_root_to_path(embed_root)

    try:
        from src.models.qwen3_vl_embedding import Qwen3VLEmbedder
    except Exception as exc:
        logger.error("Failed to import Qwen3VLEmbedder: %s", exc)
        return None

    torch_dtype = _select_torch_dtype(dtype_name)
    embed_kwargs = {"max_length": max_length}
    if torch_dtype is not None:
        embed_kwargs["torch_dtype"] = torch_dtype

    try:
        _EMBEDDER = Qwen3VLEmbedder(
            model_name_or_path=embed_model,
            fps=fps,
            max_frames=max_frames,
            **embed_kwargs,
        )
    except Exception as exc:
        logger.error("Failed to initialize Qwen3VLEmbedder: %s", exc)
        _EMBEDDER = None
        return None

    _EMBEDDER_CONFIG = config
    return _EMBEDDER


def _embedder_to_device():
    """Embedding model stays resident on its target GPU."""
    embedder = _get_embedder()
    return embedder


def _strip_materials_prefix(path: str) -> Optional[str]:
    normalized = path.replace("\\", "/")
    if normalized.startswith("materials/"):
        return normalized.split("/", 1)[1]
    return None


def _resolve_video_paths(material) -> List[str]:
    if not isinstance(material, (list, tuple)):
        material = [material] if material else []

    materials_root = _default_materials_root()
    resolved: List[str] = []

    for item in material:
        if not item:
            continue

        path = Path(item)
        if path.is_absolute() and path.exists():
            resolved.append(str(path))
            continue
        candidate = resolve_material_path(
            str(item),
            primary_root=materials_root,
            secondary_roots=[Path(PROJECT_PATH)],
        )
        resolved.append(str(Path(candidate).resolve()))

    return resolved


def _build_payload(path: str, instruction: str, fps: Optional[float], max_frames: Optional[int]):
    payload = {"instruction": instruction, "video": path}
    if fps is not None:
        payload["fps"] = fps
    if max_frames is not None:
        payload["max_frames"] = max_frames
    return payload


def _embed_one(embedder, path: str, instruction: str, fps: float, max_frames: int):
    payload = _build_payload(path, instruction, fps, max_frames)
    emb = embedder.process([payload], normalize=True)
    if hasattr(emb, "detach"):
        emb = emb.detach().cpu()
    return emb[0]


def _ordinal(n: int) -> str:
    if n % 100 in (11, 12, 13):
        return f"{n}th"
    return f"{n}{ {1:'st', 2:'nd', 3:'rd'}.get(n % 10, 'th') }"


def _REPLAY_GROUNDING_LOCAL(query=None, material=None, return_path_only: bool = False):
    # Find the replayed clip among four candidates using embedding similarity.
    video_paths = _resolve_video_paths(material)
    if len(video_paths) < 2:
        return "Replay videos not available."

    embedder = _embedder_to_device()
    if embedder is None:
        return "Replay grounding unavailable: embedding model not initialized."

    instruction = os.getenv("QWEN3_VL_EMBED_INSTRUCTION", DEFAULT_INSTRUCTION)
    fps = _get_env_float("QWEN3_VL_EMBED_FPS", DEFAULT_EMBED_FPS)
    max_frames = _get_env_int("QWEN3_VL_EMBED_MAX_FRAMES", DEFAULT_EMBED_MAX_FRAMES)

    replay_path = video_paths[0]
    candidates = video_paths[1:5]

    try:
        replay_emb = _embed_one(embedder, replay_path, instruction, fps, max_frames)
    except Exception as exc:
        logger.error("Failed to embed replay clip: %s", exc)
        return "Replay grounding failed: embedding error."

    scores = []
    for cand in candidates:
        try:
            cand_emb = _embed_one(embedder, cand, instruction, fps, max_frames)
            score = float((cand_emb @ replay_emb).item())
        except Exception as exc:
            logger.error("Failed to embed candidate clip %s: %s", cand, exc)
            score = -1.0
        scores.append(score)
    if not scores:
        return "Replay grounding failed: no candidates."

    best_idx = int(max(range(len(scores)), key=lambda i: scores[i]))
    best_path = candidates[best_idx] if best_idx < len(candidates) else None

    if not best_path:
        return "Replay grounding failed: invalid candidate selection."

    if return_path_only or os.getenv("REPLAY_GROUNDING_RETURN_PATH_ONLY") == "1":
        return best_path

    return f"the {_ordinal(best_idx + 1)} clip."


def REPLAY_GROUNDING(query=None, material=None, return_path_only: bool = False):
    if EMBED_BACKEND != "qwen":
        logger.warning("Unsupported replay grounding backend %r; using local Qwen3-VL-Embedding.", EMBED_BACKEND)
    return _REPLAY_GROUNDING_LOCAL(query=query, material=material, return_path_only=return_path_only)
