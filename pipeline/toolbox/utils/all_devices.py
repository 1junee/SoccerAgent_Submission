import os
import torch

def _detect_gpu_count() -> int:
    visible = os.getenv("CUDA_VISIBLE_DEVICES", "").strip()
    if visible:
        tokens = [tok.strip() for tok in visible.split(",")]
        tokens = [tok for tok in tokens if tok and tok.lower() not in {"none", "void"}]
        return len(tokens)
    try:
        return torch.cuda.device_count()
    except Exception:
        return 0


_gpu_count = _detect_gpu_count()
_AVAILABLE_DEVICES = [f"cuda:{idx}" for idx in range(_gpu_count)] or ["cpu"]
_VLM_RESERVE_FIRST_GPU = os.getenv("VLM_RESERVE_GPU0", "1") != "0"


def _env_device(name: str) -> str | None:
    value = os.getenv(name, "").strip()
    return value or None


def _default_aux_device() -> str:
    """
    Default placement for small always-on auxiliary models.

    Intended layout:
    - cuda:0 : UniSoccer / JNR / frame-selection and other small helpers
    - cuda:0,1 : VLM shards (gpu0 residual + gpu1)
    - cuda:2 : local embedding / replay-grounding side
    """
    if _gpu_count == 0:
        return "cpu"
    return "cuda:0"


def _default_embed_device() -> str:
    """Default placement for local embedding-style models."""
    if _gpu_count == 0:
        return "cpu"
    if _gpu_count > 2:
        return "cuda:2"
    if _gpu_count > 1:
        return "cuda:1"
    return "cuda:0"


def next_device() -> str:
    """Return the fixed auxiliary-model device; no round-robin."""
    return _default_aux_device()


def get_unisoccer_device() -> str:
    """Place UniSoccer on the auxiliary GPU by default."""
    return _env_device("UNISOCCER_DEVICE") or _default_aux_device()


def get_replay_grounding_device() -> str:
    """Place replay-grounding / local embedding models on the embedding GPU by default."""
    return (
        _env_device("QWEN3_VL_EMBED_DEVICE")
        or _env_device("EMBED_DEVICE")
        or _env_device("REPLAY_GROUNDING_DEVICE")
        or _default_embed_device()
    )


def get_vlm_device() -> str:
    """
    Return the primary VLM execution device for non-sharded single-device calls.
    Sharded Qwen loading is controlled by get_qwen_device_map(), not by this value.
    """
    if _gpu_count == 0:
        return "cpu"
    return "cuda:0"


def get_jnr_device() -> str:
    """Place Jersey Number Recognition on the auxiliary GPU by default."""
    return _env_device("JNR_DEVICE") or _default_aux_device()

def get_qwen_device_map(reserve_first_gpu: bool = True, reserve_first_n: int = 1) -> dict:
    """
    Build device_map/max_memory kwargs so Qwen can use remaining GPUs.
    If reserve_first_gpu is True and multiple GPUs exist, set cuda:0..cuda:(reserve_first_n-1) to 0GiB.
    Per-GPU max memory can always be set via VLM_GPU{N}_MAX_GIB env vars.
    """
    if _gpu_count == 0:
        return {"device_map": "auto"}

    # Check if any per-GPU override is set
    any_override = any(os.getenv(f"VLM_GPU{idx}_MAX_GIB", "").strip() for idx in range(_gpu_count))

    # No reservation and no overrides → let transformers decide freely
    if not reserve_first_gpu and not any_override:
        return {"device_map": "auto"}

    max_memory = {}
    for idx in range(_gpu_count):
        if reserve_first_gpu and idx < reserve_first_n:
            max_memory[idx] = "0GiB"
            continue
        gpu_max = os.getenv(f"VLM_GPU{idx}_MAX_GIB", "").strip()
        if gpu_max:
            # Respect explicit per-GPU caps without touching CUDA device properties.
            # This avoids triggering torch CUDA lazy-init in environments where that
            # path is unstable or crashes.
            max_memory[idx] = f"{int(gpu_max)}GiB"
            continue

        total_gib = int(torch.cuda.get_device_properties(idx).total_memory / (1024 ** 3))
        safe_gib = max(1, total_gib - 2)
        max_memory[idx] = f"{safe_gib}GiB"

    return {"device_map": "auto", "max_memory": max_memory}

# Backwards compatibility with previous constants
unisoccer_device = get_unisoccer_device()
replay_grounding_device = get_replay_grounding_device()
vlm_device = get_vlm_device()
