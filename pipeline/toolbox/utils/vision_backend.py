import os
import sys

from dotenv import load_dotenv

load_dotenv()

from project_path import PROJECT_PATH

if PROJECT_PATH not in sys.path:
    sys.path.append(PROJECT_PATH)

from llm_config import get_agent_model


_DEFAULT_VISION_BACKEND = "qwen"


def get_vision_backend() -> str:
    value = (
        os.getenv("VISION_BACKEND")
        or os.getenv("VLM_BACKEND")
        or _DEFAULT_VISION_BACKEND
    )
    backend = value.strip().lower()
    return backend if backend in {"qwen", "api"} else _DEFAULT_VISION_BACKEND


def use_api_vision_backend() -> bool:
    return get_vision_backend() == "api"


def get_vision_api_model_name() -> str:
    candidate = (
        os.getenv("VISION_API_MODEL_NAME")
        or os.getenv("API_MODEL_NAME")
        or os.getenv("AGENT_MODEL_NAME")
    )
    value = candidate.strip() if candidate else ""
    return value or get_agent_model()


def get_vision_backend_label() -> str:
    if use_api_vision_backend():
        return f"api:{get_vision_api_model_name()}"
    return os.getenv("VLM_MODEL_NAME", "Qwen/Qwen3-VL-32B-Instruct")
