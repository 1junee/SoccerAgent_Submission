import os

import sys
import torch
from dotenv import load_dotenv
from transformers import AutoProcessor
from project_path import PROJECT_PATH
sys.path.append(f"{PROJECT_PATH}/pipeline/toolbox")
from utils.all_devices import vlm_device, get_qwen_device_map

load_dotenv()

VLM_MODEL_NAME = os.getenv("VLM_MODEL_NAME", "Qwen/Qwen2.5-VL-7B-Instruct")
RESERVE_FIRST_GPU = os.getenv("VLM_RESERVE_GPU0", "1") != "0"

DEVICE = vlm_device

if "Qwen3.5" in VLM_MODEL_NAME and "VL" not in VLM_MODEL_NAME:
    # Qwen3.5-27B and similar multimodal models (AutoModelForImageTextToText)
    from transformers import AutoModelForImageTextToText

    device_map_kwargs = get_qwen_device_map(reserve_first_gpu=RESERVE_FIRST_GPU)
    vlm_model = AutoModelForImageTextToText.from_pretrained(
        VLM_MODEL_NAME,
        torch_dtype=torch.bfloat16,
        **device_map_kwargs,
    )
    print(f"VLM on: AutoModelForImageTextToText (Qwen3.5), device_map=auto (reserve cuda:0={RESERVE_FIRST_GPU})")
elif "Qwen3-VL" in VLM_MODEL_NAME or "Qwen3.5-VL" in VLM_MODEL_NAME:
    from transformers import Qwen3VLForConditionalGeneration

    device_map_kwargs = get_qwen_device_map(reserve_first_gpu=RESERVE_FIRST_GPU)
    vlm_model = Qwen3VLForConditionalGeneration.from_pretrained(
        VLM_MODEL_NAME,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        **device_map_kwargs,
    )
    print(f"VLM on: Qwen3VLForConditionalGeneration, device_map=auto (reserve cuda:0={RESERVE_FIRST_GPU})")
else:
    from transformers import Qwen2_5_VLForConditionalGeneration

    device_map_kwargs = get_qwen_device_map(reserve_first_gpu=RESERVE_FIRST_GPU)
    vlm_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        VLM_MODEL_NAME,
        torch_dtype="auto",
        **device_map_kwargs,
    )
    print(f"VLM on: Qwen2_5_VLForConditionalGeneration, device_map=auto (reserve cuda:0={RESERVE_FIRST_GPU})")

vlm_model.eval()
vlm_processor = AutoProcessor.from_pretrained(VLM_MODEL_NAME)

# Whether to use text-only LLM inference (Qwen3.5-27B and similar non-VL multimodal models)
_USE_NATIVE_PIPELINE = "Qwen3.5" in VLM_MODEL_NAME and "VL" not in VLM_MODEL_NAME


def local_llm_chat(messages: list, max_new_tokens: int = 4096) -> str:
    """Multi-turn inference using the loaded vlm_model as a text-only LLM."""
    formatted = []
    for msg in messages:
        content = msg["content"]
        if isinstance(content, str):
            content = [{"type": "text", "text": content}]
        formatted.append({"role": msg["role"], "content": content})

    with torch.no_grad():
        if _USE_NATIVE_PIPELINE:
            inputs = vlm_processor.apply_chat_template(
                formatted, tokenize=True, add_generation_prompt=True,
                return_dict=True, return_tensors="pt",
            )
        else:
            text = vlm_processor.apply_chat_template(
                formatted, tokenize=False, add_generation_prompt=True,
            )
            inputs = vlm_processor(
                text=[text], images=None, videos=None,
                padding=True, return_tensors="pt",
            )
        inputs = inputs.to(vlm_model.device)
        generated_ids = vlm_model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
        return vlm_processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
