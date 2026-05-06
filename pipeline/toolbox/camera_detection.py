import base64
import re
import os
import sys
import cv2
from io import BytesIO
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from collections import Counter
from typing import List, Optional, Sequence

from project_path import PROJECT_PATH

sys.path.append(f"{PROJECT_PATH}/pipeline/toolbox")

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

_USE_API_BACKEND = use_api_vision_backend()
VISION_BACKEND = get_vision_backend()
_VISION_API_MODEL = get_vision_api_model_name()

if not _USE_API_BACKEND:
    import torch
    from pipeline.toolbox.utils.vlm_distribution import vlm_model, vlm_processor
    from qwen_vl_utils import process_vision_info
else:
    torch = None
    vlm_model = None
    vlm_processor = None
    process_vision_info = None


CAMERA_LABELS_ALL = [
    "Main camera center", "Close-up player or field referee", "Close-up side staff",
    "Main camera left", "Main behind the goal", "Close-up behind the goal",
    "Spider camera", "Main camera right", "Public", "Goal line technology camera",
    "Close-up corner", "Inside the goal", "Other"
]

# Order matches sorted example images in example_tiny
EXAMPLE_LABELS = [
    "Close-up behind the goal", "Close-up corner", "Close-up player or field referee",
    "Close-up side staff", "Goal line technology camera", "Inside the goal",
    "Main behind the goal", "Main camera center", "Main camera left",
    "Main camera right", "Other", "Public", "Spider camera"
]

_CAMERA_DETECTION_VIDEO_MODE = os.getenv("CAMERA_DETECTION_VIDEO_MODE", "majority").strip().lower()
_CAMERA_DETECTION_FRAME_STRIDE = max(1, int(os.getenv("CAMERA_DETECTION_FRAME_STRIDE", "10")))
_CAMERA_DETECTION_CLIP_MAX_FRAMES = max(1, int(os.getenv("CAMERA_DETECTION_CLIP_MAX_FRAMES", "8")))
_CAMERA_DETECTION_MAX_NEW_TOKENS = max(
    1,
    int(os.getenv("CAMERA_DETECTION_MAX_NEW_TOKENS", os.getenv("VLM_MAX_NEW_TOKENS", "15000"))),
)


def encode_image(image_path):
    try:
        with Image.open(image_path) as img:
            img.load()
            img_byte_arr = BytesIO()
            img.save(img_byte_arr, format='PNG')
            return base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')
    except Exception:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')


def _normalize_images(img_64) -> List[str]:
    if img_64 is None:
        return []
    if isinstance(img_64, (list, tuple)):
        return [img for img in img_64 if img]
    return [img_64]


def _frame_to_base64(frame) -> str:
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)
    img_byte_arr = BytesIO()
    pil_image.save(img_byte_arr, format='PNG')
    return base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')


def _uniform_indices(total_frames: int, max_frames: int) -> List[int]:
    if total_frames <= 0:
        return []
    if total_frames <= max_frames:
        return list(range(total_frames))
    if max_frames == 1:
        return [total_frames // 2]

    step = (total_frames - 1) / float(max_frames - 1)
    indices = []
    seen = set()
    for i in range(max_frames):
        idx = int(round(i * step))
        if idx >= total_frames:
            idx = total_frames - 1
        if idx not in seen:
            indices.append(idx)
            seen.add(idx)
    return indices


def _sample_video_frames_stride(video_path: str, frame_stride: int) -> List[str]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Cannnot open video: {video_path}")
        return []

    frames_base64 = []
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        if frame_count % frame_stride != 0:
            continue
        frames_base64.append(_frame_to_base64(frame))
    cap.release()
    return frames_base64


def _sample_video_frames_uniform(video_path: str, max_frames: int) -> List[str]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Cannnot open video: {video_path}")
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total_frames <= 0:
        cap.release()
        return _sample_video_frames_stride(video_path, _CAMERA_DETECTION_FRAME_STRIDE)

    target_indices = set(_uniform_indices(total_frames, max_frames))
    frames_base64 = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx in target_indices:
            frames_base64.append(_frame_to_base64(frame))
            if len(frames_base64) >= len(target_indices):
                break
        frame_idx += 1
    cap.release()
    return frames_base64


def _normalize_label(label: str) -> str:
    return label.strip().lower()


def _resolve_prompt_labels(option_texts: Optional[Sequence[str]] = None,
                           allowed_labels: Optional[Sequence[str]] = None) -> List[str]:
    if allowed_labels:
        allowed_set = {_normalize_label(lbl) for lbl in allowed_labels}
        resolved = [lbl for lbl in CAMERA_LABELS_ALL if _normalize_label(lbl) in allowed_set]
        if resolved:
            return resolved

    if option_texts:
        found = []
        for text in option_texts:
            found.extend(extract_camera_positions_from_text(text))
        if found:
            found_keys = {_normalize_label(lbl) for lbl in found}
            resolved = [lbl for lbl in CAMERA_LABELS_ALL if _normalize_label(lbl) in found_keys]
            if resolved:
                return resolved

    return CAMERA_LABELS_ALL


def extract_camera_positions_from_text(text: str, allowed_labels: Optional[Sequence[str]] = None) -> List[str]:
    if not text:
        return []
    options = list(allowed_labels) if allowed_labels else CAMERA_LABELS_ALL
    label_map = {_normalize_label(option): option for option in options}
    pattern = "|".join([re.escape(option) for option in options])
    if not pattern:
        return []
    matches = re.findall(pattern, text, flags=re.IGNORECASE)
    parsed = []
    seen = set()
    for match in matches:
        key = _normalize_label(match)
        if key in label_map and key not in seen:
            parsed.append(label_map[key])
            seen.add(key)
    return parsed


def extract_camera_position(reply, allowed_labels: Optional[Sequence[str]] = None):
    options = list(allowed_labels) if allowed_labels else CAMERA_LABELS_ALL
    label_map = {_normalize_label(option): option for option in options}

    pattern = "|".join([re.escape(option) for option in options])

    match = re.search(pattern, reply, flags=re.IGNORECASE)

    if match:
        return label_map[_normalize_label(match.group(0))]
    else:
        return "None"

# Use GPT4o

# def send_request_with_background(prompt, img_64=None, background=[], api_key="YOUR_API_KEY"):
#     client = OpenAI(
#         base_url='YOUR_API_BASE_URL',  # Replace with your OpenAI API base URL
#         api_key=api_key
#     )
#     messages = background.copy()
#     messages.append({"role": "user", "content": []})
#     messages[-1]["content"].append({"type": "text", "text": prompt})
#     if img_64:
#         base64_image = img_64
#         messages[-1]["content"].append(
#             {
#                 "type": "image_url",
#                 "image_url": {
#                     "url": f"data:image/png;base64,{base64_image}",
#                 }
#             }
#         )
#     # print(f"request size in bytes: {sys.getsizeof(messages)}")

#     response = client.chat.completions.create(
#         model="gpt-4o",
#         messages=messages,
#         max_tokens=512,
#     )

#     model_reply = response.choices[0].message.content
#     return model_reply


def _send_request_with_background_api(prompt, img_64=None, background=None):
    background = list(background or [])
    client = make_client(_VISION_API_MODEL)
    messages = background.copy()
    messages.append({"role": "user", "content": []})
    messages[-1]["content"].append({"type": "text", "text": prompt})
    for base64_image in _normalize_images(img_64):
        messages[-1]["content"].append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{base64_image}",
                    "detail": "auto",
                }
            }
        )

    extra = {}
    if not is_gpt5_model(_VISION_API_MODEL):
        extra["temperature"] = get_temperature("agent")

    response = client.chat.completions.create(
        model=_VISION_API_MODEL,
        messages=messages,
        **get_reasoning_params(_VISION_API_MODEL, role="camera_detection"),
        **get_max_tokens_param(_CAMERA_DETECTION_MAX_NEW_TOKENS, _VISION_API_MODEL),
        **extra,
    )

    model_reply = response.choices[0].message.content
    return model_reply


# Use Qwen2.5-VL-7B-Instruct
def _send_request_with_background_qwen(prompt, img_64=None, background=None, api_key=None):
    """
    Send request using Qwen2.5-VL-7B-Instruct model
    Args:
        prompt: text prompt
        img_64: base64 encoded image
        background: conversation history
        api_key: not used, kept for compatibility
    """
    messages = []

    # Add background messages (learning examples)
    for msg in list(background or []):
        messages.append(msg)

    # Prepare current message content
    current_content = []

    # Add image if provided
    for base64_image in _normalize_images(img_64):
        current_content.append({
            "type": "image",
            "image": f"data:image/png;base64,{base64_image}"
        })

    # Add text prompt
    current_content.append({
        "type": "text",
        "text": prompt
    })

    messages.append({
        "role": "user",
        "content": current_content
    })

    # Process with Qwen model
    text = vlm_processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )

    image_inputs, video_inputs = process_vision_info(messages)

    inputs = vlm_processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(vlm_model.device)

    # Generate response
    with torch.no_grad():
        generated_ids = vlm_model.generate(
            **inputs,
            max_new_tokens=_CAMERA_DETECTION_MAX_NEW_TOKENS,
            do_sample=False,
        )
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        model_reply = vlm_processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

    return model_reply


def send_request_with_background(prompt, img_64=None, background=[], api_key=None):
    if _USE_API_BACKEND:
        return _send_request_with_background_api(prompt, img_64=img_64, background=background)
    return _send_request_with_background_qwen(prompt, img_64=img_64, background=background, api_key=api_key)


def _build_fewshot_history(prompt_labels: Sequence[str]):
    example_path = f"{PROJECT_PATH}/pipeline/toolbox/utils/example_tiny"  # Example images for learning camera positions
    example_img = [os.path.join(example_path, f) for f in os.listdir(example_path)]
    example_img = sorted(example_img)

    learn_prompt = "I want you to help me identify the camera position of a football game photo. Now I will give you some example images, each of which corresponds to a specific camera position. Please learn the characteristics of these images for classification of new photos."

    history = [{"role": "system", "content": learn_prompt}]
    for i in range(len(example_img)):
        if EXAMPLE_LABELS[i] not in prompt_labels:
            continue
        if _USE_API_BACKEND:
            base64_image = encode_image(example_img[i])
            history.append({
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}",
                            "detail": "auto",
                        }
                    },
                    {"type": "text", "text": "What is the camera position in this photo?"}
                ]
            })
            history.append({"role": "assistant", "content": EXAMPLE_LABELS[i]})
        else:
            content = [{"type": "text", "text": f"The camera position corresponding to this photo is: {EXAMPLE_LABELS[i]}"}]
            base64_image = encode_image(example_img[i])
            content.append(
                {
                    "type": "image",
                    "image": f"data:image/png;base64,{base64_image}"
                }
            )
            history.append({"role": "user", "content": content})
    return history


def _detect_single_material_label(img_path: str,
                                  prompt_labels: Sequence[str],
                                  history):
    file_extension = os.path.splitext(img_path)[1].lower()
    image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff']
    video_extensions = ['.mp4', '.mkv', '.avi', '.mov', '.flv', '.webm']
    options_str = ", ".join(prompt_labels)
    ask_prompt = f"What is the camera position in this picture? The answer should be chosen from the following options: [{options_str}]."

    if file_extension in image_extensions:
        base64_image = encode_image(img_path)
        reply = send_request_with_background(ask_prompt, base64_image, history)

        print("camera detection reply: ", reply)

        ans = extract_camera_position(reply, allowed_labels=prompt_labels)
        return ans, "image"

    elif file_extension in video_extensions:
        video_path = img_path
        video_mode = _CAMERA_DETECTION_VIDEO_MODE
        if video_mode == "clip":
            clip_prompt = (
                "These images are sampled from one short football video. "
                f"Determine the dominant camera position across the full clip. "
                f"The answer should be chosen from the following options: [{options_str}]."
            )
            frames_base64 = _sample_video_frames_uniform(video_path, _CAMERA_DETECTION_CLIP_MAX_FRAMES)
        else:
            frames_base64 = _sample_video_frames_stride(video_path, _CAMERA_DETECTION_FRAME_STRIDE)
        if not frames_base64:
            return None, None
        if video_mode == "clip":
            reply = send_request_with_background(clip_prompt, frames_base64, history)
            return extract_camera_position(reply, allowed_labels=prompt_labels), "video"
        reply = []
        for frame in frames_base64:
            reply.append(send_request_with_background(ask_prompt, frame, history))
        ans = [extract_camera_position(r, allowed_labels=prompt_labels) for r in reply]
        count = Counter(ans)
        most_common_str, _ = count.most_common(1)[0]
        return most_common_str, "video"

    return None, None


def CAMERA_DETECTION(query=None, material=[], option_texts: Optional[Sequence[str]] = None,
                     allowed_labels: Optional[Sequence[str]] = None,
                     return_mode: str = "all"):
    prompt_labels = _resolve_prompt_labels(option_texts=option_texts, allowed_labels=allowed_labels)
    history = _build_fewshot_history(prompt_labels)

    if not material:
        return "No material provided."

    if len(material) > 1:
        start_label, _ = _detect_single_material_label(material[0], prompt_labels, history)
        end_label, _ = _detect_single_material_label(material[-1], prompt_labels, history)
        if start_label and end_label:
            return f"From {start_label} to {end_label}."
        if start_label:
            return f"From {start_label} to {start_label}."
        if end_label:
            return f"From {end_label} to {end_label}."
        return "No valid material provided."

    label, kind = _detect_single_material_label(material[0], prompt_labels, history)
    if not label:
        return "No valid material provided."
    if return_mode == "from_to":
        return f"From {label} to {label}."
    if kind == "image":
        return f"The camera position in the photo is: {label}."
    if kind == "video":
        return f"The camera position in the video is: {label}."

    return "No valid material provided."
