from collections import Counter
from contextlib import contextmanager
import os
import re
import sys

from project_path import PROJECT_PATH

if PROJECT_PATH not in sys.path:
    sys.path.append(PROJECT_PATH)

os.environ.setdefault('VLM_ENABLE_THINKING', '1')
from pipeline.toolbox.utils.vision_backend import get_vision_backend, use_api_vision_backend
import pipeline.toolbox.vlm as _vlm_module
from pipeline.toolbox.vlm import VLM, _USE_AIR_FRAME_SELECTION

if not use_api_vision_backend():
    from qwen_vl_utils import process_vision_info
    import torch
    try:
        import decord
    except ImportError:
        decord = None
    from pipeline.toolbox.utils.vlm_distribution import vlm_model, vlm_processor, VLM_MODEL_NAME
    _USE_NATIVE_PIPELINE = "Qwen3.5" in VLM_MODEL_NAME and "VL" not in VLM_MODEL_NAME
else:
    process_vision_info = None
    torch = None
    decord = None
    vlm_model = None
    vlm_processor = None
    VLM_MODEL_NAME = ""
    _USE_NATIVE_PIPELINE = False


VISION_BACKEND = get_vision_backend()
_IMAGE_EXTENSIONS = {"jpg", "jpeg", "png", "bmp", "gif", "tiff", "webp"}
_VIDEO_EXTENSIONS = {"mp4", "mkv", "avi", "mov", "flv", "wmv", "webm"}

_jersey_env_fps = os.getenv("JERSEY_COLOR_VLM_FPS", "").strip()
JERSEY_COLOR_FPS = float(_jersey_env_fps) if _jersey_env_fps else 4.0
_jersey_api_max_frames = os.getenv("JERSEY_COLOR_API_MAX_FRAMES", "").strip()
JERSEY_COLOR_API_MAX_FRAMES = int(_jersey_api_max_frames) if _jersey_api_max_frames else 32


@contextmanager
def _override_jersey_vlm_fps(fps: float | None):
    previous_fps = _vlm_module._VLM_VIDEO_FPS
    if fps is not None:
        _vlm_module._VLM_VIDEO_FPS = float(fps)
    try:
        yield
    finally:
        _vlm_module._VLM_VIDEO_FPS = previous_fps


@contextmanager
def _override_jersey_api_max_frames(max_frames: int | None):
    previous_max_frames = _vlm_module._VLM_API_VIDEO_MAX_FRAMES
    if max_frames is not None:
        _vlm_module._VLM_API_VIDEO_MAX_FRAMES = int(max_frames)
    try:
        yield
    finally:
        _vlm_module._VLM_API_VIDEO_MAX_FRAMES = previous_max_frames


def _ext(path):
    return path.rsplit('.', 1)[-1].lower()


# ── fps4 best-combo 3 prompts (color_inventory_then_binary_team | camera_focus_bias_check | ball_relation_trace_v2)
# Source: experiments/q13/outputs/test_jersey_color_vlm_qwen_direct/comparison_csv/combo_vote_analysis/fps4/combo_summary_k3.csv rank=1
# Accuracy: 22/35 = 62.9% (within-fps4 majority vote)
_JERSEY_PROMPTS = [
    (
        'Watch this soccer video and solve it with a team-first decision.\n\n'
        'Step 0 — Build a clean color inventory:\n'
        '  Team A outfield color = ?\n'
        '  Team B outfield color = ?\n'
        '  Goalkeeper kit(s) = ?\n'
        '  Referee color = ?\n\n'
        'Step 1 — Decide which TEAM performs the action in the question.\n'
        '  Do not choose a player yet. Choose Team A or Team B first using who controls the action.\n\n'
        'Step 2 — Map Team A/B to its outfield color from Step 0.\n'
        '  Reject goalkeeper and referee colors.\n\n'
        'Question: {question}\n\n'
        'Options:\n{options_text}\n\n'
        'Reply with ONLY the option key (O1, O2, O3, or O4).'
    ),
    (
        'Watch this soccer video and use camera focus as a supporting clue only.\n\n'
        'Step 0 — Color inventory: Team A outfield = ?, Team B outfield = ?, goalkeeper kit(s) = ?, referee = ?.\n'
        'Step 1 — Note which player the broadcast camera centers at the key action moment.\n'
        'Step 2 — Verify that this player is actually performing the described action, not merely nearby.\n'
        'Step 3 — Read the centered actor jersey color and reject goalkeeper/referee colors.\n'
        'Step 4 — Choose the matching option.\n\n'
        'Question: {question}\n\n'
        'Options:\n{options_text}\n\n'
        'Reply with ONLY the option key (O1, O2, O3, or O4).'
    ),
    (
        'Watch this soccer video and solve it through ball relation.\n\n'
        'Step 0 — Color inventory: Team A outfield = ?, Team B outfield = ?, goalkeeper kit(s) = ?, referee = ?.\n'
        'Step 1 — At the action moment, identify which player has the strongest ball relation: touching, striking, carrying, tackling, or challenging.\n'
        'Step 2 — That player is the likely actor unless the question clearly asks about a non-ball action.\n'
        'Step 3 — Read that player jersey color and keep only outfield team colors.\n'
        'Step 4 — Select the matching option.\n\n'
        'Question: {question}\n\n'
        'Options:\n{options_text}\n\n'
        'Reply with ONLY the option key (O1, O2, O3, or O4).'
    ),
]

# open-form fallback (when no options are provided)
_JERSEY_OPEN_FORM_PROMPT = (
    'Watch this soccer video and answer step by step.\n\n'
    'Step 0 — Color inventory (do this FIRST):\n'
    '  a. Outfield Team A color: ?\n'
    '  b. Outfield Team B color: ?\n'
    '  c. Goalkeeper kit color(s): ? (distinct from outfield kits)\n'
    '  d. Referee kit color: ? (NOT a team color)\n\n'
    'Step 1 — Event: What specific action does the question describe?\n'
    'Step 2 — Actor: Which player performed that action? '
    'Are they Team A or Team B (from your inventory)? What is their outfield jersey color?\n'
    'Step 3 — Confirm: Is that color NOT a goalkeeper kit and NOT a referee kit?\n'
    'Step 4 — Answer:\n\n'
    'Question: {question}\n\n'
    'State the outfield jersey color of the player performing the action as a single color word or short phrase.'
)


def _extract_option_key(text: str) -> str:
    m = re.search(r'\b(O[1-4])\b', text, re.IGNORECASE)
    return m.group(1).upper() if m else 'NONE'


def _majority_vote_key(keys: list[str]) -> str:
    valid = [k for k in keys if k != 'NONE']
    if not valid:
        return 'NONE'
    cnt = Counter(valid)
    top = cnt.most_common()
    if len(top) > 1 and top[0][1] == top[1][1]:
        return valid[0]  # tie → first valid key
    return top[0][0]


def _run_qwen_jcr(jersey_query, material, vlm_model=vlm_model, vlm_processor=vlm_processor, fps=JERSEY_COLOR_FPS):
    if not material:
        raise ValueError("Material list is empty")

    first_ext = _ext(material[0])
    if first_ext in _VIDEO_EXTENSIONS:
        if _USE_AIR_FRAME_SELECTION:
            with _override_jersey_vlm_fps(fps):
                return VLM(jersey_query, material, vlm_model=vlm_model, vlm_processor=vlm_processor)
        video_item = {"type": "video", "video": material[0]}
        if fps is not None:
            video_item["fps"] = fps
        media_items = [video_item]
    elif first_ext in _IMAGE_EXTENSIONS:
        media_items = [{"type": "image", "image": p} for p in material if _ext(p) in _IMAGE_EXTENSIONS]
    else:
        raise ValueError(f"Unsupported file extension: {first_ext}")

    messages = [{
        "role": "user",
        "content": media_items + [{"type": "text", "text": jersey_query}],
    }]

    with torch.no_grad():
        if _USE_NATIVE_PIPELINE:
            inputs = vlm_processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
                return_dict=True,
                return_tensors="pt",
            )
            inputs = inputs.to(vlm_model.device)
        else:
            text = vlm_processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
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
            inputs = vlm_processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(vlm_model.device)

        generated_ids = vlm_model.generate(**inputs, max_new_tokens=512, do_sample=False)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = vlm_processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output_text[0] if output_text else ""


def JERSEY_COLOR_VLM(query, material, vlm_model=vlm_model, vlm_processor=vlm_processor, fps=JERSEY_COLOR_FPS, original_query=None, option_texts=None):
    effective_question = original_query if original_query else query

    # ── no options: single open-form query (fallback) ─────────────────────────
    if not option_texts:
        jersey_query = _JERSEY_OPEN_FORM_PROMPT.format(question=effective_question)
        if use_api_vision_backend():
            with _override_jersey_vlm_fps(fps):
                with _override_jersey_api_max_frames(JERSEY_COLOR_API_MAX_FRAMES):
                    return VLM(jersey_query, material)
        return _run_qwen_jcr(jersey_query, material, vlm_model=vlm_model, vlm_processor=vlm_processor, fps=fps)

    # ── options present: 3-prompt majority vote ─────────────────────────────────
    opts = list(option_texts)
    options_text = '\n'.join(f'  O{i+1}: {v}' for i, v in enumerate(opts))

    voted_keys = []
    for template in _JERSEY_PROMPTS:
        jersey_query = template.format(question=effective_question, options_text=options_text)
        if use_api_vision_backend():
            with _override_jersey_vlm_fps(fps):
                with _override_jersey_api_max_frames(JERSEY_COLOR_API_MAX_FRAMES):
                    raw = VLM(jersey_query, material)
        else:
            raw = _run_qwen_jcr(jersey_query, material, vlm_model=vlm_model, vlm_processor=vlm_processor, fps=fps)
        voted_keys.append(_extract_option_key((raw or '').strip()))

    return _majority_vote_key(voted_keys)
