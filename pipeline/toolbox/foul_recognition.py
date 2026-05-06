from contextlib import contextmanager
import re
import sys

from dotenv import load_dotenv

load_dotenv()
from llm_config import get_retrieve_model, get_chat_client_kwargs, get_reasoning_params, get_temperature, is_gpt5_model, make_client

CHAT_MODEL = get_retrieve_model()
CHAT_CLIENT_KWARGS = get_chat_client_kwargs(CHAT_MODEL)

from project_path import PROJECT_PATH
sys.path.append(f"{PROJECT_PATH}/pipeline/toolbox")
import vlm as _vlm_module


# ── LLM wrapper ───────────────────────────────────────────────────────────────

def workflow(input_text, Instruction="You are an expert of soccer referee.", follow_up_prompt=None, max_tokens_followup=1500):
    client = make_client(CHAT_MODEL)
    _temp = {} if is_gpt5_model(CHAT_MODEL) else {"temperature": get_temperature("retrieve")}
    completion = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": Instruction},
            {"role": "user", "content": input_text}
        ],
        **get_reasoning_params(CHAT_MODEL),
        **_temp,
    )
    first_round_reply = completion.choices[0].message.content

    if follow_up_prompt:
        completion = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": Instruction},
                {"role": "user", "content": input_text},
                {"role": "assistant", "content": first_round_reply},
                {"role": "user", "content": follow_up_prompt}
            ],
            max_tokens=max_tokens_followup,
            **get_reasoning_params(CHAT_MODEL),
            **_temp,
        )
        return first_round_reply, completion.choices[0].message.content
    return first_round_reply


# ── Category classification ───────────────────────────────────────────────────

def generate_prompt(question):
    """Classify the question into one of the 11 foul-attribute categories."""
    context = (
        "You are tasked with classifying a question into one of the following 11 categories:\n"
        "1. Offence - Classifies whether the foul occurred during an offensive play.\n"
        "2. Contact - Determines if there was physical contact between players during the foul.\n"
        "3. Bodypart - Identifies whether the foul involved the upper or lower body.\n"
        "4. Upper body part - Specifies the exact upper body part involved (Arms, Shoulder, Use of shoulder).\n"
        "5. Action class - Categorizes the TYPE of foul action (e.g. tackling, challenge, elbow, push, dive).\n"
        "6. Severity - Evaluates the CARD/PUNISHMENT severity of the foul (no card, yellow card, red card).\n"
        "7. Multiple fouls - Checks if multiple fouls occurred simultaneously.\n"
        "8. Try to play - Determines if the player attempted to play the ball after the foul.\n"
        "9. Touch ball - Verifies if the offending player touched the ball.\n"
        "10. Handball - Checks if the foul involved hand/arm touching the ball.\n"
        "11. Handball offence - Determines if the offence was caused by a deliberate handball.\n\n"
        "CRITICAL DISTINCTION:\n"
        "- 'Action class' = the TYPE of foul movement (tackling, challenge, elbowing, pushing, diving, holding, high leg)\n"
        "- 'Severity' = the CARD LEVEL (no card, yellow card, red card)\n"
        "If the question asks which foul TYPE/ACTION/CLASSIFICATION it is, use 'Action class'.\n"
        "If the question asks about the CARD or PUNISHMENT, use 'Severity'.\n\n"
        "Output format — strictly one line:\n"
        "[CLASS]: <Category>\n"
        "where <Category> is the EXACT name from the list (e.g., 'Offence', 'Contact', 'Upper body part', 'Action class').\n"
        "No additional text.\n\n"
        "Question: "
    )
    return context + question


def extract_category(output):
    """Extract multi-word category name from LLM classification output."""
    match = re.search(r"\[CLASS\]:\s*(.+)", output)
    if match:
        return match.group(1).strip()
    return "Unknown"


# ── Option handling ───────────────────────────────────────────────────────────

_CATEGORY_OPTIONS = {
    "Offence":          ["Offence", "No offence", "Between"],
    "Contact":          ["With", "Without"],
    "Bodypart":         ["Upperbody", "Underbody"],
    "Upper body part":  ["Arms", "Shoulder"],
    "Action class":     ["St. tackling", "Tackling", "Challenge", "Holding",
                         "Elbowing", "High leg", "Pushing", "Dive"],
    "Severity":         ["No card", "No card or yellow card", "Yellow card",
                         "Yellow card or red card", "Red card"],
    "Multiple fouls":   ["Yes", "No"],
    "Try to play":      ["Yes", "No"],
    "Touch ball":       ["Yes", "No"],
    "Handball":         ["Yes", "No"],
    "Handball offence": ["Yes", "No"],
}

_VOTE_PROMPTS = {
    "multi_angle_consensus_v1": (
        "These clips show the same foul from multiple camera angles.\n"
        "Compare the views and choose the single answer most consistent across all of them.\n"
        "If one angle is ambiguous, trust the overall agreement across views.\n"
        "Question: {question}\n"
        "Options:\n{options_text}\n"
        "{return_instruction}"
    ),
    "elimination_v1": (
        "Watch the multi-angle clips and solve by eliminating wrong options.\n"
        "Reject any option contradicted by visible contact, ball touch, body part, player action, or referee outcome.\n"
        "Keep only the best-supported option.\n"
        "Question: {question}\n"
        "Options:\n{options_text}\n"
        "{return_instruction}"
    ),
    "sequence_timeline_v1": (
        "Watch the clips as one timeline of the same incident.\n"
        "Reason in order: before contact, moment of contact, and immediately after.\n"
        "Use that sequence to answer the question.\n"
        "Question: {question}\n"
        "Options:\n{options_text}\n"
        "{return_instruction}"
    ),
}


@contextmanager
def _force_air_for_foul_vlm():
    previous = _vlm_module._USE_AIR_FRAME_SELECTION
    _vlm_module._USE_AIR_FRAME_SELECTION = True
    try:
        yield
    finally:
        _vlm_module._USE_AIR_FRAME_SELECTION = previous


def _foul_vlm(query: str, material: list):
    with _force_air_for_foul_vlm():
        return _vlm_module.VLM(query=query, material=material)


def _extract_options_from_query(query: str):
    """
    Parse explicit multiple-choice options (O1, O2, …) from the query string.

    Handles patterns like:
      "Choose from: O1 Tackling, O2 St. tackling, O3 Elbowing, O4 Dive"
      "O1: Without contact, O2: With contact. Use the two views ..."

    Returns a list of option texts in order, or None if not found.
    """
    # Split on 'O<digit>:' or 'O<digit> ' labels; capturing group keeps the digit
    parts = re.split(r'\bO(\d+)[:\s]+', query)
    # parts: [text_before, "1", "text_1...", "2", "text_2...", ...]
    if len(parts) < 5:      # need at least 2 options → 5 parts minimum
        return None

    options = []
    i = 1
    while i < len(parts) - 1:
        num_str = parts[i]
        raw_text = parts[i + 1]

        # Truncate at end-of-options markers: period followed by capital, newline
        raw_text = re.split(r'[.]\s+[A-Z]|\n', raw_text)[0]
        cleaned = re.sub(r'\s+', ' ', raw_text).strip().rstrip('.,;')

        if num_str.isdigit() and cleaned:
            options.append((int(num_str), cleaned))
        i += 2

    if len(options) >= 2:
        options.sort()
        return [text for _, text in options]
    return None


def _normalize_answer(raw: str, options: list) -> str:
    """
    Map raw VLM output to the closest valid option.
    1. Exact match
    2. Case-insensitive exact match
    3. Option text contained within raw output (e.g. "O3: Elbowing" → "Elbowing")
    4. Raw is a prefix/substring of option (e.g. "With" → "With contact")
    5. Match ignoring spaces (e.g. "Upperbody" → "Upper body")
    6. Return raw stripped as fallback
    """
    raw_stripped = raw.strip()
    raw_lower = raw_stripped.lower()
    # 1. Exact
    if raw_stripped in options:
        return raw_stripped
    # 2. Case-insensitive exact
    for opt in options:
        if opt.lower() == raw_lower:
            return opt
    # 3. Option is a substring of raw (VLM prefixes like "O3: Elbowing")
    for opt in options:
        if opt.lower() in raw_lower:
            return opt
    # 4. Raw is a substring of option ("With" → "With contact")
    for opt in options:
        if raw_lower in opt.lower():
            return opt
    # 5. Match ignoring spaces ("Upperbody" → "Upper body", "Underbody" → "Under body")
    raw_nospace = raw_lower.replace(' ', '')
    for opt in options:
        if opt.lower().replace(' ', '') == raw_nospace:
            return opt
    return raw_stripped


# ── VLM prompt generation ─────────────────────────────────────────────────────

def _format_options_text(options: list) -> str:
    return "\n".join(f"O{i + 1}: {opt}" for i, opt in enumerate(options))


def _return_instruction(options: list) -> str:
    if len(options) == 4:
        return "Return ONLY O1, O2, O3, or O4."
    key_list = ", ".join(f"O{i + 1}" for i in range(len(options)))
    return f"Return ONLY one option key: {key_list}."


def _extract_option_key(raw: str, options: list) -> str:
    m = re.search(r'\bO(\d+)\b', raw, re.IGNORECASE)
    if m:
        idx = int(m.group(1)) - 1
        if 0 <= idx < len(options):
            return f"O{idx + 1}"
    normalized = _normalize_answer(raw, options)
    if normalized in options:
        return f"O{options.index(normalized) + 1}"
    return "NONE"


def _majority_vote(option_keys: list[str]) -> str:
    valid = [key for key in option_keys if key and key != "NONE"]
    if not valid:
        return "NONE"
    counts = {}
    for key in valid:
        counts[key] = counts.get(key, 0) + 1
    max_count = max(counts.values())
    winners = sorted(
        [key for key, count in counts.items() if count == max_count],
        key=lambda item: int(item[1:]),
    )
    return winners[0]


# ── Main tool ─────────────────────────────────────────────────────────────────

def FOUL_RECOGNITION(query: str, materials: list, option_texts=None) -> str:
    """
    Recognise a foul attribute via three-prompt majority vote across all angles.

    The tool runs the same three prompts used in the AIR prompt sweep:
    multi_angle_consensus_v1, elimination_v1, and sequence_timeline_v1.
    Each prompt returns an option key, and the final answer is chosen by majority vote.
    """
    # Strip camera-speed description prefix when present
    llm_prompt_query = query.split("speed.")[-1].strip() if "speed." in query else query

    # ── Step 1: Determine effective options ───────────────────────────────────
    # Priority: option_texts (from pipeline) > parsed from query > category fallback
    if option_texts:
        explicit_options = list(option_texts)
    else:
        explicit_options = _extract_options_from_query(llm_prompt_query)

    # ── Step 2: Classify the question category (fallback when no explicit opts) ─
    category_prompt = generate_prompt(llm_prompt_query)
    llm_output = workflow(category_prompt)
    category = extract_category(llm_output)

    effective_options = explicit_options if explicit_options else _CATEGORY_OPTIONS.get(category, [])

    if not effective_options:
        raw = _foul_vlm(query=llm_prompt_query, material=materials).strip()
        return f"Three-prompt foul voting could not build options. Raw VLM answer: {raw}."

    options_text = _format_options_text(effective_options)
    prompt_votes = []
    prompt_outputs = []
    for prompt_name in (
        "multi_angle_consensus_v1",
        "elimination_v1",
        "sequence_timeline_v1",
    ):
        prompt_text = _VOTE_PROMPTS[prompt_name].format(
            question=llm_prompt_query,
            options_text=options_text,
            return_instruction=_return_instruction(effective_options),
        )
        raw = _foul_vlm(query=prompt_text, material=materials).strip()
        prompt_outputs.append((prompt_name, raw))
        prompt_votes.append(_extract_option_key(raw, effective_options))

    final_key = _majority_vote(prompt_votes)
    if final_key != "NONE":
        final_answer = effective_options[int(final_key[1:]) - 1]
    else:
        fallback = _normalize_answer(prompt_outputs[0][1], effective_options)
        final_answer = fallback if fallback in effective_options else prompt_outputs[0][1]

    return (
        f"Three-prompt voting analysis across {len(materials)} angle(s). "
        f"The inferred answer is: {final_answer}."
    )
