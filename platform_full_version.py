import sys, json, argparse
import ast
import random
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "challenge"))
import transformers
transformers.logging.disable_progress_bar()
from project_path import PROJECT_PATH
sys.path.append(f"{PROJECT_PATH}/pipeline")
import argparse
from multiagent_platform import EXECUTE_TOOL_CHAIN, set_material_root
from openai import OpenAI

from dotenv import load_dotenv
load_dotenv()
from llm_config import get_agent_model, get_chat_client_kwargs, get_responses_params, make_client, get_temperature, is_gpt5_model


# client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")
CHAT_MODEL = get_agent_model()
CHAT_CLIENT_KWARGS = get_chat_client_kwargs(CHAT_MODEL)
client = make_client(CHAT_MODEL)

# CloseQA is intentionally run on the same model as the agent.
CLOSEQA_MODEL = CHAT_MODEL
CLOSEQA_CLIENT_KWARGS = get_chat_client_kwargs(CLOSEQA_MODEL)
closeqa_client = make_client(CLOSEQA_MODEL)
REQUIRED_RESULT_FIELDS = ["answer"]
INCOMPLETE_BASE_FIELDS = [
    "id",
    "Q",
    "materials",
    "openA",
    "closeA",
    "O1",
    "O2",
    "O3",
    "O4",
]
ERROR_MARKER = "Error occurred:"

def _sanitize_model_name(model_name: str) -> str:
    return re.sub(r"[^0-9A-Za-z_.-]", "_", model_name)

def _get_metadata_path(output_file: str):
    output_dir = os.path.dirname(output_file)
    if not output_dir:
        output_dir = os.getcwd()
    os.makedirs(output_dir, exist_ok=True)
    return os.path.join(output_dir, "metadata.json")

INSTRUCTION = f"""
You are a football expert. You are provided with a question 'Q' and four options 'O1', 'O2', 'O3', and 'O4'.
Before I have used a helpful soccer multi-agent system to solve this process, I will tell you the total process of how agent deal with this problem. 
Please answer the question with one option that best matches the question (replay with 'O1', 'O2', 'O3', or 'O4'). 
Do not include any other text or explanations!!!
"""

def workflow(input_text, Instruction=INSTRUCTION, follow_up_prompt=None, max_tokens_followup=1500, return_usage=False):
    reasoning, text = get_responses_params(CHAT_MODEL, role="agent")
    _r_kwargs = {**({"reasoning": reasoning} if reasoning else {}), **({"text": text} if text else {})}
    if not is_gpt5_model(CHAT_MODEL):
        _r_kwargs["temperature"] = get_temperature("agent")
    prompt_tokens = 0
    completion_tokens = 0

    response = client.responses.create(
        model=CHAT_MODEL,
        instructions=Instruction,
        input=input_text,
        **_r_kwargs
    )
    usage = getattr(response, "usage", None)
    if usage is not None:
        prompt_tokens += getattr(usage, "input_tokens", 0) or 0
        completion_tokens += getattr(usage, "output_tokens", 0) or 0
    first_round_reply = response.output_text
    if return_usage:
        return first_round_reply, {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens
        }
    return first_round_reply


def closeqa_workflow(input_text, Instruction=INSTRUCTION, return_usage=False):
    """Open answer → option(O1~O4) mapping on the shared agent model."""
    reasoning, text = get_responses_params(CLOSEQA_MODEL, role="closeqa")
    _r_kwargs = {**({"reasoning": reasoning} if reasoning else {}), **({"text": text} if text else {})}
    if not is_gpt5_model(CLOSEQA_MODEL):
        _r_kwargs["temperature"] = get_temperature("closeqa")

    response = closeqa_client.responses.create(
        model=CLOSEQA_MODEL,
        instructions=Instruction,
        input=input_text,
        **_r_kwargs
    )
    usage = getattr(response, "usage", None)
    prompt_tokens   = getattr(usage, "input_tokens",  0) or 0 if usage else 0
    completion_tokens = getattr(usage, "output_tokens", 0) or 0 if usage else 0
    reply = response.output_text
    if return_usage:
        return reply, {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens}
    return reply

import re

TAG_PATTERN = re.compile(
    r"(<Call>.*?</(?:Call|EndCall)>|<StepResult>.*?</StepResult>|<EndCall>.*?</EndCall>)",
    re.S,
)


def _strip_block(raw: str, opening: str, closing: str) -> str:
    text = raw.strip()
    if text.startswith(opening):
        text = text[len(opening):]
    if text.endswith(closing):
        text = text[:-len(closing)]
    return text.strip()


def _extract_tag_value(text: str, tag: str) -> str:
    match = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.S)
    return match.group(1).strip() if match else ""


def _parse_material_field(raw: str):
    raw = (raw or "").strip()
    if not raw:
        return []
    try:
        parsed = ast.literal_eval(raw)
        if isinstance(parsed, list):
            return parsed
        if parsed is None:
            return ["None"]
        return [parsed]
    except Exception:
        return [raw]


def _structure_opena_process(raw: str):
    if not raw:
        return []

    steps = []
    pending_step = None
    step_num = 1
    last_end = 0
    remainder_chunks = []

    for match in TAG_PATTERN.finditer(raw):
        start, end = match.span()
        if start > last_end:
            remainder_chunks.append(raw[last_end:start])

        block = match.group(0)
        if block.startswith("<Call>"):
            content = _strip_block(block, "<Call>", "</Call>")
            pending_step = {
                "step": step_num,
                "type": "call",
                "tool": _extract_tag_value(content, "Tool"),
                "purpose": _extract_tag_value(content, "Purpose"),
                "query": _extract_tag_value(content, "Query"),
                "material": _parse_material_field(_extract_tag_value(content, "Material")),
            }
            steps.append(pending_step)
            step_num += 1
        elif block.startswith("<StepResult>"):
            content = _strip_block(block, "<StepResult>", "</StepResult>")
            answer = _extract_tag_value(content, "Answer") or content
            if pending_step is not None:
                pending_step["result"] = answer
                pending_step = None
        else:
            content = _strip_block(block, "<EndCall>", "</EndCall>")
            end_step = {
                "step": step_num,
                "type": "end_call",
                "tool": _extract_tag_value(content, "Tool"),
                "purpose": _extract_tag_value(content, "Purpose"),
                "query": _extract_tag_value(content, "Query"),
                "material": _parse_material_field(_extract_tag_value(content, "Material")),
            }
            steps.append(end_step)
            pending_step = end_step
            step_num += 1
        last_end = end

    if last_end < len(raw):
        remainder_chunks.append(raw[last_end:])

    trailing = "".join(remainder_chunks).strip()
    if trailing and steps:
        steps[-1]["result"] = trailing

    return steps

def process_football_question(input_dict):
    if "openA_process" in input_dict and "answer" in input_dict:
        return input_dict

    question = input_dict.get("Q", "")
    materials = input_dict.get("materials", "")

    options = {key: value for key, value in input_dict.items() if key.startswith("O")}
    options_str = "\n".join([f"{key}: {value}" for key, value in options.items()])

    plan, openA_process_raw, usage_summary = EXECUTE_TOOL_CHAIN(question, materials, options_str)

    prompt = f"""This football question is "{question}". The four corresponding options are:
{options_str}

The processing through the multi-agent platform is as follows:
{openA_process_raw}

Respond with the single option label that best answers the question (choose only from O1, O2, O3, O4). No explanations."""

    processed_prompt, closeqa_usage = closeqa_workflow(prompt, return_usage=True)

    print(f"CloseA: {processed_prompt}")

    answer_match = re.search(r"O\d+", processed_prompt)

    if answer_match:
        answer = answer_match.group(0)
        used_fallback = False
    else:
        option_num = len(options)
        if option_num > 0:
            answer = random.choice([f'O{i}' for i in range(1, option_num + 1)])
        else:
            answer = "O1"
        used_fallback = True

    # known_info: $$..$$, $..$ priority → Known Info: [...] section → line-by-line fallback
    known_info = re.findall(r"\$+([^$\n]+?)\$+", plan)
    if not known_info:
        _m = re.search(r"Known\s+Info[:\s]+\[([^\]]+)\]", plan, re.DOTALL)
        if _m:
            known_info = [i.strip() for i in re.split(r"[,;]", _m.group(1))
                          if i.strip() and i.strip().lower() not in ("none", "[ ]")]
    if not known_info:
        _m = re.search(r"Known\s+Info[:\s]+(.+?)(?=\nTool Chain|\n\n|$)", plan, re.DOTALL)
        if _m:
            txt = _m.group(1).strip().strip("[]")
            if txt and txt.lower() not in ("none", "[ ]"):
                known_info = [txt]

    # planned_tools: **Tool**, *Tool* priority → Tool Chain: [A -> B] → A -> B (plain)
    planned_tools = re.findall(r"\*+\s*([^*\n]+?)\s*\*+", plan)
    if not planned_tools:
        _m = re.search(r"Tool\s+Chain[:\s]+\[?([^\]\n']+)", plan, re.IGNORECASE)
        if _m:
            chain = _m.group(1).strip().rstrip("]").rstrip("'")
            planned_tools = [t.strip().strip("*[]") for t in re.split(r"\s*->\s*", chain) if t.strip()]
    if not planned_tools:
        planned_tools = re.findall(r"(?m)^[ \t]*[-•]\s+(.+)", plan)
    if not planned_tools:
        planned_tools = re.findall(r"(?m)^[ \t]*\d+[.)]\s+(.+)", plan)

    # Remove duplicates (preserve order)
    _seen, _dedup = set(), []
    for _t in planned_tools:
        if _t not in _seen:
            _seen.add(_t); _dedup.append(_t)
    planned_tools = _dedup

    executed_tools = re.findall(r"<Tool>(.*?)</Tool>", openA_process_raw)
    openA_process = _structure_opena_process(openA_process_raw)

    planner_prompt_tokens = usage_summary.get("planner_prompt_tokens", 0) if isinstance(usage_summary, dict) else 0
    planner_completion_tokens = usage_summary.get("planner_completion_tokens", 0) if isinstance(usage_summary, dict) else 0
    closeqa_prompt_tokens = closeqa_usage.get("prompt_tokens", 0) if isinstance(closeqa_usage, dict) else 0
    closeqa_completion_tokens = closeqa_usage.get("completion_tokens", 0) if isinstance(closeqa_usage, dict) else 0 

    executor_calls = usage_summary.get("executor_calls", []) if isinstance(usage_summary, dict) else []

    result_dict = input_dict.copy()
    result_dict["planner_output"] = plan
    result_dict["known_info"] = known_info
    result_dict["planned_tools"] = planned_tools
    result_dict["executed_tools"] = executed_tools
    result_dict["openA_process"] = openA_process
    result_dict["answer"] = answer

    result_dict["token_usage"] = {
        "planner_prompt_tokens": planner_prompt_tokens,
        "planner_completion_tokens": planner_completion_tokens,
        "executor_calls": executor_calls,
        "closeqa_prompt_tokens": closeqa_prompt_tokens,
        "closeqa_completion_tokens": closeqa_completion_tokens
    }

    executor_in  = sum(c.get("prompt_tokens", 0)     for c in executor_calls)
    executor_out = sum(c.get("completion_tokens", 0) for c in executor_calls)
    total_in  = planner_prompt_tokens     + executor_in  + closeqa_prompt_tokens
    total_out = planner_completion_tokens + executor_out + closeqa_completion_tokens
    print(
        f"[TOKEN] planner={planner_prompt_tokens}+{planner_completion_tokens}"
        f"  executor={executor_in}+{executor_out}"
        f"  closeqa={closeqa_prompt_tokens}+{closeqa_completion_tokens}"
        f"  total={total_in}+{total_out}"
    )

    result_dict["random_choice_used"] = used_fallback
    invalid_flag = False
    if isinstance(usage_summary, dict):
        invalid_flag = bool(usage_summary.get("invalid_call_occurred", False))
    result_dict["invalid_call_occurred"] = invalid_flag


    # For metadata
    id = input_dict.get("id", "")

    challenge_dict = {}
    challenge_dict["id"] = id
    challenge_dict["Answer"] = answer

    return result_dict, challenge_dict


def collect_incomplete_entries(entries):
    """
    Collect the original entries that are missing required result fields.
    Returned entries are the same structure as found in result.json.
    """
    incomplete = []
    for entry in entries:
        missing = False
        for field in REQUIRED_RESULT_FIELDS:
            value = entry.get(field)
            if value is None:
                missing = True
                break
            if isinstance(value, str) and not value.strip():
                missing = True
                break
            if isinstance(value, (list, dict)) and not value:
                missing = True
                break
        if missing:
            base_entry = {key: entry.get(key) for key in INCOMPLETE_BASE_FIELDS if key in entry}
            incomplete.append(base_entry)
    return incomplete


def _contains_error_marker(value):
    if isinstance(value, str):
        return ERROR_MARKER in value
    if isinstance(value, dict):
        return any(_contains_error_marker(v) for v in value.values())
    if isinstance(value, list):
        return any(_contains_error_marker(v) for v in value)
    return False


def collect_error_entries(entries):
    """
    Collect original entries whose agent/tool trace contains an explicit
    "Error occurred:" marker so they can be retried from error.json.
    """
    error_entries = []
    seen_ids = set()
    for entry in entries:
        entry_id = entry.get("id")
        if entry_id in seen_ids:
            continue
        if _contains_error_marker(entry):
            base_entry = {key: entry.get(key) for key in INCOMPLETE_BASE_FIELDS if key in entry}
            error_entries.append(base_entry)
            seen_ids.add(entry_id)
    return error_entries


from tqdm import tqdm
import json
from result_to_metadata import extract_metadata_from_result

def process_json_file(input_file, output_file):
    set_material_root(input_file)
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data_list = json.load(f)

        # Resume: merge already-processed results from output_file
        if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
            try:
                with open(output_file, 'r', encoding='utf-8') as f:
                    existing = json.load(f)
                if isinstance(existing, list):
                    existing_by_id = {e.get("id"): e for e in existing if e.get("id") is not None}
                    for i, item in enumerate(data_list):
                        item_id = item.get("id")
                        if item_id in existing_by_id:
                            data_list[i] = existing_by_id[item_id]
                    already_done = sum(1 for e in data_list if "answer" in e and e.get("answer"))
                    if already_done:
                        print(f"[RESUME] Skipping {already_done} already-completed item(s).")
            except Exception as e:
                print(f"[RESUME] Could not load existing output ({e}), starting fresh.")

        progress_bar = tqdm(data_list, desc="Processing (Accuracy: N/A)", unit="item")

        metadata_path = _get_metadata_path(output_file)

        for i, item in enumerate(progress_bar):
            try:
                if "openA_process" in item and item.get("answer"):
                    continue

                updated_item, _ = process_football_question(item)
                data_list[i] = updated_item

                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(data_list, f, ensure_ascii=False, indent=4)

            except ValueError as ve:
                print(f"ValueError processing item {i}: {ve}")
                continue

            except Exception as e:
                print(f"Unexpected error processing item {i}: {e}")
                continue

            correct_count = 0
            total_count = 0
            for entry in data_list:
                if "openA_process" in entry and "answer" in entry:
                    total_count += 1
                    if entry["answer"] == entry.get("closeA"):
                        correct_count += 1

            accuracy = correct_count / total_count if total_count > 0 else 0

            progress_bar.set_description(f"Processing (Accuracy: {accuracy:.2%})")
            progress_bar.refresh()

        print(f"Processing completed. Output saved to {output_file}")

        incomplete_entries = collect_incomplete_entries(data_list)
        if incomplete_entries:
            base, ext = os.path.splitext(output_file)
            incomplete_path = f"{base}_incomplete{ext or '.json'}"
            with open(incomplete_path, 'w', encoding='utf-8') as f:
                json.dump(incomplete_entries, f, ensure_ascii=False, indent=4)
            print(f"Incomplete entries saved to {incomplete_path}")
        else:
            print("No incomplete entries detected.")

        error_entries = collect_error_entries(data_list)
        error_path = os.path.join(os.path.dirname(output_file) or os.getcwd(), "error.json")
        if error_entries:
            with open(error_path, 'w', encoding='utf-8') as f:
                json.dump(error_entries, f, ensure_ascii=False, indent=4)
            print(f"Error entries saved to {error_path}")
        else:
            if os.path.exists(error_path):
                os.remove(error_path)
            print("No error entries detected.")

        # metadata.json is regenerated from the full result.json (guarantees complete state even on resume)
        with open(output_file, 'r', encoding='utf-8') as f:
            final_data = json.load(f)
        metadata = extract_metadata_from_result(final_data)
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=4)
        print(f"Metadata saved to {metadata_path}")

    except Exception as e:
        print(f"Error processing file: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a JSON file containing football questions.")
    parser.add_argument("--input_file", type=str, help="Path to the input JSON file. See https://huggingface.co/datasets/Homie0609/SoccerBench/raw/main/qa/q1.json as an example")
    parser.add_argument("--output_file", type=str, help="Path to save the output JSON file. You can just set an json path.")

    args = parser.parse_args()
    process_json_file(args.input_file, args.output_file)
