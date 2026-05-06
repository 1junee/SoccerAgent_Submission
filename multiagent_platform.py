import csv
import re, os
import ast
from pathlib import Path
from openai import OpenAI
import json
from tqdm import tqdm
import argparse
import sys

######################################
##                                  ##
##   PHASE 1 - Task Decomposition   ##
##                                  ##
######################################


######################## Parameters ########################
from project_path import PROJECT_PATH
sys.path.append(f"{PROJECT_PATH}/pipeline")
from pipeline.toolbox import *
from pipeline.toolbox.utils.material_path import resolve_material_path

toolbox_functions = {
    "Textual Entity Search": TEXTUAL_ENTITY_SEARCH,
    "Textual Retrieval Augment": TEXTUAL_RETRIEVAL_AUGMENT,
    "Game Search": GAME_SEARCH,
    "Game Info Retrieval": GAME_INFO_RETRIEVAL,
    "Match History Retrieval": MATCH_HISTORY_RETRIEVAL,
    "Entity Recognition": FACE_RECOGNITION,
    "Number Recognition": JERSEY_NUMBER_RECOGNITION,
    "Camera Detection": CAMERA_DETECTION,
    "Segment": SEGMENT,
    "Shot Change": SHOT_CHANGE,
    "Action Classifier": ACTION_CLASSIFICATION,
    "Commentary Generation": COMMENTARY_GENERATION,
    "Jersey Color Relevant VQA": JERSEY_COLOR_VLM,
    "Vision Language Model": VLM,
    "Replay Grounding": REPLAY_GROUNDING,
    "Score and Time Recognition": SCORE_TIME_DETECTION,
    "Frame Selection": FRAME_SELECTION,
    "Foul Recognition": FOUL_RECOGNITION,
    "Grounding Count": GROUNDING_COUNT,
}

import os
os.system('')

def _get_disabled_tools() -> set:
    """Parse DISABLED_TOOLS env var into a set of canonical tool names (lowercased)."""
    raw = os.getenv("DISABLED_TOOLS", "").strip()
    if not raw:
        return set()
    return {t.strip().lower() for t in raw.split(",") if t.strip()}

_disabled = _get_disabled_tools()
if _disabled:
    toolbox_functions = {
        k: v for k, v in toolbox_functions.items()
        if k.strip().lower() not in _disabled
    }

MATERIAL_ROOT = Path(PROJECT_PATH) / "challenge" / "test"


def set_material_root(input_file: str):
    """Determine material root from input file path: train/, valid/, test/, or challenge/.
    Can be overridden via MATERIAL_SPLIT env var (e.g. 'challenge', 'test', 'train', 'valid').
    """
    global MATERIAL_ROOT
    override = os.environ.get("MATERIAL_SPLIT", "").strip()
    if override:
        MATERIAL_ROOT = Path(PROJECT_PATH) / "challenge" / override
        return
    input_path = str(Path(input_file).resolve())
    for split in ("train", "valid", "test", "challenge"):
        if f"challenge/{split}" in input_path or f"challenge\\{split}" in input_path:
            MATERIAL_ROOT = Path(PROJECT_PATH) / "challenge" / split
            return
    MATERIAL_ROOT = Path(PROJECT_PATH) / "challenge" / "test"


######################## Helper Functions ########################

def load_toolbox(file_path):
    descriptions = []
    disabled = _get_disabled_tools()

    with open(file_path, newline='', encoding='utf-8') as csvfile:
        # reader = csv.DictReader(csvfile)
        reader = csv.DictReader(csvfile, quotechar='"', skipinitialspace=True)
        i = 0
        for row in reader:
            tool_name = row['name']
            if tool_name.strip().lower() in disabled:
                continue
            i += 1
            ability = row['ability']
            query_input = row['query input']
            material_input = row['material input']
            output = row['output']
            remark = row['remark']
            
            description = f"=== Tool Description for TOOL{i} ===\n"
            # description += f"Name: TOOL{i}\n"
            description += f"Name: {tool_name}\n"
            description += f"Ability: {ability}\n"
            description += f"Query Input: {query_input}\n"
            description += f"material Input: {material_input}\n"
            description += f"Output: {output}\n"
            description += f"Remark: {remark}\n"
            
            descriptions.append(description)
    
    return descriptions

def load_toolbox_str(file_path=os.path.join(PROJECT_PATH, "pipeline/toolbox.csv")):
    toolbox = ""
    for i in load_toolbox(file_path):
        toolbox += f"{i}\n"
    # print(toolbox)
    return toolbox


def csv_to_task_string(csv_path=os.path.join(PROJECT_PATH, "pipeline/tasks.csv")):
    disabled = _get_disabled_tools()
    result = []
    with open(csv_path, 'r') as file:
        reader = csv.reader(file, quotechar='"', skipinitialspace=True)
        next(reader)
        for i, row in enumerate(reader, start=1):
            if not row:
                continue

            task_str = f"Task{i}: **{row[0]}** {row[1]}"

            chains = []
            for j in range(2, 5):
                if j < len(row) and row[j].strip():
                    candidate_chain = row[j].strip()
                    chain_tools = [t.strip() for t in candidate_chain.split("->")]
                    if disabled and any(t.lower() in disabled for t in chain_tools):
                        continue
                    chains.append(candidate_chain)

            if len(chains) == 1:
                task_str += f"\nRecommended chain: {chains[0]}"
            elif len(chains) > 1:
                task_str += "\nRecommended chains (choose the most suitable one based on the question and available materials):"
                for k, c in enumerate(chains, 1):
                    task_str += f"\n  ({k}) {c}"

            result.append(task_str + "\n")

    return "\n".join(result)

def generate_prompt(taskdecompositionprompt, query, additional_material):
    prompt = taskdecompositionprompt
    prompt += f"\nQuery: {query}\n"
    prompt += f"Additional Material: {additional_material}\n"
    return prompt

from dotenv import load_dotenv
load_dotenv()
from llm_config import get_agent_model, get_chat_client_kwargs, get_responses_params, make_client, get_temperature, is_gpt5_model

CHAT_MODEL = get_agent_model()
CHAT_CLIENT_KWARGS = get_chat_client_kwargs(CHAT_MODEL)
_USE_LOCAL_VLM_AGENT = os.getenv("AGENT_USE_LOCAL_VLM", "0") != "0"

def workflow(input_text, Instruction, follow_up_prompt=None, max_tokens_followup=1500, return_usage=False):
    client = make_client(CHAT_MODEL)
    reasoning, text = get_responses_params(CHAT_MODEL)
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

    if follow_up_prompt:
        response = client.responses.create(
            model=CHAT_MODEL,
            instructions=Instruction,
            input=[
                {"role": "user", "content": input_text},
                {"role": "assistant", "content": first_round_reply},
                {"role": "user", "content": follow_up_prompt}
            ],
            max_output_tokens=max_tokens_followup,
            **_r_kwargs
        )
        usage = getattr(response, "usage", None)
        if usage is not None:
            prompt_tokens += getattr(usage, "input_tokens", 0) or 0
            completion_tokens += getattr(usage, "output_tokens", 0) or 0
        second_round_reply = response.output_text
        if return_usage:
            return first_round_reply, second_round_reply, {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens
            }
        return first_round_reply, second_round_reply

    if return_usage:
        return first_round_reply, {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens
        }
    return first_round_reply

def parse_input(input_str):
    known_info = re.findall(r'\$+([^$\n]+?)\$+', input_str)
    if not known_info:
        _m = re.search(r"Known\s+Info[:\s]+\[([^\]]+)\]", input_str, re.DOTALL)
        if _m:
            known_info = [i.strip() for i in re.split(r"[,;]", _m.group(1))
                          if i.strip() and i.strip().lower() not in ("none", "[ ]")]
    if not known_info:
        _m = re.search(r"Known\s+Info[:\s]+(.+?)(?=\nTool Chain|\n\n|$)", input_str, re.DOTALL)
        if _m:
            txt = _m.group(1).strip().strip("[]")
            if txt and txt.lower() not in ("none", "[ ]"):
                known_info = [txt]

    tool_chain = re.findall(r'\*+\s*([^*\n]+?)\s*\*+', input_str)
    if not tool_chain:
        _m = re.search(r"Tool\s+Chain[:\s]+\[?([^\]\n']+)", input_str, re.IGNORECASE)
        if _m:
            chain = _m.group(1).strip().rstrip("]").rstrip("'")
            tool_chain = [t.strip().strip("*[]") for t in re.split(r"\s*->\s*", chain) if t.strip()]

    # Remove duplicates (preserve order)
    _seen, _dedup = set(), []
    for _t in tool_chain:
        if _t not in _seen:
            _seen.add(_t); _dedup.append(_t)
    tool_chain = _dedup

    return (known_info, tool_chain)

def generate_prompt_execution(query, material, response, toolbox, options):
    prompt_execution = f"""As a multi-agent core in the Soccer Question Answering Assistant, you are required to execute the following tool chain to answer the question:

"{query}"

with the following additional material:

{material}

There are several options:

{options}

with the known info as:

{parse_input(response)[0]}

and you should execute the following tool chain to solve the question:

{parse_input(response)[1]}

As for the usage of the tools, you should follow the following references:

{toolbox}

For every tool above, we would input queries and materials into the tool for execution, the queries are in **text** form and the materials are in list with **file paths**. If no file path is suitable, you just write in 'None' You should determine the contents of materials and queries based on the context of the question, known info and tool descriptions.

For every steps of excution, you should return me with a clear statement of the goal of this step in the context of the overall analysis, the specific tool you are using, and the input variables you are using.

<Call>
    <Purpose>Brief, clear statement of this step’s goal in context of overall analysis</Purpose> 
    <Query>[Query/question here(string). IMPORTANT!!: Such query is highly relevant to the toolbox descriptions. you need to think carefully about your purpose this step and generate appropriate query.]</Query> 
    <Material>[Material list here(a string showing list form). Here as well, you need to think carefully considering the purpose and toolbox.]</Material>
    <Tool>[Tool name here(string)]</Tool>
</Call>

If it is the last step of the execution, you should return me with the following format:

<EndCall>
    <Purpose>Brief, clear statement of this step’s goal in context of overall analysis</Purpose> 
    <Query>[Query/question here(string)]</Query> 
    <Material>[Material list with file paths here(a string showing list form)]</Material>
    <Tool>[Tool name here(string)]</Tool>
</EndCall>

Every time you return me with the instruction as above, I will execute it and return you with the feedback of the execution in this format:

<StepResult>
    <Answer>[The results of this time's execution here(string)]</Answer>
</StepResult>

For every time of generation, you should follow the following rules:
    1. You should be clear about the tool name (must be chosen from toolbox), file path and query/question in the instruction. This part is important for me to understand the context of the execution. You cannot change any of the information in the instruction.
    2. If I have given you the feedback of the execution, you should analyze what you should write in the next call based on the feedback considering the tool chain I gave you and the task descriptions and tool descriptions. You should not repeat the same instruction again.
    3. If my prompt leaves you to generate the first call, you should directly return me with the call in the form from <> to </>. You should not add any other information in the instruction.
    4. Otherwise, if in the prompt I have given you some <StepResult>, you should consider the total process of the execution and continue to return me exactly with the form from <> to </>. You should not add any other information in the instruction.

Once again, I repreat that the question is:

"{query}"

with the following additional material:

{material}

with the known info as:

{parse_input(response)[0]}

and you should execute the following tool chain to solve the question:

{parse_input(response)[1]}

The following is all our execution history, now you can start with your call of first step:
    """

    return prompt_execution

######################################
##                                  ##
##   PHASE 2 - Tool Execution       ##
##                                  ##
######################################


def parse_call_response(model_reply):
    """
    解析 model_reply 中的 <Call> 部分，提取 <Tool>, <Query>, <Material> 的内容。
    """
    tool = re.search(r'<Tool>(.*?)</Tool>', model_reply, re.DOTALL)
    query = re.search(r'<Query>(.*?)</Query>', model_reply, re.DOTALL)
    material = re.search(r'<Material>(.*?)</Material>', model_reply, re.DOTALL)

    if not tool or not query or not material:
        # raise ValueError("Invalid <Call> format in model_reply")
        print(f"Error Reply: {model_reply}")
        print("Invalid <Call> format in model_reply")
        return None

    return normalize_tool_name(tool.group(1)), query.group(1).strip(), material.group(1).strip()


def normalize_tool_name(name: str) -> str:
    if name is None:
        return ""
    s = name.strip()
    # strip surrounding code fences/backticks
    if s.startswith("```") and s.endswith("```"):
        s = s[3:-3].strip()
    if s.startswith("`") and s.endswith("`"):
        s = s[1:-1].strip()
    if (s.startswith("\"") and s.endswith("\"")) or (s.startswith("'") and s.endswith("'")):
        s = s[1:-1].strip()
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1].strip()
    if s.startswith("(") and s.endswith(")"):
        s = s[1:-1].strip()
    if s.startswith("{") and s.endswith("}"):
        s = s[1:-1].strip()
    # collapse whitespace
    s = re.sub(r"\s+", " ", s).strip()
    return s


def resolve_tool_name(raw_name: str, toolbox_functions: dict) -> str:
    """
    Resolve tool name with robust normalization and aliasing:
    - strip wrappers/quotes/brackets
    - case-insensitive match
    - ignore separators like spaces/underscores/dashes
    - allow extra surrounding text (e.g., "Tool: Camera Detection")
    """
    s = normalize_tool_name(raw_name)
    if s in toolbox_functions:
        return s

    lower_map = {k.lower(): k for k in toolbox_functions.keys()}
    if s.lower() in lower_map:
        return lower_map[s.lower()]

    def canon(t: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", t.lower())

    canon_map = {canon(k): k for k in toolbox_functions.keys()}
    c = canon(s)
    if c in canon_map:
        return canon_map[c]

    # If raw contains extra text, try substring match on canonical form.
    candidates = [k for k in toolbox_functions.keys() if canon(k) and canon(k) in c]
    if candidates:
        # prefer the longest canonical match
        candidates.sort(key=lambda k: len(canon(k)), reverse=True)
        return candidates[0]

    return s


def normalize_material_paths(material):
    """
    Ensure relative material paths (e.g., materials/q6/xxx) point to challenge/test/materials.
    """
    if not isinstance(material, (list, tuple)):
        return material

    normalized = []
    for item in material:
        if isinstance(item, str) and item.startswith("materials/"):
            normalized.append(resolve_material_path(item, primary_root=MATERIAL_ROOT))
        else:
            normalized.append(item)
    return normalized


def parse_material_list(material_text):
    """
    Parse the <Material>...</Material> content into a list of strings.
    Handles missing quotes or malformed lists as best as possible.
    """
    if material_text is None:
        return []
    try:
        parsed = ast.literal_eval(material_text)
    except Exception:
        stripped = material_text.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            stripped = stripped[1:-1].strip()
        if not stripped:
            return []
        tokens = [token.strip().strip('\"\'') for token in stripped.split(",")]
        return [token for token in tokens if token]

    if isinstance(parsed, list):
        return [str(item) for item in parsed]
    if isinstance(parsed, str):
        return [parsed]
    return [str(parsed)]


def execute_tool_call(tool_name, query, material, toolbox_functions, option_texts=None, original_query: str = ""):
    """
    根据 tool_name 从 toolbox_functions 中找到对应的函数并执行。
    """
    tool_name = resolve_tool_name(tool_name, toolbox_functions)
    if tool_name not in toolbox_functions:
        print(f"Tool '{tool_name}' not found in toolbox_functions")
        return "<StepResult>\n    <Answer>Tool not found</Answer>\n</StepResult>"

    tool_function = toolbox_functions.get(tool_name)
    execution_retults = None
    try:
        if tool_name.strip().lower() == "camera detection":
            execution_retults = tool_function(query, material, option_texts=option_texts)
        elif tool_name.strip().lower() == "foul recognition":
            execution_retults = tool_function(query, material, option_texts=option_texts)
        elif tool_name.strip().lower() == "jersey color relevant vqa":
            effective_query = original_query if original_query else query
            execution_retults = tool_function(query, material, original_query=effective_query, option_texts=option_texts)
        elif tool_name.strip().lower() == "grounding count":
            effective_query = original_query if original_query else query
            execution_retults = tool_function(effective_query, material)
        else:
            execution_retults = tool_function(query, material)
    except Exception as e:
        execution_retults = f"Error occurred: {str(e)}"
    return f"<StepResult>\n    <Answer>{execution_retults}</Answer>\n</StepResult>"

def generate_LLM_prompt(query):
    """
    生成调用 LLM 做问题回答的 prompt。
    """
    prompt = f"""According to the total process of above conversation, now give me your answer to the question '{query}'. You should retrun me with the following form:
    
Answer: [Your answer here]
Reasoning and Explanation: [Your reasoning and explanation here]

Please make sure that your answer is consistent with the total process of the execution and the tool chain I gave you. You should not change the tool chain and the process of the execution. You should only give me the answer and the reasoning and explanation of the answer without any other words.
    """
    return prompt



# def execute_tool_chain(input_text, toolbox_functions, Instruction="You are a helpful multi-agent assistant that can answer questions about soccer.", api_key=os.getenv("DEEPSEEK_API_KEY"), return_usage=False):
def execute_tool_chain(input_text, toolbox_functions, Instruction="You are a helpful multi-agent assistant that can answer questions about soccer.", option_texts=None, return_usage=False, planned_tool_chain=None, original_query: str = ""):
    if _USE_LOCAL_VLM_AGENT:
        from pipeline.toolbox.utils.vlm_distribution import local_llm_chat as _local_chat
    client = make_client(CHAT_MODEL)
    reasoning, text = get_responses_params(CHAT_MODEL)
    _r_kwargs = {**({"reasoning": reasoning} if reasoning else {}), **({"text": text} if text else {})}
    if not is_gpt5_model(CHAT_MODEL):
        _r_kwargs["temperature"] = get_temperature("agent")

    # Initialize the conversation history with the system instruction and user input
    conversation_history = [
        {"role": "system", "content": Instruction},
        {"role": "user", "content": input_text}
    ]
    total_process = ""
    call_usages = []
    invalid_call_attempts = 0
    invalid_call_flag = False
    step_count = 0
    consecutive_tool = None
    consecutive_count = 0
    # Allow the same tool to be called up to 5 times consecutively.
    # On the 6th consecutive call, force progress to the next planned tool.
    _MAX_SAME_TOOL_CALLS = int(os.getenv("MAX_SAME_TOOL_CALLS", "5"))

    while True:
        # Generate a response from the model
        _instructions = conversation_history[0]["content"] if conversation_history and conversation_history[0]["role"] == "system" else ""
        _input_msgs = [m for m in conversation_history if m["role"] != "system"]
        current_prompt_tokens = 0
        current_completion_tokens = 0

        if _USE_LOCAL_VLM_AGENT:
            model_reply = _local_chat(conversation_history)
        else:
            response = client.responses.create(
                model=CHAT_MODEL,
                instructions=_instructions,
                input=_input_msgs,
                **_r_kwargs
            )
            usage = getattr(response, "usage", None)
            if usage is not None:
                current_prompt_tokens = getattr(usage, "input_tokens", 0) or 0
                current_completion_tokens = getattr(usage, "output_tokens", 0) or 0
            model_reply = response.output_text
        print("Model reply:", model_reply)

        conversation_history.append({"role": "assistant", "content": model_reply})
        total_process += model_reply

        parsed_call = parse_call_response(model_reply)

        if not parsed_call:
            invalid_call_attempts += 1
            invalid_call_flag = True

            if invalid_call_attempts > 5:
                raise ValueError("Invalid <Call> format detected more than 5 consecutive times.")

            if conversation_history and conversation_history[-1]["role"] == "assistant":
                conversation_history.pop()

            if len(model_reply) <= len(total_process):
                total_process = total_process[:-len(model_reply)]
            else:
                total_process = ""
            print("Invalid call attempt :", invalid_call_attempts)

            continue

        invalid_call_attempts = 0
        tool, query, material = parsed_call

        step_count += 1
        call_usages.append({
            "tool": tool if tool else "UNKNOWN",
            "prompt_tokens": current_prompt_tokens,
            "completion_tokens": current_completion_tokens,
        })

        # Allow the same tool to be called consecutively up to 5 times; intervene from the 6th call onward.
        resolved_tool = resolve_tool_name(tool, toolbox_functions)
        if resolved_tool == consecutive_tool:
            consecutive_count += 1
        else:
            consecutive_tool = resolved_tool
            consecutive_count = 1

        if (consecutive_count > _MAX_SAME_TOOL_CALLS
                and planned_tool_chain
                and "<EndCall>" not in model_reply):
            next_tool_raw = None
            for i, pt in enumerate(planned_tool_chain):
                if resolve_tool_name(pt, toolbox_functions) == resolved_tool:
                    if i + 1 < len(planned_tool_chain):
                        next_tool_raw = planned_tool_chain[i + 1]
                    break
            if next_tool_raw:
                next_resolved = resolve_tool_name(next_tool_raw, toolbox_functions)
                print(
                    f"[TOOL_REPEAT] '{resolved_tool}' called {consecutive_count}x consecutively "
                    f"(allowed: {_MAX_SAME_TOOL_CALLS}) → advancing to '{next_resolved}'"
                )
                hint = (
                    f"<StepResult>\n    <Answer>[SYSTEM] '{resolved_tool}' has been called "
                    f"{consecutive_count} times consecutively without new results. "
                    f"Please proceed to the next planned step using '{next_resolved}'.</Answer>\n</StepResult>"
                )
                conversation_history.append({"role": "user", "content": hint})
                total_process += hint
                consecutive_tool = None
                consecutive_count = 0
                continue

        # if tool != "LLM":
        #     material = ast.literal_eval(material) if material is not None else []
        #     user_execution = execute_tool_call(tool, query, material, toolbox_functions)
        #     conversation_history.append({"role": "user", "content": user_execution})
        #     total_process += user_execution
        # # Check if the reply contains <EndCall>
        # if "<EndCall>" in model_reply:
        #     if tool == "LLM":
        #         # Generate a prompt for the LLM to answer the question
        #         llm_prompt = generate_LLM_prompt(query)
        #         conversation_history.append({"role": "assistant", "content": llm_prompt})
        #         completion = client.chat.completions.create(
        #             # model="deepseek-chat",
        #             model=CHAT_MODEL,
        #             messages=conversation_history
        #         )
        #         # Get the model's reply
        #         model_reply = completion.choices[0].message.content
        #         total_process += model_reply
        #     else:
        #         tool, query, material = parse_call_response(model_reply)
        #         material = ast.literal_eval(material) if material is not None else []
        #         user_execution = execute_tool_call(tool, query, material, toolbox_functions)
        #         total_process += user_execution
        #     if return_usage:
        #         return total_process, {
        #             "executor_calls": call_usages
        #         }
        #     return total_process


        # Check if the reply contains <EndCall>
        natural_end = "<EndCall>" in model_reply or "</EndCall>" in model_reply
        if natural_end:
            if tool == "LLM":
                # Generate a prompt for the LLM to answer the question
                llm_prompt = generate_LLM_prompt(query)
                conversation_history.append({"role": "assistant", "content": llm_prompt})
                _instructions = conversation_history[0]["content"] if conversation_history[0]["role"] == "system" else ""
                _input_msgs = [m for m in conversation_history if m["role"] != "system"]
                response = client.responses.create(
                    model=CHAT_MODEL,
                    instructions=_instructions,
                    input=_input_msgs,
                    **_r_kwargs
                )
                # Get the model's reply
                model_reply = response.output_text
                total_process += model_reply
            else:
                material = parse_material_list(material)
                material = normalize_material_paths(material)
                user_execution = execute_tool_call(tool, query, material, toolbox_functions, option_texts=option_texts, original_query=original_query)
                print("User execution:", user_execution)
                total_process += user_execution
            if return_usage:
                return total_process, {
                    "executor_calls": call_usages,
                    "invalid_call_occurred": invalid_call_flag
                }
            return total_process

        # Includes handling logic for mid-chain LLM responses
        else:
            if tool != "LLM":
                material = parse_material_list(material)
                material = normalize_material_paths(material)
                user_execution = execute_tool_call(tool, query, material, toolbox_functions, option_texts=option_texts, original_query=original_query)
                print("User execution:", user_execution)
                conversation_history.append({"role": "user", "content": user_execution})
                total_process += user_execution

            # When LLM tool is in the middle of the tool chain
            elif tool == "LLM":
                llm_prompt = generate_LLM_prompt(query)
                _instructions = conversation_history[0]["content"] if conversation_history[0]["role"] == "system" else ""
                _input_msgs = [m for m in conversation_history if m["role"] != "system"] + [{"role": "user", "content": llm_prompt}]
                response = client.responses.create(
                    model=CHAT_MODEL,
                    instructions=_instructions,
                    input=_input_msgs,
                    **_r_kwargs
                )
                user_execution = f"<StepResult>\n    <Answer>{response.output_text}</Answer>\n</StepResult>"
                print("User execution:", user_execution)
                conversation_history.append({"role": "user", "content": user_execution})
                total_process += user_execution


######################## Some Basic Prompt Information ########################

toolbox_descriptions = load_toolbox_str()
tasks = csv_to_task_string()

TaskDecompositionPrompt = f"""
# Soccer Question Answering Assistant

## Task overview
You are a multi-modal agent that can answer questions about soccer knowledge.
For each question, you will reveive:
- A question about soccer considering different aspects of soccer
- You might also receive one or more video clip or image as context

Your task involves three sequential parts:
1. Problem Decomposition (Part 1) 
    - Identify available information 
    - Break down the question into sequential steps 
2. Sequential Tool Application (Part 2) 
    - Execute one tool at a time 
    - Record each tool’s output 
    - Continue until sufficient information is gathered 
3. Solution Synthesis (Part 3) 
    - Integrate all results 
    - Generate final answer


## Available Tools

For all the QA, you need to decompose them and 
Here are the tools that you can use to answer the questions:
{toolbox_descriptions}

## Common QA Tasks

Here are some common QA tasks that you might meet in the questions, for each types of questions, we provide the recommended tool chain for you to answer the questions:

{tasks}

To be noted, at this stage you only need to treat this question as open-ended QA task, you can use the common QA tasks as reference to decompose the question and identify the required tools.

## Response Format for Part 1
For each query, you should respond ONLY with: 
    Known Info: [list any categories explicitly mentioned in the query and material] 
    Tool Chain: [list required tools connected by ->]

## Examples

Query 1: "How does the viewpoint of the camera shift in the video?"
Additional Material: "video": ["materials/q8/england_epl/2014-2015/2015-02-21 - 18-00 Chelsea 1 - 1 Burnley/1_135.mkv"]

Your response:
Known Info: [$VideoClip$]
Tool Chain: [*Shot Change* -> *Camera Detection* -> *LLM*]

Query 2: "What was the final score of the game 2015-02-21 - 18-00 Chelsea vs Burnley?"
Additional Material: None

Your response:
Known Info: [$GameContext$]
Tool Chain: [*Game Search* -> *Game info Retrieval* -> *Match History Retrieval* -> *LLM*]

Query 3: "How many goals did the player who forced a corner score for Borussia Dortmund's senior team?"
Additional Material: "video": ["materials/q10/SN-Caption-test-align/germany_bundesliga_2015-2016/2015-11-08 - 17-30 Dortmund 3 - 2 Schalke/1_42_03.mp4"]

Your response:
Known Info: [$VideoClip$, $GameContext$]
Tool Chain: [*Vision Language Model* -> *Entity Recognition* -> *Textual Retrieval Augment* -> *LLM*]

## Important Rules
1. You should only use the tools provided in the toolbox to answer the questions and provide the exact tool names.
2. Use exact item category names with $$ to represent the information categories.
3. Use exact tool category names with ** as shown above to represent the tools.
4. Only respond with Part 1 analysis - Parts 2 & 3 will be addressed in subsequent interactions.
5. Connect tools using -> symbol
6. Try your best to decompose the question and identify the required tools, you can first reference the common QA tasks to get some ideas. If the template fits the question, you can directly use the recommended tool chain. If not, you can try to decompose the question and identify the required tools.
"""

def _option_texts_from_options(options):
    if options is None:
        return None
    if isinstance(options, dict):
        values = [str(v) for k, v in options.items() if str(v).strip()]
        return values or None
    if isinstance(options, (list, tuple)):
        values = [str(v) for v in options if str(v).strip()]
        return values or None
    if isinstance(options, str):
        lines = [line.strip() for line in options.splitlines() if line.strip()]
        return lines or None
    return [str(options)]


def EXECUTE_TOOL_CHAIN(query, material, options):
    _MAX_RETRIES = 2  # maximum number of retries when the invalid call limit is exceeded (total _MAX_RETRIES+1 attempts)

    for attempt in range(_MAX_RETRIES + 1):
        if attempt > 0:
            print(f"[RETRY {attempt}/{_MAX_RETRIES}] Invalid call exceeded limit. Restarting from the beginning...")

        # Planner step
        prompt = generate_prompt(TaskDecompositionPrompt, query, material)
        planner_output = workflow(input_text=prompt, Instruction="You are an expert in soccer.", return_usage=True)

        print("Planner output:", planner_output)

        if len(planner_output) == 2:
            res, planner_usage = planner_output
        else:
            res, _, planner_usage = planner_output

        planner_prompt_tokens = planner_usage.get("prompt_tokens", 0) if isinstance(planner_usage, dict) else 0
        planner_completion_tokens = planner_usage.get("completion_tokens", 0) if isinstance(planner_usage, dict) else 0

        # Executor step
        option_texts = _option_texts_from_options(options)

        planned_tool_chain = parse_input(res)[1]

        try:
            executor_output = execute_tool_chain(
                generate_prompt_execution(query, material, res, toolbox_descriptions, options),
                toolbox_functions,
                option_texts=option_texts,
                return_usage=True,
                planned_tool_chain=planned_tool_chain,
                original_query=query,
            )
        except ValueError:
            if attempt < _MAX_RETRIES:
                continue
            # All retries exhausted — return empty result
            print(f"[RETRY EXHAUSTED] All {_MAX_RETRIES} retries failed. Returning empty result.")
            usage_summary = {
                "planner_prompt_tokens": planner_prompt_tokens,
                "planner_completion_tokens": planner_completion_tokens,
                "executor_calls": [],
                "invalid_call_occurred": True,
            }
            return res, "", usage_summary

        if isinstance(executor_output, tuple):
            result, executor_usage = executor_output
        else:
            result, executor_usage = executor_output, {"prompt_tokens": 0, "completion_tokens": 0}

        executor_calls = executor_usage.get("executor_calls", []) if isinstance(executor_usage, dict) else []
        invalid_call_flag = bool(executor_usage.get("invalid_call_occurred", False)) if isinstance(executor_usage, dict) else False

        usage_summary = {
            "planner_prompt_tokens": planner_prompt_tokens,
            "planner_completion_tokens": planner_completion_tokens,
            "executor_calls": executor_calls,
            "invalid_call_occurred": invalid_call_flag,
        }

        return res, result, usage_summary
