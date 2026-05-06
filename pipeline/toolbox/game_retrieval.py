#### GAME INFO RETRIEVAL & MATCH HISTORY RETRIEVAL

import json
import json
from pathlib import Path
from tqdm import tqdm
import argparse, os
from functools import lru_cache
from typing import List, Dict
from openai import OpenAI

from project_path import PROJECT_PATH
from pipeline.toolbox.utils.material_path import resolve_material_path

######################## Parameters ########################

from dotenv import load_dotenv
load_dotenv()
from llm_config import get_retrieve_model, get_chat_client_kwargs, get_responses_params, make_client

# client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")
CHAT_MODEL = get_retrieve_model()
CHAT_CLIENT_KWARGS = get_chat_client_kwargs(CHAT_MODEL)
client = make_client(CHAT_MODEL)

def workflow(input_text, Instruction, follow_up_prompt=None, max_tokens_followup=1500):
    reasoning, text = get_responses_params(CHAT_MODEL)
    _r_kwargs = {**({"reasoning": reasoning} if reasoning else {}), **({"text": text} if text else {})}

    response = client.responses.create(
        model=CHAT_MODEL,
        instructions=Instruction,
        input=input_text,
        **_r_kwargs
    )
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
        second_round_reply = response.output_text
        return first_round_reply, second_round_reply
    else:
        return first_round_reply

# material path
def _resolve_material_path(material):
    if isinstance(material, (list, tuple)) and material:
        candidate = material[0]
    else:
        candidate = material
    if not candidate:
        return None
    candidate_path = Path(candidate)
    if candidate_path.is_absolute():
        return candidate_path
    return Path(resolve_material_path(str(candidate_path), primary_root=Path(PROJECT_PATH)))


# ============================ MATCH HISTORY RETRIEVAL ============================

# Commentary generation function for Match History Retrieval
# MatchTime JSON format
def generate_commentary_from_json_matchtime(json_file_path):
    with open(json_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    event_list = data.get("annotations", [])
    if not event_list:
        return "No annotations found in the JSON file."
    
    result = []
    
    for event in reversed(event_list): 
        timestamp = event.get("contrastive_aligned_gameTime", "")
        if not timestamp:
            timestamp = event.get("gameTime", "")
        if not timestamp:
            continue  
        
        try:
            half, time = timestamp.split(" - ")
            if half == "1":
                half_str = "1st half"
            elif half == "2":
                half_str = "2nd half"
            else:
                continue 
        except ValueError:
            continue 
        
        description = event.get("description", "")
        if not description:
            continue
        
        commentary_line = f"{half_str} - {time} \"{description}\""
        result.append(commentary_line)
    
    return "\n".join(result)

# 1988 JSON format - Commentary generation for Match History Retrieval
def generate_commentary_from_json_1988(json_file_path):
    with open(json_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    comments_list = data.get("comments", [])
    if not comments_list:
        return "No comments found in the JSON file."
    
    result = []
    
    for comment in comments_list:
        half = comment.get("half")
        if half not in [1, 2]:
            continue 
        
        timestamp = comment.get("time_stamp", "")
        if not timestamp:
            continue 
        
        comments_text = comment.get("comments_text", "")
        if not comments_text:
            continue 
        
        half_str = "1st half" if half == 1 else "2nd half"
        
        commentary_line = f"{half_str} - {timestamp} \"{comments_text}\""
        result.append(commentary_line)
    
    return "\n".join(result)

# Call the appropriate commentary generation function based on JSON file path
def generate_commentary_from_json(json_file_path):
    basename = os.path.basename(json_file_path)
    
    if basename == "Labels-caption.json":
        return generate_commentary_from_json_matchtime(json_file_path)
    else:
        return generate_commentary_from_json_1988(json_file_path)
    
# Answer the question based on the retrieved commentary information
def MATCH_HISTORY_RETRIEVAL(query, material):
    file_path = _resolve_material_path(material)
    if not file_path or not file_path.exists():
        return "Match history data not available."

    match_history = generate_commentary_from_json(file_path)
    prompt = f"""Here is a question about soccer game: 
    
    "{query}"

The match history information has been found as following shows, you need to answer the question based on the information provided:

    {match_history}

Please provide the answer based on the match history information. Please think it carefully and make sure your answer is evidence-based and accurate. Now answer the question in the following format:

[ANSWER]: [Your answer here]
[EXPLANATION & REASONING]: [Your explanation here]

You should return exactly in this form without any other words.
    """

    answer = workflow(prompt, "You are a soccer expert that answers questions based on match history information.")

    return answer

# ============================ GAME INFO RETRIEVAL ============================

# Match info generation function for Game Info Retrieval
# MatchTime JSON format
def get_match_info_matchtime(json_file_path):
    with open(json_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    if "annotations" in data:
        del data["annotations"]
    
    result = json.dumps(data, indent=4, ensure_ascii=False)
    
    return result

# 1988 JSON format
def get_match_info_1988(json_file_path):
    with open(json_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    if "comments" in data:
        del data["comments"]
    
    result = json.dumps(data, indent=4, ensure_ascii=False)
    
    return result

# Call the appropriate match info generation function based on JSON file path
def get_match_info(json_file_path):
    basename = os.path.basename(json_file_path)
    
    if basename == "Labels-caption.json":
        return get_match_info_matchtime(json_file_path)
    else:
        return get_match_info_1988(json_file_path)

# Answer the question based on the retrieved match information data
def GAME_INFO_RETRIEVAL(query, material):
    file_path = _resolve_material_path(material)
    if not file_path or not file_path.exists():
        return "Match information not available."
    match_info = get_match_info(file_path)
    prompt = f"""Here is a question about soccer game: 
    
    "{query}"

The match related information has been found as following shows, you need to answer the question based on the information provided:

    {match_info}

Please provide the answer based on the match related information. Please think it carefully and make sure your answer is evidence-based and accurate. Now answer the question in the following format:

[ANSWER]: [Your answer here]
[EXPLANATION & REASONING]: [Your explanation here]

You should return exactly in this form without any other words.
    """

    answer = workflow(prompt, "You are a soccer expert that answers questions based on match history information.")

    return answer
