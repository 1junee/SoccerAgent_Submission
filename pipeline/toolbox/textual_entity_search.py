import json
from tqdm import tqdm
import argparse
from functools import lru_cache
from typing import List, Dict
from openai import OpenAI
import os

from project_path import PROJECT_PATH

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

    
def extract_entity_info(question):
    """
    Extracts the type and exact name of a football-related entity (player, referee, team, venue) from a given question.

    Args:
        question (str): The question containing the entity.

    Returns:
        tuple: A tuple containing the type and name of the entity, e.g., ("player", "Lionel Messi").
    """
    # Define the instruction
    Instruction = """
    You are an intelligent assistant that can analyze questions related to football. Your task is to identify the type of entity mentioned in the question and extract the exact name of the entity. The entity types are: player, referee, team, venue. If the entity is a coach, classify it as a player. The name extracted should match exactly as it appears in the question.

    Output the result strictly as a tuple in the format: (type, name). Do not include any additional explanations, notes, or formatting.

    For example:
    - Question: "How many goals did Lionel Messi score last season?"
      Output: ("player", "Lionel Messi")
    - Question: "Where is the Camp Nou stadium located?"
      Output: ("venue", "Camp Nou")
    - Question: "What was the decision made by referee Michael Oliver in the last match?"
      Output: ("referee", "Michael Oliver")
    - Question: "How did Manchester United perform in the last game?"
      Output: ("team", "Manchester United")

    However, if the entity type and entity name cannot be determined, please output as: ("unknown", "unknown")

    For example:
    - Question: "Explain the 4-4-2 formation."
      Output: ("unknown", "unknown")
    - Question: "Who is the player in this image?"
      Output: ("player", "unknown")
    """

    # Call the workflow function
    result = workflow(question, Instruction)

    # Crop the output to ensure it's a tuple
    if isinstance(result, str):
        # Remove any unwanted characters or formatting
        result = result.strip().replace('```', '').replace('json', '').strip()
        if result.startswith('(') and result.endswith(')'):
            try:
                # Safely evaluate the string to a tuple
                import ast
                result = ast.literal_eval(result)
            except (ValueError, SyntaxError):
                # Fallback in case of unexpected format
                result = ("unknown", "unknown")
        else:
            # Fallback if the output is not a tuple
            result = ("unknown", "unknown")
    elif not isinstance(result, tuple):
        # Fallback if the output is not a tuple
        result = ("unknown", "unknown")

    return result

import os
import json

def find_json_path(base_folder, entity_type, entity_name):
    """
    Searches for a JSON file in the appropriate subfolder that matches the entity name.
    If no exact match is found, uses the workflow function to determine the best match.

    Args:
        base_folder (str): The base folder containing subfolders (player, referee, team, venue).
        entity_type (str): The type of entity (player, referee, team, venue).
        entity_name (str): The name of the entity to search for.

    Returns:
        str: The absolute path of the matching JSON file.
    """
    # Map entity types to their corresponding subfolder and JSON key
    type_to_key = {
        "player": "FULL_NAME",
        "referee": "DETECTED_NAME",
        "team": "TEAM",
        "venue": "VENUE"
    }

    # Get the subfolder and key based on the entity type
    subfolder = os.path.join(base_folder, entity_type)
    key = type_to_key[entity_type]

    # List to store potential matches
    potential_matches = []

    # Traverse the subfolder and search for matching JSON files
    for root, _, files in os.walk(subfolder):
        for file in files:
            if file.endswith(".json"):
                file_path = os.path.join(root, file)
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if key in data and data[key] == entity_name:
                        return file_path  # Exact match found
                    elif key in data:
                        potential_matches.append((data[key], file_path))

    # If no exact match is found, use the workflow function to determine the best match
    if potential_matches:
        # Create a prompt with all potential names
        prompt = f"I want to find some information of {entity_type} {entity_name} from my soccer database. Now I have the following possible options\n\n"
        for name, _ in potential_matches:
            prompt += f"- {name}\n"

        prompt += "\nPlease help me determine the best match in the candidate list that is the most likely to be the entity I want. You can just reply me with the exact name of the cantidate you think is the best match without any other words.Please think it carefully and don't give me the wrong answer. I trust you. If really none of them are possible, return me with 'No Matching'\n"
        # Use the workflow function to determine the best match (retry up to 3 times)
        for _ in range(5):
            best_match = workflow(prompt, "You are an assistant of entity search in soccer database. Determine the best match for the given entity name.")
            best_match = (best_match or "").strip()
            if not best_match or best_match.lower() == "no matching":
                continue
            # Find the file path corresponding to the best match
            for name, file_path in potential_matches:
                if name == best_match:
                    return file_path

    # If no match is found at all, return None
    return "No matching file found."

def TEXTUAL_ENTITY_SEARCH(question, material=None, base_folder = os.path.join(PROJECT_PATH, "database/SoccerWiki/data")):
    entity_type, entity_name = extract_entity_info(question)

    # Fallback: name unknown → use raw question text as entity name
    if entity_name == "unknown":
        entity_name = question.strip()

    # Fallback: type unknown → search across all entity types
    if entity_type == "unknown":
        for et in ("player", "referee", "team", "venue"):
            result = find_json_path(base_folder, et, entity_name)
            if result != "No matching file found.":
                return f"The wiki information of this entity could be found in {result}."
        return "No matching file found."

    json_path = find_json_path(base_folder, entity_type, entity_name)
    return f"The wiki information of this entity could be found in {json_path}."
