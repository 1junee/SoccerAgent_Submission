import json, os
from tqdm import tqdm
import argparse
from functools import lru_cache
from typing import List, Dict
from openai import OpenAI

from pathlib import Path
from project_path import PROJECT_PATH
from pipeline.toolbox.utils.material_path import resolve_material_path

######################## Parameters ########################

from dotenv import load_dotenv
load_dotenv()
from llm_config import get_retrieve_model, get_chat_client_kwargs, get_responses_params, make_client

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


def generate_textual_RAG_prompt(question, textual_material):
    """
    Generates a prompt based on the question and textual material.

    Args:
        question (str): The user's question.
        textual_material (str): The textual material (JSON file path or raw text).

    Returns:
        str: The generated prompt.
    """
    # Check if textual_material is a JSON file path
    if isinstance(textual_material, str) and textual_material.endswith(".json"):
        try:
            # Read the JSON file and convert it to a formatted string
            with open(textual_material, "r", encoding="utf-8") as f:
                json_data = json.load(f)
                formatted_json = json.dumps(json_data, indent=4, ensure_ascii=False)
                textual_material = formatted_json
        except Exception as e:
            return f"Failed to read JSON file: {e}"

    # Generate the prompt
    prompt = f"""
    Question: {question}
    Contextual Material: {textual_material}
    Please answer the question based on the provided contextual material.
    """
    return prompt

def _resolve_textual_material(textual_material):
    if isinstance(textual_material, (list, tuple)) and textual_material:
        candidate = textual_material[0]
    else:
        candidate = textual_material
    if not candidate:
        return None
    candidate_path = Path(candidate)
    if candidate_path.is_absolute():
        return candidate_path
    return Path(resolve_material_path(str(candidate_path), primary_root=Path(PROJECT_PATH)))


def TEXTUAL_RETRIEVAL_AUGMENT(question, textual_material):
    """
    Retrieves and augments textual material to answer the question using the workflow function.

    Args:
        question (str): The user's question.
        textual_material (str): The textual material (JSON file path or raw text).

    Returns:
        str: The generated answer.
    """
    # Define the instruction for the agent
    Instruction = "You are an assistant that answers questions based on provided contextual material."
    file_path = _resolve_textual_material(textual_material)
    if not file_path or not file_path.exists():
        return "No contextual material available."

    # Generate the prompt
    prompt = generate_textual_RAG_prompt(question, str(file_path))
    answer = None
    try:
    # Call the workflow function to generate the answer
        answer = workflow(prompt, Instruction)
    except:
        answer = "Failed in LLM Generation."
    # Return the QA output
    return answer
    
