import sys
from pathlib import Path
from project_path import PROJECT_PATH
if PROJECT_PATH not in sys.path:
    sys.path.append(PROJECT_PATH)
from pipeline.toolbox.unisoccer.inference.distribution import (
    preprocessor, classifier, commentary_model,
    unisoccer_to_device,
)
from pipeline.toolbox.utils.material_path import resolve_material_path
import einops

import torch
import torch.nn.functional as F


def _get_video_path(material) -> str:
    """Extract the video path string from material. Recursively unwraps nested lists (e.g., [['path']])."""
    item = material
    while isinstance(item, (list, tuple)):
        if not item:
            raise ValueError(f"Empty material: {material!r}")
        item = item[0]
    if not isinstance(item, str) or not item:
        raise ValueError(f"Could not extract path from material: {material!r}")
    resolved = resolve_material_path(item, primary_root=Path(PROJECT_PATH) / "challenge" / "test")
    if not Path(resolved).exists():
        # fallback: join directly with PROJECT_PATH
        fallback = Path(PROJECT_PATH) / item
        if fallback.exists():
            return str(fallback)
    return resolved


def classify_video(video_path, preprocessor=preprocessor, classifier=classifier):

    CLASS_NAMES = ["var", "end of half game", "clearance", "second yellow card", "injury", "ball possession", "throw in", "show added time", "shot off target", "start of half game", "substitution", "saved by goal-keeper", "red card", "lead to corner", "ball out of play", "off side", "goal", "penalty", "yellow card", "foul lead to penalty", "corner", "free kick", "foul with no card"]

    unisoccer_to_device()
    with torch.no_grad():
        video_tensor = preprocessor(video_path)
        logits = classifier.classify(video_tensor)

    probs = F.softmax(logits, dim=-1).squeeze().cpu()
    
    prob_dict = {name: float(prob) for name, prob in zip(CLASS_NAMES, probs)}
    
    sorted_dict = dict(sorted(prob_dict.items(), key=lambda x: x[1], reverse=True))
    
    return sorted_dict


def format_top_predictions(prob_dict, threshold=0.05):

    top_predictions = []
    for cls, prob in prob_dict.items():
        if prob >= threshold:
            top_predictions.append(f"{cls} - {prob*100:.1f}%")
    
    return ", ".join(top_predictions)

def ACTION_CLASSIFICATION(query, material):
    video_path = _get_video_path(material)
    video_result = classify_video(video_path)
    response = f"The classification probabilities of this soccer video clip is: {format_top_predictions(video_result)} (only above 5% mentioned)."
    return response


def commentary_video(video_path, preprocessor=preprocessor, commentary_model=commentary_model):
    unisoccer_to_device()
    with torch.no_grad():
        video_tensor = preprocessor(video_path)
        video_features = commentary_model.visual_encoder(video_tensor)
        batch_size = None
        time_length = None
        try:
            batch_size, time_length, _ = video_features.size()
        except:
            batch_size, time_length, _, _ = video_features.size()

        if len(video_features.size()) != 4:
            video_features = video_features.unsqueeze(-2)
        video_features = commentary_model.ln_vision(video_features)
        video_features = einops.rearrange(video_features, 'b t n f -> (b t) n f', b=batch_size, t=time_length)

        if commentary_model.need_temporal == "yes":
            position_ids = torch.arange(time_length, dtype=torch.long, device=video_features.device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
            frame_position_embeddings = commentary_model.video_frame_position_embedding(position_ids)
            frame_position_embeddings = frame_position_embeddings.unsqueeze(-2)
        frame_hidden_state = einops.rearrange(video_features, '(b t) n f -> b t n f', b=batch_size, t=time_length)

        if commentary_model.need_temporal == "yes":
            frame_hidden_state = frame_position_embeddings + frame_hidden_state

        frame_hidden_state = einops.rearrange(frame_hidden_state, 'b t q h -> b (t q) h', b=batch_size, t=time_length)
        frame_atts = torch.ones(frame_hidden_state.size()[:-1], dtype=torch.long).to(frame_hidden_state)
        video_query_tokens = commentary_model.video_query_tokens.expand(frame_hidden_state.shape[0], -1, -1).to(frame_hidden_state.device)

        video_query_output = commentary_model.video_Qformer.bert(
            query_embeds=video_query_tokens,
            encoder_hidden_states=frame_hidden_state,
            encoder_attention_mask=frame_atts,
            return_dict=True,
        )
        video_hidden = video_query_output.last_hidden_state

        inputs_llama = commentary_model.llama_proj(video_hidden)

        result = commentary_model.generate_text(inputs_llama)[0]

    return result
    


def COMMENTARY_GENERATION(query, material):
    video_path = _get_video_path(material)
    result = f"The according commentary to this video is: {commentary_video(video_path)}"
    return result
