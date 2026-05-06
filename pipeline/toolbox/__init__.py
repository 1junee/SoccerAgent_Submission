import importlib

_LAZY_IMPORTS = {
    "GAME_SEARCH": (".game_search", "GAME_SEARCH"),
    "TEXTUAL_ENTITY_SEARCH": (".textual_entity_search", "TEXTUAL_ENTITY_SEARCH"),
    "TEXTUAL_RETRIEVAL_AUGMENT": (".textual_retrieval_augment", "TEXTUAL_RETRIEVAL_AUGMENT"),
    "MATCH_HISTORY_RETRIEVAL": (".game_retrieval", "MATCH_HISTORY_RETRIEVAL"),
    "GAME_INFO_RETRIEVAL": (".game_retrieval", "GAME_INFO_RETRIEVAL"),
    "ACTION_CLASSIFICATION": (".unisoccer_com_cls", "ACTION_CLASSIFICATION"),
    "COMMENTARY_GENERATION": (".unisoccer_com_cls", "COMMENTARY_GENERATION"),
    "VLM": (".vlm", "VLM"),
    "JERSEY_COLOR_VLM": (".jersey_color_relevant", "JERSEY_COLOR_VLM"),
    "FRAME_SELECTION": (".frame_selection", "FRAME_SELECTION"),
    "SCORE_TIME_DETECTION": (".score_time_det", "SCORE_TIME_DETECTION"),
    "FOUL_RECOGNITION": (".foul_recognition", "FOUL_RECOGNITION"),
    "SHOT_CHANGE": (".shot_change", "SHOT_CHANGE"),
    "FACE_RECOGNITION": (".face_rec", "FACE_RECOGNITION"),
    "JERSEY_NUMBER_RECOGNITION": (".jn_rec", "JERSEY_NUMBER_RECOGNITION"),
    "CAMERA_DETECTION": (".camera_detection", "CAMERA_DETECTION"),
    "SEGMENT": (".segment", "SEGMENT"),
    "REPLAY_GROUNDING": (".replay_grounding", "REPLAY_GROUNDING"),
    "GROUNDING_COUNT": (".grounding_count", "GROUNDING_COUNT"),
}

def __getattr__(name):
    if name in _LAZY_IMPORTS:
        module_name, attr_name = _LAZY_IMPORTS[name]
        module = importlib.import_module(module_name, __name__)
        value = getattr(module, attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__} has no attribute {name}")

__all__ = list(_LAZY_IMPORTS.keys())
