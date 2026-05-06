"""
Grounding Count Tool
Three-stage VLM pipeline for counting soccer players by color/role.
  S1: VLM extracts team metadata (colors)
  S2: Rule-based query classification (target_colors, target_role, exclude_gk)
  S3: VLM detects all people → filter by S1+S2 criteria → count
"""

import os, sys, json, re, tempfile
from collections import Counter
import numpy as np
import torch
from PIL import Image
from pathlib import Path

from project_path import PROJECT_PATH
if PROJECT_PATH not in sys.path:
    sys.path.append(PROJECT_PATH)

os.environ.setdefault('VLM_ENABLE_THINKING', '1')
os.environ.setdefault('Q7_S1_PARSE_RETRIES', '1')
from pipeline.toolbox.utils.vision_backend import use_api_vision_backend

if use_api_vision_backend():
    from pipeline.toolbox.vlm import VLM_API as VLM
else:
    from pipeline.toolbox.vlm import VLM_QWEN as VLM


# ── Configuration ─────────────────────────────────────────────────────────────
MARGIN = 25
PITCH_PROMPT     = "soccer pitch . soccer field . grass field . playing field"
PITCH_BOX_THRESH = 0.25
PITCH_TXT_THRESH = 0.20

# ── S1: Team color metadata extraction prompts (3-way majority vote) ─────────
_S1_PROMPTS = {
    'gk_anchor': (
        'Task: infer team colors and attack directions from ONE soccer broadcast frame.\n\n'
        'Definitions:\n'
        '- left_team_color  = outfield shirt color of the team attacking toward the RIGHT.\n'
        '- right_team_color = outfield shirt color of the team attacking toward the LEFT.\n\n'
        'Reasoning order (strict):\n'
        '1) Detect visible goalkeeper(s) and which side they defend.\n'
        '2) A goalkeeper defends their own goal: GK on LEFT -> team attacks RIGHT (left_team).\n'
        '3) If no GK is visible, use penalty box / goal structure / defensive shape as fallback.\n'
        '4) Assign OUTFIELD team colors only (never GK kit, never referee kit).\n'
        '5) Identify referee shirt color and visible goalkeeper shirt colors.\n\n'
        'Output rules:\n'
        '- Use concise canonical color names when possible.\n'
        '- If referee not visible, set referee_color to null.\n'
        '- If no goalkeeper visible, set goalkeeper_colors to [].\n'
        '- Return ONLY one JSON object, no explanation.\n\n'
        '{"left_team_color":"...","right_team_color":"...","referee_color":"...","goalkeeper_colors":[...]}'
    ),
    'defend_side': (
        'Determine team colors by DEFENSIVE SIDE first, then convert to ATTACKING SIDE.\n\n'
        'Step A: Decide which team defends LEFT and which defends RIGHT.\n'
        'Use evidence priority:\n'
        '1) goalkeeper position,\n'
        '2) visible goal/penalty-box geometry,\n'
        '3) team defensive compactness and player orientation.\n\n'
        'Step B: Convert defensive side to attacking direction.\n'
        '- Team defending LEFT attacks RIGHT -> left_team_color.\n'
        '- Team defending RIGHT attacks LEFT -> right_team_color.\n\n'
        'Step C: Color assignment constraints.\n'
        '- Use outfield kit colors only for team colors.\n'
        '- Exclude goalkeeper kits and referee kits from team colors.\n'
        '- Also report referee_color and goalkeeper_colors.\n\n'
        'Output rules:\n'
        '- If uncertain, make the single best football-consistent guess.\n'
        '- Return ONLY one JSON object.\n\n'
        '{"left_team_color":"...","right_team_color":"...","referee_color":"...","goalkeeper_colors":[...]}'
    ),
    'narrate_extract': (
        'Do a short internal scene summary, then output structured extraction.\n\n'
        'Extraction targets:\n'
        '- left_team_color: outfield color of the team attacking RIGHT.\n'
        '- right_team_color: outfield color of the team attacking LEFT.\n'
        '- referee_color: referee shirt color (null if not visible).\n'
        '- goalkeeper_colors: list of visible goalkeeper shirt colors.\n\n'
        'Hard constraints:\n'
        '- Never use goalkeeper or referee colors as team colors.\n'
        '- Treat close synonyms as same family (white/cream/ivory, navy/blue, dark red/red).\n'
        '- Resolve conflicts by preferring goalkeeper+goal-side evidence over ball direction alone.\n\n'
        'Return ONLY one JSON object, no extra text:\n'
        '{"left_team_color":"...","right_team_color":"...","referee_color":"...","goalkeeper_colors":[...]}'
    ),
}

# ── S3: Full grounding prompt ────────────────────────────────────────────────
_PERSON_PROMPT_TEMPLATE = (
    "Locate every person on the pitch and along the sidelines. "
    "Even if they are distant or blurry, report their coordinates, "
    "role (player, goalkeeper, referee, assistant referee, or unknown), "
    "Return ONLY a JSON array:\n"
    '[{{"point_2d": [x, y], "role": "player", "shirt_color": "<shirt color>"}}]'
)


def _build_person_prompt(metadata: dict) -> str:
    voted_left  = metadata.get("left_team_color")  or "unknown"
    voted_right = metadata.get("right_team_color") or "unknown"
    gk_colors   = ", ".join(metadata.get("goalkeeper_colors") or []) or "unknown"
    ref_color   = metadata.get("referee_color") or "unknown"
    return _PERSON_PROMPT_TEMPLATE.format(
        voted_left=voted_left,
        voted_right=voted_right,
        gk_colors=gk_colors,
        referee_color=ref_color,
    )


# ── Color similarity comparison ───────────────────────────────────────────────
_COLOR_GROUPS = [
    {"red", "dark red", "crimson", "maroon", "burgundy", "scarlet", "wine", "brick red"},
    {"white", "off-white", "cream", "ivory", "light gray", "light grey", "pale"},
    {"blue", "dark blue", "navy", "navy blue", "royal blue", "cobalt", "deep blue", "sky blue"},
    {"black", "dark", "charcoal", "dark gray", "dark grey"},
    {"yellow", "gold", "golden", "amber", "lime yellow", "bright yellow"},
    {"green", "dark green", "lime green", "olive", "bright green"},
    {"orange", "orange-red", "tangerine"},
    {"purple", "violet", "lavender"},
    {"gray", "grey", "silver"},
    {"pink", "light pink"},
]

_UNCERTAIN_KEYWORDS = {"unknown", "unclear", "uncertain", "unidentified",
                       "not visible", "n/a", "none", ""}

# ── canonical color helpers (for S1 majority vote) ────────────────────────────
_CANONICAL_MAP: dict[str, str] = {}
for _grp in _COLOR_GROUPS:
    _canon = sorted(_grp)[0]
    for _item in _grp:
        _CANONICAL_MAP[_item] = _canon


def _split_color_parts(text: str) -> list[str]:
    return [p.strip() for p in re.split(r'\s*(?:and|&|with|/|,)\s*', text) if p and p.strip()]


def _canonical_color(value: str | None) -> str:
    if value is None:
        return '?'
    text = str(value).strip().lower()
    if not text or text in {'null', 'none', 'unknown', '?'}:
        return '?'
    text = text.replace('_', ' ').strip()
    parts = _split_color_parts(text)
    if len(parts) > 1:
        primary = parts[0]
        if primary in _CANONICAL_MAP:
            return _CANONICAL_MAP[primary]
        for token in sorted(_CANONICAL_MAP.keys(), key=len, reverse=True):
            if token in primary or primary in token:
                return _CANONICAL_MAP[token]
    if text in _CANONICAL_MAP:
        return _CANONICAL_MAP[text]
    for token in sorted(_CANONICAL_MAP.keys(), key=len, reverse=True):
        canon = _CANONICAL_MAP[token]
        if token in text or text in token:
            return canon
    return text


def _vote_field(values: list[str]) -> str:
    """Majority vote after excluding None/'?'. Alphabetical order takes priority on ties."""
    normalized = [_canonical_color(v) for v in values]
    valid = [v for v in normalized if v != '?']
    if not valid:
        return '?'
    counts = Counter(valid)
    best_count = max(counts.values())
    winners = {k for k, v in counts.items() if v == best_count}
    if len(winners) == 1:
        return next(iter(winners))
    return sorted(winners)[0]


def _vote_metadata(img_path: str) -> dict:
    """Run each of the 3 S1 prompts and aggregate metadata by per-field majority vote."""
    lefts: list[str] = []
    rights: list[str] = []
    refs: list[str | None] = []
    gk_lists: list[list[str]] = []
    s1_parse_retries = max(0, int(os.getenv('Q7_S1_PARSE_RETRIES', '1') or '1'))

    for prompt_name, prompt_text in _S1_PROMPTS.items():
        try:
            meta = {}
            last_err = None
            for attempt in range(s1_parse_retries + 1):
                try:
                    raw = VLM(prompt_text, [img_path]).strip()
                    meta = parse_metadata(raw)
                    if _is_valid_s1_meta(meta):
                        break
                    last_err = ValueError('S1 parse_error: invalid or empty metadata JSON')
                except Exception as e:
                    last_err = e
                if attempt < s1_parse_retries:
                    print(f'  [S1:{prompt_name}] parse failed -> retry {attempt + 1}/{s1_parse_retries}', flush=True)
            else:
                raise last_err or RuntimeError('S1 parse_error')

            lefts.append(meta.get('left_team_color') or '?')
            rights.append(meta.get('right_team_color') or '?')
            refs.append(meta.get('referee_color'))
            gk_lists.append([c for c in (meta.get('goalkeeper_colors') or []) if c])
        except Exception:
            lefts.append('?')
            rights.append('?')
            refs.append(None)
            gk_lists.append([])

    voted_left = _vote_field(lefts)
    voted_right = _vote_field(rights)

    ref_candidates = [r for r in refs if r]
    voted_ref_raw = _vote_field(ref_candidates) if ref_candidates else '?'
    voted_ref = None if voted_ref_raw == '?' else voted_ref_raw

    # goalkeeper_colors: convert to canonical form from the full list, then sort by frequency
    all_gks = [_canonical_color(gc) for gks in gk_lists for gc in gks]
    gk_counts = Counter(c for c in all_gks if c != '?')
    voted_gks = sorted(gk_counts, key=lambda x: -gk_counts[x])

    return {
        'left_team_color': voted_left,
        'right_team_color': voted_right,
        'referee_color': voted_ref,
        'goalkeeper_colors': voted_gks,
    }


def _color_match(a: str, b: str) -> bool:
    a, b = _canonical_color(a), _canonical_color(b)
    if '?' in {a, b}:
        return False
    if a == b:
        return True
    for grp in _COLOR_GROUPS:
        if a in grp and b in grp:
            return True
    return False


def _color_candidates(c: str) -> set[str]:
    text = (c or '').lower().strip().replace('_', ' ')
    if not text:
        return set()
    cands: set[str] = set()
    for part in [text, *_split_color_parts(text)]:
        part = part.strip()
        if not part:
            continue
        cands.add(part)
        cc = _canonical_color(part)
        if cc != '?':
            cands.add(cc)
    return cands


def _color_match_loose(a: str, b: str) -> bool:
    a_text, b_text = (a or '').lower().strip(), (b or '').lower().strip()
    if not a_text or not b_text:
        return False
    if a_text == b_text:
        return True
    if _color_match(a_text, b_text):
        return True

    a_cands = _color_candidates(a_text)
    b_cands = _color_candidates(b_text)
    if not a_cands or not b_cands:
        return False
    if a_cands & b_cands:
        return True
    for ac in a_cands:
        for bc in b_cands:
            if _color_match(ac, bc):
                return True
    return (a_text in b_text) or (b_text in a_text)


def _is_known_color(shirt_color: str, known_colors: set) -> bool:
    for kc in known_colors:
        if _color_match_loose(shirt_color, kc):
            return True
    return False


# ── S2: Rule-based query classification ───────────────────────────────────────
def classify_query(q: str, metadata: dict) -> dict:
    q_lower = q.lower()
    left  = metadata.get("left_team_color",  "?")
    right = metadata.get("right_team_color", "?")
    ref   = metadata.get("referee_color")
    gk    = metadata.get("goalkeeper_colors") or []

    if re.search(r"\breferee|\blinesman|linesmen|assistant.?referee", q_lower):
        return {"target_colors": [ref] if ref else [],
                "target_role": "referee",
                "exclude_gk": False}

    exclude_gk = bool(re.search(
        r'goalkeeper.{0,20}exclu|exclu.{0,20}goalkeeper|'
        r'not.{0,10}includ.{0,10}goalkeeper|GK.{0,10}exclu|excluding.{0,10}GK',
        q, re.I
    ))

    if re.search(r"right.{0,8}left|attacking left", q_lower):
        return {"target_colors": [right],
                "target_role": "player",
                "exclude_gk": exclude_gk}

    if re.search(r"left.{0,8}right|rightward|attacking right", q_lower):
        return {"target_colors": [left],
                "target_role": "player",
                "exclude_gk": exclude_gk}

    return {"target_colors": [left, right],
            "target_role": "player",
            "exclude_gk": exclude_gk}


# ── JSON parsing helpers ──────────────────────────────────────────────────────
def _strip_think_and_fence(text: str) -> str:
    text = text.strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S | re.I).strip()
    m = re.search(r"`{3}(?:json)?\s*(.*?)`{3}", text, flags=re.S | re.I)
    return m.group(1).strip() if m else text


def parse_metadata(raw: str) -> dict:
    """Extract metadata from S1 output."""
    cleaned = _strip_think_and_fence(raw)
    for c in [cleaned, cleaned[cleaned.find('{'):cleaned.rfind('}')+1]]:
        try:
            obj = json.loads(c)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
    return {}


def _is_valid_s1_meta(meta: dict) -> bool:
    if not isinstance(meta, dict) or not meta:
        return False
    return any(
        k in meta
        for k in ('left_team_color', 'right_team_color', 'referee_color', 'goalkeeper_colors')
    )


def _validate_items(raw: list) -> list:
    out = []
    for it in raw:
        if not isinstance(it, dict):
            continue
        pt = it.get("point_2d", [])
        if not isinstance(pt, list) or len(pt) != 2:
            continue
        try:
            x, y = float(pt[0]), float(pt[1])
        except (TypeError, ValueError):
            continue
        out.append({
            "point_2d":    [x, y],
            "role":        it.get("role", "unknown"),
            "shirt_color": it.get("shirt_color", "unknown"),
        })
    return out


def extract_json_payload(text: str) -> list:
    cleaned = _strip_think_and_fence(text)
    l1, r1 = cleaned.find("["), cleaned.rfind("]")
    if l1 != -1 and r1 != -1 and l1 < r1:
        try:
            obj = json.loads(cleaned[l1:r1+1])
            if isinstance(obj, list):
                result = _validate_items(obj)
                if result:
                    return result
        except:
            pass
    items = []
    for line in cleaned.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
            if isinstance(obj, dict) and "point_2d" in obj:
                items.append(obj)
        except:
            pass
    if items:
        return _validate_items(items)
    items = []
    for m in re.finditer(r"\{[^{}]+\}", cleaned):
        try:
            obj = json.loads(m.group())
            if isinstance(obj, dict) and "point_2d" in obj:
                items.append(obj)
        except:
            pass
    if items:
        return _validate_items(items)
    lower = cleaned.lower()
    if "[]" in cleaned or not cleaned.strip() or "none" in lower or lower.startswith("no "):
        return []
    return []


# ── Filtering logic ───────────────────────────────────────────────────────────
def filter_detections(all_persons: list, s2_result: dict,
                      metadata: dict) -> tuple:
    known_colors = set()
    for c in [metadata.get("left_team_color", ""),
              metadata.get("right_team_color", "")]:
        if c:
            known_colors.add(c.lower().strip())
    if metadata.get("referee_color"):
        known_colors.add(metadata["referee_color"].lower().strip())
    for gk_c in (metadata.get("goalkeeper_colors") or []):
        if gk_c:
            known_colors.add(gk_c.lower().strip())

    target_colors = s2_result["target_colors"]
    target_role   = s2_result["target_role"]
    exclude_gk    = s2_result.get("exclude_gk", False)
    gk_colors     = [c.lower().strip() for c in (metadata.get("goalkeeper_colors") or []) if c]

    filtered, reasons = [], []
    unknown_color_persons = []  # players dropped due to known_colors mismatch

    for p in all_persons:
        color = (p.get("shirt_color") or "").lower().strip()
        role  = (p.get("role") or "").lower().strip()

        if color in _UNCERTAIN_KEYWORDS:
            reasons.append(f"uncertain color '{color}'")
            continue

        if known_colors and not _is_known_color(color, known_colors):
            # exempt from known_colors gate if it's a referee query and the role is also referee
            if target_role == "referee" and role in ("referee", "assistant referee"):
                pass  # proceed to role check
            else:
                if role not in ('referee', 'assistant referee', 'unknown', 'goalkeeper') and color not in _UNCERTAIN_KEYWORDS:
                    unknown_color_persons.append(p)
                reasons.append(f"unknown color '{color}'")
                continue

        if target_role == "referee":
            if role not in ("referee", "assistant referee"):
                reasons.append(f"role '{role}' != referee")
                continue
            if target_colors:
                if not any(_color_match_loose(color, tc) for tc in target_colors):
                    reasons.append(f"ref color '{color}' mismatch")
                    continue
        else:
            if role in ("referee", "assistant referee", "unknown"):
                reasons.append(f"excluded role '{role}'")
                continue
            if exclude_gk and gk_colors:
                if any(_color_match_loose(color, gc) for gc in gk_colors):
                    if role == "goalkeeper" or not any(_color_match_loose(color, tc) for tc in target_colors):
                        reasons.append(f"excluded GK (color='{color}')")
                        continue
            if target_colors:
                if not any(_color_match_loose(color, tc) for tc in target_colors):
                    reasons.append(f"color '{color}' not in {target_colors}")
                    continue

        filtered.append(p)
        reasons.append("")

    # ── Second-team fallback (limited): only activated when S1 target color was not captured ──
    # Disabled when the target color is valid (e.g., cream/red/white) to prevent cross-team contamination.
    # Recovery is only allowed when the color is '?/unknown/none'-like.
    target_is_unknown = (
        len(target_colors) == 1
        and _canonical_color(target_colors[0]) in {'?', 'unknown', 'none', 'null', ''}
    )
    if target_role != "referee" and target_is_unknown and unknown_color_persons:
        recovered = []
        for p in unknown_color_persons:
            color = (p.get("shirt_color") or "").lower().strip()
            role = (p.get("role") or "").lower().strip()
            if role == "goalkeeper":
                continue
            if exclude_gk and gk_colors and any(_color_match_loose(color, gc) for gc in gk_colors):
                continue
            recovered.append(p)
        if recovered:
            recovered_colors = list({(p.get("shirt_color") or "").lower().strip() for p in recovered})
            print(f"  [FALLBACK] S1 miss recovery: {len(recovered)} persons added (colors={recovered_colors})", flush=True)
            filtered.extend(recovered)
            for _ in recovered:
                reasons.append("fallback:second_team")

    return filtered, reasons


# ── LLM-based color grouping (S3 API path) ───────────────────────────────────
_COLOR_GROUP_PROMPT = """\
I detected the following shirt colors in a soccer broadcast frame, with the number of \
people wearing each color shown in parentheses:

Detected colors (count): {colors}

S1 color hints (from prior analysis — use these to resolve ambiguity):
  - Left team (attacking right) jersey color: "{voted_left}"
  - Right team (attacking left) jersey color: "{voted_right}"
A detected color belongs to the hint team ONLY if it is in the same hue family: \
white/cream/ivory/off-white are equivalent; red/crimson/brick-red/dark-red are equivalent; \
blue/navy/dark-blue are equivalent; etc. Do NOT group colors across hue families \
(orange ≠ cream, yellow ≠ white, orange ≠ red).

Please group every detected color into exactly these four categories:
- team_a: one team's outfield players
- team_b: the other team's outfield players
- referee: referees / assistant referees (often a distinctive non-team color)
- unknown: sideline staff, coaches, ball boys, or colors that don't belong to either team

RULES:
1. Every detected color must appear in exactly one category.
2. Two colors belong to the same team if they are shades/synonyms of the same hue \
(e.g. "cream" and "white" are the same team; "red" and "dark red" are the same team).
3. Prioritize the S1 hints over count-based guessing. If a hint says "white" and you see \
"cream" or "light" in detections, assign it to the hint team even if count is low.
4. Referees typically wear a color clearly different from both teams (yellow, green, orange, etc.).
5. Goalkeeper kits go into whichever team category fits best.
6. If only one team color is clearly present, still return valid JSON and leave the weaker team empty if needed.

Return ONLY a valid JSON object, no extra text:
{{"team_a": ["color1", ...], "team_b": ["color2", ...], "referee": [...], "unknown": [...]}}
"""


def _group_shirt_colors(color_counts: dict, voted_left: str = "", voted_right: str = "") -> dict:
    """Group shirt colors into team_a/team_b/referee/unknown using the configured VLM backend."""
    empty = {"team_a": [], "team_b": [], "referee": [], "unknown": []}
    if not color_counts:
        return empty

    colors_str = ", ".join(
        f"{c} ({n})"
        for c, n in sorted(color_counts.items(), key=lambda x: -x[1])
        if c
    )
    if not colors_str:
        return empty

    prompt = _COLOR_GROUP_PROMPT.format(
        colors=colors_str,
        voted_left=voted_left or "unknown",
        voted_right=voted_right or "unknown",
    )
    try:
        raw = VLM(prompt, None).strip()
        cleaned = _strip_think_and_fence(raw)
        l, r = cleaned.find("{"), cleaned.rfind("}")
        if l != -1 and r != -1:
            obj = json.loads(cleaned[l:r+1])
            if isinstance(obj, dict):
                result = {}
                for k in ("team_a", "team_b", "referee", "unknown"):
                    result[k] = [
                        str(c).lower().strip()
                        for c in obj.get(k, [])
                        if c and str(c).strip()
                    ]
                print(f"  [GROUP] raw grouping: {result}", flush=True)
                return result
    except Exception as exc:
        print(f"  [GROUP] color grouping failed: {exc}", flush=True)
    return empty


def filter_detections_grouped(
    all_persons: list,
    s2_result: dict,
    grouped: dict,
    voted_left: str,
    voted_right: str,
    metadata: dict,
) -> tuple:
    """Filtering using color grouping results. Falls back to the original filter_detections for referee queries or empty groups."""
    target_role = s2_result["target_role"]

    if target_role == "referee":
        return filter_detections(all_persons, s2_result, metadata)

    team_a_colors = grouped.get("team_a", [])
    team_b_colors = grouped.get("team_b", [])
    ref_colors    = grouped.get("referee", [])

    if not team_a_colors and not team_b_colors:
        print("  [GROUP] grouping empty → fallback to original filter", flush=True)
        return filter_detections(all_persons, s2_result, metadata)

    def find_group(voted_color: str):
        if not voted_color or voted_color == "?":
            return None
        for c in team_a_colors:
            if _color_match(voted_color, c):
                return "team_a"
        for c in team_b_colors:
            if _color_match(voted_color, c):
                return "team_b"
        return None

    left_group  = find_group(voted_left)
    right_group = find_group(voted_right)

    if left_group and right_group and left_group == right_group:
        print(f"  [GROUP] left/right both map to {left_group} → reset", flush=True)
        left_group = right_group = None

    if left_group and not right_group:
        right_group = "team_b" if left_group == "team_a" else "team_a"
        print(f"  [GROUP] right_group inferred as {right_group}", flush=True)
    elif right_group and not left_group:
        left_group = "team_b" if right_group == "team_a" else "team_a"
        print(f"  [GROUP] left_group inferred as {left_group}", flush=True)

    group_map = {"team_a": team_a_colors, "team_b": team_b_colors}

    orig_target = s2_result["target_colors"]
    expanded_target: list = []
    for tc in orig_target:
        tc_canon = _canonical_color(tc)
        if _color_match(tc_canon, voted_left) and left_group:
            expanded_target.extend(group_map.get(left_group, [tc]))
        elif _color_match(tc_canon, voted_right) and right_group:
            expanded_target.extend(group_map.get(right_group, [tc]))
        else:
            expanded_target.append(tc)
    seen: set = set()
    unique_target: list = []
    for c in expanded_target:
        if c not in seen:
            seen.add(c); unique_target.append(c)
    expanded_target = unique_target or orig_target

    print(
        f"  [GROUP] left_group={left_group} right_group={right_group} "
        f"orig_target={orig_target} → expanded={expanded_target}",
        flush=True,
    )

    known_colors: set = set()
    for c in [metadata.get("left_team_color", ""), metadata.get("right_team_color", "")]:
        if c:
            known_colors.add(c.lower().strip())
    if metadata.get("referee_color"):
        known_colors.add(metadata["referee_color"].lower().strip())
    for gc in (metadata.get("goalkeeper_colors") or []):
        if gc:
            known_colors.add(gc.lower().strip())
    for c in team_a_colors + team_b_colors + ref_colors:
        if c:
            known_colors.add(c)

    exclude_gk = s2_result.get("exclude_gk", False)
    gk_colors  = [c.lower().strip() for c in (metadata.get("goalkeeper_colors") or []) if c]

    filtered, reasons = [], []
    for p in all_persons:
        color = (p.get("shirt_color") or "").lower().strip()
        role  = (p.get("role") or "").lower().strip()

        if color in _UNCERTAIN_KEYWORDS:
            reasons.append(f"uncertain color '{color}'"); continue

        in_known = (not known_colors) or _is_known_color(color, known_colors)
        if not in_known:
            reasons.append(f"unknown color '{color}'"); continue

        if role in ("referee", "assistant referee", "unknown"):
            reasons.append(f"excluded role '{role}'"); continue

        if exclude_gk and gk_colors:
            if any(_color_match(color, gc) for gc in gk_colors):
                if role == "goalkeeper" or not any(_color_match(color, tc) for tc in expanded_target):
                    reasons.append(f"excluded GK (color='{color}')"); continue

        if expanded_target and not any(_color_match(color, tc) for tc in expanded_target):
            reasons.append(f"color '{color}' not in expanded {expanded_target}"); continue

        filtered.append(p)
        reasons.append("")

    return filtered, reasons


# ── GroundingDINO crop (lazy-load, GPU-resident) ──────────────────────────────
_gdino_model = None
_GDINO_DEVICE = os.getenv("GDINO_DEVICE", "cuda:0")


def _load_gdino():
    """Load GroundingDINO once and keep it on the configured GPU."""
    global _gdino_model
    if _gdino_model is not None:
        return _gdino_model

    GDINO_ROOT = Path(PROJECT_PATH) / "pipeline/toolbox/utils/GroundingDINO"
    if str(GDINO_ROOT) not in sys.path:
        sys.path.insert(0, str(GDINO_ROOT))

    from groundingdino.util.inference import load_model as _gdino_load_model

    config = str(GDINO_ROOT / "groundingdino/config/GroundingDINO_SwinB_cfg.py")
    ckpt   = str(GDINO_ROOT / "groundingdino/config/groundingdino_swinb_cogcoor.pth")

    _gdino_model = _gdino_load_model(config, ckpt, device=_GDINO_DEVICE)
    return _gdino_model


def _gdino_to_device():
    """GroundingDINO stays resident on the target GPU."""
    return _load_gdino()


def get_pitch_crop(img_path: str, margin: int = MARGIN):
    from torchvision.ops import box_convert as _box_convert

    GDINO_ROOT = Path(PROJECT_PATH) / "pipeline/toolbox/utils/GroundingDINO"
    if str(GDINO_ROOT) not in sys.path:
        sys.path.insert(0, str(GDINO_ROOT))
    from groundingdino.util.inference import predict as _gdino_predict
    from groundingdino.util.inference import load_image as _gdino_load_image

    # Model is already on _GDINO_DEVICE (moved by the GROUNDING_COUNT wrapper).
    model = _load_gdino()
    device = _GDINO_DEVICE

    img_pil = Image.open(img_path).convert("RGB")
    W, H    = img_pil.size
    _, img_tensor = _gdino_load_image(img_path)
    boxes_cx, logits, _ = _gdino_predict(
        model=model, image=img_tensor, caption=PITCH_PROMPT,
        box_threshold=PITCH_BOX_THRESH, text_threshold=PITCH_TXT_THRESH,
        device=device,
    )
    if len(boxes_cx) == 0:
        return img_pil, (0, 0, W, H), False

    boxes_px = _box_convert(
        boxes_cx * torch.Tensor([W, H, W, H]),
        in_fmt="cxcywh", out_fmt="xyxy"
    ).numpy().astype(int)
    areas = [(b[2]-b[0])*(b[3]-b[1]) for b in boxes_px]
    b     = boxes_px[int(np.argmax(areas))]
    x1 = int(np.clip(b[0] - margin, 0, W))
    y1 = int(np.clip(b[1] - margin, 0, H))
    x2 = int(np.clip(b[2] + margin, 0, W))
    y2 = int(np.clip(b[3] + margin, 0, H))
    return img_pil.crop((x1, y1, x2, y2)), (x1, y1, x2, y2), True


def get_stage_image(img_path: str, use_crop: bool, margin: int = MARGIN):
    orig_pil = Image.open(img_path).convert("RGB")
    W, H = orig_pil.size
    if use_crop:
        crop_pil, crop_bbox, crop_ok = get_pitch_crop(img_path, margin=margin)
        return orig_pil, crop_pil, crop_bbox, crop_ok
    else:
        return orig_pil, orig_pil.copy(), (0, 0, W, H), False


# ── Main function ─────────────────────────────────────────────────────────────
def GROUNDING_COUNT(query, material=None):
    """
    Three-stage VLM pipeline for counting soccer players/referees.
    Input: query (text question), material (list with image path)
    Output: text string with detected count
    """
    img_path = material[0] if material else None
    if not img_path or not os.path.exists(img_path):
        return "Error: No valid image material provided."

    # Temporarily override VLM_REASONING_EFFORT if GROUNDING_COUNT_REASONING_EFFORT is set
    _gc_effort = os.getenv("GROUNDING_COUNT_REASONING_EFFORT")
    _prev_effort = os.environ.get("VLM_REASONING_EFFORT") if _gc_effort else None
    if _gc_effort:
        os.environ["VLM_REASONING_EFFORT"] = _gc_effort

    try:
        # S1: Extract color metadata with Qwen (3-prompt majority vote, original image)
        metadata = _vote_metadata(img_path)

        # S2: Rule-based query classification
        s2_result = classify_query(query, metadata)
        voted_left  = metadata.get("left_team_color",  "?")
        voted_right = metadata.get("right_team_color", "?")

        # S3: Prompt with S1 color hints → VLM full-person grounding → direct S1 color filtering
        person_prompt = _build_person_prompt(metadata)
        raw_det = VLM(person_prompt, [img_path]).strip()
        all_persons = extract_json_payload(raw_det)

        kept, _ = filter_detections(all_persons, s2_result, metadata)
        count = len(kept)

        return f"The detected count of matching players is: {count}"
    finally:
        if _gc_effort:
            if _prev_effort is None:
                os.environ.pop("VLM_REASONING_EFFORT", None)
            else:
                os.environ["VLM_REASONING_EFFORT"] = _prev_effort
