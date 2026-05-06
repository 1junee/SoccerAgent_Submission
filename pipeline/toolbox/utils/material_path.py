import re
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Optional

from project_path import PROJECT_PATH


def _normalized_part(part: str) -> str:
    return re.sub(r"[-_]+", "", part.lower())


def _path_score(candidate: Path, rel_parts: list[str]) -> int:
    cand_parts = list(candidate.parts)
    tail = cand_parts[-len(rel_parts):] if len(cand_parts) >= len(rel_parts) else cand_parts
    score = 0
    for a, b in zip(tail, rel_parts[-len(tail):]):
        if a == b:
            score += 4
        elif _normalized_part(a) == _normalized_part(b):
            score += 2
    return score


def _candidate_part_variants(rel_parts: list[str]) -> list[list[str]]:
    variants = [rel_parts]
    rewritten = [
        re.sub(r"(?<=[A-Za-z])-(?=20\d{2}-20\d{2}\b)", "_", part)
        for part in rel_parts
    ]
    if rewritten != rel_parts:
        variants.append(rewritten)
    return variants


@lru_cache(maxsize=4096)
def _resolve_material_path_cached(item: str, primary_root_str: str, secondary_roots: tuple[str, ...]) -> str:
    rel = Path(item)
    if rel.is_absolute():
        return str(rel)

    primary_root = Path(primary_root_str) if primary_root_str else None
    other_roots = [Path(root) for root in secondary_roots if root]

    rel_parts = list(rel.parts)
    variants = _candidate_part_variants(rel_parts)

    direct_candidates: list[Path] = []
    if primary_root is not None:
        if item.startswith("materials/"):
            stripped = item.replace("\\", "/").split("/", 1)[1]
            direct_candidates.append(primary_root / stripped)
        direct_candidates.extend(primary_root.joinpath(*parts) for parts in variants)

    project_root = Path(PROJECT_PATH)
    direct_candidates.extend(project_root.joinpath(*parts) for parts in variants)
    for root in other_roots:
        direct_candidates.extend(root.joinpath(*parts) for parts in variants)

    seen = set()
    ordered_candidates = []
    for cand in direct_candidates:
        s = str(cand)
        if s not in seen:
            seen.add(s)
            ordered_candidates.append(cand)

    for cand in ordered_candidates:
        if cand.exists():
            return str(cand)

    filename = rel.name
    search_roots: list[Path] = []
    if primary_root is not None:
        search_roots.append(primary_root)
    search_roots.append(project_root)
    search_roots.extend(other_roots)

    hits: list[Path] = []
    for root in search_roots:
        if not root.exists():
            continue
        try:
            hits.extend(root.rglob(filename))
        except Exception:
            continue

    if hits:
        hits.sort(key=lambda p: (_path_score(p, rel_parts), len(p.parts)), reverse=True)
        best = hits[0]
        if _path_score(best, rel_parts) > 0:
            return str(best)

    return str(ordered_candidates[0]) if ordered_candidates else str(rel)


def resolve_material_path(
    item: str,
    primary_root: Optional[Path] = None,
    secondary_roots: Optional[Iterable[Path]] = None,
) -> str:
    if not item:
        return item
    primary_root_str = str(primary_root) if primary_root else ""
    secondary_root_strs = tuple(str(root) for root in (secondary_roots or []) if root)
    return _resolve_material_path_cached(item, primary_root_str, secondary_root_strs)

