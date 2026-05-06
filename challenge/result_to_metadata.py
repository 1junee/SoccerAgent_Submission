from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=4)

def extract_metadata_from_result(result: list) -> dict:
    """
    Extract metadata from a result list.
    """
    metadata = []
    for item in result:
        entry = {
            "id": item.get("id", ""),
            "Answer": item.get("answer", "")
        }
        metadata.append(entry)
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert result.json to metadata.json")
    parser.add_argument("input", type=Path, help="Path to the input result.json file")
    parser.add_argument("output", type=Path, help="Path to the output metadata.json file")
    args = parser.parse_args()

    result_data = load_json(args.input)
    metadata = extract_metadata_from_result(result_data)

    write_json(args.output, metadata)
