import re
from pathlib import Path
from typing import List, Tuple


def open_logfile(filepath: Path) -> str:
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()


def detect_error_items(log_content: str) -> Tuple[int, List[int]]:
    pattern = r"Unexpected error processing item (\d+): 'NoneType' object has no attribute 'group'"
    item_numbers = [int(match) for match in re.findall(pattern, log_content)]
    return len(item_numbers), item_numbers


def extract_error_items(log_path: Path) -> Tuple[int, List[int]]:
    """지정된 로그 파일에서 에러 개수와 인덱스 목록을 추출한다."""
    log_content = open_logfile(log_path)
    return detect_error_items(log_content)


if __name__ == "__main__":
    log_path = Path(__file__).with_name("run.log")
    error_count, error_items = extract_error_items(log_path)
    print("Error count:", error_count)
    if error_items:
        print("Error items:", ", ".join(map(str, error_items)))
