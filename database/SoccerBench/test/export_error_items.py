import json  # JSON 데이터를 다루기 위한 표준 라이브러리를 불러온다.
import sys
from pathlib import Path  # 경로를 객체 형태로 다루기 위해 Path 클래스를 사용한다.
from typing import Iterable, List  # 타입 힌트를 위해 Iterable과 List를 가져온다.

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from project_path import PROJECT_PATH

from detect_error_num import extract_error_items  # 로그에서 에러 인덱스를 추출하는 함수.

SOURCE_PATH = Path(f"{PROJECT_PATH}/database/SoccerBench/subqa/random100/q8.json")  # 원본 문제 세트의 절대 경로.
DEST_PATH = Path(f"{PROJECT_PATH}/database/SoccerBench/test/test_qa.json")  # 추출 결과를 저장할 경로.
LOG_PATH = Path(f"{PROJECT_PATH}/database/SoccerBench/run_outputs/scAgent/subqa/random100_1/q8/run.log")  # 에러가 기록된 로그 위치.
_, ERROR_ITEMS = extract_error_items(LOG_PATH)  # 로그를 분석해 에러 인덱스 목록을 계산한다.


def load_questions(path: Path) -> List[dict]:
    """지정된 경로에서 문제 데이터를 읽어 리스트로 반환한다."""
    with path.open("r", encoding="utf-8") as fh:  # 파일을 읽기 모드로 연다.
        return json.load(fh)  # JSON 내용을 파싱해 파이썬 객체로 반환한다.


def select_questions(data: List[dict], indices: Iterable[int]) -> List[dict]:
    """주어진 인덱스에 해당하는 문제들만 추려서 새 리스트로 만든다."""
    return [data[i] for i in indices]  # 리스트 내포로 원하는 항목만 골라낸다.


def save_questions(path: Path, items: List[dict]) -> None:
    """선택된 문제 리스트를 지정된 경로에 저장한다."""
    with path.open("w", encoding="utf-8") as fh:  # 파일을 쓰기 모드로 연다.
        json.dump(items, fh, ensure_ascii=False, indent=2)  # 보기 좋게 들여쓰기와 함께 JSON으로 기록한다.


def main() -> None:
    """모든 단계를 순서대로 실행하는 메인 함수."""
    questions = load_questions(SOURCE_PATH)  # 원본 문제 집합을 읽어온다.
    selected = select_questions(questions, ERROR_ITEMS)  # 에러 항목만 걸러낸다.
    save_questions(DEST_PATH, selected)  # 결과를 출력 파일에 저장한다.
    print(f"Saved {len(selected)} questions to {DEST_PATH}")  # 처리 결과를 콘솔에 알린다.


if __name__ == "__main__":  # 스크립트를 직접 실행했는지 확인한다.
    main()  # 메인 함수를 호출해 작업을 수행한다.
