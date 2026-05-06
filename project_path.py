import os
from pathlib import Path


def get_project_path() -> str:
    """Resolve the SoccerAgent project root path.

    Priority:
    1) SOCCERAGENT_HOME env var if set
    2) The directory containing this file (the SoccerAgent folder)
    """
    env = os.getenv("SOCCERAGENT_HOME")
    if env:
        return env
    return str(Path(__file__).resolve().parent)


PROJECT_PATH = get_project_path()

