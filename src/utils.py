from pathlib import Path


def get_project_root_path() -> Path:
    return str(Path(__file__).parent.parent)
