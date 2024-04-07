from pathlib import Path
import json
import os

def get_project_root() -> Path:
    return Path(__file__)

def load_json(path: str):
    if os.path.exists(path) and path.endswith('.json'):
        with open(path, "r") as f:
            data = json.load(f)
            return data
    else:
        print("Invalid path.")

def listdir_nonhidden(path: str) -> list[str]:
    return [file for file in os.listdir(path) if not file.startswith('.')]
