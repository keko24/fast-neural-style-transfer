import os
import json

from PIL import Image
from pathlib import Path
from torchvision import transforms

def get_project_root() -> Path:
    return Path(__file__).parent

def load_json(path: str):
    if os.path.exists(path) and path.endswith('.json'):
        with open(path, "r") as f:
            data = json.load(f)
            return data
    else:
        print("Invalid path.")

def listdir_nonhidden(path: str) -> list[str]:
    return [file for file in os.listdir(path) if not file.startswith('.')]

preprocessor = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
    ])

def load_image(path):
    content = Image.open(path)
    content = preprocessor(content)
    content = content.unsqueeze(0)
    return content

