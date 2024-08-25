import json
import os
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms


def get_project_root() -> Path:
    return Path(__file__).parent


def load_json(path: str):
    if os.path.exists(path) and path.endswith(".json"):
        with open(path, "r") as f:
            data = json.load(f)
            return data
    else:
        print("Invalid path.")


def listdir_nonhidden(path: str) -> list[str]:
    return [file for file in os.listdir(path) if not file.startswith(".")]


preprocessor = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255)),
    ]
)


def load_image(path):
    content = Image.open(path).convert("RGB")
    content = preprocessor(content)
    return content


def load_style(path, batch_size, DEVICE):
    style = Image.open(path)
    style = preprocessor(style)
    style = torch.stack([style] * batch_size, dim=0).to(DEVICE)
    return style
