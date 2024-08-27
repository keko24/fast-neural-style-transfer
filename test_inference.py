import os

import torch

from model import TransformationNetwork
from utils import load_image, save_image

if __name__ == "__main__":
    model = TransformationNetwork()
    checkpoint = torch.load(os.path.join("results", "model.pt"))
    model.load_state_dict(checkpoint)
    model.eval()

    content = load_image(os.path.join("data", "test", "original.jpg"))

    content = content.unsqueeze(0)
    output = model(content)
    output = output.squeeze(0)

    save_image(output, os.path.join("results", "result.jpg"))
