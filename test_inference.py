import os

import torch
from PIL import Image
from torchvision.transforms.functional import to_pil_image

from model import TransformationNetwork
from preprocess import Preprocess

if __name__ == "__main__":
    model = TransformationNetwork()
    checkpoint = torch.load(os.path.join("results", "model.pt"))
    model.load_state_dict(checkpoint)
    model.eval()
    content = Image.open(os.path.join("data", "test", "original.jpg")).convert("RGB")
    transform = Preprocess()
    content = transform(content)
    content = content.unsqueeze(0)
    output = model(content)
    output = output.squeeze(0)
    output = to_pil_image(output)
    output.save(os.path.join("results", "result.jpg"))
