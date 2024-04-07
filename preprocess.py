import torch
from torch import nn
from torchvision import transforms

class Preprocess(nn.Module):
    def __init__(self) -> None:
        super(Preprocess, self).__init__()
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        self.preprocess = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.preprocess(inputs)
