import os
from PIL import Image

import torch
from torch.utils.data import Dataset

from utils import (
    load_image, 
    listdir_nonhidden
)

class ImageDataset(Dataset):
    def __init__(self, img_dir: str) -> None:
        super().__init__()
        self.img_dir = img_dir
        self.img_list = listdir_nonhidden(self.img_dir)

    def __len__(self) -> int:
        return len(self.img_list)

    def __getitem__(self, idx) -> tuple[torch.Tensor, None]:
        img_path = os.path.join(self.img_dir, self.img_list[idx])
        image = load_image(img_path)
        return image
