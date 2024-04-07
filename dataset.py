import os

import torch
from torch.utils.data import Dataset
from torchvision.io import read_image

from utils import listdir_nonhidden
from preprocess import Preprocess

class ImageDataset(Dataset):
    def __init__(self, img_dir: str) -> None:
        super().__init__()
        self.img_dir = img_dir
        self.img_list = listdir_nonhidden(self.img_dir)
        self.preprocess = Preprocess()

    def __len__(self) -> int:
        return len(self.img_list)

    def __getitem__(self, idx) -> tuple[torch.Tensor, None]:
        img_path = os.path.join(self.img_dir, self.img_list[idx])
        image = read_image(img_path)
        image = self.preprocess(image)
        return image, None
        
