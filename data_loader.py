from torch.utils.data import DataLoader

from dataset import ImageDataset
from utils import load_json


class ImageDataLoader:
    def __init__(self, paths: dict[str, str]):
        super(ImageDataLoader, self).__init__()
        self.data = ImageDataset(paths["data_path"])
        self.setup = load_json(paths["setup_path"])

    def train_data(self):
        train_loader = DataLoader(
            self.data, batch_size=self.setup["batch_size"], shuffle=True, drop_last=True
        )
        return train_loader

    def test_data(self):
        test_loader = DataLoader(
            self.data,
            batch_size=self.setup["batch_size"],
            shuffle=False,
            drop_last=False,
        )
        return test_loader
