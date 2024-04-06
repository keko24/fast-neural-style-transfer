from torch.utils.data import DataLoader

class ImageDataLoader:
    def __init__(self, setup, data):
        super(ImageDataLoader, self).__init__() 
        self.data = data
        self.setup = setup

    def train_data(self):
        train_loader = DataLoader(
            self.data,
            batch_size=self.setup["batch_size"],
            shuffle=True,
            drop_last=True
        )
        return train_loader
