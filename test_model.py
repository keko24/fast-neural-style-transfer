import os
import torch
from torchvision.io import read_image

from data_loader import ImageDataLoader
from model import TransformationNetwork
from loss import LossNetwork
from train import Trainer
from utils import (
    get_project_root,
    load_json
)

if __name__ == "__main__":
    paths = {
        "setup_path": os.path.join(
            get_project_root(),
            "setup_files",
            "setup.json"
        ),
        "data_path": os.path.join(
            get_project_root(),
            "data",
            "content"
        ),
        "style_path": os.path.join(
            get_project_root(),
            "data",
            "style",
            "style.jpg"
        ),
        "results_path": os.path.join(
            get_project_root(),
            "results"
        )
    }
    setup = load_json(paths["setup_path"])
    data = ImageDataLoader(paths)
    style = read_image(paths["style_path"])
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = TransformationNetwork()
    model.to(DEVICE)
    loss_network = LossNetwork(style, DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=setup["lr"])
    trainer = Trainer(data, model, loss_network, optimizer, setup, DEVICE)
    trainer.train()
    torch.save(model.state_dict(), os.path.join("results_path", "model.pt")) 
