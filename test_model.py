import os

import torch

from data_loader import ImageDataLoader
from loss import LossNetwork
from model import TransformationNetwork
from train import Trainer
from utils import get_project_root, load_json, load_style, save_image

if __name__ == "__main__":
    paths = {
        "setup_path": os.path.join(
            get_project_root(), "setup_files", "setup.json"
        ),
        "data_path": os.path.join(get_project_root(), "data", "content"),
        "style_path": os.path.join(
            get_project_root(), "data", "style", "style.jpg"
        ),
        "results_path": os.path.join(get_project_root(), "results"),
    }
    setup = load_json(paths["setup_path"])
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data = ImageDataLoader(paths)
    style = load_style(paths["style_path"], setup["batch_size"], DEVICE)

    model = TransformationNetwork()
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=setup["lr"])
    loss_network = LossNetwork(DEVICE)
    style_outputs = loss_network(style).style_outputs
    trainer = Trainer(
        data, style_outputs, model, loss_network, optimizer, setup, DEVICE
    )
    trainer.train()

    os.makedirs(paths["results_path"], exist_ok=True)
    style = style[0].squeeze(0)
    save_image(style, os.path.join("results", "style.jpg"))
    torch.save(
        model.state_dict(), os.path.join(paths["results_path"], "model.pt")
    )
