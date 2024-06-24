import os

import torch
from torchvision.transforms.functional import to_pil_image

from data_loader import ImageDataLoader
from loss import LossNetwork
from model import TransformationNetwork
from preprocess import Preprocess
from train import Trainer
from utils import get_project_root, load_json, load_style

if __name__ == "__main__":
    paths = {
        "setup_path": os.path.join(get_project_root(), "setup_files", "setup.json"),
        "data_path": os.path.join(get_project_root(), "data", "content"),
        "style_path": os.path.join(get_project_root(), "data", "style", "style.jpg"),
        "results_path": os.path.join(get_project_root(), "results"),
    }
    setup = load_json(paths["setup_path"])
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data = ImageDataLoader(paths)
    style = load_style(paths["style_path"], setup["batch_size"], DEVICE)

    transform = Preprocess()
    model = TransformationNetwork()
    model.to(DEVICE)
    loss_network = LossNetwork(style, DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=setup["lr"])
    trainer = Trainer(data, model, loss_network, optimizer, setup, DEVICE)
    trainer.train()

    style = style[0].detach().cpu().clone().squeeze(0)
    style = to_pil_image(style)
    os.makedirs(paths["results_path"], exist_ok=True)
    style.save(os.path.join("results", "style.jpg"))
    torch.save(model.state_dict(), os.path.join(paths["results_path"], "model.pt"))
