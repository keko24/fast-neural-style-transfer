import torch

from model import TransformationNetwork
from loss import LossNetwork

if __name__ == "__main__":
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = TransformationNetwork()
    model.to(DEVICE)
    loss_network = LossNetwork(DEVICE)

