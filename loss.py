from collections import namedtuple

import torch
from torch import nn
from torch.nn.functional import mse_loss
from torchvision.models import vgg16

from utils import Normalize


class LossNetwork(nn.Module):
    def __init__(self, DEVICE):
        super().__init__()
        style_layers = ["relu1_2", "relu2_2", "relu3_3", "relu4_3"]
        content_layers = ["relu2_2"]

        vgg = vgg16(weights="DEFAULT").features.eval().to(DEVICE)
        vgg.requires_grad_(False)

        self.OutputTuple = namedtuple(
            "LossNetworkOutputs", ["content_outputs", "style_outputs"]
        )

        self.style_models = []
        self.content_models = []

        self.normalize = Normalize()
        self.model = nn.Sequential(self.normalize)

        pool_idx, conv_idx, relu_idx, bn_idx = (1, 1, 1, 1)

        style_model = nn.Sequential(Normalize())
        content_model = nn.Sequential(Normalize())
        for layer in vgg.children():
            if isinstance(layer, nn.MaxPool2d):
                name = f"pool_{pool_idx}"
                layer = nn.AvgPool2d(
                    kernel_size=layer.kernel_size,
                    stride=layer.stride,
                    padding=layer.padding,
                    ceil_mode=layer.ceil_mode,
                )
                pool_idx += 1
                relu_idx, conv_idx, bn_idx = (1, 1, 1)
            elif isinstance(layer, nn.Conv2d):
                name = f"conv{pool_idx}_{conv_idx}"
                conv_idx += 1
            elif isinstance(layer, nn.ReLU):
                name = f"relu{pool_idx}_{relu_idx}"
                layer = nn.ReLU(inplace=False)
                relu_idx += 1
            elif isinstance(layer, nn.BatchNorm2d):
                name = f"bn_{pool_idx}_{bn_idx}"
                bn_idx += 1
            else:
                raise RuntimeError(
                    "Unrecognized layer: {}".format(layer.__class__.__name__)
                )

            content_model.add_module(name, layer)
            style_model.add_module(name, layer)

            if name in style_layers:
                self.style_models.append(style_model)
                style_model = nn.Sequential()

            if name in content_layers:
                self.content_models.append(content_model)
                content_model = nn.Sequential()

    def forward(self, inputs):
        content_outputs, style_outputs = [], []
        x = inputs
        for content_model in self.content_models:
            x = content_model(x)
            content_outputs.append(x)

        x = inputs
        for style_model in self.style_models:
            x = style_model(x)
            style_outputs.append(x)

        return self.OutputTuple(content_outputs, style_outputs)


def compute_gram_matrix(inputs):
    batch_size, cnn_channels, height, width = inputs.size()
    features = inputs.view(batch_size, cnn_channels, height * width)
    return features.bmm(features.transpose(1, 2)).div(
        cnn_channels * height * width
    )


def calculate_style_loss(inputs, targets):
    inputs_gram, targets_gram = (
        compute_gram_matrix(inputs),
        compute_gram_matrix(targets),
    )
    return mse_loss(inputs_gram, targets_gram)


def calculate_content_loss(inputs, targets):
    return mse_loss(inputs, targets)


def calculate_total_variation_loss(inputs):
    tv_height = torch.sum(
        torch.square(inputs[:, :, 1:, :] - inputs[:, :, :-1, :])
    )
    tv_width = torch.sum(
        torch.square(inputs[:, :, :, 1:] - inputs[:, :, :, :-1])
    )
    return tv_width + tv_height
