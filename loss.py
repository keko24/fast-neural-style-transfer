import copy

import torch
from torch import nn
from torch.nn.functional import mse_loss
from torchvision.models import vgg16

from utils import Normalize


class LossNetwork(nn.Module):
    def __init__(self, style, DEVICE):
        super().__init__()
        style_layers = ["relu1_2", "relu2_2", "relu3_3", "relu4_3"]
        content_layers = ["relu2_2"]

        vgg = vgg16(weights="DEFAULT").features.eval().to(DEVICE)
        vgg.requires_grad_(False)

        self.style_losses = []
        self.content_losses = []
        self.models = []

        self.normalize = Normalize()
        self.model = nn.Sequential(self.normalize)

        pool_idx, conv_idx, relu_idx, bn_idx = (1, 1, 1, 1)
        style_loss_idx, content_loss_idx = (1, 1)

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

            self.model.add_module(name, layer)

            if name in style_layers:
                target_style = self.model(style).detach()
                style_loss = StyleLoss(target_style)
                self.model.add_module(
                    f"style_loss_{style_loss_idx}", style_loss
                )
                self.style_losses.append(style_loss)
                style_loss_idx += 1

            if name in content_layers:
                target_content = self.model(
                    torch.zeros_like(style).detach()
                ).detach()
                content_loss = ContentLoss(
                    target_content, copy.deepcopy(self.model)
                )
                self.model.add_module(
                    f"content_loss_{content_loss_idx}", content_loss
                )
                self.content_losses.append(content_loss)
                content_loss_idx += 1

        for i in range(len(self.model) - 1, -1, -1):
            if isinstance(self.model[i], StyleLoss) or isinstance(
                self.model[i], ContentLoss
            ):
                self.model = self.model[: (i + 1)]
                break

    def forward(self, inputs, content):
        for content_loss in self.content_losses:
            content_loss.set_targets(content)
        return self.model(inputs)

def compute_gram_matrix(inputs):
    batch_size, cnn_channels, height, width = inputs.size()
    features = inputs.view(batch_size, cnn_channels, height * width)
    return features.bmm(features.transpose(1, 2)).div(
        cnn_channels * height * width
    )


class ContentLoss(nn.Module):
    def __init__(self, targets, model):
        super().__init__()
        self.targets = targets
        self.model = model

    def set_targets(self, targets):
        self.targets = self.model(targets).detach()

    def forward(self, inputs):
        self.loss = mse_loss(inputs, self.targets)
        return inputs


class StyleLoss(nn.Module):
    def __init__(self, target):
        super().__init__()
        self.target = compute_gram_matrix(target).detach()

    def forward(self, inputs):
        gram_matrix = compute_gram_matrix(inputs)
        self.loss = mse_loss(gram_matrix, self.target)
        return inputs


def calculate_total_variation_loss(inputs):
    tv_height = torch.sum(
        torch.square(inputs[:, :, 1:, :] - inputs[:, :, :-1, :])
    )
    tv_width = torch.sum(
        torch.square(inputs[:, :, :, 1:] - inputs[:, :, :, :-1])
    )
    return tv_width + tv_height
