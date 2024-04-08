import torch
from torch import nn
from torch.nn.functional import mse_loss
from torchvision.models import vgg16

class LossNetwork(nn.Module):
    def __init__(self, style, DEVICE):
        super(LossNetwork, self).__init__()
        vgg = vgg16().features.eval().to(DEVICE)
        for parameter in vgg.parameters():
            parameter.requires_grad_(False)
        style_layers = [3, 8, 15, 22]
        content_layer = 15

        self.content_extractor = nn.Sequential()
        self.content_losses = []
        for curr_layer in range(content_layer + 1):
            self.content_extractor.add_module(str(curr_layer), vgg[curr_layer])
        content = self.content_extractor(style)
        content_loss = ContentLoss(content)
        self.content_extractor.add_module("content_loss", content_loss)
        self.content_losses.append(content_loss)

        self.style_extractor = nn.ModuleList()
        self.style_losses = []
        for i, end_layer in enumerate(style_layers):
            start_layer = 0 if i == 0 else style_layers[i - 1] + 1
            layer_subset = nn.Sequential()
            for curr_layer in range(start_layer, end_layer + 1):
                layer_subset.add_module(str(curr_layer), vgg[curr_layer])
            style = layer_subset(style)
            style_loss = StyleLoss(style)
            layer_subset.add_module(f"style_loss_{i + 1}", style_loss)
            self.style_losses.append(style_loss)
            self.style_extractor.append(layer_subset)

    def forward(self, inputs, content):
        content = self.content_extractor(content)
        content_loss = ContentLoss(content)
        self.content_extractor[-1] = content_loss
        self.content_losses[-1] = content_loss
        content_feature_maps = self.content_extractor(inputs)
        style_feature_maps = [] 
        for style_layer in self.style_extractor:
            inputs = style_layer(inputs)
            style_feature_maps.append(inputs)
        return {"content": content_feature_maps, "style": style_feature_maps}

def compute_gram_matrix(inputs):
    batch_size, cnn_channels, height, width = inputs.size()
    features = inputs.view(batch_size, cnn_channels, height * width)
    return features.bmm(features.transpose(1, 2)).div(cnn_channels * height * width)

class ContentLoss(nn.Module):
    def __init__(self, target):
        super().__init__()
        self.target = target.detach()
    
    def forward(self, x):
        self.loss = mse_loss(x, self.target)
        return x
 
class StyleLoss(nn.Module):
    def __init__(self, target):
        super().__init__()
        self.target = compute_gram_matrix(target).detach()
    
    def forward(self, x):
        gram_matrix = compute_gram_matrix(x)
        self.loss = mse_loss(gram_matrix, self.target)
        return x

def calculate_total_variation_loss(inputs):
    tv_height = torch.sum(torch.square(inputs[:, :, 1:, :] - inputs[:, :, :-1, :]))
    tv_width = torch.sum(torch.square(inputs[:, :, :, 1:] - inputs[:, :, :, :-1]))
    return tv_width + tv_height
