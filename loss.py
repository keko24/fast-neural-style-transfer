import torch
from torch import nn
from torch.nn.functional import mse_loss
from torchvision.models import vgg16

class LossNetwork(nn.Module):
    def __init__(self, content, style, DEVICE):
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
            style_loss = StyleLoss(style)
            self.layer_subset.add_module(f"style_loss_{i + 1}", style_loss)
            self.style_losses.append(style_loss)
            self.style_extractor.append(layer_subset)

    def forward(self, x):
        content_feature_maps = self.content_extractor(x)
        style_feature_maps = [] 
        for style_layer in self.style_extractor:
            x = style_layer(x)
            style_feature_maps.append(x)
        return {"content": content_feature_maps, "style": style_feature_maps}

def compute_gram_matrix(input):
    batch_size, cnn_channels, height, width = input.size()
    features = input.view(batch_size, cnn_channels, height * width)
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

def compute_style_and_content_loss(input_img, model, DEVICE, style_weights=None):
    if style_weights == None:
        style_weights = torch.FloatTensor([1 / model.num_style_layers] * model.num_style_layers)
    assert isinstance(style_weights, torch.FloatTensor) and len(style_weights) == model.num_style_layers, "style_weights should be a torch.FloatTensor containing a weight for each style layer."
    input_features = model(input_img)
    content_loss = model.content_loss(input_features["content"])
    style_loss = torch.zeros(1).to(DEVICE, torch.float)
    for features, sl, weight in zip(input_features["style"], model.style_losses, style_weights):
        style_loss += sl(features) * weight
    return content_loss, style_loss
