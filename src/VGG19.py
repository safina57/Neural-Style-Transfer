import torch
from torchvision import models

import torch.nn as nn

class VGG19(nn.Module):
    def __init__(self):
        super(VGG19, self).__init__()
        self.vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features
        self.style_layers = [0, 5, 10, 19, 28]
        self.content_layers = [21]

    def forward(self, x : torch.tensor, is_style : bool = False ) -> list:
        """
        Forward pass through the network
        Parameters:
            x : torch.tensor : input image
            is_style : bool : whether to extract style or content features
        Returns:
            features : list : list of features extracted from the network
        """
        features = []
        layers = self.style_layers if is_style else self.content_layers
        for i , layer in enumerate(self.vgg):
            x = layer(x)
            if i in layers:
                features.append(x)

        return features