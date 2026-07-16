import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


class ResNetEncoder(nn.Module):

    def __init__(self):
        super().__init__()

        model = resnet50(weights=ResNet50_Weights.DEFAULT)

        # Remove classification layer
        self.backbone = nn.Sequential(*list(model.children())[:-1])

    def forward(self, x):

        x = self.backbone(x)

        x = x.view(x.size(0), -1)

        return x