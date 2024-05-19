import numpy as np
import torch
from torch import nn
from torch.nn import functional
from PIL import Image
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
from torchvision import transforms

class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.mobile_net = mobilenet_v3_large(weights = MobileNet_V3_Large_Weights.DEFAULT)
        self.model = self.mobile_net
        self.model.classifier = self.model.classifier[:-1]
        
        for param in self.model.parameters():
            param.requires_grad = False
    def forward(self, x):
        return self.model(x)

class YoloLP(nn.Module):
    def __init__(self, S=7, B=2, num_classes=3):
        super(YoloLP, self).__init__()
        self.S = S
        self.B = B
        self.num_classes = num_classes
        self.backbone = Backbone()

        # convolution
        self.conv_layers = nn.Sequential(
            nn.Conv2d(960, 480, 3, stride=1, padding=1),
            nn.BatchNorm2d(480),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(480, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=1),

            nn.Conv2d(128, 128, 5, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=1),
        )

      
        self.fc_layers = nn.Sequential(
            nn.Linear(1280, 560),
            nn.SiLU(inplace=True),
            nn.Dropout(p=0.25),
            nn.Linear(560, self.S * self.S * (self.B * 5 + self.num_classes)),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        out = self.backbone(x)
        # out = self.conv_layers(out)
        # out = out.view(out.size()[0], -1)
        out = self.fc_layers(out)
        out = out.reshape(-1, self.S, self.S, self.B * 5 + self.num_classes)
        return out

