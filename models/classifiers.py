import torch
import torch.nn as nn
from torchvision.models import resnet50


class VAEClassifier(nn.Module):
    def __init__(self, vae, latent_dim, classes):
        super().__init__()
        self.backbone = vae
        self.backbone.requires_grad = False
        self.fc = nn.Linear(latent_dim, classes)

    def forward(self, x):
        with torch.no_grad():
            z, _ = self.backbone.encode(x)
        output = self.fc(z)
        return output


class CNNClassifier(nn.Module):
    def __init__(self, classes, input_channels):
        super().__init__()
        self.input_channels = input_channels
        self.conv1 = nn.Conv2d(self.input_channels, 32, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1)
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, stride=1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(512, classes)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = self.pool(self.relu(self.conv4(x)))
        x = self.avgpool(self.relu(self.conv5(x)))
        x = self.flatten(x)
        x = self.fc(x)
        return x


class ResnetClassifier(nn.Module):
    def __init__(self, classes, input_channels):
        super().__init__()
        self.input_channels = input_channels
        resnet = resnet50(pretrained=True)
        modules = list(resnet.children())[:-1]
        self.encoder = nn.Sequential(*modules)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(2048, classes)

    def forward(self, x):
        x = self.encoder(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x
