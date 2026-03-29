# backend/app/inference.py
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np

# === Manual Xception Model Definition ===
class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class EntryFlow(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = ConvBNReLU(3, 32, 3, 2, 1)
        self.conv2 = ConvBNReLU(32, 64, 3, 1, 1)
        self.block1 = self._make_block(64, 128, stride=2)
        self.block2 = self._make_block(128, 256, stride=2)
        self.block3 = self._make_block(256, 728, stride=2)

    def _make_block(self, in_channels, out_channels, stride):
        return nn.Sequential(
            nn.ReLU(inplace=True),
            ConvBNReLU(in_channels, out_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            ConvBNReLU(out_channels, out_channels, 3, 1, 1),
            nn.MaxPool2d(3, stride, 1)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return x

class MiddleFlowBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.layers = nn.Sequential(
            nn.ReLU(inplace=True),
            ConvBNReLU(channels, channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            ConvBNReLU(channels, channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            ConvBNReLU(channels, channels, 3, 1, 1)
        )

    def forward(self, x):
        return self.layers(x) + x

class MiddleFlow(nn.Module):
    def __init__(self, channels=728):
        super().__init__()
        self.blocks = nn.Sequential(*[MiddleFlowBlock(channels) for _ in range(8)])

    def forward(self, x):
        return self.blocks(x)

class ExitFlow(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.ReLU(inplace=True),
            ConvBNReLU(728, 728, 3, 1, 1),
            nn.ReLU(inplace=True),
            ConvBNReLU(728, 1024, 3, 1, 1),
            nn.MaxPool2d(3, 2, 1)
        )
        self.block2 = nn.Sequential(
            ConvBNReLU(1024, 1536, 3, 1, 1),
            nn.ReLU(inplace=True),
            ConvBNReLU(1536, 2048, 3, 1, 1),
            nn.ReLU(inplace=True)
        )
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        x1 = self.block1(x)
        x2 = F.max_pool2d(x, 3, 2, 1)
        x2 = F.pad(x2, (0, 1, 0, 1))  # align
        x = x1 + x2
        x = self.block2(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

class Xception(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        self.entry = EntryFlow()
        self.middle = MiddleFlow()
        self.exit = ExitFlow(num_classes)

    def forward(self, x):
        x = self.entry(x)
        x = self.middle(x)
        x = self.exit(x)
        return x

# === Transform ===
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# === Detector Class ===
class DeepfakeDetector:
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = Xception(num_classes=1)
        # Create dummy weights if file doesn't exist
        if not os.path.exists(model_path):
            torch.save(self.model.state_dict(), model_path)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        self.model.to(self.device)

    def predict(self, face_img):
        input_tensor = transform(face_img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = torch.sigmoid(self.model(input_tensor))
        prob_fake = output.item()
        return {"real": 1 - prob_fake, "fake": prob_fake}