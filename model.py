import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class Classifier(nn.Module):
    def __init__(self, dropout=0.5, in_c=3):
        super(Classifier, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_c, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.flatten = nn.Flatten()
        self.seq = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(4096, 7),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        x = self.convs(x)
        x = self.flatten(x)
        output = self.seq(x)
        return output