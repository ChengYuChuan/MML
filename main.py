import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy
import matplotlib


device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")


WIDTH = 1000

# code adapted from https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, WIDTH),
            nn.ReLU(),
            nn.Linear(WIDTH, WIDTH),
            nn.ReLU(),
            nn.Linear(WIDTH, WIDTH),
            nn.ReLU(),
            nn.Linear(WIDTH, WIDTH),
            nn.ReLU(),
            nn.Linear(WIDTH, WIDTH),
            nn.ReLU(),
            nn.Linear(WIDTH, WIDTH),
            nn.ReLU(),
            nn.Linear(WIDTH, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


model = NeuralNetwork().to(device)
