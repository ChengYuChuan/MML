import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

# argparser:
parser = argparse.ArgumentParser(
                    prog="Checking the Xavier Paper",
                    description="pass which activation and initialisation to use and the distribution of weights will be returned in a plot",
                    epilog="")
parser.add_argument("-a", "--activation")
parser.add_argument("-i", "--initialisation")
parser.add_argument("--learningrate")
parser.add_argument("-e", "--epochs")
parser.add_argument("-d", "--depth")
parser.add_argument("-w", "--width")
parser.add_argument("-s", "--datasetsize")


args = parser.parse_args()




class NeuralNetwork(nn.Module):
    def __init__(self, WIDTH, DEPTH, ACTIVATION, INIT):
        super().__init__()
        match ACTIVATION:
            case "ReLu":
                activation = nn.ReLU()
            case "Sigmoidal":
                activation = nn.Sigmoid()
            case "Tanh":
                activation = nn.Tanh()
            case "Softsign":
                activation = nn.Softsign()
            case _:
                print("Error in activation function name")
                activation = nn.ReLU()
        self.flatten = nn.Flatten()
        self.linearStack = nn.Sequential(nn.Linear(28*28, WIDTH))
        self.linearStack.add_module(name="2", module=activation)
        for i in range(DEPTH):
            self.linearStack.add_module(name=str(2*i+2), module=nn.Linear(WIDTH, WIDTH))
            self.linearStack.add_module(name=str(2*i+3), module=activation)
        self.linearStack.add_module(name=str(2*DEPTH+2), module=nn.Linear(WIDTH, 10))
        # initialize values
        for layer in self.linearStack:
            classname = layer.__class__.__name__
            if classname.find('Linear') != -1:
                match INIT: # .bernoulli_; .cauchy_; geometric_; normal_; xavier_uniform
                    case "uniform":
                        layer.weight.data.uniform_(-1.0, 1.0)
                    case "normal":
                        layer.weight.data.normal_(mean=0, std=1)
                    case "uniform/10":
                        layer.weight.data.uniform_(-0.1, 0.1)
                    case "normal/10":
                        layer.weight.data.normal_(mean=0, std=1/10)
                    case "xavier":
                        nn.init.xavier_uniform(layer.weight)
                    case _:
                        print("Error in initialisation name. Choosing default.")
                layer.bias.data.fill_(0)

    def forward(self, x):
        x = self.flatten(x)
        result = F.log_softmax(self.linearStack(x),dim=1) # COM: this is where the softmax is applyed. 
        return result
    
# adapted from https://github.com/pytorch/examples/blob/main/mnist/main.py    
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 500 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def trainAndVisualize(depth: int, width: int, initDistr: str, actFunc: str):
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
        )
    print(f"Using {device} device")
    
    model = NeuralNetwork(WIDTH=width, DEPTH=depth, ACTIVATION=actFunc, INIT=initDistr).to(device)

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)) # what does this do?
        ])
    dataset = datasets.MNIST('../data', train=True, download=True,
                       transform=transform)
    dataset_truncated = torch.utils.data.Subset(dataset, list(range(int(args.datasetsize))))
    trainLoader = torch.utils.data.DataLoader(dataset_truncated, batch_size=10, shuffle=True)
    
    optimizer = torch.optim.Adam(model.parameters(), lr = LR)
    for epoch in range(EPOCHS):
        train(model, device, trainLoader, optimizer, epoch)
    
    # get layer values
    layerValues = []
    for idx,layer in enumerate(model.parameters()):
        if idx%2 == 0:
            layerValues.append(np.array(layer.data.flatten().cpu()))
    
    # visualize the layer values
    for layer in range(depth):
        counts, bins = np.histogram(layerValues[layer], bins = 100)
        plt.plot(counts/np.sum(counts), label=str(layer))
        plt.legend()
        plt.xlabel("weight value")
        plt.ylabel("value probability")
    path = os.path.join("images", f"Weight_Distribution_depth{args.depth}_width{args.width}_init={args.initialisation}_act={args.activation}.png".replace('/', '_'))
    plt.savefig(path)

LR = float(args.learningrate)
EPOCHS = int(args.epochs)
trainAndVisualize(int(args.depth),int(args.width),args.initialisation,args.activation)
    
