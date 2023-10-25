import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt





class NeuralNetwork(nn.Module):
    def __init__(self, WIDTH, DEPTH, ACTIVATION):
        super().__init__()
        match ACTIVATION:
            case "ReLu":
                activation = nn.ReLU()
            case "Sigmoidal":
                activation = nn.Sigmoid()
            case "Tanh":
                activation = nn.Tanh()
            case "Softsign":
                activation = lambda x: x/(1+abs(x))
            case _:
                print("Error in activation function name") # TODO
                activation = nn.ReLU()
        self.flatten = nn.Flatten()
        self.linearStack = nn.Sequential(
            nn.Linear(28*28, WIDTH),
            nn.ReLU(),
        )
        for i in range(DEPTH):
            self.linearStack.add_module(name=str(2*i+2), module=nn.Linear(WIDTH, WIDTH))
            self.linearStack.add_module(name=str(2*i+3), module=activation)
        self.linearStack.add_module(name=str(2*DEPTH+2), module=nn.Linear(WIDTH, 10))
        # initialize values here ???

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
        if batch_idx % 100 == 0:
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
    
    model = NeuralNetwork(WIDTH=width, DEPTH=depth, ACTIVATION=actFunc).to(device)

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)) # what does this do?
        ])
    dataset = datasets.MNIST('../data', train=True, download=True,
                       transform=transform)
    dataset_truncated = torch.utils.data.Subset(dataset, list(range(6000)))
    trainLoader = torch.utils.data.DataLoader(dataset_truncated, batch_size=10, shuffle=True)
    
    optimizer = torch.optim.Adam(model.parameters(), lr = LR)
    # lrDecreacer = torch.optim.lr_scheduler(optimizer, step_size=3, gamma=0.005)
    for epoch in range(EPOCHS):
        train(model, device, trainLoader, optimizer, epoch)
        # lrDecreacer.step()
    
    # get layer values
    layerValues = []
    print(model.linearStack)
    for layer in model.parameters(): # TODO!!!
        layerValues.append(np.array(layer.data.flatten().cpu()))
        print(len(list(layerValues[-1])))
        # layerValues.append(model.linearStack[layer*2])
        # print(layerValues[-1].type, layerValues[-1])
    print(len(layerValues))
    
    # visualize the layer values
    for layer in range(depth):
        counts, bins = np.histogram(layerValues[layer], bins = 100)
        # print(counts, bins)
        plt.plot(counts/np.sum(counts), label=str(layer))
        plt.legend()
        plt.xlabel("weight value")
        plt.ylabel("value probability")
        # plt.hist(layerValues[layer])
    plt.show()
    # plt.savefig("test.png")

LR = 0.01
EPOCHS = 2#80
trainAndVisualize(5,1000,"normalized","Tanh")
    
