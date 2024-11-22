import torch
import torchvision
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import datetime
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from flows import *

# i'll conduct tests on my methods in this file
# first test: autoregressive neural spline flow on mnist

class DataProc():
    def __init__(self, eps=1e-6, bound=5.0, features=784):
        self.eps = eps
        self.features = features
        self.bound = bound
    
    def __call__(self, x: torch.Tensor):
        x = x.view(self.features).float()
        x = x + (torch.rand_like(x)/255)
        x = x / (1+1/255)
        x = self.eps + (1 - 2 * self.eps) * x
        # x = (x - x.mean(dim=-1)) / x.std(dim=-1)
        x = torch.log(x / (1.0 - x))
        return x

n_in = 784
context_in = 10
num_bins = 2
tail_bound = 14
batch_size = 64
hidden_dims = [64]

transform = transforms.Compose([
    transforms.ToTensor(),
    DataProc(bound=tail_bound)
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
model = Flow([
    ConditionalAutoregressiveSpline('rqs', num_bins, tail_bound, n_in, context_in, hidden_dims), 
    BatchNorm(n_in, 2, 1),
    ConditionalAutoregressiveSpline('rqs', num_bins, 3, n_in, context_in, hidden_dims), 
    BatchNorm(n_in, 2, 1),
    ConditionalAutoregressiveSpline('rqs', num_bins, 3, n_in, context_in, hidden_dims), 
    BatchNorm(n_in, 2, 1),
    ConditionalAutoregressiveSpline('rqs', num_bins, 3, n_in, context_in, hidden_dims), 
    BatchNorm(n_in, 2, 1),
    ])
optimizer = optim.Adam(model.parameters(), lr=1e-4)

def train(model, train_loader, optimizer, epoch):
    model.train()
    running_loss = 0.0
    for batch_idx, (data, context) in enumerate(train_loader):
        optimizer.zero_grad()
        context = F.one_hot(context, num_classes=10).to(torch.float32)
        z, log_prob, log_det = model(data, context)
        nll = -torch.mean(log_prob + log_det)
        running_loss += nll.item()
        nll.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(datetime.datetime.now(), nll)
    return running_loss / len(train_loader)

def sample_and_plot(model, num_samples=16):
    model.eval()
    with torch.no_grad():
        z = torch.randn(num_samples, n_in)
        labels = torch.randint(0, 10, (num_samples,), dtype=torch.long)
        print(labels)
        labels = F.one_hot(labels, num_classes=10).float()
        samples, _ = model.inverse(z, labels)
        samples = torch.clamp(samples, 1e-6, 1-1e-6)
        samples = samples.view(num_samples, 1, 28, 28).cpu()
        grid = torchvision.utils.make_grid(samples, nrow=4, padding=2, normalize=False)
        plt.imshow(grid.permute(1, 2, 0))
        plt.axis('off')
        plt.show()

def test(model, train_loader):
    model.eval()
    with torch.no_grad():
        data, labels = next(iter(train_loader))
        labels = F.one_hot(labels, num_classes=10).float()
        data = data.view(data.size(0), -1)
        z, _, _ = model(data, labels)
        z = z.view(-1, 1, 28, 28).cpu()
        print(z.mean(), z.std())
        # recon, _ = model.inverse(z, labels)
        # recon = recon.view(-1, 1, 28, 28).cpu()
        # recon = torch.clamp(recon, 1e-6, 1-1e-6)
        grid = torchvision.utils.make_grid(z, nrow=8, padding=2, normalize=True)
        fig = plt.imshow(grid.permute(1, 2, 0))
        plt.axis('off')
        plt.show()


num_epochs = 10
for epoch in range(1, num_epochs + 1):
    train_loss = train(model, train_loader, optimizer, epoch)
    print(datetime.datetime.now(), f"epoch {epoch} training loss: {train_loss}")
    sample_and_plot(model)
    test(model, train_loader)