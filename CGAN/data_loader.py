import os
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.datasets import MNIST

def MNIST_dataset(input_size=28, batch_size=64):
    transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    data_loader = DataLoader(MNIST(os.getcwd(), train=True, download=True, transform=transform), batch_size=batch_size, shuffle=True)
    return data_loader