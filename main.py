import matplotlib.pyplot as plt
import numpy as np
import torch
from numpy._core.defchararray import mod
from torch import batch_norm_stats, mode, nn, optim
from torch.utils import data
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms.transforms import ToTensor

from classes import mean
from loops import test_loop, train_loop
from nn import NeuralNetwork

plt.rcParams["font.size"] = 18

torch.manual_seed(1006)

device = torch.device("cuda")

training_data = datasets.FashionMNIST(root="data", train=True, download=True, transform=ToTensor())

test_data = datasets.FashionMNIST(root="data", train=False, download=True, transform=ToTensor())

BATCH_SIZE = 128
LEARNING_RATE = 1e-3
EPOCHS = 32

train_dataloader = DataLoader(training_data, batch_size=BATCH_SIZE)
test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE)

model = NeuralNetwork().to(device)

loss_fn = nn.CrossEntropyLoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)

training_losses = []
training_accuracies = []

test_losses = []
test_accuracies = []

for t in range(EPOCHS):
    print(f"EPOCH {t+1}\n-----------------------------------")
    training_loss, training_accuracy = train_loop(train_dataloader, model, loss_fn, optimizer, device, BATCH_SIZE)
    test_loss, test_accuracy = test_loop(test_dataloader, model, loss_fn, device)

    training_losses += training_loss
    training_accuracies += training_accuracy
    test_losses.append(test_loss)
    test_accuracies.append(test_accuracy)


N = 10

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

ax1.plot(mean(training_losses, N), label="training loss", color="tab:blue")
ax2.plot(mean(training_accuracies, N), label="training accuracy", color="tab:orange")

epochs_end_x = [len(train_dataloader) * (i + 1) / N for i in range(EPOCHS)]
for i, x in enumerate(epochs_end_x):
    ax1.axvline(x, linestyle=":")


ax1.plot(epochs_end_x, test_losses, label="test loss", color="tab:green")
ax2.plot(epochs_end_x, test_accuracies, label="test accuracy", color="tab:purple")

fig.legend()
fig.tight_layout()
plt.show()
