from net.NeuralNetwork import NeuralNetwork

import torch
from torch import nn
from torch.utils.data import DataLoader


class Trainer:
    r"""Performer of all model train tasks"""

    def __init__(self, device, lr=1e-3):
        self.device = device
        # Create the model
        self.model = NeuralNetwork().to(device)
        print(self.model)
        # To train a model, we need a loss function and an optimizer.
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)

    def train(self, dataloader, mi):
        size = len(dataloader.dataset)
        self.model.train()
        for batch, (X, y) in enumerate(dataloader):

            if mi:
                bx = batch * 64
                for i in range(X.shape[0]):
                    from main import mkimage # TODO
                    mkimage(bx + i, X[i, 0,].numpy(), "train", y[i])

            # Compute prediction error
            pred = self.model(X.to(self.device))
            loss = self.loss_fn(pred, y.to(self.device))

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if batch % 100 == 0:
                loss, current = loss.item(), batch * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    def test(self, dataloader):
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        self.model.eval()
        test_loss, correct = 0, 0
        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(self.device), y.to(self.device)
                pred = self.model(X)
                test_loss += self.loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        test_loss /= num_batches
        correct /= size
        print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    def run(self, train_dataloader: DataLoader, test_dataloader: DataLoader, epochs: int) -> NeuralNetwork:
        for t in range(epochs):
            print(f"Epoch {t + 1}\n-------------------------------")
            self.train(train_dataloader, t == 0)
            self.test(test_dataloader)
        return self.model
