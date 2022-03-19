# This is a sample Python script fom https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html.

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import os.path

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.

# Define model
class NeuralNetwork(nn.Module):

    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

def train(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

def train_and_save(training_data,test_data,device,epochs,fname):

    # Create the model
    model = NeuralNetwork().to(device)
    print(model)

    # To train a model, we need a loss function and an optimizer.
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    train_dataloader = DataLoader(training_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)
    for X, y in test_dataloader:
        print(f"Shape of X [N, C, H, W]: {X.shape}")
        print(f"Shape of y: {y.shape} {y.dtype}")
        break

    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer, device)
        test(test_dataloader, model, loss_fn)
    print("Training Done!")

    torch.save(model.state_dict(), fname)
    print(f"Saved PyTorch Model State to {fname}")

def read_and_run(fname,test_data,device):

    model = NeuralNetwork().eval()
    model.load_state_dict(torch.load(fname))
    # model.to(device)
    # RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu! (when checking argument for argument mat1 in method wrapper_addmm)
    print(f"Loaded PyTorch Model State from {fname}")
    print(model)

    classes = [
        "T-shirt/top",
        "Trouser",
        "Pullover",
        "Dress",
        "Coat",
        "Sandal",
        "Shirt",
        "Sneaker",
        "Bag",
        "Ankle boot",
    ]

    for i in range(test_data.data.size(dim=0)):
        x, y = test_data[i][0], test_data[i][1]
        with torch.no_grad():
            pred = model(x)
            predicted, actual = int(pred[0].argmax(0)), y
            if predicted == actual:
                print(f'{i:5d} Matched:   "{classes[predicted]}" {pred[0,predicted]:3.1f}')
            else:
                print(f'{i:5d} Predicted: "{classes[predicted]}" {pred[0,predicted]:3.1f}, Actual: "{classes[actual]}" {pred[0,actual]:3.1f}')

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    batch_size = 64
    epochs = 5
    ds_data_root = "data"
    model_pth = ds_data_root + "/model.pth"

    # Download training data from open datasets.
    training_data = datasets.FashionMNIST(
        root=ds_data_root,
        train=True,
        download=True, # once
        transform=ToTensor(),
    )

    # Download test data from open datasets.
    test_data = datasets.FashionMNIST(
        root=ds_data_root,
        train=False,
        download=True, # once
        transform=ToTensor(),
    )

    # Get cpu or gpu device for training.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    # train the model
    if not os.path.exists(model_pth):
        train_and_save(training_data,test_data,device, epochs, model_pth)

    # run the model
    read_and_run(model_pth,test_data,device)

    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/