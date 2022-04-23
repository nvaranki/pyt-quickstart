# This is a sample Python script fom https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html.

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import numpy as np
import os.path

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


def train_and_save(training_data, test_data, batch_size, device, epochs, fname):

    from net.Trainer import Trainer
    trainer = Trainer(device, 1e-3)

    train_dataloader = DataLoader(training_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)
    for X, y in test_dataloader:
        print(f"Shape of X [N, C, H, W]: {X.shape}")
        print(f"Shape of y: {y.shape} {y.dtype}")
        break

    model = trainer.run(train_dataloader, test_dataloader, epochs)
    print("Training Done!")

    torch.save(model.state_dict(), fname)
    print(f"Saved PyTorch Model State to {fname}")


def read_and_run(fname, test_data, device, good=False, bad=False, hist=False, summ=True):

    from net.Runner import Runner
    runner = Runner(device, fname)
    classes = test_data.classes
    print(classes)
    print(f"Loaded PyTorch Model State from {fname}")

    h_cat_matched = []
    h_cat_missed  = []
    h_val_matched = []
    h_val_missed  = []
    for i in range(test_data.data.size(dim=0)):
        x, y = test_data[i][0], test_data[i][1]
        mkimage(i,x[0,].numpy(),"test",y)
        with torch.no_grad():
            pred = runner.infer(x.expand(1,-1,-1,-1))
            predicted, actual = int(pred[0].argmax(0)), y
            if predicted == actual:
                if good:
                    print(f'{i:5d} Matched:   "{classes[predicted]}" {pred[0,predicted]:3.1f}')
                h_cat_matched.append(predicted)
                h_val_matched.append(pred[0,predicted].item())
            else:
                if bad:
                    print(f'{i:5d} Predicted: "{classes[predicted]}" {pred[0,predicted]:3.1f}, Actual: "{classes[actual]}" {pred[0,actual]:3.1f}')
                h_cat_missed.append(predicted)
                h_val_missed.append(pred[0,predicted].item())
    if hist:
        np.set_printoptions(precision=1)
        print('Category Guess Histogram')
        t_cat_matched = print_hist('Matched', h_cat_matched, range(len(classes)+1))[0]
        t_cat_missed  = print_hist('Missed',  h_cat_missed,  range(len(classes)+1))[0]
        print(f'Quality\n{np.multiply(100,np.divide(t_cat_matched,np.add(t_cat_matched,t_cat_missed)))}')
        print('Value Guess Histogram')
        print_hist('Matched', h_val_matched, range(0, 22, 2))
        print_hist('Missed', h_val_missed, range(0, 22, 2))
        np.set_printoptions()
    if summ:
        n_cat_matched = len(h_cat_matched)
        n_cat_missed = len(h_cat_missed)
        n_cat_total = n_cat_matched + n_cat_missed
        print(f'Summary: {n_cat_matched}={float(n_cat_matched)/n_cat_total:.1%} matched, {n_cat_missed}={float(n_cat_missed)/n_cat_total:.1%} missed, {n_cat_total} total')


def print_hist(title: str, data, *args, **kwargs) -> tuple:
    hist, bins = np.histogram(data, *args, **kwargs)
    print(title)
    print(hist)
    print(bins)
    return (hist,bins)


def mkimage(i,bmp,prefix,suffix):
    import skimage.io
    import numpy as np

    fname = f"image/{prefix}"
    if not os.path.exists(fname):
        os.makedirs(fname, exist_ok = False)
    fname += f"/{i:05d}-{suffix}.png"
    if os.path.exists(fname):
        return

    h = bmp.shape[1]
    w = bmp.shape[0]
    img = np.zeros((w, h, 3), dtype="uint8")
    for y in range(h):
        for x in range(w):
            g = int(np.floor(bmp[x,y]*255))
            img[x,y,] = [g, g, g]
    try:
        skimage.io.imsave(fname, img, check_contrast=False)
    except Exception:
        print(f"Failed to save \"{fname}\".")


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    batch_size = 64
    epochs = 5
    # 4m:        5 epochs Accuracy: 75.2%, Avg loss: 0.767384;
    #           25 epochs Accuracy: 80.9%, Avg loss: 0.534439;
    #           50 epochs Accuracy: 82.5%, Avg loss: 0.488360;
    #           90 epochs Accuracy: 83.5%, Avg loss: 0.462558
    #          125 epochs Accuracy: 83.7%, Avg loss: 0.451608
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
        train_and_save(training_data, test_data, batch_size, device, epochs, model_pth)

    # run the model
    read_and_run(model_pth,test_data,device,hist=True)

    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
