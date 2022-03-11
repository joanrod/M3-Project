import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from model import NetSquared
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from utils import get_loaders

# Hyperparameters etc.
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
NUM_EPOCHS = 10
NUM_WORKERS = 2
PIN_MEMORY = True       # To speed up the file transfer to CPU to GPU
LOAD_MODEL = False
TRAIN_IMG_DIR = "MIT_split/train/"
TEST_IMG_DIR = "MIT_split/test/"

def train_fn(loader, model, optimizer, loss_fn, scaler, device, epoch_num):
    """
    Function to train the model
    :param loader:
    :param model:
    :param optimizer:
    :param loss_fn:
    :param scaler:
    :return:
    """

    model = model.to(device=device)     # Put model in DEVICE (recomended in GPU)
    model.train()                       # Train mode

    loop = tqdm(loader, desc=f'EPOCH {epoch_num} TRAIN')  # Create the tqdm bar for visualizing the progress.

    correct = 0     # accumulated correct predictions
    total = 0       # accumulated total predictions
    acc_loss = 0    # accumulated loss

    # Loop to obtain the batches of images and labels
    for (data, targets) in loop:

        data = data.to(device=device)       # Batch of images to DEVICE, where the model is
        targets = targets.to(device=device) # Batch of labels to DEVICE, where the model is

        optimizer.zero_grad()

        output = model(data)                # Output of the model (logits).
        loss = loss_fn(output, targets)     # Compute the loss between the output and the ground truth

        _, predictions = torch.max(output.data, 1)  # Obtain the classes with higher probability

        total += data.size(0)                               # subtotal of the predictions
        correct += (predictions == targets).sum().item()    # subtotal of the correct predictions
        acc_loss += loss.item()

        scaler.scale(loss).backward()   # compute the backward stage updating the weights of the model
        scaler.step(optimizer)
        scaler.update()

        loop.set_postfix(acc=correct/total, loss=acc_loss/total)  # set current loss
    # print(f'Train: acc: {correct / total} - loss: {acc_loss/total}')

def eval_fn(loader, model, loss_fn, device, epoch_num):

    model = model.to(device=device)
    model.eval()

    loop = tqdm(loader, desc=f'EPOCH {epoch_num}  TEST')  # Create the tqdm bar for visualizing the progress.

    correct = 0     # accumulated correct predictions
    total = 0       # accumulated total predictions
    acc_loss = 0    # accumulated loss

    with torch.no_grad():
        for (data, targets) in loop:
            data = data.to(device=device)
            targets = targets.to(device=device)

            output = model(data)
            loss = loss_fn(output, targets)  # Compute the loss between the output and the ground truth

            _, predictions = torch.max(output.data, 1)  # Obtain the classes with higher probability
            acc_loss += loss.item()

            total += data.size(0)
            correct += (predictions == targets).sum().item()
            loop.set_postfix(acc=correct/total, loss=acc_loss/total)

def main():
    if torch.cuda.is_available():
        print(f'Trianing the model in {torch.cuda.get_device_name(torch.cuda.current_device())}')
    else:
        print('CAREFUL!! Training the model with CPU')

    tf = transforms.Compose([
        transforms.ToTensor()
    ])

    model = NetSquared()

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler()

    train_dataset = ImageFolder(TRAIN_IMG_DIR, transform=tf)
    test_dataset = ImageFolder(TEST_IMG_DIR, transform=tf)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, pin_memory=True, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, pin_memory=True)

    for epoch in range(NUM_EPOCHS):

        train_fn(train_loader, model, optimizer, loss_fn, scaler, DEVICE, epoch)
        eval_fn(test_loader, model, loss_fn, DEVICE, epoch)




if __name__ == "__main__":
    main()