import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from model import NetSquared
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

def train_fn(loader, model, optimizer, loss_fn, scaler):
    """
    Function to train the model
    :param loader:
    :param model:
    :param optimizer:
    :param loss_fn:
    :param scaler:
    :return:
    """

    loop = tqdm(loader)

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        print(data.shape)
        #targets = targets

        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())


def main():
    model = NetSquared().to(DEVICE)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Create Dataloaders
    train_loader, test_loader = get_loaders(
        TRAIN_IMG_DIR,
        TEST_IMG_DIR,
        BATCH_SIZE,
        NUM_WORKERS,
        PIN_MEMORY,
    )

    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(NUM_EPOCHS):
        train_fn(train_loader, model, optimizer, loss_fn, scaler)


if __name__ == "__main__":
    main()