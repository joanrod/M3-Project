import torch
from torchvision.transforms import RandomHorizontalFlip, RandomRotation
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from model import NetSquared
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter  # tensorboard --logdir=folder logs
from datetime import datetime
import wandb

wandb.init(project="NetSquared", entity="celulaeucariota")
wandb.config = {
  "learning_rate": 0.001,
  "epochs": 100,
  "batch_size": 128
}

# Hyperparameters etc.
LEARNING_RATE = 0.01
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
NUM_EPOCHS = 200
NUM_WORKERS = 2
PIN_MEMORY = True       # To speed up the file transfer to CPU to GPU
LOAD_MODEL = False      # If True the stored weights will be laoded
ROOT_PATH = ""
TRAIN_IMG_DIR = ROOT_PATH + "MIT_split/train/"
TEST_IMG_DIR = ROOT_PATH + "MIT_split/test/"

# Timer
start_time = datetime.today().strftime('%d_%m_%Y_%H_%M_%S')

path_tensorboard = "tb"
writer = SummaryWriter(path_tensorboard)

model_id = "baseline"
def train_fn(loader, model, optimizer, loss_fn, scaler, device, epoch_num):
    """
    Function to train the model with one epoch
    :param loader: Dataloder, train dataloader
    :param model: object, model to train
    :param optimizer: optimizer object
    :param loss_fn: loss
    :param scaler: scaler object
    :param device: string ('cuda' or cpu')
    :param epoch_num: int number of epoch in which the model is going to be trained
    :return: float, float (accuracy, loss)
    """

    model.train()                       # Train mode

    loop = tqdm(loader, desc=f'EPOCH {epoch_num} TRAIN')  # Create the tqdm bar for visualizing the progress.

    correct = 0     # accumulated correct predictions
    total = 0       # accumulated total predictions
    acc_loss = 0    # accumulated loss

    # Loop to obtain the batches of images and labels
    for (data, targets) in loop:

        data = data.to(device=device)           # Batch of images to DEVICE, where the model is
        targets = targets.to(device=device)     # Batch of labels to DEVICE, where the model is

        optimizer.zero_grad()               # Initialize the gradients

        output = model(data)                # Output of the model (logits).
        loss = loss_fn(output, targets)     # Compute the loss between the output and the ground truth

        _, predictions = torch.max(output.data, 1)  # Obtain the classes with higher probability (predicted classes)

        total += data.size(0)                               # subtotal of the predictions
        correct += (predictions == targets).sum().item()    # subtotal of the correct predictions
        acc_loss += loss.item()                             # subtotal of the correct losses

        scaler.scale(loss).backward()   # compute the backward stage updating the weights of the model
        scaler.step(optimizer)          # using the Gradient Scaler
        scaler.update()

        loop.set_postfix(acc=correct/total, loss=loss.item())  # set current accuracy and loss

        # Tensorboard: the object writer will add the batch metrics to plot in real time
    write_to_tensorboard(model_id, 'train', acc_loss/total, correct/total, epoch_num)

    return correct/total, acc_loss/total
def write_to_tensorboard(model_name, phase, epoch_loss, epoch_acc, counter):
    writer.add_scalars(model_name + "/loss", {phase: epoch_loss}, counter)
    writer.add_scalars(model_name + "/accuracy", {phase: epoch_acc}, counter)
    wandb.log({"loss": epoch_loss})
    wandb.log({"accuracy": epoch_acc})


def eval_fn(loader, model, loss_fn, device, epoch_num):
    """
    Function to evaluate the model
    :param loader: Dataloader, test dataloader
    :param model: object, model to evaluate
    :param loss_fn: loss
    :param device: string ('cuda' or cpu')
    :param epoch_num: int number of epoch in which the model is going to be evaluated
    :return: float, float (accuracy, loss)
    """

    model.eval()                        # Put the model in evaluation mode

    loop = tqdm(loader, desc=f'EPOCH {epoch_num}  TEST')  # Create the tqdm bar for visualizing the progress.

    correct = 0     # accumulated correct predictions
    total = 0       # accumulated total predictions
    acc_loss = 0    # accumulated loss

    # Do not compute the gradients to go faster as we are not in training
    with torch.no_grad():
        # Iterate through the batches
        for (data, targets) in loop:

            data = data.to(device=device)           # Batch of images to DEVICE, where the model is
            targets = targets.to(device=device)     # Batch of labels to DEVICE, where the model is

            output = model(data)                    # Output of the model (logits).
            loss = loss_fn(output, targets)         # Compute the loss between the output and the ground truth

            _, predictions = torch.max(output.data, 1)  # Obtain the classes with higher probability

            total += data.size(0)                               # subtotal of the predictions
            correct += (predictions == targets).sum().item()    # subtotal of the correct predictions
            acc_loss += loss.item()                             # subtotal of the correct losses

            loop.set_postfix(acc=correct/total, loss=loss.item())    # set current accuracy and loss
    write_to_tensorboard(model_id, 'train', acc_loss / total, correct / total, epoch_num)
    return correct/total, loss.item()

def main():
    # Find which device is used
    if torch.cuda.is_available() and DEVICE=="cuda":
        print(f'Training the model in {torch.cuda.get_device_name(torch.cuda.current_device())}')
    else:
        print('CAREFUL!! Training the model with CPU')

    # Transform the output of the Dataset object into Tensor

    transform = transforms.Compose(
        [
            RandomHorizontalFlip(),
            RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    model = NetSquared()    # Create the model
    model.to(DEVICE)
    # Load the stored weight if LOAD_MODEL==True
    if LOAD_MODEL:
        checkpoint = torch.load("NetSquared_checkpoint.pth.tar")
        model.load_state_dict(checkpoint["state_dict"])
        print('Model loaded...')

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)    # Initialize the model
    loss_fn = nn.CrossEntropyLoss()                                 # Initialize the loss
    scaler = torch.cuda.amp.GradScaler()                            # Initialize the Scaler

    # SCALER EXPLANATION: If the forward pass for a particular operation has float16 inputs, the backward pass will
    # produce float16 gradients, which with small magnitudes may not be representable for float16. Gradient Scaling
    # multiply the the network "losses" by a scale factor and invokes a backward pass on the scaled loss(es)
    # https://pytorch.org/docs/stable/amp.html#torch.cuda.amp.GradScaler

    train_dataset = ImageFolder(TRAIN_IMG_DIR, transform=transform)    # Create the train dataset
    test_dataset = ImageFolder(TEST_IMG_DIR, transform=transform)      # Create the test dataset

    print("Training Data: ", train_dataset.__len__(), "images")
    print("Validation Data: ", test_dataset.__len__(), "images")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, pin_memory=True, shuffle=True)  # Create the train dataloader
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, pin_memory=True)                  # Create the test dataloader

    train_metrics = {'accuracy': [], 'loss': []}
    test_metrics = {'accuracy': [], 'loss': []}

    print("")
    print("----------- MODEL: {} --------------".format(model_id))
    print("----------- TRAINING START {} --------------".format(start_time))
    print("")

    # Iterate as many times as NUM_EPOCHS
    for epoch in range(NUM_EPOCHS):

        acc, loss = train_fn(train_loader, model, optimizer, loss_fn, scaler, DEVICE, epoch)    # Train the model
        train_metrics['accuracy'].append(acc)   # Append train accuracy
        train_metrics['loss'].append(loss)      # Append train accuracy

        acc, loss = eval_fn(test_loader, model, loss_fn, DEVICE, epoch)                         # Validate the model in the test set
        test_metrics['accuracy'].append(acc)    # Append test accuracy
        test_metrics['loss'].append(loss)       # Append test accuracy

        # If the validation accuracy is the best one so far, save the model
        if (acc == max(test_metrics['accuracy'])) and (epoch > 0):
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            torch.save(checkpoint, "NetSquared_checkpoint.pth.tar")
            print('\nModel saved...\n')

    # Plot accuracies and losses
    plt.subplot(121)
    plt.plot(train_metrics['accuracy'])
    plt.plot(test_metrics['accuracy'])
    plt.subplot(122)
    plt.plot(train_metrics['loss'])
    plt.plot(test_metrics['loss'])
    plt.show()

if __name__ == "__main__":
    main()
