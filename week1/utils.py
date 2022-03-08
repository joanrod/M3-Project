import torch
import torchvision
from dataset import MITSceneDataset
from torch.utils.data import DataLoader

def get_loaders(
        train_dir,
        test_dir,
        batch_size,
        num_workers=4,
        pin_memory=True,
):
    train_dataset = MITSceneDataset(
        image_dir=train_dir
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    test_dataset = MITSceneDataset(
        image_dir=test_dir
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return train_loader, test_loader

