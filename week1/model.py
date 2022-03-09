import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from collections import OrderedDict

class NetSquared(nn.Module):
    def __init__(self):
        super(NetSquared, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3,3)),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.BatchNorm2d(num_features=16),
            nn.Dropout(p=0.1),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3)),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(num_features=32),
            nn.Dropout(p=0.3),
            nn.Linear()
        )

    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":
    """
    Tests to see if there are errors
    """
    x = torch.randn((16, 3, 224, 224))      # 16 random images
    model = NetSquared()                    # model init
    preds = model(x)                        # model predictions
