import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from collections import OrderedDict

class NetSquared(nn.Module):
    def __init__(self):
        super(NetSquared, self).__init__()

        self.relu = nn.ReLU()
        self.maxp = nn.MaxPool2d(kernel_size=2)

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3,3))
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3))

        self.bnorm1 = nn.BatchNorm2d(num_features=32)
        self.bnorm2 = nn.BatchNorm2d(num_features=64)

        self.dropout1 = nn.Dropout2d(p=0.1)
        self.dropout2 = nn.Dropout2d(p=0.3)

        self.linear = nn.Linear(in_features=186624, out_features=8)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxp(x)
        x = self.bnorm1(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxp(x)
        x = self.bnorm2(x)

        x = torch.squeeze(x)
        x = torch.flatten(x,1)
        x = self.linear(x)

        return x

if __name__ == "__main__":
    """
    Tests to see if there are errors
    """
    x = torch.randn((16, 3, 224, 224))      # 16 random images
    model = NetSquared()                    # model init
    preds = model(x)                        # model predictions
