import torch
from torch import nn
from torch.nn.modules import Conv2d
from torch.nn.modules.activation import Tanh

class Generator(nn.Module):
    def __init__(self, ngpu) -> None:
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.model = nn.Sequential(
            # x105
            nn.Linear(105, 48, dtype=torch.float),
            nn.LayerNorm(48),
            nn.ReLU(inplace=True),
            # x48
            nn.Unflatten(dim=1, unflattened_size=(3, 4, 4)),
            # 4x4x3
            nn.ConvTranspose2d(3, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # 8x8x3
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),

            nn.ReLU(inplace=True),
            # 16x16x3
            nn.Conv2d(128,64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.Conv2d(64,3, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x, y):

        x = torch.cat((x, y), dim=1)

        return self.model(x)
