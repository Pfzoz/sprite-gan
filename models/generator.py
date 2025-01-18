import torch
from torch import nn

class Generator(nn.Module):
    def __init__(self, ngpu) -> None:
        super(Generator, self).__init__()
        self.ngpu = ngpu
        linear_output = 28 * 28 * 128
        self.model = nn.Sequential(
            # x105

            nn.Linear(105, linear_output, dtype=torch.float),
            nn.LayerNorm(linear_output),
            nn.ReLU(inplace=True),
            nn.Unflatten(dim=1, unflattened_size=(128, 28, 28)),            # nn.Upsample(scale_factor=2),

            # 28x28x256
            nn.Conv2d(128, 64, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            # 25x25x256
            nn.Conv2d(64, 32, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),
            # 22x22x64
            nn.Conv2d(32, 16, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(inplace=True),
            # 19x19x32
            #16
            nn.Conv2d(16, 3, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x, y):

        x = torch.cat((x, y), dim=1)
        output = self.model(x)
        return output
