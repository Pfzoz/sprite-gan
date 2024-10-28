from torch import nn
from torch.nn.modules import Conv2d
from torch.nn.modules.activation import Tanh

class Generator(nn.Module):
    def __init__(self, ngpu) -> None:
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.model = nn.Sequential(
            # x105
            nn.Linear(105, 48),
            nn.ReLU(inplace=True),
            # x48
            nn.Unflatten(dim=0, unflattened_size=(3, 4, 4)),
            # 4x4x3
            nn.ConvTranspose2d(3, 64, 4, 2, 1, bias=False),
            nn.ReLU(inplace=True),
            # 8x8x3
            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),
            nn.ReLU(inplace=True),
            # 16x16x3
            nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)
