import torch
from torch import nn
from torch.nn.modules import Conv2d
from torch.nn.modules.activation import LeakyReLU
from transformers.models.roformer.tokenization_roformer import List

class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.convolutional_model = nn.Sequential(
            # 3x16x16
            nn.Conv2d(3, 32, 3, 2, 1, bias=False),
            nn.Dropout(0.1),
            nn.LeakyReLU(0.2, inplace=True),
            # 32x8x8
            nn.Conv2d(32, 64, 3, 2, 1, bias=False),
            nn.Dropout(0.1),
            nn.LeakyReLU(0.2, inplace=True),
            # 64x4x4
            nn.Conv2d(64, 64, 3, 1, 1, bias=False),
            nn.Dropout(0.1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Flatten()
        )
        self.dense_model = nn.Sequential(
            # x(64*4*4+5)
            nn.Linear(64*4*4+5, 128, dtype=torch.float),
            nn.Dropout(0.1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 1, dtype=torch.float),
            nn.Sigmoid()
        )

    def forward(self, x, y):
        x = self.convolutional_model(x)

        x = torch.cat((x, y), dim=1)

        x = self.dense_model(x)
        return x
