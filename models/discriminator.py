import torch
from torch import nn
from torch.nn.modules.activation import LeakyReLU

class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.convolutional_model = nn.Sequential(
            # 3x16x16
            nn.utils.spectral_norm(nn.Conv2d(3, 16, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(inplace=True),
            # 32x8x8
            nn.utils.spectral_norm(nn.Conv2d(16, 32, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),
            #
            nn.utils.spectral_norm(nn.Conv2d(32, 64, 3, 2, 1, bias=False)),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            #
            nn.utils.spectral_norm(nn.Conv2d(64, 128, 3, 2, 1, bias=False)),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),
            # 128x2x2
            nn.Flatten()
        )
        # self.dense_layer = nn.Sequential(
        #     nn.Linear(133, 25, dtype=torch.float),
        #     nn.LayerNorm(25),
        #     nn.LeakyReLU(inplace=True),
        # )
        self.final_dense_layer = nn.Sequential(
            nn.Linear(133, 1, dtype=torch.float),
            nn.Sigmoid()
        )

    def forward(self, x, y):
        x = self.convolutional_model(x)
        x = torch.cat((x, y), dim=1)
        # x = self.dense_layer(x)
        # x = torch.cat((x, y), dim=1)

        x = self.final_dense_layer(x)
        return x
