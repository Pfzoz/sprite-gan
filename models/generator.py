import torch
from torch import nn

class Generator(nn.Module):
    def __init__(self, ngpu, num_classes: int, latent_dim: int, embed_dim: int = 32) -> None:
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        self.embed_dim = embed_dim

        self.label_embedding = nn.Embedding(num_classes, embed_dim)

        linear_input = latent_dim + embed_dim

        linear_output = 16 * 16 * 256

        self.model = nn.Sequential(
            # x105
            nn.Linear(linear_input, linear_output, dtype=torch.float),
            nn.BatchNorm1d(linear_output),
            nn.LeakyReLU(),
            nn.Unflatten(dim=1, unflattened_size=(256, 16, 16)),

            # 128x1x1
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            # 128x5x5
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            # 128x13x13
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            #
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1, bias=False),
            # 128x16x16
            nn.Tanh()
        )

    def forward(self, x, y):
        y_embed = self.label_embedding(y).squeeze(1)
        x = torch.cat((x, y_embed), dim=1)
        output = self.model(x)
        return output
