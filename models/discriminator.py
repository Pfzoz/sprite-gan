import torch
from torch import nn

class Discriminator(nn.Module):
    def __init__(self, ngpu, num_classes: int, embed_dim: int = 64):
        super(Discriminator, self).__init__()

        self.num_classes = num_classes
        self.embed_dim = embed_dim

        self.convolutional_model = nn.Sequential(
            # 3x16x16
            nn.utils.spectral_norm(nn.Conv2d(3, 16, 4, 2, 1, bias=False)),
            nn.LeakyReLU(),
            # 16x8x8
            nn.utils.spectral_norm(nn.Conv2d(16, 32, 4, 2, 1, bias=False)),
            nn.LeakyReLU(),
            # 32x4x4
            nn.utils.spectral_norm(nn.Conv2d(32, 64, 4, 2, 1, bias=False)),
            nn.LeakyReLU(),
            # 64x2x2
            nn.Flatten()
        )

        self.label_embedding = nn.Embedding(num_classes, embed_dim)

        self.final_dense_layer = nn.Sequential(
            nn.Linear(256 + embed_dim, 1, dtype=torch.float),
        )

    def forward(self, x, y):
        x = self.convolutional_model(x)

        y_embed = self.label_embedding(y).squeeze(1)
        x = torch.cat((x, y_embed), dim=1)

        x = self.final_dense_layer(x)
        return x
