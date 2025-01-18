import itertools
import torch
from sys import argv
import os
import torchvision.utils as vutils
from matplotlib import pyplot as plt
import numpy as np

from models.generator import Generator

if len(argv) < 2:
    print("Missing load path")

if not os.path.exists(argv[1]):
    print("Load path doesn't exist")

print("Searching for device.")
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Device: {device}")

model = Generator(1).to(device)

model.load_state_dict(torch.load(argv[1], weights_only=True))
model.eval()

command: str | None = None

y_classes = torch.Tensor([y for y in set(itertools.permutations([5, 0, 0, 0, 0]))]).to(device)

Z_VECTOR_LENGTH = 100

with torch.no_grad():
    while not command == 'q':
        command = input("'g' to generate, 'q' to quit\n")
        if command == 'g':
            fixed_noise = torch.randn(5, Z_VECTOR_LENGTH, device=device)
            result = model(fixed_noise, y_classes).detach().cpu()
            plt.imshow(np.transpose(vutils.make_grid(result[:5])))
            plt.show()
