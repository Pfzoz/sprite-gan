import itertools
import torch
from sys import argv
import os
from torchvision.transforms.functional import to_pil_image
from matplotlib import pyplot as plt
import numpy as np

from data.dataset import get_class
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

Z_VECTOR_LENGTH = 100
N_CLASSES = 5

model = Generator(1, N_CLASSES, Z_VECTOR_LENGTH).to(device)

model.load_state_dict(torch.load(argv[1], weights_only=True))
model.eval()

command: str | None = None

y_classes = torch.Tensor([i for i in range(5)]).long().to(device)

with torch.no_grad():
    while not command == 'q':
        command = input("'g' to generate, 'q' to quit\n")
        if command == 'g':
            fixed_noise = torch.randn(5, Z_VECTOR_LENGTH, device=device)
            result = (torch.clamp(model(fixed_noise, y_classes).detach().cpu(), -1, 1) + 1) / 2
            pillow_images = [to_pil_image(image) for image in result[:5]]
            fig, axes = plt.subplots(1, 5)
            plt.subplots_adjust(wspace=0.3)
            for i, ax in enumerate(axes.flat):
                ax.imshow(np.array(pillow_images[i]))
                ax.set_title(f"Class {get_class(y_classes[i])}")
            plt.show()
