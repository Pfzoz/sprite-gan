### IMPORTS ###

import torch
import torchvision.utils as vutils
from torch.autograd import no_grad
from torch.cuda import is_available
from torch.utils.data import DataLoader, Dataset
from torch import nn
from torch import optim
import numpy as np

import os
from dotenv import load_dotenv

from matplotlib import pyplot as plt

from torchvision import transforms
from transformers import BlipProcessor, BlipForConditionalGeneration

from models import Generator, Discriminator

import itertools

### PREPARATIONS ###

# Load .env
LOCAL_ENV = load_dotenv(".env")
IMAGES_PATH = os.getenv("IMAGES_NPY_PATH")
LABELS_PATH = os.getenv("LABELS_NPY_PATH")

if (not IMAGES_PATH or not LABELS_PATH):
    print(f"Empty ENV variables.")
    exit()

images_np = np.load(IMAGES_PATH)
labels_np = np.load(LABELS_PATH)

# Get device
print("Searching for device.")
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Device: {device}")

### FUNCTIONS ###

def translate_label(label: np.ndarray) -> str:
    return "Front Character" if label[0] else "Monster/Animal" if label[1] else "Food" if label[2] else "Item" if label[3] else "Side Character"

class LabeledDataset(Dataset):
    def __init__(self, images, labels, transform):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        return transform(image), label

### DATA ANALYSIS ###

# Classes separation

characters_np = images_np[np.all(labels_np == np.array([1, 0, 0, 0, 0]), axis=1)]
monster_animal_np = images_np[np.all(labels_np == np.array([0, 1, 0, 0, 0]), axis=1)]
food_np = images_np[np.all(labels_np == np.array([0, 0, 1, 0, 0]), axis=1)]
item_np = images_np[np.all(labels_np == np.array([0, 0, 0, 1, 0]), axis=1)]
characeters_side_view_np = images_np[np.all(labels_np == np.array([0, 0, 0, 0, 1]), axis=1)]

# Showcase

plt.imshow(characters_np[0])
plt.show()
plt.imshow(monster_animal_np[0])
plt.show()
plt.imshow(food_np[0])
plt.show()
plt.imshow(item_np[0])
plt.show()
plt.imshow(characeters_side_view_np[0])
plt.show()

# Classes histogram

categories = ["Personagem", "Monstro/Animal", "Comida", "Item", "Personagem (Visão Lateral)"]
values = [characters_np.shape[0], monster_animal_np.shape[0], food_np.shape[0], item_np.shape[0], characeters_side_view_np.shape[0]]

plt.bar(categories, values)

for i, category in enumerate(categories):
    plt.text(i, values[i] + 0.5, category, ha='center', va='bottom', rotation=45)

plt.xticks([])
plt.show()

del characters_np
del food_np
del item_np
del characeters_side_view_np
del monster_animal_np

### MAIN STUFF ###

# Pre Processing

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0, 0, 0], std=[1, 1, 1]),
])

# CONSTANTS

Z_VECTOR_LENGTH = 100
Y_LENGTH = 5
G_INPUT_LENGTH = 105

# EXTERNAL HYPERPARAMETERS

BATCH_SIZE=600
NUM_WORKERS=2
LEARNING_RATE=0.002
EPOCHS=100

# Weight Initial

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 1)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# Load Data

dataset = LabeledDataset(images_np, labels_np, transform=transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

# Intialization

generator = Generator(1).to(device)
discriminator = Discriminator(1).to(device)

generator.apply(weights_init)
# print(generator)
discriminator.apply(weights_init)
# print(discriminator)

loss_fn = nn.BCELoss()

fixed_noise = torch.randn(5, Z_VECTOR_LENGTH, device=device)
y_classes = torch.Tensor([y for y in set(itertools.permutations([1, 0, 0, 0, 0]))]).to(device)

optimizer_D = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
optimizer_G = optim.Adam(generator.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))

for epoch in range(EPOCHS):
    print(f"Epoch: {epoch}")
    if (epoch % 1 == 0):
        with torch.no_grad():
            generated_fixed = generator(fixed_noise, y_classes).detach().cpu()
            # images = []
            # for image in generated_fixed:
            #     images.append(image.permute(1, 2, 0))
            plt.imshow(np.transpose(vutils.make_grid(generated_fixed[:5])))
            plt.show()
    for i, (image_data, label_data) in enumerate(dataloader):
        print(f"Batch: {i}")
        # Optimize D
        image_data_gpu = image_data.to(device)
        label_data_gpu = label_data.to(device).float()

        # print( "Images shape:", image_data_gpu.shape, image_data_gpu.dtype)
        # print("Labels shape:", label_data_gpu.shape, label_data_gpu.dtype)

        # Real
        discriminator.zero_grad()
        labels_tensor = torch.full((BATCH_SIZE, 1), 1, device=device, dtype=torch.float).to(device)
        output = discriminator(image_data_gpu, label_data_gpu)
        discriminator_real_errors = loss_fn(output, labels_tensor)
        discriminator_real_errors.backward()
        discriminator_x = output.mean().item()

        # Fake
        noise = torch.randn(BATCH_SIZE, Z_VECTOR_LENGTH, dtype=torch.float).to(device)
        generated_samples = generator(noise, label_data_gpu)
        labels_tensor.fill_(0)
        output = discriminator(generated_samples.detach(), label_data_gpu)
        discriminator_fake_errors = loss_fn(output, labels_tensor)
        discriminator_fake_errors.backward()
        discriminator_generator_z1 = discriminator_fake_errors.mean().item()
        discriminator_error = discriminator_generator_z1 + discriminator_x
        optimizer_D.step()

        # Optimize Generator

        generator.zero_grad()
        labels_tensor.fill_(1)
        output = discriminator(generated_samples, label_data_gpu)
        generator_error = loss_fn(output, labels_tensor)
        generator_error.backward()
        discriminator_generator_z2 = output.mean().item()

        optimizer_G.step()
