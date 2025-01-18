from posix import write
from genericpath import exists
### IMPORTS ###

import torch
import torchvision.utils as vutils
from torch.utils.data import DataLoader
from torch import nn
from torch import optim
import numpy as np

import os
from dotenv import load_dotenv

from matplotlib import pyplot as plt

from data.dataset import LabeledDataset
from models import Generator, Discriminator

import itertools
from data.preprocessing import dataset_transform

### PREPARATIONS ###

# Load .env
LOCAL_ENV = load_dotenv(".env")
IMAGES_PATH = os.getenv("IMAGES_NPY_PATH")
LABELS_PATH = os.getenv("LABELS_NPY_PATH")

if (not IMAGES_PATH or not LABELS_PATH):
    print("Empty ENV variables.")
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

### MAIN STUFF ###

# CONSTANTS

Z_VECTOR_LENGTH = 100
Y_LENGTH = 5
G_INPUT_LENGTH = 105
MODEL_SAVING_PATH="results"

if not os.path.exists(MODEL_SAVING_PATH):
    os.mkdir(MODEL_SAVING_PATH)

# EXTERNAL HYPERPARAMETERS

BATCH_SIZE=150
NUM_WORKERS=2
LEARNING_RATE=0.0003
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

dataset = LabeledDataset(images_np, labels_np, transform=dataset_transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

# Intialization

generator = Generator(1).to(device)
discriminator = Discriminator(1).to(device)

generator.apply(weights_init)
# print(generator)
discriminator.apply(weights_init)
# print(discriminator)

loss_fn = nn.BCELoss()

# Fixed noise for testing same input. 5 Images that we'll be generating so 5 input latent vectors
# Making vector of all 5 possible classes to generate one of each

fixed_noise = torch.randn(5, Z_VECTOR_LENGTH, device=device)
y_classes = torch.Tensor([y for y in set(itertools.permutations([5, 0, 0, 0, 0]))]).to(device)

optimizer_D = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
optimizer_G = optim.Adam(generator.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))

# Path for Fixed Noise Image

final_path = "model-0"
i = 0
while os.path.exists(f"{MODEL_SAVING_PATH}/{final_path}"):
    i += 1
    final_path = f"model-{i}"

os.mkdir(f"{MODEL_SAVING_PATH}/{final_path}")

with open(f"{MODEL_SAVING_PATH}/{final_path}/info.txt", "w+") as info_file:
    info_file.write(f"Seed: {torch.seed()} Cuda seed: {torch.cuda.seed()}\n")
    info_file.write("Generator:\n\n")
    info_file.write(str(generator))
    info_file.write("Discriminator\n\n")
    info_file.write(str(discriminator))

for epoch in range(EPOCHS):
    if (epoch % 1 == 0):
        generator.eval()
        with torch.no_grad():
            generated_fixed = generator(fixed_noise, y_classes).detach().cpu()
            plt.imshow(np.transpose(vutils.make_grid(generated_fixed[:5])))
            plt.savefig(fname=f"{MODEL_SAVING_PATH}/{final_path}/e-{epoch}.jpg")
            plt.close()
        generator.train()
    for i, (image_data, label_data) in enumerate(dataloader):

        # Optimize D
        image_data_gpu = image_data.to(device)
        label_data_gpu = (label_data * 5).to(device).float()

        # print( "Images shape:", image_data_gpu.shape, image_data_gpu.dtype)
        # print("Labels shape:", label_data_gpu.shape, label_data_gpu.dtype)

        # Real
        discriminator.zero_grad()
        labels_tensor = torch.full((BATCH_SIZE, 1), 1, device=device, dtype=torch.float).to(device)
        output = discriminator(image_data_gpu, label_data_gpu)
        discriminator_real_errors = loss_fn(output, labels_tensor)
        discriminator_real_errors.backward()
        # discriminator_x = output.mean().item()

        # Fake
        noise = torch.randn(BATCH_SIZE, Z_VECTOR_LENGTH, dtype=torch.float).to(device)
        generated_samples = generator(noise, label_data_gpu)
        labels_tensor.fill_(0)
        output = discriminator(generated_samples.detach(), label_data_gpu)
        discriminator_fake_errors = loss_fn(output, labels_tensor)
        discriminator_fake_errors.backward()
        # discriminator_generator_z1 = discriminator_fake_errors.mean().item()
        # discriminator_error = discriminator_generator_z1 + discriminator_x
        optimizer_D.step()

        # Optimize Generator

        generator.zero_grad()
        labels_tensor.fill_(1)
        output = discriminator(generated_samples, label_data_gpu)
        generator_error = loss_fn(output, labels_tensor)
        generator_error.backward()
        # discriminator_generator_z2 = output.mean().item()

        optimizer_G.step()
        print(f"Epoch: {epoch} - Batch: {i}; G-Loss: {generator_error} D-Loss: {discriminator_real_errors}")

torch.save(generator.state_dict(), f"{MODEL_SAVING_PATH}/{final_path}/state_dict.pt")

generator.eval()

with torch.no_grad():
    generated_fixed = generator(fixed_noise, y_classes).detach().cpu()
    # images = []
    # for image in generated_fixed:
    #     images.append(image.permute(1, 2, 0))
    plt.imshow(np.transpose(vutils.make_grid(generated_fixed[:5])))
    plt.show()
