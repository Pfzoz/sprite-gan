### IMPORTS ###

import time
from torchvision.transforms.functional import to_pil_image

import torch
from torch.utils.data import DataLoader
from torch import nn
from torch import optim
import numpy as np

import os
from dotenv import load_dotenv

from matplotlib import pyplot as plt

from analysis.scores import get_fid
from data.dataset import LabeledDataset
from models import Generator, Discriminator

from data.preprocessing import dataset_transform

from sys import argv

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
Y_LENGTH = 1
N_CLASSES = 5
G_INPUT_LENGTH = 105
MODEL_SAVING_PATH="results"

if not os.path.exists(MODEL_SAVING_PATH):
    os.mkdir(MODEL_SAVING_PATH)

# EXTERNAL HYPERPARAMETERS

BATCH_SIZE=128
NUM_WORKERS=2
LEARNING_RATE=0.003
EPOCHS=50000
CHECKPOINT_INTERVAL=20

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
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=dataset.sampler, num_workers=NUM_WORKERS, drop_last=True)

# Intialization

generator = Generator(1, N_CLASSES, Z_VECTOR_LENGTH).to(device)
discriminator = Discriminator(1, N_CLASSES, Z_VECTOR_LENGTH).to(device)

generator.apply(weights_init)
discriminator.apply(weights_init)

loss_fn = nn.BCELoss()

fixed_noise = torch.randn(5, Z_VECTOR_LENGTH, device=device)
y_classes = torch.Tensor([i for i in range(5)]).long().to(device)

optimizer_D = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE/10, betas=(0.5, 0.999))
optimizer_G = optim.Adam(generator.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))

if len(argv) != 2:
    print("No name informed for the model. Exiting.")
    exit()

final_path = argv[1]
i = 0

if not os.path.exists(f"{MODEL_SAVING_PATH}/{final_path}"):
    os.mkdir(f"{MODEL_SAVING_PATH}/{final_path}")

with open(f"{MODEL_SAVING_PATH}/{final_path}/info.txt", "w+") as info_file:
    info_file.write(f"Seed: {torch.seed()} Cuda seed: {torch.cuda.seed()}\n")
    info_file.write(f"Batch-size: {BATCH_SIZE}\n")
    info_file.write(f"Learning-rate: {LEARNING_RATE}\n")
    info_file.write("Generator:\n\n")
    info_file.write(str(generator))
    info_file.write("Discriminator\n\n")
    info_file.write(str(discriminator))

real_images_for_fid = []
fake_images_for_fid = []

generator_error = None
d_loss_real = None
d_loss_fake = None
d_loss = None
g_loss = None

def get_class(class_n):
    labels_v = ["Front-Character", "Monster/NPC", "Food/Prop", "Equippable/Item", "Side-Character"]
    labels_v = ["FC", "M", "F", "I", "SC"]
    return labels_v[class_n]

time_start = time.time()

inference_speed_count = 0
inference_speed_sum = 0

def make_checkpoint():
    checkpoint_path = f"{MODEL_SAVING_PATH}/{final_path}/checkpoint-{epoch}"
    os.mkdir(checkpoint_path)
    generator.eval()
    FID_SAMPLES_AMOUNT = 1000
    has_generated_fake_example: bool = False
    has_generated_real_example: bool = False

    real_class_count = [0] * 5
    fake_class_count = [0] * 5
    samples_per_class = FID_SAMPLES_AMOUNT // 5

    cycles = 5
    attempts = 0

    while sum(real_class_count) < FID_SAMPLES_AMOUNT and attempts < cycles:
        attempts += 1
        print("Attempt: ", attempts)
        for image_data, label_data in dataloader:
            batch_size = image_data.size(0)
            z = torch.randn(batch_size, Z_VECTOR_LENGTH, device=device)
            generated_images = generator(z, label_data.long().to(device)).detach().cpu()
            z.detach()
            denormalized_generated_images = (torch.clamp(generated_images, -1, 1) + 1) / 2

            for i, image in enumerate(image_data):
                class_id = label_data[i]
                denormalized_image = (torch.clamp(image, -1, 1) + 1) / 2

                if real_class_count[class_id] >= samples_per_class:
                    continue

                fake_img_pil = to_pil_image(denormalized_generated_images[i])
                img_pil = to_pil_image(denormalized_image)

                if not has_generated_fake_example:
                    plt.imshow(fake_img_pil)
                    plt.savefig(fname=f"{checkpoint_path}/fake_example.jpg")
                    plt.close()
                    has_generated_fake_example = True
                if not has_generated_real_example:
                    plt.imshow(to_pil_image(denormalized_image))
                    plt.savefig(fname=f"{checkpoint_path}/real_example.jpg")
                    plt.close()
                    has_generated_real_example = True

                real_images_for_fid.append(img_pil)
                fake_images_for_fid.append(fake_img_pil)
                real_class_count[class_id] += 1
                fake_class_count[class_id] += 1

    fid = get_fid(real_images_for_fid, fake_images_for_fid)
    real_images_for_fid.clear()
    fake_images_for_fid.clear()
    torch.save(generator.state_dict(), f"{checkpoint_path}/state_dict.pt")
    with open(f"{checkpoint_path}/info.txt", "w+") as info_file:
        info_file.write(f"Epoch: {epoch}\n")
        info_file.write(f"FID: {fid} with {FID_SAMPLES_AMOUNT} samples amount\n")
        info_file.write(f"Generator loss: {g_loss}\n")
        info_file.write(f"Discriminator loss (real): {d_loss_real}\n")
        info_file.write(f"Discriminator loss (fake): {d_loss_fake}\n")
        info_file.write(f"Discriminator loss (total): {d_loss}\n")
        info_file.write(f"Time began: {time_start / 1000}; Time passed: {time.time() - time_start / 1000}")
    generator.train()

for epoch in range(EPOCHS):
    if (epoch % 1 == 0):
        torch.save(generator.state_dict(), f"{MODEL_SAVING_PATH}/{final_path}/state_dict.pt")
        generator.eval()
        with torch.no_grad():
            generated_fixed = torch.clamp(generator(fixed_noise, y_classes), -1, 1).detach().cpu()
            start_dummy_gen = time.time()
            dummy_gen = generator(fixed_noise[:1], y_classes[:1])
            inference_time = time.time() - start_dummy_gen
            inference_speed_sum += inference_time
            inference_speed_count += 1
            with open("info-inference.txt", "a+") as inference_file:
                inference_file.write(f"Epoch: {epoch} Inference time: {inference_time} Average inference time: {inference_speed_sum / inference_speed_count}s")
            denormalized_fixed = (generated_fixed + 1) / 2
            pillow_images = [to_pil_image(image) for image in denormalized_fixed[:5]]
            plt.clf()
            fig, axes = plt.subplots(1, 5)
            plt.subplots_adjust(wspace=0.3)
            for i, ax in enumerate(axes.flat):
                ax.imshow(np.array(pillow_images[i]))
                ax.set_title(f"Class {get_class(y_classes[i])}")
            plt.savefig(fname=f"{MODEL_SAVING_PATH}/{final_path}/e-{epoch}-pillow.jpg")
            plt.close()
        generator.train()
    if (epoch % CHECKPOINT_INTERVAL == 0):
        make_checkpoint()
    for i, (image_data, label_data) in enumerate(dataloader):
        # Optimize D
        image_data_gpu = image_data.to(device)
        label_data_gpu = (label_data).long().to(device)

        # Real
        discriminator.zero_grad()
        output_real = discriminator(image_data_gpu, label_data_gpu)

        # Fake
        noise = torch.randn(BATCH_SIZE, Z_VECTOR_LENGTH, dtype=torch.float).to(device)
        generated_samples = generator(noise, label_data_gpu)
        output_fake = discriminator(generated_samples.detach(), label_data_gpu)

        d_loss_real = torch.mean(torch.relu(1 - output_real))
        d_loss_fake = torch.mean(torch.relu(1 + output_fake))
        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        optimizer_D.step()

        # Optimize Generator

        generator.zero_grad()
        # labels_tensor.fill_(1)
        output = discriminator(generated_samples, label_data_gpu)

        g_loss = -torch.mean(output)
        g_loss.backward()
        optimizer_G.step()

        print(f"Epoch: {epoch} - Batch: {i}; G-Loss: {g_loss} D-Loss (Real): {d_loss};")
