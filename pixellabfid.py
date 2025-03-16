import numpy as np
import os
import torch

from dotenv import load_dotenv
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms.functional import to_pil_image
from analysis.scores import get_fid
from data.dataset import LabeledDataset
from PIL import Image
from data.preprocessing import dataset_transform


LOCAL_ENV_LOADED = load_dotenv(".env")
IMAGES_PATH = os.getenv("IMAGES_NPY_PATH")
LABELS_PATH = os.getenv("LABELS_NPY_PATH")

if (not IMAGES_PATH or not LABELS_PATH):
    print("Empty ENV PATH variables.")
    exit()

images_np = np.load(IMAGES_PATH)
labels_np = np.load(LABELS_PATH)

pixel_lab_images = []

for path in os.listdir("pixellab-images"):
    pixel_lab_images.append(Image.open("pixellab-images/" + path))

dataset = LabeledDataset(images_np, labels_np, dataset_transform)
dataloader = DataLoader(dataset, batch_size=1000, sampler=dataset.sampler, num_workers=2, drop_last=True)

for images, labels in dataloader:
    denormalized_images = (torch.clamp(images, -1, 1) + 1) / 2
    fid = get_fid([to_pil_image(img) for img in denormalized_images], pixel_lab_images)

    print(fid)
    break
