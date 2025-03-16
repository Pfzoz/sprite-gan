import pixellab

import numpy as np
import os
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from PIL import Image

LOCAL_ENV_LOADED = load_dotenv(".env")
PIXELLAB_KEY = os.getenv("PIXELLAB_KEY")
IMAGES_PATH = os.getenv("IMAGES_NPY_PATH")
LABELS_PATH = os.getenv("LABELS_NPY_PATH")

PIXELLAB_FOLDER_PATH="pixellab-images"

if (type(PIXELLAB_KEY) != str):
    print("PixelLab key not found")
    exit()

client = pixellab.Client(secret=PIXELLAB_KEY)

if (not IMAGES_PATH or not LABELS_PATH):
    print("Empty ENV PATH variables.")
    exit()

images_np = np.load(IMAGES_PATH)
labels_np = np.load(LABELS_PATH)

pil_init_image = Image.fromarray(images_np[0]).resize((32, 32), Image.Resampling.NEAREST)
init_image_label = np.where(labels_np[0] == 1)[0][0]

class_bins_count = [0] * 5
class_bins = [[] for _ in range(5)]

for image_np, label_np in zip(images_np, labels_np):
    label = np.where(label_np == 1)[0][0]
    if len(class_bins[label]) < (1000 / 5):
        class_bins[label].append(image_np)
        class_bins_count[label] += 1




for i, class_bin in enumerate(class_bins):
    for j, image_np in enumerate(class_bin):
        if i == 0 or (i == 1 and j < 82):
            continue
        tries = 0
        max_tries = 3
        success = False
        while not success and tries < max_tries:
            response = None
            try:
                response = client.generate_image_pixflux(
                    description="",
                    image_size=dict(width=32, height=32),
                    init_image=Image.fromarray(image_np).resize((32, 32), Image.Resampling.NEAREST),
                    init_image_strength=190
                )
                pil_img = response.image.pil_image()

                plt.imshow(pil_img)
                plt.savefig(f"{PIXELLAB_FOLDER_PATH}/class-{i}-sample-{j}.jpg")
                success = True
            except:
                print("Error occured, trying again", response)
                tries += 1
