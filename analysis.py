import numpy as np

import os
from dotenv import load_dotenv

from matplotlib import pyplot as plt

# Load .env
LOCAL_ENV = load_dotenv(".env")
IMAGES_PATH = os.getenv("IMAGES_NPY_PATH")
LABELS_PATH = os.getenv("LABELS_NPY_PATH")

if (not IMAGES_PATH or not LABELS_PATH):
    print("Empty ENV variables.")
    exit()

def translate_label(label: np.ndarray) -> str:
    return "Front Character" if label[0] else "Monster/Animal" if label[1] else "Food" if label[2] else "Item" if label[3] else "Side Character"

images_np = np.load(IMAGES_PATH)
labels_np = np.load(LABELS_PATH)

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
    plt.text(i, values[i] + 0.5, values[i], ha='center', va='bottom', rotation=0)


plt.show()

del characters_np
del food_np
del item_np
del characeters_side_view_np
del monster_animal_np
