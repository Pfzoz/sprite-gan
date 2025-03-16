import torch
import numpy as np
from torch.utils.data.dataset import Dataset
from torch.utils.data import WeightedRandomSampler

def get_class(class_n):
    labels_v = ["Front-Character", "Monster/NPC", "Food/Prop", "Equippable/Item", "Side-Character"]
    labels_v = ["FC", "M", "F", "I", "SC"]
    return labels_v[class_n]

def get_class_name(class_n):
    labels_v = ["Character", "Entity", "Object", "Item", "Character"]
    return labels_v[class_n]

CLASS_INTENSITY=1

class LabeledDataset(Dataset):
    def __init__(self, images, labels: list[np.ndarray], transform):
        self.images = [transform(image) for image in images]
        self.labels = torch.tensor([np.where(label_v == 1)[0] for label_v in labels]).long().view(-1)

        class_counts = torch.bincount(self.labels)
        class_weights = 1.0 / class_counts.float()

        sample_weights = class_weights[self.labels]

        self.sampler = WeightedRandomSampler(sample_weights.tolist(), num_samples=len(labels), replacement=True)


        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        return image, label
