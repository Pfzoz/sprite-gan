from torch.utils.data.dataset import Dataset

class LabeledDataset(Dataset):
    def __init__(self, images, labels, transform):
        self.images = [transform(image) for image in images]
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        return image, label
