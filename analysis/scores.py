from typing import Any
import numpy as np
from scipy.linalg import sqrtm
import torch
from torch.nn import Module
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset
from torchvision.models import inception_v3
from torch.utils.data import DataLoader
from torchvision import transforms

class NormalDataset(Dataset):
    def __init__(self, images, transform):
        self.images = [transform(image) for image in images]
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]

        return image

def compute_statistics(features):
    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)
    return mu, sigma

def calculate_fid(mu1, sigma1, mu2, sigma2):
    diff = mu1 - mu2
    covmean: Any = sqrtm(sigma1.dot(sigma2))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return np.sum(diff**2) + np.trace(sigma1 + sigma2 - 2 * covmean)

def extract_features(dataloader, model, device):
    features = []
    with torch.no_grad():
        for images in dataloader:
            images = images.to(device)
            preds = model(images)
            features.append(preds.cpu().numpy())
    return np.concatenate(features, axis=0)

def get_fid(real_images, fake_images):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    inception_model = inception_v3(pretrained=True, transform_input=False).to(device)
    inception_model.fc = torch.nn.Identity()
    inception_model.eval()

    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    real_dataset = NormalDataset(real_images, transform)
    fake_dataset = NormalDataset(fake_images, transform)
    print("Real:", len(real_dataset))
    print("Fake:",len(fake_dataset))

    real_loader = DataLoader(real_dataset, batch_size=16, shuffle=False, num_workers=2, drop_last=True)
    gen_loader = DataLoader(fake_dataset, batch_size=16, shuffle=False, num_workers=2, drop_last=True)

    real_features = extract_features(real_loader, inception_model, device)
    gen_features = extract_features(gen_loader, inception_model, device)

    mu_real, sigma_real = compute_statistics(real_features)
    mu_gen, sigma_gen = compute_statistics(gen_features)

    return calculate_fid(mu_real, sigma_real, mu_gen, sigma_gen)
