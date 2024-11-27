from torchvision.transforms import transforms

dataset_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0, 0, 0], std=[1, 1, 1]),
])
