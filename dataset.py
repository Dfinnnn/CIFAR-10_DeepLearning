#Custom CIFAR-10 dataset loader using images folder and trainLabels.csv
import os
# torch must import before pandas on Windows or native DLLs can conflict (WinError 1114).
import torch
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image

class CIFAR10FolderDataset(Dataset):
    def __init__(self, images_dir, labels_csv, transform=None):
        """
        images_dir: path to image folder (e.g. 'train/')
        labels_csv: path to labels csv (e.g. 'trainLabels.csv')
        transform: torchvision transforms to apply to images
        """
        self.images_dir = images_dir
        self.transform = transform

        # Load CSV
        df = pd.read_csv(labels_csv)
        self.image_ids = df['id'].astype(str).tolist()
        self.labels = df['label'].tolist()
        
        # Map label names to class indices
        self.label_to_idx = {label: idx for idx, label in enumerate(sorted(set(self.labels)))}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}
        self.numeric_labels = [self.label_to_idx[label] for label in self.labels]

    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_path = os.path.join(self.images_dir, f"{image_id}.png")
        image = Image.open(image_path).convert("RGB")
        label = self.numeric_labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(label)


def load_idx_to_label(labels_csv):
    """Same label order as CIFAR10FolderDataset (sorted unique names from CSV)."""
    df = pd.read_csv(labels_csv)
    labels = df["label"].tolist()
    label_to_idx = {label: idx for idx, label in enumerate(sorted(set(labels)))}
    return {idx: label for label, idx in label_to_idx.items()}


class CIFAR10TestDataset(Dataset):
    """Unlabeled test images: folder of `{id}.png` only (no labels CSV)."""

    def __init__(self, images_dir, transform=None):
        self.images_dir = images_dir
        self.transform = transform
        self.image_ids = []
        for name in os.listdir(images_dir):
            if not name.lower().endswith(".png"):
                continue
            stem = os.path.splitext(name)[0]
            self.image_ids.append(stem)
        self.image_ids.sort(key=lambda x: int(x))

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        path = os.path.join(self.images_dir, f"{image_id}.png")
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, image_id


def get_dataloader(images_dir, labels_csv, batch_size=64, shuffle=True, num_workers=0):
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader

    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                             std=[0.2023, 0.1994, 0.2010])
    ])

    dataset = CIFAR10FolderDataset(images_dir, labels_csv, transform=transform)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )
    return dataloader


def get_test_dataloader(images_dir, batch_size=64, num_workers=0):
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465],
                std=[0.2023, 0.1994, 0.2010],
            ),
        ]
    )
    dataset = CIFAR10TestDataset(images_dir, transform=transform)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
