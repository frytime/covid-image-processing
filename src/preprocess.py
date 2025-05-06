import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, random_split, DataLoader
from torchvision import transforms

# Configuration
IMG_SIZE = 224
BATCH_SIZE = 32
# Relative path based on script location
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/COVID-19_Radiography_Dataset"))
CATEGORIES = ["COVID", "Normal", "Viral Pneumonia", "Lung_Opacity"]
label_map = {category: idx for idx, category in enumerate(CATEGORIES)}

# Custom PyTorch dataset
class CovidXRayDataset(Dataset):
    def __init__(self, data_dir, categories, transform=None):
        self.data = []
        self.transform = transform
        for category in categories:
            folder = os.path.join(data_dir, category, "images")  # all use "images/" now
            label = label_map[category]
            for img_file in os.listdir(folder):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(folder, img_file)
                    self.data.append((img_path, label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        image = np.stack([image] * 3, axis=-1)  # convert to 3-channel
        if self.transform:
            image = self.transform(image)
        return image, label

# Image preprocessing pipeline
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)
])

# Load dataset
dataset = CovidXRayDataset(DATA_DIR, CATEGORIES, transform=transform)

# Split dataset
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Display stats
print("✅ Dataset loaded successfully!")
print(f"Total samples: {len(dataset)}")
print(f"Training samples: {len(train_dataset)}")
print(f"Testing samples: {len(test_dataset)}")
print(f"Label mapping: {label_map}")
