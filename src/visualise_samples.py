import os
import cv2
import matplotlib.pyplot as plt
import random

IMG_SIZE = 224
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/COVID-19_Radiography_Dataset"))
CATEGORIES = ["COVID", "Normal", "Viral Pneumonia", "Lung_Opacity"]
label_map = {category: idx for idx, category in enumerate(CATEGORIES)}

# Collect image paths
samples = []
for category in CATEGORIES:
    folder = os.path.join(DATA_DIR, category, "images")
    label = label_map[category]
    for f in os.listdir(folder):
        if f.lower().endswith(('.png', '.jpg', '.jpeg')):
            samples.append((os.path.join(folder, f), label))

# Pick 8 samples at random
subset = random.sample(samples, 8)

# Plot
plt.figure(figsize=(16, 8))
for i, (img_path, label_idx) in enumerate(subset):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    label = [k for k, v in label_map.items() if v == label_idx][0]

    plt.subplot(2, 4, i + 1)
    plt.imshow(img, cmap='gray')
    plt.title(f"Label: {label}")
    plt.axis('off')

plt.tight_layout()
plt.show()
