import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from preprocess import test_loader, label_map
from cnn_model import CovidCNN
import os

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CovidCNN(num_classes=4).to(device)

model_path = os.path.join(os.path.dirname(__file__), "covid_cnn.pth")
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Class labels
idx_to_label = {v: k for k, v in label_map.items()}

# Collect predictions
y_true = []
y_pred = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())

# Print report
print("\n📊 Classification Report:")
print(classification_report(y_true, y_pred, target_names=[idx_to_label[i] for i in range(4)]))

# Print confusion matrix
print("🧩 Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))
