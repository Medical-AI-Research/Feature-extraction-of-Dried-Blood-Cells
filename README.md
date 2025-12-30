# Liver Biopsy Image Classification (Research)

## Description
Deep learning–based classification of liver biopsy images into five pathological classes.
Research-only implementation.

Classes: Healthy | Inflammation | Steatosis | Ballooning | Fibrosis

---

## Environment
Python >= 3.9

pip install torch torchvision timm numpy pandas matplotlib scikit-learn grad-cam

---

## Dataset Structure
Liver_Biopsies/
├── Healthy/
├── Inflammation/
├── Steatosis/
├── Ballooning/
└── Fibrosis/

---

## Notebook
Untitled11.ipynb

---

## Dataset Loading
from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

dataset = datasets.ImageFolder(root=dataset_path, transform=transform)

---

## Model
import timm
import torch.nn as nn

model = timm.create_model(
    "regnety_016",
    pretrained=True,
    num_classes=5
)

---

## Training
import torch

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(num_epochs):
    model.train()
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

---

## Evaluation
from sklearn.metrics import classification_report

model.eval()
y_true, y_pred = [], []

with torch.no_grad():
    for images, labels in val_loader:
        outputs = model(images)
        preds = outputs.argmax(dim=1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

print(classification_report(y_true, y_pred))

---

## Explainability (Grad-CAM)
from grad_cam import GradCAM

cam = GradCAM(model=model, target_layer="blocks.3")
heatmap = cam(input_tensor, target_class)

---

## Outputs
outputs/
├── trained_model.pth
├── confusion_matrix.png
├── gradcam_samples/
└── metrics.json

---

## License
Apache License 2.0

---

## Disclaimer
For research and educational use only. Not for clinical use.

---

## Author
Kethan — Biomedical AI / Medical Imaging Research
