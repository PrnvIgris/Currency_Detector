# train.py
import os
import json
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Config
NUM_CLASSES = 7
EPOCHS = 25
BATCH_SIZE = 16
LR = 0.001
VAL_SPLIT = 0.2
TRAIN_PATH = 'dataset/train'
MODEL_PATH = 'models/resnet_currency.pth'
CLASS_JSON = 'models/classes.json'

# Transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(0.3, 0.3, 0.3, 0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Dataset
full_dataset = datasets.ImageFolder(TRAIN_PATH, transform=transform)
class_names = full_dataset.classes
train_size = int((1 - VAL_SPLIT) * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# Save class map
os.makedirs("models", exist_ok=True)
with open(CLASS_JSON, 'w') as f:
    json.dump(class_names, f)

# Model
model = models.resnet18(pretrained=True)
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, NUM_CLASSES)
)
model.to(device)

# Loss & Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# Train Loop
train_losses, val_losses, val_accuracies = [], [], []
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for imgs, lbls in train_loader:
        imgs, lbls = imgs.to(device), lbls.to(device)
        optimizer.zero_grad()
        out = model(imgs)
        loss = criterion(out, lbls)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    avg_train = running_loss / len(train_loader)
    train_losses.append(avg_train)

    model.eval()
    val_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for imgs, lbls in val_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            out = model(imgs)
            loss = criterion(out, lbls)
            val_loss += loss.item()
            _, pred = torch.max(out, 1)
            total += lbls.size(0)
            correct += (pred == lbls).sum().item()

    avg_val = val_loss / len(val_loader)
    acc = correct / total
    val_losses.append(avg_val)
    val_accuracies.append(acc)
    scheduler.step()

    print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f} | Val Acc: {acc:.4f}")

# Save Model
torch.save(model.state_dict(), MODEL_PATH)
print("\n✅ Model trained and saved!")

# Plot
plt.figure(figsize=(10,5))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.plot(val_accuracies, label='Val Acc')
plt.legend()
plt.grid()
plt.title("Training Progress")
plt.savefig("training_log.png")
plt.show()
