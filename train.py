import warnings
warnings.filterwarnings("ignore")

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models

# Parameters
batch_size    = 32
learning_rate = 1e-3
num_epochs    = 5

# Paths
train_dir = 'output/train'
val_dir   = 'output/test'

# 0) Guard: ensure splits exist
missing = []
for d in (train_dir, val_dir):
    if not os.path.isdir(d):
        missing.append(d)
if missing:
    print("❌ The following directories are missing:")
    for d in missing:
        print(f"  • {d}")
    print("\nPlease run the Streamlit app’s “➡️ Split & Export Data” button first to create them.")
    sys.exit(1)

# 1) Data transforms
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]),
    'test': transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
    ]),
}

# 2) Datasets & loaders
datasets_dict = {
    'train': datasets.ImageFolder(train_dir, transform=data_transforms['train']),
    'test':  datasets.ImageFolder(val_dir,   transform=data_transforms['test'])
}
loaders = {
    'train': torch.utils.data.DataLoader(datasets_dict['train'],
                                         batch_size=batch_size, shuffle=True),
    'test':  torch.utils.data.DataLoader(datasets_dict['test'],
                                         batch_size=batch_size, shuffle=False)
}

# 3) Model setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(datasets_dict['train'].classes))
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 4) Training loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in loaders['train']:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    epoch_loss = running_loss / len(datasets_dict['train'])
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

print("🎉 Training complete.")

# 5) Evaluation on test set
model.eval()
correct, total = 0, 0
with torch.no_grad():
    for inputs, labels in loaders['test']:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)

acc = correct / total * 100
print(f"Test Accuracy: {acc:.2f}%")

# 6) Save model
torch.save(model.state_dict(), "resnet18_finetuned.pth")
print("Model weights saved to resnet18_finetuned.pth")
