# predict.py

import os
import sys
import torch
from torchvision import transforms, models
from PIL import Image

# 0) Figure out how many classes you have by listing your train folders
TRAIN_DIR = 'output/train'
classes = [
    entry.name
    for entry in os.scandir(TRAIN_DIR)
    if entry.is_dir()
]
NUM_CLASSES = len(classes)

# 1) Load your trained model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = models.resnet18(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, NUM_CLASSES)
model.load_state_dict(torch.load("resnet18_finetuned.pth", map_location=device))
model.to(device)
model.eval()

# 2) Preprocessing pipeline must match train.py
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# 3) Get the image path from the command line
if len(sys.argv) != 2:
    print(f"Usage: python {sys.argv[0]} path/to/image.jpg")
    sys.exit(1)

img_path = sys.argv[1]
if not os.path.isfile(img_path):
    print(f"❌ File not found: {img_path}")
    sys.exit(1)

# 4) Run inference
img = Image.open(img_path).convert('RGB')
inp = preprocess(img).unsqueeze(0).to(device)

with torch.no_grad():
    out = model(inp)
    _, pred = torch.max(out, 1)
    class_idx = pred.item()

# 5) Print result
print(f"Predicted class index: {class_idx}")
print(f"Predicted class name: {classes[class_idx]}")
