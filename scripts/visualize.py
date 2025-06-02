import sys
import os
import torch

# Set up project root and device
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
from models.colorize_unet import UNetColorize

# Load the trained model
model_path = os.path.join(project_root, 'models', 'saved_models', 'colorize_epoch_5.pth')
model = UNetColorize().to(device)
try:
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

# Dataset paths
val_gray_dir = os.path.join(project_root, 'data', 'processed', 'val', 'gray')
val_color_dir = os.path.join(project_root, 'data', 'processed', 'val', 'color')

# List validation grayscale images
try:
    val_gray_files = sorted(f for f in os.listdir(val_gray_dir) if f.endswith(('.png', '.jpg', '.jpeg')))
    if not val_gray_files:
        raise FileNotFoundError("No valid images found in validation gray directory.")
except Exception as e:
    print(f"Error reading validation images: {e}")
    exit(1)

# Pick a random image
random_idx = random.randint(0, len(val_gray_files) - 1)
random_gray_path = os.path.join(val_gray_dir, val_gray_files[random_idx])
random_color_path = os.path.join(val_color_dir, val_gray_files[random_idx])

# Load and validate images
gray_img = cv2.imread(random_gray_path, cv2.IMREAD_GRAYSCALE)
if gray_img is None:
    print(f"Error: Could not read grayscale image at {random_gray_path}")
    exit(1)
color_img = cv2.imread(random_color_path, cv2.IMREAD_COLOR)
if color_img is None:
    print(f"Error: Could not read color image at {random_color_path}")
    exit(1)

# Preprocess grayscale image
gray_resized = cv2.resize(gray_img, (256, 256)) / 255.0
input_tensor = torch.FloatTensor(gray_resized[None, None, ...]).to(device)  # (1,1,256,256)

# Predict ab channels
with torch.no_grad():
    pred_ab = model(input_tensor).cpu().numpy()[0]  # (2,256,256)

# Postprocess prediction
pred_ab = (pred_ab + 1) * 128  # Denormalize to [0,255]
pred_ab = pred_ab.transpose(1, 2, 0).astype(np.uint8)  # (256,256,2)

# Prepare LAB image
L = (gray_resized * 255).astype(np.uint8)
LAB_pred = np.zeros((256, 256, 3), dtype=np.uint8)
LAB_pred[:, :, 0] = L
LAB_pred[:, :, 1:] = pred_ab

# Convert LAB to BGR
colorized_bgr = cv2.cvtColor(LAB_pred, cv2.COLOR_LAB2BGR)

# Convert BGR to RGB for matplotlib
colorized_rgb = cv2.cvtColor(colorized_bgr, cv2.COLOR_BGR2RGB)

# Prepare original color image for comparison
color_img_resized = cv2.resize(color_img, (256, 256))
color_img_rgb = cv2.cvtColor(color_img_resized, cv2.COLOR_BGR2RGB)

# Plot images
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.title('Input Grayscale')
plt.imshow(gray_resized, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title('Original Color')
plt.imshow(color_img_rgb)
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title('Colorized Output')
plt.imshow(colorized_rgb)
plt.axis('off')

plt.tight_layout()
plt.show()

print(f"Random image used: {val_gray_files[random_idx]}")