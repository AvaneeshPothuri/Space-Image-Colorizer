import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import torch
import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from models.colorize_unet import UNetColorize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNetColorize().to(device)
model.load_state_dict(torch.load('models/saved_models/colorize_epoch_5.pth', map_location=device))
model.eval()

class SpaceColorDataset:
    def __init__(self, data_dir, split='train'):
        self.gray_dir = os.path.join(data_dir, split, 'gray')
        self.color_dir = os.path.join(data_dir, split, 'color')
        self.gray_files = sorted(os.listdir(self.gray_dir))
        self.color_files = sorted(os.listdir(self.color_dir))
    
    def __len__(self):
        return len(self.gray_files)
    
    def __getitem__(self, idx):
        gray_path = os.path.join(self.gray_dir, self.gray_files[idx])
        color_path = os.path.join(self.color_dir, self.color_files[idx])
        gray = cv2.imread(gray_path, cv2.IMREAD_GRAYSCALE)
        color = cv2.imread(color_path, cv2.IMREAD_COLOR)
        gray = cv2.resize(gray, (256, 256)) / 255.0
        color = cv2.resize(color, (256, 256))
        # Convert color to LAB and extract a,b channels
        color_lab = cv2.cvtColor(color, cv2.COLOR_BGR2LAB)
        gray = gray[None, ...]  # Add channel dim
        ab = color_lab[..., 1:].transpose(2, 0, 1) / 128.0 - 1.0  # Normalize a,b to [-1,1]
        gray = torch.FloatTensor(gray)
        ab = torch.FloatTensor(ab)
        return gray, ab, color

def evaluate_model(model, data_dir, split='val'):
    dataset = SpaceColorDataset(data_dir, split)
    psnr_list = []
    ssim_list = []
    
    for idx in range(len(dataset)):
        gray, ab, color = dataset[idx]
        gray = gray.to(device)
        with torch.no_grad():
            pred_ab = model(gray.unsqueeze(0)).cpu().numpy()[0]
        
        # Reconstruct LAB image
        pred_ab = (pred_ab + 1) * 128  # Denormalize to [0,255]
        pred_ab = pred_ab.transpose(1, 2, 0).astype(np.uint8)
        L = (gray.cpu().numpy()[0] * 255).astype(np.uint8)
        LAB_pred = np.zeros((256, 256, 3), dtype=np.uint8)
        LAB_pred[:, :, 0] = L
        LAB_pred[:, :, 1:] = pred_ab
        
        # Convert LAB to BGR
        colorized_bgr = cv2.cvtColor(LAB_pred, cv2.COLOR_LAB2BGR)
        colorized_rgb = cv2.cvtColor(colorized_bgr, cv2.COLOR_BGR2RGB)
        color_rgb = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
        
        # Compute PSNR and SSIM
        psnr_val = psnr(color_rgb, colorized_rgb)
        ssim_val = ssim(color_rgb, colorized_rgb, multichannel=True, channel_axis=2)
        psnr_list.append(psnr_val)
        ssim_list.append(ssim_val)
    
    avg_psnr = np.mean(psnr_list)
    avg_ssim = np.mean(ssim_list)
    return avg_psnr, avg_ssim

data_dir = 'data/processed'

# Evaluate on validation set
val_psnr, val_ssim = evaluate_model(model, data_dir, split='val')
print(f"Validation PSNR: {val_psnr:.2f} dB")
print(f"Validation SSIM: {val_ssim:.4f}")

# Evaluate on training set
train_psnr, train_ssim = evaluate_model(model, data_dir, split='train')
print(f"Training PSNR: {train_psnr:.2f} dB")
print(f"Training SSIM: {train_ssim:.4f}")