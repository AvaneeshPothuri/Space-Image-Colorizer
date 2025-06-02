import sys
import os
import torch
import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# Set up paths
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
from models.colorize_unet import UNetColorize

def validate_paths(data_dir, split):
    """Ensure required directories exist"""
    gray_dir = os.path.join(data_dir, split, 'gray')
    color_dir = os.path.join(data_dir, split, 'color')
    
    if not os.path.exists(gray_dir):
        raise FileNotFoundError(f"Gray directory not found: {gray_dir}")
    if not os.path.exists(color_dir):
        raise FileNotFoundError(f"Color directory not found: {color_dir}")
        
    return gray_dir, color_dir

class SpaceColorDataset:
    def __init__(self, data_dir, split='val'):
        self.gray_dir, self.color_dir = validate_paths(data_dir, split)
        
        # Get sorted file pairs
        self.gray_files = sorted(f for f in os.listdir(self.gray_dir) if f.endswith(('.png', '.jpg', '.jpeg')))
        self.color_files = sorted(f for f in os.listdir(self.color_dir) if f.endswith(('.png', '.jpg', '.jpeg')))
        
        # Verify file correspondence
        if len(self.gray_files) != len(self.color_files):
            raise ValueError(f"Mismatched files: {len(self.gray_files)} grayscale vs {len(self.color_files)} color")
            
        print(f"Found {len(self.gray_files)} images in {split} set")

    def __len__(self):
        return len(self.gray_files)

    def __getitem__(self, idx):
        gray_path = os.path.join(self.gray_dir, self.gray_files[idx])
        color_path = os.path.join(self.color_dir, self.color_files[idx])
        
        # Load and validate images
        gray = cv2.imread(gray_path, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise ValueError(f"Failed to read grayscale image: {gray_path}")
            
        color = cv2.imread(color_path, cv2.IMREAD_COLOR)
        if color is None:
            raise ValueError(f"Failed to read color image: {color_path}")

        # Preprocess
        gray = cv2.resize(gray, (256, 256)) / 255.0  # [0,1]
        color = cv2.resize(color, (256, 256))
        
        # Convert to LAB and extract AB channels
        lab = cv2.cvtColor(color, cv2.COLOR_BGR2LAB)
        ab = lab[..., 1:].transpose(2, 0, 1) / 128.0 - 1.0  # [-1,1]

        return (
            torch.FloatTensor(gray).unsqueeze(0),  # Add channel dim
            torch.FloatTensor(ab),
            color
        )

def evaluate_model(model, data_dir, split='val', device='cpu'):
    dataset = SpaceColorDataset(data_dir, split)
    psnr_values = []
    ssim_values = []
    
    model.eval()
    with torch.no_grad():
        for gray, ab_true, color_orig in dataset:
            # Move data to device
            gray = gray.unsqueeze(0).to(device)  # Add batch dim
            
            # Forward pass
            ab_pred = model(gray).cpu().numpy()[0]  # (2, H, W)
            
            # Denormalize AB channels
            ab_pred = (ab_pred + 1) * 128  # [0,255]
            ab_pred = ab_pred.transpose(1, 2, 0).astype(np.uint8)
            
            # Reconstruct LAB image
            L = (gray.squeeze().cpu().numpy() * 255).astype(np.uint8)
            lab_pred = np.zeros((256, 256, 3), dtype=np.uint8)
            lab_pred[..., 0] = L
            lab_pred[..., 1:] = ab_pred
            
            # Convert to RGB
            colorized_bgr = cv2.cvtColor(lab_pred, cv2.COLOR_LAB2BGR)
            colorized_rgb = cv2.cvtColor(colorized_bgr, cv2.COLOR_BGR2RGB)
            target_rgb = cv2.cvtColor(color_orig, cv2.COLOR_BGR2RGB)
            
            # Calculate metrics
            psnr_values.append(psnr(target_rgb, colorized_rgb))
            ssim_values.append(ssim(target_rgb, colorized_rgb, channel_axis=2))
    
    return np.mean(psnr_values), np.mean(ssim_values)

if __name__ == "__main__":
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = os.path.join(project_root, 'models', 'saved_models', 'colorize_epoch_5.pth')
    data_dir = os.path.join(project_root, 'data', 'processed')

    # Load model
    model = UNetColorize().to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except Exception as e:
        print(f"Error loading model: {e}")
        exit(1)

    # Run evaluation
    try:
        print("Evaluating validation set...")
        val_psnr, val_ssim = evaluate_model(model, data_dir, 'val', device)
        print(f"Validation PSNR: {val_psnr:.2f} dB")
        print(f"Validation SSIM: {val_ssim:.4f}")

        print("\nEvaluating training set...")
        train_psnr, train_ssim = evaluate_model(model, data_dir, 'train', device)
        print(f"Training PSNR: {train_psnr:.2f} dB")
        print(f"Training SSIM: {train_ssim:.4f}")
    except Exception as e:
        print(f"Evaluation failed: {e}")
        exit(1)