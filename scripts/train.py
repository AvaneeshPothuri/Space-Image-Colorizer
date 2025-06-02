import sys
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.cuda.amp import autocast, GradScaler

# Ensure we're working from the correct directory
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
os.chdir(project_root)  # Change to project root
sys.path.insert(0, project_root)

print(f"Working directory: {os.getcwd()}")
print(f"Project root: {project_root}")

from models.colorize_unet import UNetColorize

class Config:
    data_dir = os.path.join(project_root, 'data', 'processed')  # Use absolute path
    batch_size = 4
    lr = 3e-4
    epochs = 10
    patience = 7
    weight_decay = 1e-5
    mix_precision = torch.cuda.is_available()  # Only enable if CUDA available
    augment = True

def create_model(device):
    model = UNetColorize()    
    return model.to(device)

class EnhancedSpaceColorDataset(Dataset):
    def __init__(self, data_dir, split='train'):
        # Use os.path.join for proper path construction
        self.gray_dir = os.path.join(data_dir, split, 'gray')
        self.color_dir = os.path.join(data_dir, split, 'color')
        
        # Validate directories exist
        self._validate_directories()
        
        # Get file pairs
        self.file_pairs = self._get_file_pairs()
        
        print(f"Found {len(self.file_pairs)} image pairs in {split} set")
        
        self.augmentations = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.3),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
            transforms.RandomAffine(degrees=0, translate=(0.15, 0.15)),
            transforms.ToTensor()
        ]) if Config.augment else None

    def _validate_directories(self):
        """Check if required directories exist"""
        if not os.path.exists(self.gray_dir):
            raise FileNotFoundError(f"Gray directory not found: {self.gray_dir}")
        if not os.path.exists(self.color_dir):
            raise FileNotFoundError(f"Color directory not found: {self.color_dir}")
        
        print(f"Gray dir: {self.gray_dir} (exists: {os.path.exists(self.gray_dir)})")
        print(f"Color dir: {self.color_dir} (exists: {os.path.exists(self.color_dir)})")

    def _get_file_pairs(self):
        """Get matching file pairs from gray and color directories"""
        try:
            gray_files = set(os.listdir(self.gray_dir))
            color_files = set(os.listdir(self.color_dir))
            
            # Find common files
            common_files = gray_files.intersection(color_files)
            
            # Filter out non-image files
            valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
            file_pairs = []
            
            for file in common_files:
                if any(file.lower().endswith(ext) for ext in valid_extensions):
                    file_pairs.append((file, file))
            
            if not file_pairs:
                raise ValueError(f"No matching image pairs found!\n"
                               f"Gray files: {list(gray_files)[:5]}...\n"
                               f"Color files: {list(color_files)[:5]}...")
            
            return file_pairs
            
        except Exception as e:
            raise FileNotFoundError(f"Error reading directories: {e}")

    def __len__(self):
        return len(self.file_pairs)

    def __getitem__(self, idx):
        gray_file, color_file = self.file_pairs[idx]
        
        gray_path = os.path.join(self.gray_dir, gray_file)
        color_path = os.path.join(self.color_dir, color_file)
        
        # Read images
        gray = cv2.imread(gray_path, cv2.IMREAD_GRAYSCALE)
        color = cv2.imread(color_path, cv2.IMREAD_COLOR)
        
        if gray is None:
            raise ValueError(f"Cannot read gray image: {gray_path}")
        if color is None:
            raise ValueError(f"Cannot read color image: {color_path}")
        
        # Resize and convert
        gray = cv2.resize(gray, (256, 256))
        color = cv2.resize(color, (256, 256))
        color_lab = cv2.cvtColor(color, cv2.COLOR_BGR2LAB)
        ab = color_lab[..., 1:].transpose(2, 0, 1) / 128.0 - 1.0

        # Normalize gray
        gray = gray / 255.0

        # Apply augmentations
        if self.augmentations:
            gray = self.augmentations(gray).squeeze(0)

        return torch.FloatTensor(gray).unsqueeze(0), torch.FloatTensor(ab)

def train_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = create_model(device)
    
    # Optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=Config.lr, weight_decay=Config.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
    scaler = GradScaler(enabled=Config.mix_precision)

    # Create datasets
    try:
        train_dataset = EnhancedSpaceColorDataset(Config.data_dir, 'train')
        val_dataset = EnhancedSpaceColorDataset(Config.data_dir, 'val')
    except Exception as e:
        print(f"Error creating datasets: {e}")
        print(f"Make sure you have the following structure:")
        print(f"  {Config.data_dir}/train/gray/")
        print(f"  {Config.data_dir}/train/color/")
        print(f"  {Config.data_dir}/val/gray/")
        print(f"  {Config.data_dir}/val/color/")
        return None
    
    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=Config.batch_size, 
                            shuffle=True, pin_memory=True, num_workers=0)  # num_workers=0 for Windows
    val_loader = DataLoader(val_dataset, batch_size=Config.batch_size,
                          shuffle=False, pin_memory=True, num_workers=0)

    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0

    # Ensure models directory exists
    os.makedirs(os.path.join(project_root, 'models', 'saved_models'), exist_ok=True)

    for epoch in range(Config.epochs):
        model.train()
        train_loss = 0.0
        
        for gray, ab in train_loader:
            gray, ab = gray.to(device), ab.to(device)
            
            optimizer.zero_grad()
            
            with autocast(enabled=Config.mix_precision):
                pred = model(gray)
                loss = nn.L1Loss()(pred, ab)
            
            if Config.mix_precision:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            
            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for gray, ab in val_loader:
                gray, ab = gray.to(device), ab.to(device)
                with autocast(enabled=Config.mix_precision):
                    pred = model(gray)
                    val_loss += nn.L1Loss()(pred, ab).item()
        
        # Update metrics
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        scheduler.step(val_loss)

        print(f'Epoch {epoch+1:03d} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | LR: {optimizer.param_groups[0]["lr"]:.2e}')

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            model_path = os.path.join(project_root, 'models', 'saved_models', 'best_model.pth')
            torch.save(model.state_dict(), model_path)
            print(f"Saved best model to {model_path}")
        else:
            patience_counter += 1
            if patience_counter >= Config.patience:
                print(f'Early stopping at epoch {epoch+1}')
                break

    # Load best model
    best_model_path = os.path.join(project_root, 'models', 'saved_models', 'best_model.pth')
    model.load_state_dict(torch.load(best_model_path))
    return model

if __name__ == '__main__':
    train_model()