import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
from models.colorize_unet import UNetColorize

class SpaceColorDataset(Dataset):
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
        
        # Read and resize
        gray = cv2.imread(gray_path, cv2.IMREAD_GRAYSCALE)
        gray = cv2.resize(gray, (256, 256)) / 255.0  # (256, 256)
        gray = gray[None, ...]  # Add channel dimension: (1, 256, 256)
        
        # Convert color to LAB (use uint8 for conversion)
        color = cv2.imread(color_path, cv2.IMREAD_COLOR)
        color = cv2.resize(color, (256, 256))
        color_lab = cv2.cvtColor(color, cv2.COLOR_BGR2LAB)
        ab = color_lab[..., 1:].transpose(2, 0, 1) / 128.0 - 1.0  # (2, 256, 256)
        
        return torch.FloatTensor(gray), torch.FloatTensor(ab)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNetColorize().to(device)
    criterion = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_dataset = SpaceColorDataset('data/processed', 'train')
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

    val_dataset = SpaceColorDataset('data/processed', 'val')
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

    for epoch in range(5):  # Adjust as needed
        model.train()
        for gray, ab in train_loader:
            print("Input shape:", gray.shape)
            gray, ab = gray.to(device), ab.to(device)
            optimizer.zero_grad()
            pred_ab = model(gray)
            loss = criterion(pred_ab, ab)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch+1}, Train Loss: {loss.item():.4f}')

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for gray, ab in val_loader:
                gray, ab = gray.to(device), ab.to(device)
                pred_ab = model(gray)
                val_loss += criterion(pred_ab, ab).item()
        val_loss /= len(val_loader)
        print(f'Epoch {epoch+1}, Val Loss: {val_loss:.4f}')

        # Save model checkpoint
        torch.save(model.state_dict(), f'models/saved_models/colorize_epoch_{epoch+1}.pth')

if __name__ == '__main__':
    main()