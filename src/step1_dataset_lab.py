%%writefile /kaggle/working/step1_dataset_lab.py
# PASTE THE ENTIRE CONTENTS OF step1_dataset_lab.py HERE
"""
STEP 1: Lab Color Space Dataset
=================================
WHY THIS CHANGE:
- Original code: grayscale(1ch) -> RGB(3ch). Network must predict all detail + color.
- New code: L channel(1ch) -> ab channels(2ch). Network only predicts smooth color fields.
- L channel (sharpness/structure) bypasses the network entirely and is recombined at the end.
- This is the single most important change for SNN viability because:
  * Spiking neurons are bad at high-frequency continuous output
  * Color (ab) is naturally smooth and low-frequency -> perfect for spikes
  * All sharp edges are preserved exactly via the L bypass, regardless of spike quality
"""

import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from skimage.color import rgb2lab, lab2rgb


class LabColorizationDataset(Dataset):
    """
    Returns:
        L_tensor  : shape [1, H, W], range [-1, 1]  — this is your grayscale input
        ab_tensor : shape [2, H, W], range [-1, 1]  — this is what the SNN predicts
        L_raw     : shape [1, H, W], range [0, 100] — needed for reconstruction later
    """

    def __init__(self, image_dir, image_size=128):
        # We drop from 256 to 128. Reason: your U-Net has ~2M spiking neurons at 256.
        # At 128, it's ~500K — much more feasible for a custom chip with limited SRAM.
        self.image_paths = sorted(glob.glob(os.path.join(image_dir, "*")))
        self.image_size = image_size
        self.resize = transforms.Resize((image_size, image_size), 
                                         interpolation=transforms.InterpolationMode.BICUBIC)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load as RGB PIL image
        image = Image.open(self.image_paths[idx]).convert("RGB")
        image = self.resize(image)
        
        # Convert to numpy for skimage Lab conversion
        img_np = np.array(image, dtype=np.float32) / 255.0  # [0,1]
        
        # Convert RGB -> Lab
        # L: 0-100 (luminance / your grayscale)
        # a: -128 to 127 (green-red axis)
        # b: -128 to 127 (blue-yellow axis)
        lab = rgb2lab(img_np).astype(np.float32)
        
        # Normalize L to [-1, 1] for network input
        L = lab[:, :, 0]           # shape [H, W]
        ab = lab[:, :, 1:]         # shape [H, W, 2]
        
        L_raw = torch.from_numpy(L).unsqueeze(0)               # [1,H,W] range [0,100]
        L_norm = (L_raw / 50.0) - 1.0                          # [1,H,W] range [-1,1]
        
        # Normalize ab to [-1, 1]
        ab_norm = torch.from_numpy(ab.transpose(2, 0, 1))      # [2,H,W]
        ab_norm = ab_norm / 110.0                               # ab range is ~[-110,110]
        
        return L_norm, ab_norm, L_raw


def lab_to_rgb_tensor(L_raw, ab_pred):
    """
    Reconstruct RGB from L (raw, [0,100]) and predicted ab (normalized, [-1,1]).
    
    Args:
        L_raw   : torch tensor [B, 1, H, W] range [0, 100]
        ab_pred : torch tensor [B, 2, H, W] range [-1, 1]
    Returns:
        rgb     : torch tensor [B, 3, H, W] range [0, 1]
    """
    # Denormalize ab
    ab_denorm = ab_pred * 110.0  # back to [-110, 110]
    
    batch_rgb = []
    for i in range(L_raw.shape[0]):
        L_np = L_raw[i, 0].cpu().numpy()            # [H,W]
        ab_np = ab_denorm[i].cpu().numpy()           # [2,H,W]
        
        # Stack into [H,W,3] Lab image
        lab_np = np.stack([L_np, ab_np[0], ab_np[1]], axis=2)
        
        # Convert to RGB, clip to [0,1]
        rgb_np = lab2rgb(lab_np).astype(np.float32)
        rgb_np = np.clip(rgb_np, 0.0, 1.0)
        
        batch_rgb.append(torch.from_numpy(rgb_np.transpose(2, 0, 1)))
    
    return torch.stack(batch_rgb, dim=0)


# ==========================================
# Quick dataset test — run this block alone
# ==========================================
if __name__ == "__main__":
    # Replace with your actual path
    TEST_DIR = "/content/drive/MyDrive/Pix2Pix_Colorization/dataset/Landscape_Pictures_256/train"
    
    dataset = LabColorizationDataset(TEST_DIR, image_size=128)
    loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)
    
    L_norm, ab_norm, L_raw = next(iter(loader))
    
    print("L_norm  shape:", L_norm.shape,   "  range:", L_norm.min().item(), "to", L_norm.max().item())
    print("ab_norm shape:", ab_norm.shape,  "  range:", ab_norm.min().item(), "to", ab_norm.max().item())
    print("L_raw   shape:", L_raw.shape,    "  range:", L_raw.min().item(),  "to", L_raw.max().item())
    
    # Test reconstruction
    rgb_reconstructed = lab_to_rgb_tensor(L_raw, ab_norm)
    print("RGB reconstructed shape:", rgb_reconstructed.shape, " range:", 
          rgb_reconstructed.min().item(), "to", rgb_reconstructed.max().item())
    
    print("\nStep 1 complete: Dataset working correctly.")