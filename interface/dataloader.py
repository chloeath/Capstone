from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2
import random
import torch

class HyperspectralDataset(Dataset):
    def __init__(self, images, masks, hflip=None, vflip=None):
        self.images = images  # Hyperspectral tiles of shape (270, 32, 32)
        self.masks = masks    # Ground truth tiles of shape (32, 32)
        self.hflip = hflip
        self.vflip = vflip

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = torch.tensor(self.images[idx], dtype=torch.float32)  # (270, 32, 32)
        mask = torch.tensor(self.masks[idx], dtype=torch.long)      # (32, 32)
        if self.hflip:
            if random.random() < 0.5:
                image = v2.functional.horizontal_flip(image)
                mask = v2.functional.horizontal_flip(mask)
        if self.vflip:
            if random.random() < 0.5:
                image = v2.functional.vertical_flip(image)
                mask = v2.functional.vertical_flip(mask)
        return image, mask
    
class SingleImageDataset(Dataset):
    def __init__(self, image, mask):
        self.image = torch.tensor(image, dtype=torch.float32)  # (270, H, W)
        self.mask = torch.tensor(mask, dtype=torch.long)       # (H, W)

    def __len__(self):
        return 1  # Only one sample

    def __getitem__(self, idx):
        return self.image, self.mask