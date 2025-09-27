from torch.utils.data import Dataset
import cv2
import constants
from PIL import Image
import torch

class ImageDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        image = Image.open(img_path).convert("RGB")
        
        # print(image.shape)
        if self.transform:
            image = self.transform(image)

        # print(label)
        return image, torch.tensor(label)