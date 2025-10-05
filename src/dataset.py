from torch.utils.data import Dataset
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
        img_path : str= self.image_paths[idx]
        label = self.labels[idx]
        
        images = [Image.open(img_path[:-4] + f"_{i}.png").convert("RGB") for i in range(0, 3)]
        if self.transform:
            images = [self.transform(image) for image in images]
        # print(image.shape)
        # print(f"{img_path} : {label}")
        return images, torch.tensor(label)