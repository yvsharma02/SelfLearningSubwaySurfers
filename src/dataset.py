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
        image = cv2.imread(img_path)
#        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        nothing_y = [1.0, 0] if label != constants.ACTION_NOTHING else [0.0, 1.0]
        action_y = [0] * 4
        if (label is not constants.ACTION_NOTHING):
            action_y[label - 1] = 1.0 # Since we removed NOTHING action, we need to shift indices.
        
        return image, (torch.tensor(nothing_y), torch.tensor(action_y))