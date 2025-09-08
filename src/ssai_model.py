import torch.nn as nn
import torch
from torchvision import transforms

class SSAICNN(nn.Module):

    IMAGE_TRANSFORM = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
    ])

    def __init__(self, num_classes=5):
        super(SSAICNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128*8*8, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

    def infer(self, img, device):
        image_tensor = SSAICNN.IMAGE_TRANSFORM(img).unsqueeze(0)
        image_tensor = image_tensor.to(device)

        with torch.no_grad():
            outputs = self(image_tensor)
            _, predicted_class = outputs.max(1)

        return predicted_class.item()
    

def load(path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dict = torch.load(path, map_location=device)
    model = SSAICNN()
    model.load_state_dict(state_dict)
    model.to(device)
    return model