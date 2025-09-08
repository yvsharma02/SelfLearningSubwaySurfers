import torch.nn as nn
import torch
import torch.nn.functional as F
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
            nn.Conv2d(3, 12, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(12, 24, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(24, 36, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2304, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
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
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted_class = torch.max(probabilities, 1)

        return confidence.item(), predicted_class.item()
        

    def save_to_file(self, path):
        torch.save(self.state_dict(), path)

def load(path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    state_dict = torch.load(path, map_location=device)
    model = SSAICNN()
    model.load_state_dict(torch.load(path))
    model.to(device)
    return model, device
