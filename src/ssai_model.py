import torch.nn as nn
import torch
import torch.nn.functional as F
from torchvision import transforms

class SSAIModel(nn.Module):

    IMAGE_TRANSFORM = transforms.Compose([
        transforms.Resize((60, 100)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
    ])

    def calculate_loss(pred_nothing_y, pred_action_y, actual_nothing_y, actual_action_y):
        actual_action_y = torch.argmax(actual_action_y, dim=1)
        actual_nothing_y = torch.argmax(actual_nothing_y, dim=1)

        action_prob = F.log_softmax(pred_action_y, dim=1)
        action_loss = F.nll_loss(action_prob, actual_action_y)

        nothing_prob = F.log_softmax(pred_nothing_y, dim=1)
        nothing_loss = F.nll_loss(nothing_prob, actual_nothing_y)

        return action_loss + nothing_loss

    def __init__(self):
        super(SSAIModel, self).__init__()

        self.stage1 = nn.Sequential(
            nn.Conv2d(3, 12, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(12, 24, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(24, 36, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(3024, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 2) # (TAKE_ACTION, DO_NOT_TAKE_ACTION)
        )

        self.stage2 = nn.Sequential(
            nn.Conv2d(3, 12, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(12, 24, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(24, 36, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(3024, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 4) # Up Down Left Right
        )

    def forward(self, x):
        do_something = self.stage1(x)
        action = self.stage2(x)
        return do_something, action

    def infer(self, img, device):
        image_tensor = SSAIModel.IMAGE_TRANSFORM(img).unsqueeze(0)
        image_tensor = image_tensor.to(device)

        with torch.no_grad():
            nothing, action = self(image_tensor)
            nothing = F.softmax(nothing, dim=1)
            action = F.softmax(action, dim=1)

            nothing_confidence = nothing[1]
            confidence, predicted_class = torch.max(action, 1)

        return nothing_confidence.item(), confidence.item(), predicted_class.item()
        

    def save_to_file(self, path):
        torch.save(self.state_dict(), path)

def load(path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model = SSAIModel()
    model.load_state_dict(torch.load(path))
    model.to(device)
    return model, device