import torch.nn as nn
import torch
import torch.nn.functional as F
from torchvision import transforms

# Instead of predicting which choice to make, predict which choices to elimiate.
# (The choice least likely to be elimiated is our result.)

class SSAIModel(nn.Module):

    IMAGE_TRANSFORM = transforms.Compose([
        transforms.Resize((60, 100)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
    ])

    def calculate_loss_of_batch(pred, required):
        required = required.argmax(dim=1)

        prob = F.log_softmax(pred, dim=1)
        loss = F.nll_loss(prob, required)

        return loss

    def __init__(self):
        super(SSAIModel, self).__init__()

        self.cnn_stage = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.AvgPool2d(2),
            nn.Conv2d(16, 48, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(3),
            nn.Conv2d(48, 54, kernel_size=5, padding=1),
            nn.MaxPool2d(3),
            nn.Dropout(p=0.3),
            nn.Flatten(),
        )
        self.fully_connected_stage = nn.Sequential(
#            nn.Linear(432 + 1, 216),
            nn.Linear(432, 216),
            nn.ReLU(),
            nn.Linear(216, 128),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(64, 32), # NOTHING UP DOWN LEFT RIGHT
            nn.ReLU(),
            nn.Linear(32, 5)
        )

    def forward(self, img):
        flattened = self.cnn_stage(img)
        # flattened_with_velocity = torch.cat((flattened, time), dim=1)
        action = self.fully_connected_stage(flattened)
        return action

    # Returns the action to take.
    def infer(self, img, time, device):
        if type(time) is float:
            time_tensor = torch.tensor([time])

        image_tensor = SSAIModel.IMAGE_TRANSFORM(img).unsqueeze(0)
        image_tensor = image_tensor.to(device)
        time_tensor = time_tensor.unsqueeze(0)
        time_tensor = time_tensor.to(device)
        # print(image_tensor.shape)
        with torch.no_grad():
            action = self(image_tensor)
            action = F.softmax(action, dim=1)
            # print(f"nothing shape: {nothing.shape}")
            # print(f"action shape: {action.shape}")
            confidence, predicted_class = torch.min(action, 1)

        return predicted_class.item()
        

    def save_to_file(self, path):
        torch.save(self.state_dict(), path)

def load(path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model = SSAIModel()
    model.load_state_dict(torch.load(path))
    model.to(device)
    return model, device