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

    def calculate_loss_of_batch(pred_nothing, pred_action, actual_nothing_y, actual_action_y):
        actual_action_y = torch.argmax(actual_action_y, dim=1)
        actual_nothing_y = torch.argmax(actual_nothing_y, dim=1)

        action_prob = F.log_softmax(pred_action, dim=1)
        action_loss = F.nll_loss(action_prob, actual_action_y)

        nothing_prob = F.log_softmax(pred_nothing, dim=1)
        nothing_loss = F.nll_loss(nothing_prob, actual_nothing_y)

        return action_loss + nothing_loss

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
            nn.Linear(432 + 1, 216),
            nn.ReLU(),
            nn.Linear(216, 128),
            nn.ReLU(),
        )

        self.nothing_predictor = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 2) # (TAKE_ACTION, DO_NOT_TAKE_ACTION)
        )

        self.action_predictor = nn.Sequential(
            nn.Dropout(p=0.25),
            nn.Linear(128, 48), # Up Down Left Right
            nn.ReLU(),
            nn.Linear(48, 4)
        )

    def forward(self, img, time):
        flattened = self.cnn_stage(img)
        flattened_with_velocity = torch.cat(flattened, torch.tensor([time]), dim=0)

        stage1res = self.fully_connected_stage(flattened_with_velocity)
        
        do_something = self.nothing_predictor(stage1res)
        action = self.action_predictor(stage1res)
        return do_something, action

    def infer(self, img, device):
        image_tensor = SSAIModel.IMAGE_TRANSFORM(img).unsqueeze(0)
        image_tensor = image_tensor.to(device)
        # print(image_tensor.shape)
        with torch.no_grad():
            nothing, action = self(image_tensor)
            nothing = F.softmax(nothing, dim=1)
            action = F.softmax(action, dim=1)
            # print(f"nothing shape: {nothing.shape}")
            # print(f"action shape: {action.shape}")
            nothing_confidence = nothing[0, 1]
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