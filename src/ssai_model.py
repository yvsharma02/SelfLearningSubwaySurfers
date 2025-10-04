import torch.nn as nn
import torch
import torch.nn.functional as F
from torchvision import transforms
import constants
import time

# Instead of predicting which choice to make, predict which choices to elimiate.
# (The choice least likely to be elimiated is our result.)

class SSAIModel(nn.Module):

    IMAGE_TRANSFORM = transforms.Compose([
        transforms.Resize((100, 60)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.25,0.25,0.25])
    ])

    def calculate_loss_of_batch(pred, required):
        return F.mse_loss(pred, required)        

    def __init__(self):
        super(SSAIModel, self).__init__()

        self.cnn_stage = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.AvgPool2d(2),

            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AvgPool2d(2),

            nn.Conv2d(32, 72, kernel_size=5, padding=2),
            nn.BatchNorm2d(72),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Flatten(),
            nn.Dropout(p=0.15),
        )

        self.fully_connected_stage = nn.Sequential(
            nn.Linear(6048, 916),
            nn.BatchNorm1d(916),
            nn.ReLU(),
            nn.Dropout(p=0.5),

            nn.Linear(916, 96),
            nn.BatchNorm1d(96),
            nn.ReLU(),
            nn.Dropout(p=0.3),

            nn.Linear(96, 5),
        )

    def forward(self, img):
        flattened = self.cnn_stage(img)
        # flattened_with_velocity = torch.cat((flattened, time), dim=1)
        action = self.fully_connected_stage(flattened)
        return action

    # Returns the action to take.
    def infer(self, img, run_time, device, randomize = False):
        if type(run_time) is float:
            time_tensor = torch.tensor([run_time])

        image_tensor = SSAIModel.IMAGE_TRANSFORM(img).unsqueeze(0)
        image_tensor = image_tensor.to(device)
        time_tensor = time_tensor.unsqueeze(0)
        time_tensor = time_tensor.to(device)
        # print(image_tensor.shape)
        self.eval()
        with torch.no_grad():
            eliminations_logits = self(image_tensor)
            eliminations = F.softmax(eliminations_logits, dim=1)

            if (not randomize):
                confidence, predicted_class = torch.min(eliminations, 1)
            else:
                for i in range(0, 4):
                    elim = torch.multinomial(eliminations, 1)
                    eliminations[0, elim] = 0
                
                # print(eliminations)
                confidence, predicted_class = torch.max(eliminations, 1)
                # print(constants.action_to_name(predicted_class))


        return predicted_class.item(), eliminations_logits[0, :]
        

    def save_to_file(self, path):
        torch.save(self.state_dict(), path)

def load(path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model = SSAIModel()
    model.load_state_dict(torch.load(path))
    model.to(device)
    return model, device