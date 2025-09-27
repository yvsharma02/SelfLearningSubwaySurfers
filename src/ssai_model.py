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
        prob = F.log_softmax(pred, dim=1)
        loss = F.kl_div(prob, required, reduction="batchmean")
        return loss

    def __init__(self):
        super(SSAIModel, self).__init__()

        self.cnn_stage = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.AvgPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.AvgPool2d(2),
            nn.Conv2d(32, 64, kernel_size=5, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(p=0.1),
            nn.Flatten(),
        )
        self.fully_connected_stage = nn.Sequential(
            nn.Linear(4224, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.45),
            nn.Linear(1024, 128),
            nn.ReLU(),
            nn.Dropout(p=0.33),
            nn.Linear(128, 5), # NOTHING UP DOWN LEFT RIGHT
        )

    def forward(self, img):
        flattened = self.cnn_stage(img)
        # flattened_with_velocity = torch.cat((flattened, time), dim=1)
        action = self.fully_connected_stage(flattened)
        return action

    # Returns the action to take.
    def infer(self, img, time, device, randomize = True):
        if type(time) is float:
            time_tensor = torch.tensor([time])

        image_tensor = SSAIModel.IMAGE_TRANSFORM(img).unsqueeze(0)
        image_tensor = image_tensor.to(device)
        time_tensor = time_tensor.unsqueeze(0)
        time_tensor = time_tensor.to(device)
        # print(image_tensor.shape)
        with torch.no_grad():
            eliminations_logits = self(image_tensor)
            eliminations = F.softmax(eliminations_logits, dim=1)

            if (not randomize):
                confidence, predicted_class = torch.min(eliminations, 1)
            else:
                inv = 1.0 / (eliminations + 1e-9)
                actions = F.softmax(inv, dim=1)
                predicted_class = torch.multinomial(actions, 1)
                confidence = actions[0, predicted_class.item()]

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