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

    def calculate_loss_of_batch(pred_nothing, pred_action, actual_nothing_y, actual_action_y):
        actual_action_y = torch.argmax(actual_action_y, dim=1)
        actual_nothing_y = torch.argmax(actual_nothing_y, dim=1)

        action_prob = F.log_softmax(pred_action, dim=1)
        action_loss = F.nll_loss(action_prob, actual_action_y)

        nothing_prob = F.log_softmax(pred_nothing, dim=1)
        nothing_loss = F.nll_loss(nothing_prob, actual_nothing_y)

        # print("Nothing:")
        # print ("Labl: [" + ",".join([str(x.item()) for x in actual_nothing_y]) + "]")
        # print ("Pred: [" + ",".join([str(x.item()) for x in pred_nothing.argmax(dim=1)]) + "]")

        # print("Action:")
        # print ("Labl: [" + ",".join([str(x.item()) for x in actual_action_y]) + "]")
        # print ("Pred: [" + ",".join([str(x.item()) for x in pred_action.argmax(dim=1)]) + "]")
        # print("_____________________________________________________________________")

        # print(f"Action Loss: {action_loss:.4f}, Nothing Loss: {nothing_loss:.4f}")
        return action_loss + nothing_loss

    def __init__(self):
        super(SSAIModel, self).__init__()

        self.common_stage = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.AvgPool2d(2),
            nn.Conv2d(16, 64, kernel_size=5, padding=1),
            nn.ReLU(),
            # nn.Dropout(p=0.5),
            # nn.AvgPool2d(2),
            # nn.Conv2d(32, 64, kernel_size=3, padding=1),
            # nn.ReLU(),
            nn.MaxPool2d(5),
            nn.Dropout(p=0.3),
            nn.Flatten(),
            nn.Linear(2880, 1024),
            nn.ReLU(),
            nn.Linear(1024, 128),
            nn.ReLU(),
        )

        self.nothing_predictor = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 2) # (TAKE_ACTION, DO_NOT_TAKE_ACTION)
        )

        self.action_predictor = nn.Sequential(
            nn.Linear(128, 4),
#            nn.ReLU(),
#            nn.Linear(32, 4) # Up Down Left Right
        )

    def forward(self, x):
        x = self.common_stage(x)
        do_something = self.nothing_predictor(x)
        action = self.action_predictor(x)
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