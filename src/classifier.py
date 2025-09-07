import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# -------------------
# Define the CNN
# -------------------
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=4):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)  # assuming input 64x64
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # [B,16,32,32]
        x = self.pool(F.relu(self.conv2(x)))  # [B,32,16,16]
        x = self.pool(F.relu(self.conv3(x)))  # [B,64,8,8]
        x = x.view(-1, 64 * 8 * 8)           # flatten
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# -------------------
# Custom Loss Function
# -------------------
class CustomLoss(nn.Module):
    def __init__(self, lambda_penalty=0.01):
        super(CustomLoss, self).__init__()
        self.ce = nn.CrossEntropyLoss()
        self.lambda_penalty = lambda_penalty

    def forward(self, outputs, targets):
        # Standard cross-entropy
        loss = self.ce(outputs, targets)
        # Example penalty: encourage smaller outputs
        penalty = self.lambda_penalty * torch.mean(outputs**2)
        return loss + penalty

# -------------------
# Training Loop Example
# -------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SimpleCNN(num_classes=4).to(device)
criterion = CustomLoss(lambda_penalty=0.01)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Dummy data (replace with real DataLoader)
for epoch in range(5):
    inputs = torch.randn(8, 3, 64, 64).to(device)  # batch_size=8
    targets = torch.randint(0, 4, (8,)).to(device) # 4 classes

    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# Test with a dummy input
dummy_input = torch.randn(1, 3, 64, 64).to(device)
pred = model(dummy_input)
pred_class = torch.argmax(pred, dim=1)
print("Predicted class:", pred_class.item())
