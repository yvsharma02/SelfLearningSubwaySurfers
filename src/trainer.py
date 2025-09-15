import os
import random
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from dataset import ImageDataset
from ssai_model import SSAICNN
import run_validator

def read_data(path):
    res = [[], [], [], [], []]
    for subdir, _, files in os.walk(path):
        if "metadata.txt" in files:
            if (not run_validator.is_valid(subdir)):
                continue
            metadata_path = os.path.join(subdir, "metadata.txt")
            with open(metadata_path, "r") as f:
                lines = f.readlines()[:-10] # Ignore the last 10 images, because they lead to the end of the game (mistakes were probabbly made.)
                for idx, line in enumerate(lines):
                    line = line.strip()
                    index, time, eliminations = line.split(';')
                    index = int(index.strip())
                    time = float(time.strip())
                    eliminations = eliminations.strip("[] \n").split(",")
                    eliminations = [int(x) for x in eliminations]
                    action = [x for x in range(0, 5) if x not in eliminations][0]
                    res[action].append(os.path.join(subdir, f"{index}.png"))

    # Undersample biased data.
    nothing_target_size = int(sum(len(res[x]) for x in range(1, 5)) / 3)
    nothing_target_size = min(nothing_target_size, len(res[0]))
    rs = random.sample(range(0, len(res[0])), nothing_target_size)
    res[0] = [res[0][x] for x in rs]

    return res

def labelify(data):
    images = []
    labels = []

    for c in range(0, len(data)):
        for img in data[c]:
            images.append(img)
            labels.append(c)

    return images, labels

def randomize_data(balanced_image_paths, balanced_labels):
    combined = list(zip(balanced_image_paths, balanced_labels))
    random.shuffle(combined)
    return zip(*combined)

def create_train_test_split(balanced_image_paths, balanced_labels):
    train_paths, test_paths, train_labels, test_labels = train_test_split(
        balanced_image_paths, balanced_labels, test_size=0.2, random_state=42, stratify=balanced_labels
    )

    return train_paths, test_paths, train_labels, test_labels

def create_datasets(train_paths, test_paths, train_labels, test_labels, transform):
    train_dataset = ImageDataset(train_paths, train_labels, transform=transform)
    test_dataset = ImageDataset(test_paths, test_labels, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    return train_dataset, train_loader, test_dataset, test_loader

def train(model, train_loader, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 50

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        train_loss = running_loss / total
        train_acc = correct / total
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")

def test(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    test_acc = correct / total
    print(f"Test Accuracy: {test_acc:.4f}")

def main():
    PATH = "generated/runs/dataset/"
    print("Reading Data...")
    data = read_data(PATH)
    path, classes = labelify(data)
    train_dataset, train_loader, test_dataset, test_loader = create_datasets(*create_train_test_split(*randomize_data(path, classes)), transform=SSAICNN.IMAGE_TRANSFORM)
    print("Starting Training...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model = SSAICNN(num_classes=5).to(device)
    train(model, train_loader, device)
    test(model, test_loader, device)

    model.save_to_file("generated/models/test.pth")

if __name__ == "__main__":
    main()