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

def read_data(path, class_no):
    all_classes = set(range(class_no))
    image_classes = {}  # {image_path: class_number}

    for subdir, _, files in os.walk(path):
        if "metadata.txt" in files:
            metadata_path = os.path.join(subdir, "metadata.txt")
            with open(metadata_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split(";")
                    if len(parts) != 3:
                        print(f"Skipping malformed line: {line}")
                        continue

                    image_no = parts[0].strip()
                    int_list_str = parts[2].strip().strip("[]")
                    if int_list_str:
                        int_list = set(map(int, int_list_str.split(",")))
                    else:
                        int_list = set()

                    img_class = list(all_classes - int_list)
                    if len(img_class) != 1:
                        print(f"Warning: ambiguous class for {image_no} in {metadata_path}")
                        continue
                    img_class = img_class[0]

                    image_path = os.path.join(subdir, f"{image_no}.png")
                    image_classes[image_path] = img_class

        
    return image_classes

def undersample_and_unmap(unbiased_data):
    class_to_images = defaultdict(list)
    for path, cls in unbiased_data.items():
        class_to_images[cls].append(path)


    non_zero_classes = [cls for cls in class_to_images if cls != 0]
    # print([len(class_to_images[cls]) for cls in class_to_images])
    target_size = max(sum(len(class_to_images[cls]) for cls in non_zero_classes) / 20, len(class_to_images[0]))

    undersampled_class_0 = random.sample(class_to_images[0], target_size)

    balanced_image_paths = undersampled_class_0
    balanced_labels = [0]*target_size

    for cls in non_zero_classes:
        balanced_image_paths.extend(class_to_images[cls])
        balanced_labels.extend([cls]*len(class_to_images[cls]))

    return balanced_image_paths, balanced_labels

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
    num_epochs = 25

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
    CLASSES = 5
    PATH = "generated/runs/dataset/"

    biased_data = read_data(PATH, CLASSES)
    path, classes = undersample_and_unmap(biased_data)
    train_dataset, train_loader, test_dataset, test_loader = create_datasets(*create_train_test_split(*randomize_data(path, classes)), transform=SSAICNN.IMAGE_TRANSFORM)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model = SSAICNN(num_classes=5).to(device)
    train(model, train_loader, device)
    test(model, test_loader, device)

    model.save_to_file("generated/models/test.pth")

main()