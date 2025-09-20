import os
import random
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from dataset import ImageDataset
from ssai_model import SSAIModel
import run_validator
import custom_enums

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
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 30

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        nothing_correct = 0
        action_corrent = 0
        nothing_count = 0
        action_count = 0
        # correct = 0
        # total = 0
        for images, labels in train_loader:
            images = images.to(device)
            nothing_labels, action_labels = labels[0], labels[1]
            # print(type(images))

            nothing_labels = nothing_labels.to(device)
            action_labels = action_labels.to(device)
            optimizer.zero_grad()
            nothing_pred, action_pred = model(images)
            loss = SSAIModel.calculate_loss_of_batch(nothing_pred, action_pred, nothing_labels, action_labels)
            loss.backward()
            optimizer.step()

            nothing_labels = nothing_labels.argmax(dim = 1)
            action_labels = action_labels.argmax(dim = 1)

            nothing_pred = nothing_pred.argmax(dim = 1)
            action_pred = action_pred.argmax(dim = 1)

            # print("Nothing:")
            # print ("Labl: [" + ",".join([str(x.item()) for x in nothing_labels]) + "]")
            # print ("Pred: [" + ",".join([str(x.item()) for x in nothing_pred]) + "]")

            # print("Action:")
            # print ("Labl: [" + ",".join([str(x.item()) for x in action_labels]) + "]")
            # print ("Pred: [" + ",".join([str(x.item()) for x in action_pred]) + "]")
            # print("_____________________________________________________________________")

            for i in range(0, len(nothing_pred)):
                # print(nothing_labels[i].item())
                if ((nothing_pred[i].item() == 1 and nothing_labels[i].item() == 1) or (nothing_pred[i].item() == 0 and nothing_labels[i].item() == 0)):
                    nothing_correct += 1
                if (nothing_labels[i].item() == 0 and action_pred[i].item() == action_labels[i].item()):
                    action_corrent += 1

                nothing_count += 1

                if (nothing_labels[i].item() == 0):
                    action_count += 1
            
            running_loss += loss.item() * images.size(0)
            # _, predicted = outputs.max(1)
            # total += images.size(0)
            # correct += predicted.eq(labels).sum().item()
        
        train_loss = running_loss / (nothing_count + action_count)
        train_acc = (nothing_correct + action_corrent) / (nothing_count + action_count)
        nothing_acc = "NA" if nothing_count == 0 else f"{(nothing_correct / nothing_count):.4f}"
        action_acc = "NA" if action_count == 0 else f"{(action_corrent / action_count):.4f}"
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}, Nothing Accuracy: {nothing_acc}, Action Accuracy: {action_acc}")
        # print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss:.4f}")

# def test(model, test_loader, device):
#     model.eval()
#     correct = 0
#     total = 0
#     with torch.no_grad():
#         for images, labels in test_loader:

#             nothing_labels, action_labels = labels[0], labels[1]

#             nothing_labels = nothing_labels.to(device)
#             action_labels = action_labels.to(device)
#             images = images.to(device)

#             nothing, action = model(images)
#             _, act_pred = action.max(1)
#             _, act_label = action_labels.max(1)
#             total += images.size(0)

#             if (nothing[0, 1].item() > .5 and nothing_labels[0, 1].item() > .5):
#                 correct += 1
#             elif (act_pred == act_label):
#                 correct += 1


#             correct += act_pred.eq(labels).sum().item()
#     test_acc = correct / total
#     print(f"Test Accuracy: {test_acc:.4f}")

def main():
    PATH = "generated/runs/dataset/"
    print("Reading Data...")
    data = read_data(PATH)
    path, classes = labelify(data)
    train_dataset, train_loader, test_dataset, test_loader = create_datasets(*create_train_test_split(*randomize_data(path, classes)), transform=SSAIModel.IMAGE_TRANSFORM)
    print("Starting Training...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model = SSAIModel().to(device)
    train(model, train_loader, device)
    # test(model, test_loader, device)

    model.save_to_file("generated/models/test.pth")

if __name__ == "__main__":
    main()