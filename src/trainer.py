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
import constants
import numpy as np
import shutil
from PIL import Image
import pipeline

MULTI_ELIM_PERCENTAGE_OF_SINGLE_ELIM = 8
MULTI_ELIM_NOTHING_LIMIT = 0.475 # This percent of single elim can be nothing multi elims

# Returns (img_path, label (tensor with length 5 denoting elimination confidence.))
def read_data(path):
    single_elim = {

    }
    multi_elim = {

    }
    for subdir, _, files in os.walk(path):
        if "metadata.txt" in files:
            if (not run_validator.is_valid(subdir)):
                continue

            metadata_path = os.path.join(subdir, "metadata.txt")
            with open(metadata_path, "r") as f:
                lines = f.readlines()
                for idx, line in enumerate(lines):
                    line = line.strip()
                    index, time, eliminations, logits, debug_log = line.split(';')
                    index = int(index.strip())
                    time = float(time.strip())
                    eliminations = eliminations.strip("[] \n").split(",")
                    if (len(eliminations) == 0 or not eliminations[0]):
                        continue
                    eliminations = [int(x) for x in eliminations]
                    label = [(1.0 / len(eliminations) if i in eliminations else 0.0) for i in range(0, 5)]
                    imname = os.path.join(subdir, f"{index}.png")

                    if (len(eliminations) > 1):
                        multi_elim[imname] = label
                    else: 
                        single_elim[imname] = label

    res_mp = {}

    no_action_multi_elims = [k for k in multi_elim.keys() if multi_elim[k][0] < 0.25]
    action_multi_elims = [k for k in multi_elim.keys() if k not in no_action_multi_elims]
    sampled_no_actions = np.random.choice(no_action_multi_elims, size=min(int(len(single_elim.keys()) * MULTI_ELIM_PERCENTAGE_OF_SINGLE_ELIM * MULTI_ELIM_NOTHING_LIMIT), len(no_action_multi_elims)), replace=False)
    sampled_actions = np.random.choice(action_multi_elims, size=min(int(len(single_elim.keys()) * MULTI_ELIM_PERCENTAGE_OF_SINGLE_ELIM) - len(sampled_no_actions), len(action_multi_elims)), replace=False)

    for k in sampled_actions:
        single_elim[k] = multi_elim[k]
    for k in sampled_no_actions:
        single_elim[k] = multi_elim[k]
    for k in single_elim.keys():
        res_mp[k] = single_elim[k]

    return res_mp

# def write_data(data, output):
#     c = 0
#     for k in data:
#         # action = torch.argmax(torch.tensor(data[k]), dim=0).item()
#         # os.makedirs(os.path.join(output, str(action)), exist_ok=True)
#         pipeline.detect_whitest_large_greys(k, os.path.join(output, f"{c}.png"))
#         # shutil.copy(k, os.path.join(output, str(action), f"{counts[action]}.png"))
#         c += 1

def create_train_test_split(data):
    train_paths, test_paths, train_labels, test_labels = train_test_split(
        list(data.keys()), list(data.values()), test_size=0.2, random_state=42
    )

    return train_paths, test_paths, train_labels, test_labels

def create_datasets(train_paths, test_paths, train_labels, test_labels, transform):
    train_dataset = ImageDataset(train_paths, train_labels, transform=transform)
    test_dataset = ImageDataset(test_paths, test_labels, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, drop_last=True)

    return train_dataset, train_loader, test_dataset, test_loader

def train(model, train_loader, test_loader, device):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    MAX_EPOCH = 50
    test_losses = []
    for epoch in range(MAX_EPOCH):
        model.train()
        running_loss = 0.0

        single_elim_correct = 0
        single_elim_total = 0
        multi_elim_correct = 0
        multi_elim_total = 0

        for images, labels in train_loader:
            images = images.to(device)

            eliminations = labels.to(device)
            optimizer.zero_grad()
            elim_pred = model(images)
            loss = SSAIModel.calculate_loss_of_batch(elim_pred, eliminations)
            loss.backward()
            optimizer.step()
            for i in range(0, elim_pred.shape[0]):
                # print(eliminations[i, :])
                elim_count = sum(1 if eliminations[i, x].item() > 0.9 else 0 for x in range(0, 5))
                label_confidence, label_choice = torch.max(eliminations[i, :], 0)
                if (elim_count == 1):
                    pred_confidence, pred_choice = torch.max(elim_pred[i, :], 0)
                    if (pred_choice == label_choice):
                        single_elim_correct += 1
                    single_elim_total += 1
                else:
                    label_confidence, label_choice = torch.min(eliminations[i, :], 0)
                    pred_confidence, pred_choice = torch.min(elim_pred[i, :], 0)
                    if (pred_choice == label_choice):
                        multi_elim_correct += 1
                    multi_elim_total += 1


            running_loss += loss.item() * images.size(0)

        if ((single_elim_total + multi_elim_total) > 0):
            train_loss = running_loss / (single_elim_total + multi_elim_total)
            single_elim_acc = f"{((single_elim_correct) / (single_elim_total)):.4f}" if single_elim_total > 0 else "NA"
            multi_elim_acc = f"{((multi_elim_correct) / (multi_elim_total)):.4f}" if multi_elim_total > 0 else "NA"
            total_acc = (single_elim_correct + multi_elim_correct) / (single_elim_total + multi_elim_total)
            print(f"Epoch [{epoch+1}/{MAX_EPOCH}], Loss: {train_loss:.4f}, Accuracy: {total_acc:.4f}, Single Elim Accuracy: {single_elim_acc}, Multi Elim Accuracy: {multi_elim_acc}")
        else:
            print(f"Empty Empoch: {epoch+1}/{MAX_EPOCH}")

        model.save_to_file("generated/models/test.pth")
        test_loss = test(model, test_loader, device)[0]
        window = test_loss[-max(len(test_loss, 7)):]
        avg_loss = 0 if len(window) == 0 else sum(window) / len(window)
        test_loss.append(test_loss)

        if (len(window) > 7 and avg_loss - test_loss < 0.001):
            break
    print("Training Finished!")

def test(model, test_loader, device):
    model.eval()
    running_loss = 0.0

    single_elim_correct = 0
    single_elim_total = 0
    multi_elim_correct = 0
    multi_elim_total = 0

    c = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            eliminations = labels.to(device)

            elim_pred = model(images)
            loss = SSAIModel.calculate_loss_of_batch(elim_pred, eliminations)
            running_loss += loss.item() * images.size(0)

            for i in range(0, elim_pred.shape[0]):
                elim_count = sum(1 if eliminations[i, x].item() > 0.05 else 0 for x in range(0, 5))
                # img_tensor = images[i, :, :, :].cpu()
                # unnorm = img_tensor * 0.5 + 0.5

                # Convert to numpy (HWC) and uint8
                # np_img = (unnorm.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

                # Back to PIL
                # pil_img = Image.fromarray(np_img, mode="RGB")

                # pred_str = "[" + ",".join([f"{x.item():.3f}" for x in elim_pred[i, :]]) + "]"
                # label_str = "[" +  ",".join([f"{x.item():.3f}" for x in eliminations[i, :]]) + "]"
                # pil_img.save(f"generated/test/{c}_{label_str}_{pred_str}.png")
                c += 1
                label_confidence, label_choice = torch.max(eliminations[i, :], 0)
                if elim_count == 1:
                    pred_confidence, pred_choice = torch.max(elim_pred[i, :], 0)
                    if pred_choice == label_choice:
                        single_elim_correct += 1
                    single_elim_total += 1
                else:
                    label_confidence, label_choice = torch.min(eliminations[i, :], 0)
                    pred_confidence, pred_choice = torch.min(elim_pred[i, :], 0)
                    if pred_choice == label_choice:
                        multi_elim_correct += 1
                    multi_elim_total += 1

    total_samples = single_elim_total + multi_elim_total
    test_loss = running_loss / total_samples if total_samples > 0 else 0
    single_elim_acc = f"{(single_elim_correct / single_elim_total):.4f}" if single_elim_total > 0 else "NA"
    multi_elim_acc = f"{(multi_elim_correct / multi_elim_total):.4f}" if multi_elim_total > 0 else "NA"
    total_acc = (single_elim_correct + multi_elim_correct) / total_samples if total_samples > 0 else 0

    print(f"Test Loss: {test_loss:.4f}, Accuracy: {total_acc:.4f}, "
          f"Single Elim Accuracy: {single_elim_acc}, Multi Elim Accuracy: {multi_elim_acc}")

    return test_loss, total_acc, single_elim_acc, multi_elim_acc


def main():
    PATH = "generated/runs/dataset/"
    print("Reading Data...")
    # data = read_data(PATH)

    train_dataset, train_loader, test_dataset, test_loader = create_datasets(*create_train_test_split(read_data(PATH)), transform=SSAIModel.IMAGE_TRANSFORM)
    print("Starting Training...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model = SSAIModel().to(device)
    train(model, train_loader, test_loader, device)
    # test(model, test_loader, device)


if __name__ == "__main__":
    main()