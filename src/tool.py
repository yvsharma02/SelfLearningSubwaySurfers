import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F

CHOICES = ["NOTHING", "UP", "DOWN", "LEFT", "RIGHT"]

def parse_md(path):
    res = []
    # Does os.walk work in ordered way?
    for subdir, _, files in os.walk(path):
        if "metadata.txt" in files:
            metadata_path = os.path.join(subdir, "metadata.txt")
            with open(metadata_path, "r") as f:
                lines = f.readlines()
                for idx, line in enumerate(lines):
                    line = line.strip()
                    index, time, eliminations, logits = line.split(';')
                    index = int(index.strip())
                    time = float(time.strip())
                    eliminations = eliminations.strip("[] \n").split(",")
                    logits = logits.strip("[] \n").split(",")
                    eliminations = [int(x) for x in eliminations]
                    label = [(1.0 / len(eliminations) if i in eliminations else 0.0) for i in range(0, 5)]
                    res.append(os.path.join(subdir, f"{index}.png", label, eliminations, logits))

    return res

def get_arrow_pos(action, cx, cy):
    if (action == "UP"):
        return ((cx, cy - 20), (cx, cy - 40))
    if (action == "DOWN"):
        return ((cx, cy + 20), (cx, cy + 40))
    if (action == "LEFT"):
        return ((cx - 20, cy), (cx - 40, cy))
    if (action == "RIGHT"):
        return ((cx + 20, cy), (cx + 40, cy))


def visualize(image_path, logits, eliminated, save_path, model_number):
    img = cv2.imread(image_path)
    if img is None:
        return
    print(logits)
    logits_t = torch.tensor(logits)
    probs = F.softmax(logits_t, dim=0).numpy()

    h, w, _ = img.shape
    canvas = np.zeros((h + 100, w + 250, 3), dtype=np.uint8)  # black background
    canvas[100:100+h, :w] = img

    # Add model marker
    cv2.putText(canvas, f"Model #{model_number}", (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA)

    # Joystick center
    cx, cy = w + 120, 100 + h // 2
    circle_radius = 20
    arrow_len = 30  # shorter arrows
    gap = 10        # distance from circle

    # Spread arrows around the circle (offsets to avoid overlapping)
    positions = {
        "NOTHING": (cx, cy),  # center
        "UP": (cx, cy - circle_radius - arrow_len - gap),
        "DOWN": (cx, cy + circle_radius + arrow_len + gap),
        "LEFT": (cx - circle_radius - arrow_len - gap, cy),
        "RIGHT": (cx + circle_radius + arrow_len + gap, cy)
    }

    # Identify most probable choice
    max_idx = int(np.argmax(probs))

    # Draw indicators
    for i, label in enumerate(CHOICES):
        prob = probs[i]
        color = (0, 255, 0) if i == max_idx else (0, 0, 255)  # green for max prob, red otherwise
        x, y = positions[label]

        if label == "NOTHING":
            cv2.circle(canvas, (x, y), circle_radius, color, 3)
            # Probability inside circle
            cv2.putText(canvas, f"{prob*100:.1f}%", (x - 18, y + 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
        else:
            # Draw arrow from edge of circle to tip
            start_x = get_arrow_pos(label, cx, cy)[0][0]
            start_y = get_arrow_pos(label, cx, cy)[0][1]
            end_x = get_arrow_pos(label, cx, cy)[1][0]
            end_y = get_arrow_pos(label, cx, cy)[1][1]
            cv2.arrowedLine(canvas, (start_x, start_y), (end_x, end_y), color, 2, tipLength=0.2)
            # Probability at arrow tip
            cv2.putText(canvas, f"{prob*100:.1f}%", (end_x - 15, end_y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

    cv2.imwrite(save_path, canvas)




def main(root_dir, output_dir="output_viz"):
    os.makedirs(output_dir, exist_ok=True)
    dirs = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])

    counter = 0
    for dir_idx, d in enumerate(dirs):
        meta_path = os.path.join(root_dir, d, "metadata.txt")
        model_number = dir_idx // 10
        if not os.path.exists(meta_path):
            continue

        with open(meta_path, "r") as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()
            index, time, eliminations, logits = line.split(';')
            index = int(index.strip())
            time = float(time.strip())
            eliminations = eliminations.strip("[] \n").split(",")
            logits = logits.strip("() \n").split(",")
            eliminations = [int(x) for x in eliminations]
            logits = [float(x) for x in logits]
            label = [(1.0 / len(eliminations) if i in eliminations else 0.0) for i in range(0, 5)]

            visualize(os.path.join(root_dir, d, f"{index}.png"), logits, eliminations, os.path.join(output_dir, f"{counter}.png"), model_number)
            counter += 1

    print(f"Visualization saved to {output_dir}, {counter} images created.")

if __name__ == "__main__":
    main("generated/runs/dataset", "generated/analysis")  # Change "D" to your root folder
