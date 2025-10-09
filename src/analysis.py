# This file was completely generated.

import os
import cv2
import torch
import argparse
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torchvision.transforms as transforms
from ssai_model import SSAIModel  # import your model definition
from torchvision.utils import save_image
from PIL import Image

# ===== CONFIG =====
MODEL_PATH = "generated/models/test copy.pth"
ANALYSIS_DIR = "generated/analysis"
FPS = 60
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ===== UTILS =====
def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def get_sorted_indices(folder):
    """Get sorted integer indices for .png files like 0.png, 1.png, 2.png, ..."""
    return sorted([
        int(os.path.splitext(f)[0]) for f in os.listdir(folder)
        if f.endswith(".png") and os.path.splitext(f)[0].isdigit()
    ])

def normalize_img(tensor):
    tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min() + 1e-8)
    return tensor


# ===== MAIN =====
def main(stream_dir):
    print(f"[INFO] Starting analysis for stream folder: {stream_dir}")
    processed_dir = os.path.join(ANALYSIS_DIR, "processed")
    video_dir = os.path.join(ANALYSIS_DIR, "videos")

    for d in [processed_dir, video_dir]:
        ensure_dir(d)

    # ---- Step 1: Get list of frames ----
    indices = get_sorted_indices(stream_dir)
    if not indices:
        print("[ERROR] No .png files found in the given folder!")
        return
    print(f"[INFO] Found {len(indices)} frames in stream.")

    imgs = [cv2.imread(os.path.join(stream_dir, f"{i}.png")) for i in indices]
    h, w, _ = imgs[0].shape

    # ---- Step 2: Make stream video ----
    stream_video_path = os.path.join(video_dir, "stream.mp4")
    out = cv2.VideoWriter(stream_video_path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (w, h))
    for img in imgs:
        out.write(img)
    out.release()
    print(f"[INFO] Stream video saved to {stream_video_path}")

    # ---- Step 3: Model setup ----
    model = SSAIModel().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    # Hook to capture activations after each ReLU
    activation_stages = {}
    def get_hook(name):
        def hook(model, input, output):
            activation_stages[name] = output.detach()
        return hook

    relu_layers = [m for m in model.cnn_stage if isinstance(m, nn.ReLU)]
    for i, relu in enumerate(relu_layers, start=1):
        relu.register_forward_hook(get_hook(f"stage_{i}"))

    # ---- Step 4: Process each image ----
    print("[INFO] Processing CNN feature maps...")
    stage_outputs = {f"stage_{i}": [] for i in range(1, 5)}

    transform = transforms.Compose([
        transforms.Resize((100, 60)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.25, 0.25, 0.25])
    ])

    for idx in tqdm(indices):
        img_path = os.path.join(stream_dir, f"{idx}.png")
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        tensor = transform(Image.fromarray(img)).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            _ = model.cnn_stage(tensor)

        for stage, feat in activation_stages.items():
            feat_cpu = feat.squeeze(0).cpu()  # (C, H, W)
            stage_outputs[stage].append(feat_cpu)

            # Save all channels as grayscale
            for ch in range(feat_cpu.shape[0]):
                ch_dir = os.path.join(processed_dir, stage, f"channel_{ch+1}")
                ensure_dir(ch_dir)
                save_image(normalize_img(feat_cpu[ch]), os.path.join(ch_dir, f"{idx}.png"))

    # ---- Step 5: Brightest 3 channels per stage ----
    print("[INFO] Finding brightest channels per stage...")
    for stage, feats_over_time in stage_outputs.items():
        num_channels = feats_over_time[0].shape[0]
        mean_brightness = []
        for ch in range(num_channels):
            mean_val = np.mean([feat[ch].mean().item() for feat in feats_over_time])
            mean_brightness.append(mean_val)
        top3 = np.argsort(mean_brightness)[-3:][::-1]
        print(f"[STAGE {stage}] Brightest channels: {top3 + 1}")

        # Create combined RGB frames
        rgb_frames = []
        for feats in feats_over_time:
            chans = []
            for ch in top3:
                img = normalize_img(feats[ch])
                img = (img * 255).byte().numpy()
                chans.append(img)
            # Match sizes and stack into RGB
            rgb = np.stack([
                cv2.resize(chans[0], (w, h)),
                cv2.resize(chans[1], (w, h)),
                cv2.resize(chans[2], (w, h))
            ], axis=2)
            rgb_frames.append(rgb)

        # Write RGB combined video
        out_path = os.path.join(video_dir, f"{stage}_brightest.mp4")
        out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (w, h))
        for frame in rgb_frames:
            out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        out.release()
        print(f"[INFO] Saved {stage} brightest RGB video → {out_path}")

    print("\n✅ Analysis complete! All outputs stored in 'generated/analysis/'")


# ===== ENTRY POINT =====
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True, help="Path to the stream folder containing x.png images")
    args = parser.parse_args()
    main(args.dir)
