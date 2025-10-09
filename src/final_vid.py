import os
import cv2
import torch
import argparse
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.utils import save_image
from PIL import Image
from ssai_model import SSAIModel  # import your model class

# ===== CONFIG =====
MODEL_PATH = "generated/models/test copy.pth"
ANALYSIS_DIR = "generated/analysis"
FPS = 30
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLASS_NAMES = ["Nothing", "Up", "Down", "Left", "Right"]

# ===== UTILS =====
def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def get_sorted_indices(folder):
    return sorted([
        int(os.path.splitext(f)[0]) for f in os.listdir(folder)
        if f.endswith(".png") and os.path.splitext(f)[0].isdigit()
    ])

def normalize_img(tensor):
    tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min() + 1e-8)
    return tensor

def interp_red_green(p):
    """Normal red→green map (we'll reverse for classifier)."""
    p = float(np.clip(p, 0.0, 1.0))
    r = int((1 - p) * 255)
    g = int(p * 255)
    return (0, g, r)

# ===== MAIN =====
def main(stream_dir):
    print(f"[INFO] Starting analysis for stream folder: {stream_dir}")
    processed_dir = os.path.join(ANALYSIS_DIR, "processed")
    video_dir = os.path.join(ANALYSIS_DIR, "videos")
    ensure_dir(processed_dir)
    ensure_dir(video_dir)

    # ---- Step 1: Read stream frames ----
    indices = get_sorted_indices(stream_dir)
    print(f"[INFO] Found {len(indices)} frames in stream.")
    imgs = [cv2.imread(os.path.join(stream_dir, f"{i}.png")) for i in indices]
    h, w, _ = imgs[0].shape

    # ---- Step 2: Save stream video ----
    stream_video_path = os.path.join(video_dir, "stream.mp4")
    out = cv2.VideoWriter(stream_video_path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (w, h))
    for img in imgs:
        out.write(img)
    out.release()
    print(f"[INFO] Stream video saved to {stream_video_path}")

    # ---- Step 3: Load model ----
    model = SSAIModel().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    # Register hooks to capture activations
    activation_stages = {}
    def get_hook(name):
        def hook(m, i, o):
            activation_stages[name] = o.detach()
        return hook
    relus = [m for m in model.cnn_stage if isinstance(m, nn.ReLU)]
    for i, r in enumerate(relus, 1):
        r.register_forward_hook(get_hook(f"stage_{i}"))

    # Transform
    transform = transforms.Compose([
        transforms.Resize((100, 60)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.25]*3)
    ])

    stage_outputs = {f"stage_{i}": [] for i in range(1, 5)}

    # ---- Step 4: Process each frame ----
    for idx in tqdm(indices, desc="Extracting CNN features"):
        img = cv2.cvtColor(cv2.imread(os.path.join(stream_dir, f"{idx}.png")), cv2.COLOR_BGR2RGB)
        t = transform(Image.fromarray(img)).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            _ = model.cnn_stage(t)

        for stage, feat in activation_stages.items():
            feat_cpu = feat.squeeze(0).cpu()
            stage_outputs[stage].append(feat_cpu)
            for ch in range(feat_cpu.shape[0]):
                ch_dir = os.path.join(processed_dir, stage, f"channel_{ch+1}")
                ensure_dir(ch_dir)
                save_image(normalize_img(feat_cpu[ch]), os.path.join(ch_dir, f"{idx}.png"))

    # ---- Step 5: Find brightest channels + make videos ----
    for stage, feats in stage_outputs.items():
        num_channels = feats[0].shape[0]
        mean_brightness = [np.mean([f[ch].mean().item() for f in feats]) for ch in range(num_channels)]
        top3 = np.argsort(mean_brightness)[-3:][::-1]
        print(f"[{stage}] Brightest channels: {top3 + 1}")
        rgb_frames = []
        for f in feats:
            chans = [(normalize_img(f[ch]) * 255).byte().numpy() for ch in top3]
            rgb = np.stack([cv2.resize(ch, (w, h)) for ch in chans], axis=2)
            rgb_frames.append(rgb)
        out_path = os.path.join(video_dir, f"{stage}_brightest.mp4")
        out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (w, h))
        for frame in rgb_frames:
            out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        out.release()

    print("[INFO] Stage videos saved. Generating final pipeline visualization...")

    # ---- Step 6: Combine all stage videos ----
    video_paths = [
        os.path.join(video_dir, "stream.mp4"),
        os.path.join(video_dir, "stage_1_brightest.mp4"),
        os.path.join(video_dir, "stage_2_brightest.mp4"),
        os.path.join(video_dir, "stage_3_brightest.mp4"),
        os.path.join(video_dir, "stage_4_brightest.mp4")
    ]

    caps = [cv2.VideoCapture(p) for p in video_paths]
    heights = [int(c.get(cv2.CAP_PROP_FRAME_HEIGHT)) for c in caps]
    widths = [int(c.get(cv2.CAP_PROP_FRAME_WIDTH)) for c in caps]
    tile_h = 240
    tile_ws = [int(w * (tile_h / h)) for w, h in zip(widths, heights)]
    panel_w = 220
    total_w = sum(tile_ws) + 16 * (len(tile_ws) + 1) + panel_w
    total_h = tile_h + 40
    out_path = os.path.join(ANALYSIS_DIR, "final_pipeline.mp4")
    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (total_w, total_h))

    # Helper transform for classification
    tform = transform

    for f_idx in tqdm(range(int(caps[0].get(cv2.CAP_PROP_FRAME_COUNT))), desc="Compositing pipeline"):
        base = np.zeros((total_h, total_w, 3), dtype=np.uint8)
        x = 16
        y = 20
        centers = []
        frames = []

        # ---- read and label stage tiles ----
        for i, cap in enumerate(caps):
            ret, frame = cap.read()
            if not ret:
                frame = np.zeros((heights[i], widths[i], 3), dtype=np.uint8)
            tile = cv2.resize(frame, (tile_ws[i], tile_h))

            if i == 0:
                lbl_main, lbl_sub = "Input", "(last 3 frames)"
            elif i < len(caps) - 1:
                lbl_main, lbl_sub = f"CNN-{i}", "(brightest 3 channels)"
            # else:
                # lbl_main, lbl_sub = "CNN-x", "(Action with lowest score is taken)"

            cv2.putText(tile, lbl_main, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
            cv2.putText(tile, lbl_sub, (1, tile_h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200,200,200), 1, cv2.LINE_AA)

            fade_stage_start = [0, 20, 40, 60, 80]
            fade = 1.0
            if i > 0:
                fade = np.clip((f_idx - fade_stage_start[i]) / 20.0, 0, 1)
            tile = (tile.astype(np.float32) * fade).astype(np.uint8)

            base[y:y+tile_h, x:x+tile_ws[i]] = tile
            centers.append((x + tile_ws[i] // 2, y + tile_h // 2))
            x += tile_ws[i] + 16
            frames.append(tile)

        # ---- arrows (white, spaced) ----
        for i in range(len(centers)-1):
            start = (centers[i][0] + 50, centers[i][1])
            end   = (centers[i+1][0] - 50, centers[i+1][1])
            cv2.arrowedLine(base, start, end, (255,255,255), 2, tipLength=0.05)
            if i == len(centers)-2:
                midx = (start[0] + end[0]) // 2
                cv2.putText(base, "Fully Connected", (midx-60, centers[i][1]-35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

        # ---- Run classifier on last 3 frames ----
        frame_indices = [max(0, f_idx - 2), max(0, f_idx - 1), f_idx]
        img_tensors = []
        for fi in frame_indices:
            caps[0].set(cv2.CAP_PROP_POS_FRAMES, fi)
            ret_prev, frame_prev = caps[0].read()
            if not ret_prev:
                frame_prev = np.zeros((heights[0], widths[0], 3), dtype=np.uint8)
            frame_prev_rgb = cv2.cvtColor(frame_prev, cv2.COLOR_BGR2RGB)
            pil_prev = Image.fromarray(frame_prev_rgb)
            t_prev = tform(pil_prev).unsqueeze(0).to(DEVICE)
            img_tensors.append(t_prev)

        with torch.no_grad():
            logits = model(img_tensors)
            probs = F.softmax(logits.squeeze(0), dim=0).cpu().numpy()

        # ---- Classifier panel ----
        px = total_w - panel_w + 10
        py = 10
        ph = tile_h
        cv2.rectangle(base, (px, py), (px + panel_w - 10, py + ph + 20), (20,20,20), -1)

        # Reverse mapping: lowest → green, highest → red
        norm_probs = (probs - probs.min()) / (probs.max() - probs.min() + 1e-8)
        bar_colors = [interp_red_green(1 - p) for p in norm_probs]

        bar_h = (ph - 50) // len(CLASS_NAMES)
        bar_w = panel_w - 40
        max_conf = probs.max()
        for i, cname in enumerate(CLASS_NAMES):
            val = probs[i]
            color = bar_colors[i]
            y0 = py + 40 + i*(bar_h + 5)
            cv2.rectangle(base, (px+20, y0), (px+20+int(bar_w*val), y0+bar_h), color, -1)
            cv2.putText(base, f"{cname} {val*100:4.1f}%", (px+20, y0+bar_h-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1, cv2.LINE_AA)

        # Tint classifier green when confidence is low
        if max_conf < 0.5:
            tint = np.zeros_like(base, dtype=np.uint8)
            cv2.rectangle(tint, (px, py), (px+panel_w-10, py+ph+20), (0,80,0), -1)
            base = cv2.addWeighted(base, 0.7, tint, 0.3, 0)
            cv2.putText(base, "Eliminate Choice\n(Lowest is performed)", (px+5, py+25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,255,0), 2, cv2.LINE_AA)

        out.write(base)

    out.release()
    for c in caps: c.release()
    print(f"\n✅ Final pipeline visualization saved → {out_path}")

# ===== ENTRY =====
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True, help="Path to the stream folder containing x.png images")
    args = parser.parse_args()
    main(args.dir)
