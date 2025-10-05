import os
import matplotlib.pyplot as plt
import numpy as np

root_dir = "generated/runs/dataset"

times = []
dirs = []

subdirs = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])[25:]

for subdir in subdirs:
    metadata_file = os.path.join(root_dir, subdir, "metadata.txt")
    
    if not os.path.exists(metadata_file):
        continue
    
    with open(metadata_file, "r") as f:
        lines = f.readlines()
        if not lines:
            continue
        first_line = lines[0].strip()
        last_line = lines[-1].strip()
        
        start_time = float(first_line.split(";")[1])
        parts = last_line.split(";")
        if len(parts) >= 2:
            try:
                time_val = float(parts[1])
                times.append(time_val - start_time)
                dirs.append(subdir)
            except ValueError:
                continue

times = np.array(times)
x = np.arange(len(times))

# --- Moving average ---
def moving_average(x, w=5):
    return np.convolve(x, np.ones(w)/w, mode="valid")

window_size = 15
smoothed_times = moving_average(times, window_size)

# --- Trend line ---
coeffs = np.polyfit(x, times, 1)
slope, intercept = coeffs[0], coeffs[1]
trend = np.polyval(coeffs, x)

# --- Count runs above thresholds in a rolling window ---
def rolling_count_above_threshold(data, threshold, window):
    result = []
    for i in range(len(data)):
        start = max(0, i - window + 1)
        count = np.sum(data[start:i+1] > threshold)
        result.append(count)
    return np.array(result)

window_for_counts = 50
thresholds = [20, 30, 40]
counts = {thr: rolling_count_above_threshold(times, thr, window_for_counts) for thr in thresholds}

# --- Plot setup ---
fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True, height_ratios=[2, 1])

# ---------------- Top plot (time + trend) ----------------
axes[0].plot(x, times, marker="o", label="Original")
axes[0].plot(x[window_size-1:], smoothed_times, marker="s", label=f"Moving Avg (window={window_size})")
axes[0].plot(x, trend, color="red", linestyle="--", label=f"Trend Line (slope={slope:.4f})")

# Conditional horizontal lines
y_lines = range(5, 65, 5)
for y in y_lines:
    if np.any((times >= y - 2.5) & (times <= y + 2.5)):
        axes[0].axhline(y=y, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)

axes[0].set_ylabel("Time (s)")
axes[0].set_title("Run durations with smoothing and trend")
axes[0].legend()

# ---------------- Bottom plot (counts above thresholds) ----------------
for thr, cnt in counts.items():
    axes[1].plot(x, cnt, label=f"> {thr}s (window={window_for_counts})")

axes[1].set_xlabel("Subdirectory index")
axes[1].set_ylabel("Count in window")
axes[1].set_title("Number of runs exceeding thresholds")
axes[1].legend()
axes[1].grid(True, linestyle="--", alpha=0.4)

# Simplify x-axis labels (only 5 evenly spaced ticks)
num_points = len(x)
if num_points > 0:
    tick_positions = np.linspace(0, num_points - 1, 5, dtype=int)
    tick_labels = [str(i) for i in tick_positions]
    axes[1].set_xticks(tick_positions)
    axes[1].set_xticklabels(tick_labels)

plt.tight_layout()
plt.savefig("graph.png")
# plt.show()
