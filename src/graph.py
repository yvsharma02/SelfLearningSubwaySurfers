import os
import matplotlib.pyplot as plt
import numpy as np

root_dir = "generated/runs/dataset_old10"

times = []
dirs = []

subdirs = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])[25:216]

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
x = np.arange(len(times))  # numeric x-axis instead of subdir strings

def moving_average(x, w=5):
    return np.convolve(x, np.ones(w)/w, mode="valid")

window_size = 15  # adjust for smoother / less smooth curve
smoothed_times = moving_average(times, window_size)

coeffs = np.polyfit(x, times, 1)   # degree=1 for linear
slope, intercept = coeffs[0], coeffs[1]
trend = np.polyval(coeffs, x)

plt.figure(figsize=(10,5))
plt.plot(x, times, marker="o", label="Original")
plt.plot(x[window_size-1:], smoothed_times, marker="s", label=f"Moving Avg (window={window_size})")
plt.plot(x, trend, color="red", linestyle="--", label=f"Trend Line (slope={slope:.4f})")

y_lines = range(5, 65, 5)
for y in y_lines:
    # Check if any data point lies within ±2.5 of this line
    if np.any((times >= y - 2.5) & (times <= y + 2.5)):
        plt.axhline(y=y, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)

num_points = len(x)
if num_points > 0:
    tick_positions = np.linspace(0, num_points - 1, 5, dtype=int)
    tick_labels = [str(i) for i in tick_positions]
    plt.xticks(tick_positions, tick_labels)
else:
    plt.xticks([])

plt.xlabel("Subdirectory index")
plt.ylabel("Time (last line - start time)")
plt.title("Last line time per subdirectory (with smoothing + trend)")
plt.legend()
plt.tight_layout()
plt.savefig("graph.png")
