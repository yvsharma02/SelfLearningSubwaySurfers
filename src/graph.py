import os
import matplotlib.pyplot as plt
import numpy as np

root_dir = "generated/runs/dataset"

times = []
dirs = []

# Get subdirectories sorted by name
subdirs = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])

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

# Convert times to numpy for easier handling
times = np.array(times)

# --- Moving average smoothing ---
def moving_average(x, w=5):
    return np.convolve(x, np.ones(w)/w, mode="valid")

window_size = 50  # adjust for smoother / less smooth curve
smoothed_times = moving_average(times, window_size)

# Plot
plt.figure(figsize=(10,5))
plt.plot(dirs, times, marker="o", label="Original")
plt.plot(dirs[window_size-1:], smoothed_times, marker="s", label=f"Moving Avg (window={window_size})")

plt.xticks(rotation=45, ha="right")
plt.xlabel("Subdirectory")
plt.ylabel("Time (last line)")
plt.title("Last line time per subdirectory (with smoothing)")
plt.legend()
plt.tight_layout()
# plt.show()
plt.savefig("graph.png")
