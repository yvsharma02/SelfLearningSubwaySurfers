import os
import matplotlib.pyplot as plt
import numpy as np

root_dir = "generated/runs/dataset"

dirs = []
times = []
subdirs = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])[25:]
lengths = [len(os.listdir(os.path.join(root_dir, d))) for d in subdirs]

#print(subdirs[lengths.index(max(lengths))])

for subdir in subdirs:
    metadata_file = os.path.join(root_dir, subdir, "metadata.txt")
    
    if not os.path.exists(metadata_file):
        continue
    
    with open(metadata_file, "r") as f:
        lines = f.readlines()
        if not lines:
            times.append(0)
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


sorted_dirs = [(x, times[i]) for i, x in enumerate(subdirs)]
sorted_dirs.sort(key=lambda x: x[1])

print(sorted_dirs[-10:])

def graph():
    global times
    times = np.array(times)
    x = np.arange(len(times))

    # --- Moving average ---
    def moving_average(x, w=5):
        return np.convolve(x, np.ones(w)/w, mode="valid")

    window_size = 15
    smoothed_times = moving_average(times, window_size)

    # --- Trend line (overall) ---
    coeffs = np.polyfit(x, times, 1)
    slope, intercept = coeffs[0], coeffs[1]
    trend = np.polyval(coeffs, x)

    # --- Trend line for top 10% slowest runs ---
    top_percentile_mask = times >= np.percentile(times, 90)
    top_x = x[top_percentile_mask]
    top_times = times[top_percentile_mask]
    if len(top_x) >= 2:
        top_coeffs = np.polyfit(top_x, top_times, 1)
        top_slope, top_intercept = top_coeffs[0], top_coeffs[1]
        top_trend = np.polyval(top_coeffs, x)
    else:
        top_slope, top_intercept = 0, 0
        top_trend = np.zeros_like(x)

    # --- Configurable thresholds ---
    thresholds = {
        10: 50,
        20: 50,  # threshold: window size
        30: 50,
        40: 50,
    }

    def rolling_count_above_threshold(data, threshold, window):
        result = []
        for i in range(len(data)):
            start = max(0, i - window + 1)
            count = np.sum(data[start:i+1] > threshold)
            result.append(count)
        return np.array(result)

    counts = {thr: rolling_count_above_threshold(times, thr, win) for thr, win in thresholds.items()}

    # --- Plot setup ---
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True, height_ratios=[2, 1])

    # ---------------- Top plot (scatter + trend lines) ----------------
    axes[0].scatter(x, times, s=25, alpha=0.8, label="Original")
    axes[0].plot(x[window_size-1:], smoothed_times, color="orange", label=f"Moving Avg (window={window_size})")
    axes[0].plot(x, trend, color="red", linestyle="--", label=f"Trend (slope={slope:.4f})")
    axes[0].plot(x, top_trend, color="purple", linestyle="--", label=f"Top 10% Trend (slope={top_slope:.4f})")

    # --- Horizontal dashed lines every 5s up to +15 of max ---
    y_max = np.max(times)
    y_limit = int(np.ceil((y_max + 15) / 5) * 5)
    for y in range(5, y_limit + 1, 5):
        axes[0].axhline(y=y, color='gray', linestyle='--', linewidth=0.8, alpha=0.4)

    axes[0].set_ylabel("Time (s)")
    axes[0].set_title("Run durations with smoothing and trend")
    axes[0].legend()

    # ---------------- Bottom plot (solid threshold lines) ----------------
    for thr, cnt in counts.items():
        axes[1].plot(x, cnt, label=f"> {thr}s (window={thresholds[thr]})")

    axes[1].set_xlabel("Subdirectory index")
    axes[1].set_ylabel("Count in window")
    axes[1].set_title("Number of runs exceeding thresholds")
    axes[1].legend()
    axes[1].grid(True, linestyle="--", alpha=0.4)

    # Simplify x-axis labels (5 evenly spaced ticks)
    num_points = len(x)
    if num_points > 0:
        tick_positions = np.linspace(0, num_points - 1, 5, dtype=int)
        tick_labels = [str(i) for i in tick_positions]
        axes[1].set_xticks(tick_positions)
        axes[1].set_xticklabels(tick_labels)

    plt.tight_layout()
    plt.savefig("graph.png")
    # plt.show()


def get_total_and_average_runtime(root_dir: str, skip_first_n: int = 25):
    if not times:
        print("⚠️ No valid runs found.")
        return 0.0, 0.0

    total_runtime = sum(times)
    average_runtime = total_runtime / len(times)

    print(f"✅ Total runtime: {total_runtime:.2f}s")
    print(f"✅ Average runtime: {average_runtime:.2f}s over {len(times)} runs")

    return total_runtime, average_runtime

graph()
# get_total_and_average_runtime(root_dir)