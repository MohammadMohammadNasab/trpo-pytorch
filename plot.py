import os
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt

def extract_scalars_from_event_file(logdir, tag):
    """Extract scalars for a specific tag from a TensorBoard log directory."""
    event_acc = EventAccumulator(logdir)
    event_acc.Reload()
    if tag in event_acc.Tags()["scalars"]:
        scalars = event_acc.Scalars(tag)
        steps = [scalar.step for scalar in scalars]
        values = [scalar.value for scalar in scalars]
        return np.array(steps), np.array(values)
    else:
        return None, None

def plot_mean_with_deviation(logdirs, tag, xlabel, ylabel, title):
    """Plot mean with shaded deviation for a specific tag across multiple logs."""
    all_values = []
    common_steps = None

    # Extract data from each log
    for logdir in logdirs:
        steps, values = extract_scalars_from_event_file(logdir, tag)
        if steps is not None and values is not None:
            if common_steps is None:
                common_steps = steps
            elif not np.array_equal(common_steps, steps):
                print(f"Warning: Steps do not match in logdir: {logdir}")
                continue
            all_values.append(values)
    
    if not all_values:
        print(f"No data found for tag: {tag}")
        return

    # Compute mean and standard deviation
    all_values = np.array(all_values)
    mean_values = np.mean(all_values, axis=0)
    std_values = np.std(all_values, axis=0)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(common_steps, mean_values, label=f"Mean {tag}")
    plt.fill_between(common_steps, mean_values - std_values, mean_values + std_values, alpha=0.3, label="Standard Deviation")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

# Example usage
logdirs = ["/home/bamdad/rl/trpo-pytorch/experiments/hopper_2024-12-27 20:32:05_seed_42_ver_filtering_high_t_0.09_low_t_-0.10_damp_f_0.90",
           "/home/bamdad/rl/trpo-pytorch/experiments/hopper_2024-12-27 20:33:42_seed_42_ver_filtering_high_t_0.09_low_t_-0.10_damp_f_0.90",
           "/home/bamdad/rl/trpo-pytorch/experiments/hopper_2024-12-28 11:57:49_seed_10_ver_filtering_high_t_0.09_low_t_-0.10_damp_f_0.90"]  # Replace with your TensorBoard log directories
tag = "Reward/MeanReward"  # Replace with your desired tag
plot_mean_with_deviation(logdirs, tag, xlabel="Steps", ylabel="Mean Reward", title="Mean Reward with Deviation")
