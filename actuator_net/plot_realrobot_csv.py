"""Quick visualisation of the real-robot CSV.

Plots q_rel (relative joint position), qdot (velocity), and overlays action
(policy target) for each of the 12 actuated joints. Joint 12 (waist/platform)
is shown separately because it has no `action` channel.

Outputs three PNGs alongside this script:
  realrobot_position.png
  realrobot_velocity.png
  realrobot_torque.png
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
CSV_PATH = HERE / "data" / "realrobot_260402_152500_trimmed.csv"

JOINT_NAMES = [
    "left_hip_roll", "left_hip_pitch", "left_hip_yaw",
    "left_knee_pitch", "left_ankle_pitch", "left_ankle_roll",
    "right_hip_roll", "right_hip_pitch", "right_hip_yaw",
    "right_knee_pitch", "right_ankle_pitch", "right_ankle_roll",
]

DOWNSAMPLE = 10  # plot every 10th sample (1 kHz → 100 Hz visual)

cols_needed = (
    ["time"]
    + [f"q_rel_{i}" for i in range(12)]
    + [f"qdot_{i}" for i in range(12)]
    + [f"action_{i}" for i in range(12)]
    + [f"tau_joint_{i}" for i in range(12)]
)

print(f"Loading {CSV_PATH.name} ...")
df = pd.read_csv(CSV_PATH, usecols=cols_needed)
df = df.iloc[::DOWNSAMPLE].reset_index(drop=True)
print(f"  {len(df)} rows after {DOWNSAMPLE}x downsample, "
      f"t = [{df['time'].iloc[0]:.2f}, {df['time'].iloc[-1]:.2f}] s")

t = df["time"].to_numpy()


def grid_plot(title, ylabel, channels, out_path):
    """channels[j] is a list of (label, series) tuples for joint j."""
    fig, axes = plt.subplots(3, 4, figsize=(20, 10), sharex=True)
    for j, ax in enumerate(axes.flat):
        for label, series in channels[j]:
            ax.plot(t, series, linewidth=0.6, label=label)
        ax.set_title(JOINT_NAMES[j], fontsize=9)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc="upper right")
        if j >= 8:
            ax.set_xlabel("time (s)")
        if j % 4 == 0:
            ax.set_ylabel(ylabel)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    print(f"  saved {out_path.name}")
    plt.close(fig)


print("Plotting ...")
grid_plot(
    "Joint position: q_rel vs action (policy target)",
    "rad",
    [[("q_rel", df[f"q_rel_{j}"]), ("action", df[f"action_{j}"])]
     for j in range(12)],
    HERE / "realrobot_position.png",
)

grid_plot(
    "Joint velocity (qdot)",
    "rad/s",
    [[("qdot", df[f"qdot_{j}"])] for j in range(12)],
    HERE / "realrobot_velocity.png",
)

grid_plot(
    "Joint torque (tau_joint)",
    "Nm",
    [[("tau_joint", df[f"tau_joint_{j}"])] for j in range(12)],
    HERE / "realrobot_torque.png",
)

print("Done.")
