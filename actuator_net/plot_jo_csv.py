"""Plot every CSV in data/jo/ so we can decide which time ranges to keep.

For each CSV produces 3 PNGs (3x4 grid, one panel per leg joint):
  <stem>_position.png : q_raw (measured) vs q_des (commanded)
  <stem>_velocity.png : qdot
  <stem>_torque.png   : tau_meas_joint

Joint 12 (waist yaw) is dropped — the actuator network only covers the 12 leg
joints. Raw logs are 1 kHz; we downsample 10x for plotting only.
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

HERE     = Path(__file__).parent
JO_DIR   = HERE / "data" / "jo"
OUT_DIR  = HERE / "plots_jo"
OUT_DIR.mkdir(exist_ok=True)

JOINT_NAMES = [
    "left_hip_roll", "left_hip_pitch", "left_hip_yaw",
    "left_knee_pitch", "left_ankle_pitch", "left_ankle_roll",
    "right_hip_roll", "right_hip_pitch", "right_hip_yaw",
    "right_knee_pitch", "right_ankle_pitch", "right_ankle_roll",
]

DOWNSAMPLE = 10  # 1 kHz -> 100 Hz visual

COLS_NEEDED = (
    ["time"]
    + [f"q_raw_{i}"          for i in range(12)]
    + [f"q_des_{i}"          for i in range(12)]
    + [f"qdot_{i}"           for i in range(12)]
    + [f"tau_meas_joint_{i}" for i in range(12)]
)


def grid_plot(t, title, ylabel, channels, out_path):
    """channels[j] = list of (label, series) tuples for joint j."""
    fig, axes = plt.subplots(3, 4, figsize=(22, 11), sharex=True)
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
    plt.close(fig)


def process_csv(csv_path: Path):
    print(f"\n{csv_path.name}")
    df = pd.read_csv(csv_path, usecols=COLS_NEEDED)
    df = df.iloc[::DOWNSAMPLE].reset_index(drop=True)
    t = df["time"].to_numpy()
    t = t - t[0]  # zero-base time so cuts are easy to read off
    print(f"  {len(df)} rows after {DOWNSAMPLE}x downsample, "
          f"duration = {t[-1]:.2f} s")

    stem = csv_path.stem

    grid_plot(
        t,
        f"{stem}  —  position (q_raw vs q_des)",
        "rad",
        [[("q_raw", df[f"q_raw_{j}"]),
          ("q_des", df[f"q_des_{j}"])] for j in range(12)],
        OUT_DIR / f"{stem}_position.png",
    )
    grid_plot(
        t,
        f"{stem}  —  velocity (qdot)",
        "rad/s",
        [[("qdot", df[f"qdot_{j}"])] for j in range(12)],
        OUT_DIR / f"{stem}_velocity.png",
    )
    grid_plot(
        t,
        f"{stem}  —  measured joint torque (tau_meas_joint)",
        "Nm",
        [[("tau_meas_joint", df[f"tau_meas_joint_{j}"])] for j in range(12)],
        OUT_DIR / f"{stem}_torque.png",
    )
    print(f"  saved 3 PNGs to {OUT_DIR.relative_to(HERE)}/")


csv_files = sorted(JO_DIR.glob("*.csv"))
print(f"Found {len(csv_files)} CSVs in {JO_DIR.relative_to(HERE)}/")
for csv in csv_files:
    process_csv(csv)
print("\nDone.")
