"""Convert data/jo/*.csv into pkl files in the same format as convert_to_pkl.py.

Each (csv, time-range) pair produces one pkl. CSVs with two ranges produce two
pkls (`_a`, `_b`). Time ranges are zero-based seconds relative to the start of
the CSV (matching plot_jo_csv.py).

Pipeline per range: crop @ 1 kHz -> 5x anti-aliased decimation -> 200 Hz pkl.
Joint 12 (waist yaw) is dropped; col 0 of every joint list is a dummy
"platform" placeholder that utils.py strips via [1:].
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import decimate

HERE       = Path(__file__).parent
JO_DIR     = HERE / "data" / "jo"
OUTPUT_DIR = HERE / "data" / "pkl"
OUTPUT_DIR.mkdir(exist_ok=True)

DECIMATION_FACTOR = 5  # 1 kHz -> 200 Hz

JOINT_NAMES = [
    "platform",
    "left_hip_roll", "left_hip_pitch", "left_hip_yaw",
    "left_knee_pitch", "left_ankle_pitch", "left_ankle_roll",
    "right_hip_roll", "right_hip_pitch", "right_hip_yaw",
    "right_knee_pitch", "right_ankle_pitch", "right_ankle_roll",
]

# (csv_stem, [(t_start, t_end, suffix), ...])
CUTS = [
    ("realrobot_260506_152154pdcurr2",          [(6.0,  15.0,  "")]),
    ("realrobot_260506_152521DWMpdcurr2",       [(8.0,  48.0,  "_a"), (100.0, 240.0, "_b")]),
    ("realrobot_260506_153149DWMacnetcurr1",    [(4.0,  80.0,  "")]),
    ("realrobot_260506_154208DWMpdcurr2",       [(2.0,  33.0,  "")]),
    ("realrobot_260506_160921DWMbasicacnetcurr2", [(15.0, 110.0, "_a"), (140.0, 180.0, "_b")]),
    ("realrobot_260506_161626acnetcurr3",       [(5.0,  65.0,  "_a"), (83.0,  130.0, "_b")]),
]

COLS_NEEDED = (
    ["time"]
    + [f"q_raw_{i}"          for i in range(12)]
    + [f"q_des_{i}"          for i in range(12)]
    + [f"qdot_{i}"           for i in range(12)]
    + [f"tau_meas_joint_{i}" for i in range(12)]
)


def build_records(df: pd.DataFrame, t_start: float, t_end: float):
    t = df["time"].to_numpy()
    t_rel = t - t[0]
    mask = (t_rel >= t_start) & (t_rel <= t_end)
    if mask.sum() < DECIMATION_FACTOR * 4:
        raise ValueError(f"slice [{t_start}, {t_end}] too short ({mask.sum()} samples)")

    sub = df[mask].reset_index(drop=True)
    pos = sub[[f"q_raw_{i}"          for i in range(12)]].to_numpy()
    des = sub[[f"q_des_{i}"          for i in range(12)]].to_numpy()
    vel = sub[[f"qdot_{i}"           for i in range(12)]].to_numpy()
    trq = sub[[f"tau_meas_joint_{i}" for i in range(12)]].to_numpy()
    ts  = sub["time"].to_numpy() - sub["time"].iloc[0]

    pos = decimate(pos, DECIMATION_FACTOR, axis=0, ftype="iir", zero_phase=True)
    des = decimate(des, DECIMATION_FACTOR, axis=0, ftype="iir", zero_phase=True)
    vel = decimate(vel, DECIMATION_FACTOR, axis=0, ftype="iir", zero_phase=True)
    trq = decimate(trq, DECIMATION_FACTOR, axis=0, ftype="iir", zero_phase=True)
    ts  = ts[::DECIMATION_FACTOR]

    n = min(len(ts), len(pos))
    pos, des, vel, trq, ts = pos[:n], des[:n], vel[:n], trq[:n], ts[:n]

    records = []
    for i in range(n):
        t_sec  = int(ts[i])
        t_nsec = int(round((ts[i] - t_sec) * 1e9))
        records.append({
            "joint_names":            JOINT_NAMES,
            "joint_positions":        [0.0] + pos[i].tolist(),
            "joint_velocities":       [0.0] + vel[i].tolist(),
            "joint_efforts":          [0.0] + trq[i].tolist(),
            "joint_position_command": [0.0] + des[i].tolist(),
            "time_sec":               t_sec,
            "time_nsec":              t_nsec,
        })
    return records


for stem, ranges in CUTS:
    csv_path = JO_DIR / f"{stem}.csv"
    if not csv_path.exists():
        print(f"!! missing: {csv_path.name}")
        continue
    print(f"\n{csv_path.name}")
    df = pd.read_csv(csv_path, usecols=COLS_NEEDED)
    for t_start, t_end, suffix in ranges:
        records  = build_records(df, t_start, t_end)
        out_name = f"jo_{stem.replace('realrobot_', '')}{suffix}.pkl"
        out_path = OUTPUT_DIR / out_name
        with open(out_path, "wb") as f:
            pickle.dump(records, f)
        print(f"  [{t_start:6.1f}, {t_end:6.1f}] s  ->  "
              f"{len(records):5d} steps  ->  {out_name}")

print(f"\nDone. PKLs saved to {OUTPUT_DIR.relative_to(HERE)}/")
