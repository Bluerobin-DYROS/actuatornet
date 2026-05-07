"""Crop data/jo/*.csv into pace-style txt-log directories.

For each (csv, time-range) pair this writes a directory under data/jo/ with
4 txt files matching the pace format (no timestamp, 13 columns):
  joint_position_log.txt   (q_raw_0..11  + platform=0.0)
  joint_velocity_log.txt   (qdot_0..11   + platform=0.0)
  joint_desired_log.txt    (q_des_0..11  + platform=0.0)
  torque_joint_log.txt     (tau_meas_joint_0..11 + platform=0.0)

The rows stay at the raw 1 kHz rate; convert_to_pkl.py is what decimates
5x -> 200 Hz when it builds the pkls.
"""

from pathlib import Path

import numpy as np
import pandas as pd

HERE   = Path(__file__).parent
JO_DIR = HERE / "data" / "jo"

# (csv_stem, [(t_start, t_end, suffix), ...])  — t in zero-based seconds
CUTS = [
    ("realrobot_260506_152154pdcurr2",            [(6.0,  15.0,  "")]),
    ("realrobot_260506_152521DWMpdcurr2",         [(8.0,  48.0,  "_a"), (100.0, 240.0, "_b")]),
    ("realrobot_260506_153149DWMacnetcurr1",      [(4.0,  80.0,  "")]),
    ("realrobot_260506_154208DWMpdcurr2",         [(2.0,  33.0,  "")]),
    ("realrobot_260506_160921DWMbasicacnetcurr2", [(15.0, 110.0, "_a"), (140.0, 180.0, "_b")]),
    ("realrobot_260506_161626acnetcurr3",         [(5.0,  65.0,  "_a"), (83.0,  130.0, "_b")]),
]

COLS_NEEDED = (
    ["time"]
    + [f"q_raw_{i}"          for i in range(12)]
    + [f"q_des_{i}"          for i in range(12)]
    + [f"qdot_{i}"           for i in range(12)]
    + [f"tau_meas_joint_{i}" for i in range(12)]
)


def write_pace_format(out_dir: Path, df_slice: pd.DataFrame):
    """Write 4 txt files with 13 cols each (12 joints + platform placeholder=0)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    n = len(df_slice)
    platform = np.zeros((n, 1))

    pos = np.hstack([df_slice[[f"q_raw_{i}"          for i in range(12)]].to_numpy(), platform])
    vel = np.hstack([df_slice[[f"qdot_{i}"           for i in range(12)]].to_numpy(), platform])
    des = np.hstack([df_slice[[f"q_des_{i}"          for i in range(12)]].to_numpy(), platform])
    trq = np.hstack([df_slice[[f"tau_meas_joint_{i}" for i in range(12)]].to_numpy(), platform])

    np.savetxt(out_dir / "joint_position_log.txt", pos, fmt="%.7g")
    np.savetxt(out_dir / "joint_velocity_log.txt", vel, fmt="%.7g")
    np.savetxt(out_dir / "joint_desired_log.txt",  des, fmt="%.7g")
    np.savetxt(out_dir / "torque_joint_log.txt",   trq, fmt="%.7g")


for stem, ranges in CUTS:
    csv_path = JO_DIR / f"{stem}.csv"
    if not csv_path.exists():
        print(f"!! missing: {csv_path.name}")
        continue
    print(f"\n{csv_path.name}")
    df = pd.read_csv(csv_path, usecols=COLS_NEEDED)
    t_rel = df["time"].to_numpy() - df["time"].iloc[0]

    for t_start, t_end, suffix in ranges:
        mask = (t_rel >= t_start) & (t_rel <= t_end)
        sub  = df[mask].reset_index(drop=True)
        # Drop any row containing NaN/Inf in the columns we use. Even one bad
        # sample destroys the whole pkl: convert_to_pkl.py uses scipy IIR
        # zero-phase decimation, which has infinite impulse response and
        # smears a single NaN across every output sample.
        used = [c for c in sub.columns if c != "time"]
        finite = np.isfinite(sub[used].to_numpy()).all(axis=1)
        n_dropped = (~finite).sum()
        sub = sub[finite].reset_index(drop=True)
        out_dir = JO_DIR / f"jo_{stem.replace('realrobot_', '')}{suffix}"
        write_pace_format(out_dir, sub)
        extra = f"  (dropped {n_dropped} non-finite rows)" if n_dropped else ""
        print(f"  [{t_start:6.1f}, {t_end:6.1f}] s  ->  "
              f"{len(sub):6d} rows  ->  {out_dir.relative_to(HERE)}/{extra}")

print("\nDone.")
