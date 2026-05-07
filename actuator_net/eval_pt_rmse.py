"""Evaluate the 12 TorchScript (.pt) LSTM actuator networks.

Runs inference on a single held-out pkl file (not used in training),
reports per-joint RMSE in Nm, and saves a plot.
"""
import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
import torch

from utils import JOINT_GROUPS, LSTM_WARMUP_LEN, load_single_experiment

# ── Configuration ────────────────────────────────────────────────────────────
_HERE          = os.path.dirname(os.path.abspath(__file__))
EXPERIMENT_DIR = os.path.join(_HERE, 'data', 'pkl')
MODEL_DIR      = _HERE

# Choose a pkl file that was NOT used during training.
# Disturbance datasets are new and were never included in training.
EVAL_PKL_NAME  = 'data_chirp_amplitude0.3_f00.1_f10.5_disturbance.pkl'

TORQUE_SCALE   = 0.01   # same scaling used during training (Nm → model units)
# ─────────────────────────────────────────────────────────────────────────────

eval_pkl_path = os.path.join(EXPERIMENT_DIR, EVAL_PKL_NAME)
if not os.path.exists(eval_pkl_path):
    raise FileNotFoundError(f"Eval pkl not found: {eval_pkl_path}")

jpe, jv, te = load_single_experiment(eval_pkl_path, torque_scaling=TORQUE_SCALE)
N = jpe.shape[0]
print(f"Eval dataset : {EVAL_PKL_NAME}")
print(f"Timesteps    : {N}\n")


n_joints = len(JOINT_GROUPS)
n_cols   = 2
n_rows   = (n_joints + n_cols - 1) // n_cols
fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 3.5 * n_rows))
axes = axes.flatten()

results = {}

for idx, (joint_indices, group_name) in enumerate(JOINT_GROUPS):
    model_path = os.path.join(MODEL_DIR, f"p73_lstm_{group_name}.pt")
    ax = axes[idx]

    if not os.path.exists(model_path):
        ax.set_title(f"{group_name}\n(no model)", fontsize=8)
        ax.axis('off')
        print(f"[{group_name}] No model found — skipped: {model_path}")
        continue

    model = torch.jit.load(model_path, map_location='cpu')
    model.eval()

    ji = joint_indices[0]

    # Shape (1, N, 2): one batch, full trajectory as the sequence dim, so the
    # LSTM carries hidden state across all N timesteps from a single h=0 init.
    # Using (N, 1, 2) instead would reset h, c every step → predictions become
    # context-free and RMSE blows up.
    xs = torch.stack([jpe[:, ji], jv[:, ji]], dim=1).unsqueeze(0)  # (1, N, 2)

    with torch.no_grad():
        y_pred_scaled, _ = model(xs)   # (1, N, 1)

    y_pred_scaled = y_pred_scaled[0, :, 0]  # (N,)
    y_true_scaled = te[:, ji]               # (N,)

    # Convert to Nm
    y_pred_Nm = y_pred_scaled / TORQUE_SCALE
    y_true_Nm = y_true_scaled / TORQUE_SCALE

    # Skip the first LSTM_WARMUP_LEN steps: hidden state is still building
    # up from h=0 there, and training does not score that region either.
    diff = y_pred_Nm[LSTM_WARMUP_LEN:] - y_true_Nm[LSTM_WARMUP_LEN:]
    rmse = float(torch.sqrt((diff ** 2).mean()))
    print(f"[{group_name:22s}]  RMSE={rmse:.4f} Nm  (skipped first {LSTM_WARMUP_LEN} steps)")
    results[group_name] = {"rmse": rmse}

    t_axis = np.arange(N) * 0.005  # 200 Hz → seconds
    ax.plot(t_axis, y_true_Nm.numpy(), label="Measured",   color="green", linewidth=1.5)
    ax.plot(t_axis, y_pred_Nm.numpy(), label="LSTM pred",  color="red",   linewidth=0.8, linestyle="--")
    ax.set_title(f"{group_name}  RMSE={rmse:.3f} Nm", fontsize=8)
    ax.set_ylabel("Torque [Nm]", fontsize=7)
    ax.set_xlabel("Time [s]", fontsize=7)
    ax.legend(fontsize=6)
    ax.grid(True, alpha=0.3)

for i in range(n_joints, len(axes)):
    axes[i].axis('off')

fig.suptitle(f"LSTM ActuatorNet (.pt) Evaluation — {EVAL_PKL_NAME}", fontsize=11)
plt.tight_layout()

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
out_path  = os.path.join(MODEL_DIR, f"eval_pt_rmse_{timestamp}.png")
plt.savefig(out_path, dpi=120)
plt.close()
print(f"\nSaved plot: {out_path}")

print("\n─── RMSE Summary ───────────────────────────────────────────")
for name, r in results.items():
    print(f"  {name:22s}  RMSE={r['rmse']:.4f} Nm")

if results:
    avg_rmse = np.mean([r["rmse"] for r in results.values()])
    print(f"\n  {'Average':22s}  RMSE={avg_rmse:.4f} Nm")
