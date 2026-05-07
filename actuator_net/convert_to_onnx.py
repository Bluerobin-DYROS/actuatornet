import torch
import os

_HERE    = os.path.dirname(os.path.abspath(__file__))
PT_DIR   = _HERE
ONNX_DIR = os.path.join(_HERE, "onnx")
os.makedirs(ONNX_DIR, exist_ok=True)

input_sizes = {
    "p73_left_hip_roll":      6,
    "p73_left_hip_pitch":     6,
    "p73_left_hip_yaw":       6,
    "p73_left_knee_pitch":    6,
    "p73_left_ankle_pitch":   6,
    "p73_left_ankle_roll":    6,
    "p73_right_hip_roll":     6,
    "p73_right_hip_pitch":    6,
    "p73_right_hip_yaw":      6,
    "p73_right_knee_pitch":   6,
    "p73_right_ankle_pitch":  6,
    "p73_right_ankle_roll":   6,
}

for name, in_size in input_sizes.items():
    pt_path   = os.path.join(PT_DIR,   f"{name}.pt")
    onnx_path = os.path.join(ONNX_DIR, f"{name}.onnx")

    if not os.path.exists(pt_path):
        print(f"Skipped (not found): {pt_path}")
        continue

    m = torch.jit.load(pt_path, map_location="cpu")
    m.eval()
    torch.onnx.export(
        m,
        torch.zeros(1, in_size),
        onnx_path,
        input_names=["input"],
        output_names=["output"],
        opset_version=11,
    )
    print(f"Converted: {name}.onnx")
