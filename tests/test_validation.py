import os
import sys
import time
import numpy as np

# Path setup
PUPIL_SRC = "pupil_src"
SHARED_MODULES = "pupil_src/shared_modules"
if PUPIL_SRC not in sys.path:
    sys.path.insert(0, PUPIL_SRC)
if SHARED_MODULES not in sys.path:
    sys.path.insert(0, SHARED_MODULES)

import torch
# Apply monkeypatch for float4/float8 if needed
if not hasattr(torch, "float4_e2m1fn_x2"):
    torch.float4_e2m1fn_x2 = "dummy_float4_e2m1fn_x2"
if not hasattr(torch, "float8_e8m0fnu"):
    torch.float8_e8m0fnu = "dummy_float8_e8m0fnu"

import gaze_mapping.gazer_2d
from gaze_mapping.gazer_2d import Gazer2D
from accuracy_visualizer import Accuracy_Visualizer

class DummyIntrinsics:
    resolution = (640, 480)
    def unprojectPoints(self, pts, normalize=True):
        pts_3d = np.zeros((len(pts), 3))
        pts_3d[:, 0] = pts[:, 0] / 640.0 - 0.5
        pts_3d[:, 1] = pts[:, 1] / 480.0 - 0.5
        pts_3d[:, 2] = 1.0
        norms = np.linalg.norm(pts_3d, axis=1, keepdims=True)
        return pts_3d / norms

class DummyGPool:
    app = "capture"
    min_calibration_confidence = 0.6
    ipc_pub = type('IPCPub', (), {'notify': lambda self, n: None})()
    capture = type('Capture', (), {'intrinsics': DummyIntrinsics(), 'frame_size': (640, 480)})()
    user_dir = "."
    def get_timestamp(self):
        return time.time()

def test_validation_calibration_flow():
    print("Testing Validation execution via Calibration evaluation pipeline...")
    g_pool = DummyGPool()
    
    # 1. Create 5 calibration points to properly fit 2D gazer
    ref_init = []
    pupil_init = []
    positions = [(0.2, 0.2), (0.8, 0.2), (0.5, 0.5), (0.2, 0.8), (0.8, 0.8)]
    for i, pos in enumerate(positions):
        ts = 1000.0 + i
        ref_init.append({"norm_pos": pos, "screen_pos": (pos[0]*640, pos[1]*480), "timestamp": ts})
        pupil_init.append({"id": 0, "timestamp": ts, "confidence": 0.9, "method": "Mamba3 (T=5)", "norm_pos": pos, "ellipse": {"center": (pos[0]*640, pos[1]*480), "axes": (20, 20), "angle": 0}})
        
    calib_data_init = {"ref_list": ref_init, "pupil_list": pupil_init}
    gazer = Gazer2D(g_pool, calib_data=calib_data_init)
    print("Gazer created and fitted successfully!")
    
    # 2. Simulate validation data (Test button)
    pupil_list = []
    ref_list = []
    base_ts = 2000.0
    for i in range(10):
        ts = base_ts + i * 0.1
        p = {
            "id": 0,
            "timestamp": ts,
            "confidence": 0.9,
            "method": "Mamba3 (T=5)",
            "norm_pos": (0.5 + i*0.001, 0.5 + i*0.001),
            "ellipse": {"center": (320 + i, 240 + i), "axes": (20, 20), "angle": 0}
        }
        r = {
            "timestamp": ts,
            "norm_pos": (0.5, 0.5),
            "screen_pos": (320, 240)
        }
        pupil_list.append(p)
        ref_list.append(r)
        
    calib_data = {"ref_list": ref_list, "pupil_list": pupil_list}
    
    # Accuracy Visualizer instance
    acc_vis = Accuracy_Visualizer(g_pool)
    
    # Simulate notification dispatch when Test ("T") button finishes
    gazer._announce_calibration_setup(calib_data)
    gazer._announce_calibration_result(gazer.get_params())
    
    note_setup = {
        "subject": "calibration.setup",
        "gazer_class_name": gazer.__class__.__name__, # "Gazer2D"
        "calib_data": calib_data,
        "timestamp": g_pool.get_timestamp(),
        "record": True
    }
    note_result = {
        "subject": "calibration.result",
        "gazer_class_name": gazer.__class__.__name__, # "Gazer2D"
        "params": gazer.get_params(),
        "timestamp": g_pool.get_timestamp(),
        "record": True
    }
    
    acc_vis.on_notify(note_setup)
    acc_vis.on_notify(note_result)
    
    print(f"Is recent input complete: {acc_vis.recent_input.is_complete}")
    print(f"Accuracy visualizer calculation accuracy: {acc_vis.accuracy}")
    assert acc_vis.accuracy is not None, "Accuracy visualizer failed to compute accuracy for validation data!"
    print(f"Computed Angular Accuracy: {acc_vis.accuracy.result:.3f} degrees")
    print("Validation mapping test passed successfully!")

if __name__ == "__main__":
    test_validation_calibration_flow()
