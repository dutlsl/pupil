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
from calibration_choreography import ChoreographyMode, ChoreographyAction, ChoreographyNotification

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
    def __init__(self):
        self.active_gaze_mapping_plugin = None
    def get_timestamp(self):
        return time.time()

def test_validation_workflow():
    print("=== Testing Complete Calibration & Validation Workflow with Mamba3 ===")
    g_pool = DummyGPool()
    
    # 1. Calibration Phase (Route A): Train Gazer2D on 5 points
    ref_calib = []
    pupil_calib = []
    positions = [(0.2, 0.2), (0.8, 0.2), (0.5, 0.5), (0.2, 0.8), (0.8, 0.8)]
    for i, pos in enumerate(positions):
        ts = 1000.0 + i
        ref_calib.append({"norm_pos": pos, "screen_pos": (pos[0]*640, pos[1]*480), "timestamp": ts})
        pupil_calib.append({
            "id": 0,
            "timestamp": ts,
            "confidence": 0.9,
            "method": "Mamba3 (T=7)",
            "norm_pos": pos,
            "ellipse": {"center": (pos[0]*640, pos[1]*480), "axes": (20, 20), "angle": 0}
        })
        
    calib_data = {"ref_list": ref_calib, "pupil_list": pupil_calib}
    gazer = Gazer2D(g_pool, calib_data=calib_data)
    assert gazer.alive, "Gazer2D failed to fit on calibration data!"
    print("✅ Calibration Gazer fitted successfully with Mamba3 pupil data.")
    
    # Register active gazer plugin in g_pool
    g_pool.active_gaze_mapping_plugin = gazer
    gazer_params = gazer.get_params()
    
    # 2. Validation Phase (Route B): Independent Test Data
    pupil_val = []
    ref_val = []
    base_ts = 2000.0
    for i in range(30):
        ts = base_ts + i * 0.05
        pupil_val.append({
            "id": 0,
            "timestamp": ts,
            "confidence": 0.9,
            "method": "Mamba3 (T=7)",
            "norm_pos": (0.5 + i * 0.001, 0.5 + i * 0.001),
            "ellipse": {"center": (320 + i, 240 + i), "axes": (20, 20), "angle": 0}
        })
        ref_val.append({
            "timestamp": ts,
            "norm_pos": (0.5, 0.5),
            "screen_pos": (320, 240)
        })
        
    # 3. Accuracy_Visualizer receives validation.data notification (Route B - Calibration ON)
    acc_vis = Accuracy_Visualizer(g_pool)
    
    val_notification = ChoreographyNotification(
        mode=ChoreographyMode.VALIDATION,
        action=ChoreographyAction.DATA,
        gazer_class_name=gazer.__class__.__name__,
        gazer_params=gazer_params,
        pupil_list=pupil_val,
        ref_list=ref_val,
        timestamp=g_pool.get_timestamp(),
        record=True,
    ).to_dict()
    
    acc_vis.on_notify(val_notification)
    
    print(f"Is validation input complete: {acc_vis.recent_input.is_complete}")
    assert acc_vis.accuracy is not None, "Accuracy visualizer failed to compute accuracy for validation data!"
    print(f"✅ [Calibration ON] Validation Angular Accuracy: {acc_vis.accuracy.result:.3f} degrees (used: {acc_vis.accuracy.num_used}/{acc_vis.accuracy.num_total})")
    print(f"✅ [Calibration ON] Validation Angular Precision: {acc_vis.precision.result:.3f} degrees (used: {acc_vis.precision.num_used}/{acc_vis.precision.num_total})")

    # 4. Test Calibration OFF mode (Bypass)
    print("\n--- Testing Calibration OFF (Bypass) Mode ---")
    gazer.enable_calibration = False
    g_pool.enable_calibration = False
    gaze_mapped = list(gazer.map_pupil_to_gaze(pupil_val))
    assert len(gaze_mapped) > 0, "Gazer2D failed to output gaze in bypass mode!"
    print(f"✅ [Calibration OFF] Raw Bypass Gaze Sample: norm_pos={gaze_mapped[0]['norm_pos']}")
    assert gaze_mapped[0]["norm_pos"] == pupil_val[0]["norm_pos"], "Bypass mode did not preserve raw pupil norm_pos!"
    print("✅ Verified that Calibration OFF accurately passes raw pupil norm_pos without regression.")
    
    print("=== All Validation Tests Passed Successfully! ===")

if __name__ == "__main__":
    test_validation_workflow()
