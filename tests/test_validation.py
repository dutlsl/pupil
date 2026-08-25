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
from calibration_choreography.screen_marker_plugin import ScreenMarkerChoreographyPlugin

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
    capture = type('Capture', (), {'intrinsics': DummyIntrinsics(), 'frame_size': (640, 480), 'online': True})()
    user_dir = "."
    def __init__(self):
        self.active_gaze_mapping_plugin = None
    def get_timestamp(self):
        return time.time()

def test_validation_workflow():
    print("=== Testing Complete Calibration & Validation Workflow with Mamba3 ===")
    g_pool = DummyGPool()
    
    # Mock GLFW for headless test execution
    try:
        import glfw
        glfw.init()
    except Exception:
        pass

    # 1. Test Choreography Marker Patterns
    patterns = ["5-Point (Pupil Labs Default)", "9-Point (3x3 Grid / Ours)", "12-Point (4x3 Dense Grid / New)"]
    for pat in patterns:
        markers = ScreenMarkerChoreographyPlugin.CALIBRATION_PATTERNS[pat]
        expected_len = int(pat.split("-")[0])
        assert len(markers) == expected_len, f"Pattern {pat} expected {expected_len} markers, got {len(markers)}"
        print(f"✅ Verified calibration pattern '{pat}': {len(markers)} markers configured.")
        
    for val_name, val_targets in ScreenMarkerChoreographyPlugin.VALIDATION_PATTERNS.items():
        assert len(val_targets) == 4, f"Validation pattern {val_name} expected 4 targets, got {len(val_targets)}"
        print(f"✅ Verified validation pattern '{val_name}': {val_targets}")
    val_markers = ScreenMarkerChoreographyPlugin.VALIDATION_PATTERNS["Diamond (Inward Cross / Default)"]

    # 2. Calibration Phase: Train Gazer2D on 9 points
    ref_calib = []
    pupil_calib = []
    positions = ScreenMarkerChoreographyPlugin.CALIBRATION_PATTERNS["9-Point (3x3 Grid / Ours)"]
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
    
    # 3. Validation Phase (Vanilla 4-point cross): Independent Test Data
    pupil_val = []
    ref_val = []
    base_ts = 2000.0
    for i, vpos in enumerate(val_markers):
        for rep in range(5):
            ts = base_ts + i * 1.0 + rep * 0.05
            pupil_val.append({
                "id": 0,
                "timestamp": ts,
                "confidence": 0.9,
                "method": "Mamba3 (T=7)",
                "norm_pos": (vpos[0] + rep * 0.0005, vpos[1] + rep * 0.0005),
                "ellipse": {"center": (vpos[0]*640, vpos[1]*480), "axes": (20, 20), "angle": 0}
            })
            ref_val.append({
                "timestamp": ts,
                "norm_pos": vpos,
                "screen_pos": (vpos[0]*640, vpos[1]*480)
            })
        
    # 4. Accuracy_Visualizer receives validation.data notification (Route B - Calibration ON)
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

    # 5. Test Calibration OFF mode (Bypass)
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
