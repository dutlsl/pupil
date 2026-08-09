import os
import sys
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

from gaze_mapping.gazer_2d import Gazer2D
from accuracy_visualizer import Accuracy_Visualizer, CorrelationError

class DummyIntrinsics:
    resolution = (640, 480)
    def unprojectPoints(self, pts, normalize=True):
        # pts: Nx2
        pts_3d = np.zeros((len(pts), 3))
        pts_3d[:, 0] = pts[:, 0] / 640.0 - 0.5
        pts_3d[:, 1] = pts[:, 1] / 480.0 - 0.5
        pts_3d[:, 2] = 1.0
        norms = np.linalg.norm(pts_3d, axis=1, keepdims=True)
        return pts_3d / norms

class DummyGPool:
    capture = type('Capture', (), {'intrinsics': DummyIntrinsics(), 'frame_size': (640, 480)})()
    user_dir = "."

def test_validation_mapping():
    print("Testing Validation pupil mapping & accuracy calculation with Mamba3 method tag...")
    
    # 1. Create simulated pupil_list with method='Mamba3 (T=5)'
    pupil_list = []
    ref_list = []
    base_ts = 1000.0
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
            "timestamp": ts + 0.02, # 20ms offset
            "norm_pos": (0.5, 0.5),
            "screen_pos": (320, 240)
        }
        pupil_list.append(p)
        ref_list.append(r)
        
    g_pool = DummyGPool()
    
    # Check gazer_2d filter_pupil_data
    gazer = Gazer2D(g_pool, params={})
    filtered = list(gazer.filter_pupil_data(pupil_list))
    print(f"Filtered pupil data count: {len(filtered)} / {len(pupil_list)}")
    assert len(filtered) == len(pupil_list), "filter_pupil_data failed to retain Mamba3 pupil data!"
    
    # Check accuracy visualizer calc_acc_prec_errlines
    res = Accuracy_Visualizer.calc_acc_prec_errlines(
        g_pool=g_pool,
        gazer_class=Gazer2D,
        gazer_params={"eye_0": {"calib_points_3d": [], "gaze_points_3d": []}}, # dummy params
        pupil_list=pupil_list,
        ref_list=ref_list,
        intrinsics=g_pool.capture.intrinsics,
        outlier_threshold=5.0
    )
    print(f"Accuracy result valid: {res.is_valid}")
    print(f"Accuracy: {res.accuracy.result:.3f} degrees")
    print(f"Precision: {res.precision.result:.3f} degrees")
    print("Test passed successfully!")

if __name__ == "__main__":
    test_validation_mapping()
