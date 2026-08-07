import os
import sys

# Path setup
PUPIL_SRC = "pupil_src"
SHARED_MODULES = "pupil_src/shared_modules"
if PUPIL_SRC not in sys.path:
    sys.path.insert(0, PUPIL_SRC)
if SHARED_MODULES not in sys.path:
    sys.path.insert(0, SHARED_MODULES)

import torch
# Apply monkeypatch
if not hasattr(torch, "float4_e2m1fn_x2"):
    torch.float4_e2m1fn_x2 = "dummy_float4_e2m1fn_x2"
if not hasattr(torch, "float8_e8m0fnu"):
    torch.float8_e8m0fnu = "dummy_float8_e8m0fnu"

# Create a dummy pupil_capture.log to test RMSE extraction
with open("pupil_capture.log", "w", encoding="utf-8") as f:
    f.write("Some logging info...\n")
    f.write("Fitting. RMSE =    3.45px in final iteration.\n")

from pupil_detector_plugins.detector_2d_plugin import Detector2DPlugin

class DummyRoi:
    bounds = (0, 0, 640, 400)

class DummyGPool:
    eye_id = 0
    roi = DummyRoi()
    display_mode = "video"
    capture = None

def test():
    print("Testing Detector2DPlugin on_notify accuracy logging...")
    g_pool = DummyGPool()
    plugin = Detector2DPlugin(g_pool=g_pool)
    plugin.active_model = "nnUNet Vivim (Mamba)"
    
    # Clean up old logs to be precise
    if os.path.exists("recordings"):
        for f in os.listdir("recordings"):
            if f.endswith(".log"):
                os.remove(os.path.join("recordings", f))
    
    # 1. Trigger calibration.successful
    print("Triggering calibration.successful...")
    plugin.on_notify({"subject": "calibration.successful"})
    
    # 2. Trigger accuracy_visualizer.data
    print("Triggering accuracy_visualizer.data...")
    plugin.on_notify({
        "subject": "accuracy_visualizer.data",
        "accuracy": 0.456,
        "precision": 0.123
    })
    
    # Check created logs
    recs = [f for f in os.listdir("recordings") if f.endswith(".log")]
    print("\nCreated log files:")
    for f in recs:
        print(f" - {f}")
        with open(os.path.join("recordings", f), "r") as fh:
            print(fh.read())
            print("-" * 40)

if __name__ == "__main__":
    test()
