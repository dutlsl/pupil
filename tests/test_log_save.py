import os
import sys
import time

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
    plugin.active_model = "Mamba3 (T=5)"
    
    # Clean up old logs to be precise
    if os.path.exists("recordings"):
        for f in os.listdir("recordings"):
            if f.endswith(".log"):
                os.remove(os.path.join("recordings", f))
                
    # 1. Simulate Calibration Run
    print("Simulating Calibration logging...")
    with open("pupil_capture.log", "w", encoding="utf-8") as f:
        f.write("Some logging info...\n")
        f.write("Starting  Calibration\n")
        f.write("Fitting. RMSE =    3.45px in final iteration.\n")
        f.write("accuracy_visualizer: Angular accuracy: 0.721 degrees\n")
        f.write("accuracy_visualizer: Angular precision: 0.221 degrees\n")
        
    print("Triggering calibration.successful...")
    plugin.on_notify({"subject": "calibration.successful"})
    time.sleep(0.6) # Let thread finish
    
    # 2. Simulate Validation Run
    print("Simulating Validation logging...")
    with open("pupil_capture.log", "a", encoding="utf-8") as f:
        f.write("Starting  Validation\n")
        f.write("accuracy_visualizer: Angular accuracy: 1.028 degrees\n")
        f.write("accuracy_visualizer: Angular precision: 0.195 degrees\n")
        
    print("Triggering validation.stopped...")
    plugin.on_notify({"subject": "validation.stopped"})
    time.sleep(0.6) # Let thread finish
    
    # Check created logs
    recs = sorted([f for f in os.listdir("recordings") if f.endswith(".log")])
    print("\nCreated log files:")
    for f in recs:
        print(f" - {f}")
        with open(os.path.join("recordings", f), "r") as fh:
            print(fh.read())
            print("-" * 40)

if __name__ == "__main__":
    test()
