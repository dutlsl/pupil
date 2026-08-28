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
    calibration_counter = 1
    def __init__(self):
        self.active_gaze_mapping_plugin = None
    def get_timestamp(self):
        return time.time()

def make_validation_data(base_ts=2000.0, offset=0.0):
    val_markers = ScreenMarkerChoreographyPlugin.VALIDATION_PATTERNS["Diamond (Inward Cross / Default)"]
    pupil_val = []
    ref_val = []
    for i, vpos in enumerate(val_markers):
        for rep in range(5):
            ts = base_ts + i * 1.0 + rep * 0.05
            pupil_val.append({
                "id": 0,
                "timestamp": ts,
                "confidence": 0.9,
                "method": "Mamba3 (T=7)",
                "norm_pos": (vpos[0] + offset + rep * 0.0005, vpos[1] + offset + rep * 0.0005),
                "ellipse": {"center": ((vpos[0] + offset)*640, (vpos[1] + offset)*480), "axes": (20, 20), "angle": 0}
            })
            ref_val.append({
                "timestamp": ts,
                "norm_pos": vpos,
                "screen_pos": (vpos[0]*640, vpos[1]*480)
            })
    return pupil_val, ref_val

def test_5stack_demo():
    print("\n================================================================================")
    print("🧪 TEST 1: Full 5-Stack Validation Scenario (5 Successful Rounds)")
    print("================================================================================")
    g_pool = DummyGPool()
    
    # 1. Fit Gazer2D on 9 points
    positions = ScreenMarkerChoreographyPlugin.CALIBRATION_PATTERNS["9-Point (3x3 Grid / Ours)"]
    ref_calib = []
    pupil_calib = []
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
    g_pool.active_gaze_mapping_plugin = gazer
    gazer_params = gazer.get_params()

    acc_vis = Accuracy_Visualizer(g_pool, outlier_threshold=1.2, enable_5stack_summary=True)

    # 2. Trigger Calibration Start event
    acc_vis.on_notify({"subject": "calibration.started"})
    assert acc_vis._stack_active is True
    assert len(acc_vis._val_stack) == 0
    print("✅ Calibration start event triggered 5-stack session initialization.")

    # Calibration result notification
    acc_vis.on_notify({
        "subject": "calibration.result",
        "gazer_class_name": gazer.__class__.__name__,
        "params": gazer_params
    })

    # 3. Send 5 Validation rounds
    offsets = [0.001, 0.003, 0.0005, 0.004, 0.002]
    for r_idx, off in enumerate(offsets, start=1):
        p_val, r_val = make_validation_data(base_ts=2000.0 + r_idx * 100, offset=off)
        val_note = ChoreographyNotification(
            mode=ChoreographyMode.VALIDATION,
            action=ChoreographyAction.DATA,
            gazer_class_name=gazer.__class__.__name__,
            gazer_params=gazer_params,
            pupil_list=p_val,
            ref_list=r_val,
            timestamp=g_pool.get_timestamp(),
            record=True,
        ).to_dict()
        acc_vis.on_notify(val_note)
        if r_idx < 5:
            assert len(acc_vis._val_stack) == r_idx, f"Expected {r_idx} rounds in stack, got {len(acc_vis._val_stack)}"
    
    # After round 5, stack should be printed and reset to 0
    assert len(acc_vis._val_stack) == 0, "Stack was not reset after 5-stack report!"
    print("✅ Successfully verified 5-stack completion, console output, and reset.")

    print("\n================================================================================")
    print("🧪 TEST 2: 5-Stack with Fails Included in Count (e.g. 1 Fail + 4 OK)")
    print("================================================================================")
    acc_vis.on_notify({"subject": "calibration.started"})
    
    # Round 1: OK
    p_val, r_val = make_validation_data(base_ts=3000.0, offset=0.001)
    acc_vis.on_notify(ChoreographyNotification(
        mode=ChoreographyMode.VALIDATION, action=ChoreographyAction.DATA,
        gazer_class_name=gazer.__class__.__name__, gazer_params=gazer_params,
        pupil_list=p_val, ref_list=r_val, timestamp=g_pool.get_timestamp(), record=True,
    ).to_dict())
    assert len(acc_vis._val_stack) == 1

    # Round 2: FAIL (Empty/Outlier pupil data)
    fail_p_val = [{"id": 0, "timestamp": 3100.0, "confidence": 0.1, "method": "Mamba3 (T=7)", "norm_pos": (999.0, 999.0), "ellipse": {"center": (0, 0), "axes": (0, 0), "angle": 0}}]
    acc_vis.on_notify(ChoreographyNotification(
        mode=ChoreographyMode.VALIDATION, action=ChoreographyAction.DATA,
        gazer_class_name=gazer.__class__.__name__, gazer_params=gazer_params,
        pupil_list=fail_p_val, ref_list=r_val, timestamp=g_pool.get_timestamp(), record=True,
    ).to_dict())
    assert len(acc_vis._val_stack) == 2
    assert np.isnan(acc_vis._val_stack[1]), "Failed round should be recorded as NaN!"

    # Rounds 3, 4, 5: OK
    for r_idx in [3, 4, 5]:
        p_val, r_val = make_validation_data(base_ts=3000.0 + r_idx * 100, offset=0.002)
        acc_vis.on_notify(ChoreographyNotification(
            mode=ChoreographyMode.VALIDATION, action=ChoreographyAction.DATA,
            gazer_class_name=gazer.__class__.__name__, gazer_params=gazer_params,
            pupil_list=p_val, ref_list=r_val, timestamp=g_pool.get_timestamp(), record=True,
        ).to_dict())
    
    assert len(acc_vis._val_stack) == 0, "Stack was not reset after 5-stack report!"
    print("✅ Verified that Fail round is counted as 1 of the 5 stacks without crashing.")

    print("\n================================================================================")
    print("🧪 TEST 3: Interruption Reset (New Calibration Starts after 2 Rounds)")
    print("================================================================================")
    acc_vis.on_notify({"subject": "calibration.started"})
    
    # 2 rounds
    for r_idx in [1, 2]:
        p_val, r_val = make_validation_data(base_ts=4000.0 + r_idx * 100, offset=0.001)
        acc_vis.on_notify(ChoreographyNotification(
            mode=ChoreographyMode.VALIDATION, action=ChoreographyAction.DATA,
            gazer_class_name=gazer.__class__.__name__, gazer_params=gazer_params,
            pupil_list=p_val, ref_list=r_val, timestamp=g_pool.get_timestamp(), record=True,
        ).to_dict())
    assert len(acc_vis._val_stack) == 2
    print(f"Stack has {len(acc_vis._val_stack)} rounds before interruption.")

    # User starts a new calibration!
    acc_vis.on_notify({"subject": "calibration.should_start"})
    assert len(acc_vis._val_stack) == 0, "Incomplete stack was not cleared on new calibration start!"
    print("✅ Successfully verified interruption reset when new calibration starts.")

    print("\n================================================================================")
    print("🧪 TEST 4: Interruption Reset on Cancel / Stopping Event")
    print("================================================================================")
    acc_vis.on_notify({"subject": "calibration.started"})
    p_val, r_val = make_validation_data(base_ts=5000.0, offset=0.001)
    acc_vis.on_notify(ChoreographyNotification(
        mode=ChoreographyMode.VALIDATION, action=ChoreographyAction.DATA,
        gazer_class_name=gazer.__class__.__name__, gazer_params=gazer_params,
        pupil_list=p_val, ref_list=r_val, timestamp=g_pool.get_timestamp(), record=True,
    ).to_dict())
    assert len(acc_vis._val_stack) == 1

    # User cancels calibration / stops
    acc_vis.on_notify({"subject": "calibration.stopped"})
    assert len(acc_vis._val_stack) == 0, "Incomplete stack was not cleared on calibration.stopped!"
    print("✅ Successfully verified interruption reset on calibration.stopped.")

    print("\n================================================================================")
    print("🧪 TEST 5: Persistent Option State Across Plugin Re-instantiations")
    print("================================================================================")
    # Start with default: False
    fresh_gpool = DummyGPool()
    acc_inst1 = Accuracy_Visualizer(fresh_gpool)
    assert acc_inst1.enable_5stack_summary is False, "Initial default must be False"
    print("✅ Default initial state is False.")

    # User toggles switch ON in UI
    acc_inst1.enable_5stack_summary = True
    assert fresh_gpool.enable_5stack_summary is True, "g_pool must reflect toggled state"
    print("✅ User toggled switch to True -> stored in g_pool.")

    # Simulate start_plugin("Accuracy_Visualizer") recreating instance with empty args
    acc_inst2 = Accuracy_Visualizer(fresh_gpool)
    assert acc_inst2.enable_5stack_summary is True, "Re-created instance must preserve user's True toggle!"
    print("✅ Re-instantiated Accuracy_Visualizer preserves enable_5stack_summary=True!")

    # Verify get_init_dict preserves the setting
    init_dict = acc_inst2.get_init_dict()
    assert init_dict.get("enable_5stack_summary") is True
    print("✅ get_init_dict correctly includes enable_5stack_summary=True.")

    # User toggles switch OFF in UI
    acc_inst2.enable_5stack_summary = False
    assert fresh_gpool.enable_5stack_summary is False
    acc_inst3 = Accuracy_Visualizer(fresh_gpool)
    assert acc_inst3.enable_5stack_summary is False
    print("✅ User toggled switch to False -> preserved across re-instantiations.")

    print("\n================================================================================")
    print("🎉 ALL 5-STACK VALIDATION DEMO TESTS PASSED SUCCESSFULLY!")
    print("================================================================================")

if __name__ == "__main__":
    test_5stack_demo()
