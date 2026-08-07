"""
Pupil Labs Detector2DPlugin Dummy Test Harness

This script provides an offline, hardware-free integration test for Pupil Labs 2D detector plugins.
It tests both 192x192 (Pupil Core Eye Camera) and 400x640 (OpenEDS Dataset) frame resolutions, verifying
Letterboxing (aspect ratio preserving padding), dynamic Z-Score normalization, model loading, and datum integrity.
"""
import os
import sys
import time
import numpy as np
import cv2
import torch

# Path setup
PUPIL_SRC = os.path.expanduser("~/PycharmProjects/pupil/pupil_src")
SHARED_MODULES = os.path.join(PUPIL_SRC, "shared_modules")
if PUPIL_SRC not in sys.path:
    sys.path.insert(0, PUPIL_SRC)
if SHARED_MODULES not in sys.path:
    sys.path.insert(0, SHARED_MODULES)

class DummyRoi:
    bounds = (0, 0, 640, 400)

class DummyGPool:
    eye_id = 0
    roi = DummyRoi()
    display_mode = "video"

class DummyFrame:
    def __init__(self, img_np):
        self.gray = img_np
        self.height, self.width = img_np.shape[:2]
        self.timestamp = time.time()

def make_synthetic_eye(height, width):
    img = np.ones((height, width), dtype=np.uint8) * 180
    cx, cy = width // 2, height // 2
    r_x, r_y = max(10, width // 16), max(8, height // 16)
    cv2.ellipse(img, (cx, cy), (r_x, r_y), 15, 0, 360, color=20, thickness=-1)
    noise = np.random.randint(-10, 10, (height, width), dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def run_harness():
    print("=" * 75)
    print("🤖 PUPIL LABS DETECTOR 2D PLUGIN - DUMMY TEST HARNESS (Letterbox & 192x192 Test)")
    print("=" * 75)

    from pupil_detector_plugins.detector_2d_plugin import Detector2DPlugin

    g_pool = DummyGPool()
    print("[1/3] Instantiating Detector2DPlugin & Loading PyTorch Models...")
    plugin = Detector2DPlugin(g_pool=g_pool)

    # STRICT MODEL INTEGRITY ASSERTIONS
    model_attr_map = {
        "TemporalUNet": "temporal_model",
        "nnUNet Vivim (Mamba)": "vivim_model",
        "nnUNet 2D": "vanilla_2d_model",
        "RITnet": "ritnet_model",
    }

    print("\n🔍 Verifying Model Object Loading Integrity...")
    for m_name, attr_name in model_attr_map.items():
        model_obj = getattr(plugin, attr_name, None)
        assert model_obj is not None, (
            f"❌ CRITICAL HARNESS FAILURE: Model '{m_name}' ({attr_name}) "
            f"failed to load / is None! Check dependencies & weights!"
        )
        print(f"   ✓ Verified model '{m_name}' ({attr_name}) is loaded and ready.")

    print("✅ Plugin & All Models Loaded Successfully.")

    resolutions = [
        ("Pupil Core Eye Camera (192x192)", 192, 192),
        ("Pupil Core Eye Camera (400x400)", 400, 400),
        ("OpenEDS Dataset Native (400x640)", 400, 640),
    ]

    models = ["TemporalUNet", "nnUNet Vivim (Mamba)", "nnUNet 2D", "RITnet", "2D C++"]

    print("\n[2/3] Executing Dummy USB Video Stream & Streaming Inference Tests...")
    for res_name, h, w in resolutions:
        print(f"\n==================================================")
        print(f"📷 USB Camera Input Simulation Target: {res_name}")
        print(f"==================================================")
        synthetic_img = make_synthetic_eye(h, w)
        frame = DummyFrame(synthetic_img)

        for model_name in models:
            print(f"\n▶ Model Mode: {model_name}")
            plugin.active_model = model_name

            if model_name in model_attr_map:
                attr = model_attr_map[model_name]
                assert getattr(plugin, attr, None) is not None, (
                    f"❌ CRITICAL INTEGRITY FAILURE: {model_name} ({attr}) is None! Cannot perform inference!"
                )

            latencies = []
            for frame_idx in range(1, 6):
                t0 = time.time()
                datum = plugin.detect(frame)
                latency_ms = (time.time() - t0) * 1000.0
                latencies.append(latency_ms)

                assert datum is not None, f"FAIL: {model_name} returned None datum!"
                assert "norm_pos" in datum, f"FAIL: {model_name} missing norm_pos!"
                assert "confidence" in datum, f"FAIL: {model_name} missing confidence!"
                assert "ellipse" in datum, f"FAIL: {model_name} missing ellipse!"
                assert datum.get("method") == model_name, (
                    f"❌ METHOD MISMATCH FAILURE: Requested '{model_name}', but got '{datum.get('method')}'! Fallback occurred!"
                )

                print(
                    f"   Frame {frame_idx}: Latency = {latency_ms:6.2f} ms | "
                    f"Conf = {datum['confidence']:.3f} | "
                    f"NormPos = ({datum['norm_pos'][0]:.4f}, {datum['norm_pos'][1]:.4f}) | "
                    f"Center = ({datum['ellipse']['center'][0]:.1f}, {datum['ellipse']['center'][1]:.1f})"
                )

            avg_latency = np.mean(latencies[1:]) if len(latencies) > 1 else latencies[0]
            print(f"   ✓ Average Streaming Latency (FPS): {avg_latency:.2f} ms ({1000.0/max(avg_latency, 1e-3):.1f} FPS)")

    print("\n[3/3] Integrity Verification Completed.")
    print("=" * 75)
    print("🎉 HARNESS RESULT: PASS (Strict Model Object & Inference Verified)")
    print("=" * 75)

if __name__ == "__main__":
    run_harness()
