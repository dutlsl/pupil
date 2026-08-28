"""
Pupil - eye tracking platform
Copyright (C) Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
"""
import collections
import logging
import numpy as np
import os
import queue
import sys
import threading
import time
import types
import cv2
import torch
# Monkeypatch missing low-precision float types in PyTorch <= 2.6.0
if not hasattr(torch, "float4_e2m1fn_x2"):
    torch.float4_e2m1fn_x2 = "dummy_float4_e2m1fn_x2"
if not hasattr(torch, "float8_e8m0fnu"):
    torch.float8_e8m0fnu = "dummy_float8_e8m0fnu"
import PIL

import glfw
from gl_utils import (
    GLFWErrorReporting,
    adjust_gl_view,
    basic_gl_setup,
    clear_gl_screen,
    make_coord_system_norm_based,
    make_coord_system_pixel_based,
)
from pupil_detectors import Detector2D, DetectorBase, Roi
from pyglui import ui
from pyglui.cygl.utils import draw_gl_texture

GLFWErrorReporting.set_default()

from methods import normalize
from plugin import Plugin

from . import color_scheme
from .detector_base_plugin import PupilDetectorPlugin
from .eyelid_filter import fit_eyelid_robust_ellipse
from .visualizer_2d import draw_pupil_outline
try:
    from pupil_detector_plugins import deepvog
except Exception:
    deepvog = None

try:
    from pupil_detector_plugins import edgaze
except Exception:
    edgaze = None
from draw_ellipse import fit_ellipse
from CheckEllipse import computeEllipseConfidence
from pupil_detector_plugins.utils import get_predictions
from pupil_detector_plugins.models import model_dict
import torchvision

# Dynamic setup for nnUNet paths
NNUNET_DIR = os.path.expanduser("~/PycharmProjects/nnUNet")
NNUNET_LEGACY_DIR = os.path.expanduser("~/PycharmProjects/nnUNet_legacy")
NNUNET_AGENT_DIR = os.path.join(NNUNET_DIR, "agent")

for p in [NNUNET_LEGACY_DIR, NNUNET_DIR, NNUNET_AGENT_DIR]:
    if os.path.exists(p) and p not in sys.path:
        sys.path.insert(0, p)

if "numpy._core" not in sys.modules and hasattr(np, "core"):
    sys.modules["numpy._core"] = np.core
    if hasattr(np.core, "multiarray"):
        sys.modules["numpy._core.multiarray"] = np.core.multiarray
# TODO: Should I change nnUNet dataset num to 250?
PUPIL_CLASS_ID = 3  # OpenEDS labels: background=0, sclera=1, iris=2, pupil=3
MEAN_VAL = 86.45
STD_VAL = 39.94
COLOR_MAX = 255
COLOR_CAP = 256
CLIP_LIMIT = 1.5
TILE_GRID_SIZE = 8
# Temporal context size of the active Mamba3 variant. The Vivim weights are
# T-agnostic (Mamba3 selective scan), so the same checkpoint serves any T;
# T=3 keeps the per-frame inference cost low for the 120 Hz eye loop.
MAMBA3_T = 3
MAMBA3_LABEL = f"Mamba3 (T={MAMBA3_T})"

logger = logging.getLogger(__name__)


def select_nir_torch_device():
    """Select the NIR inference GPU, preserving normal single-GPU behavior.

    ``main_int.py`` sets ``PUPIL_HYBRID_NIR_GPU_ID`` so NIR Eye0/Eye1 work can
    be separated from the TDTracker GPU.  The normal launcher leaves that
    variable unset and therefore keeps the historical CUDA:0 behavior.
    """

    if not torch.cuda.is_available():
        return torch.device("cpu")

    raw_index = os.getenv("PUPIL_HYBRID_NIR_GPU_ID")
    if raw_index is None:
        device_index = 0
    else:
        try:
            device_index = int(raw_index)
        except ValueError as error:
            raise RuntimeError(
                "PUPIL_HYBRID_NIR_GPU_ID must be a non-negative CUDA device "
                f"index, got {raw_index!r}."
            ) from error
        if device_index < 0:
            raise RuntimeError(
                "PUPIL_HYBRID_NIR_GPU_ID must be a non-negative CUDA device "
                f"index, got {raw_index!r}."
            )

    device_count = torch.cuda.device_count()
    if device_index >= device_count:
        raise RuntimeError(
            "PUPIL_HYBRID_NIR_GPU_ID="
            f"{device_index} requested, but only {device_count} CUDA device(s) "
            "are visible to this process."
        )
    device = torch.device(f"cuda:{device_index}")
    # AMP calls in the RITnet path use the current CUDA device. Explicitly set
    # it so tensors and autocast never accidentally split across GPU 0/GPU N.
    torch.cuda.set_device(device)
    return device


class Detector2DPlugin(PupilDetectorPlugin):
    pupil_detection_identifier = "2d"
    pupil_detection_method = "2d c++"

    label = "C++ 2d detector"
    icon_font = "pupil_icons"
    icon_chr = chr(0xEC18)
    order = 0.100

    @property
    def pretty_class_name(self):
        return "Pupil Detector 2D"

    @property
    def pupil_detector(self) -> DetectorBase:
        return self.detector_2d

    def __init__(
        self,
        g_pool=None,
        properties=None,
        detector_2d: Detector2D = None,
        active_model=MAMBA3_LABEL,
        flip_vertically: bool = False,
        flip_horizontally: bool = False,
        enable_calibration: bool = True,
        **kwargs,
    ):
        super().__init__(g_pool=g_pool)
        self.detector_2d = detector_2d or Detector2D(properties or {})

        self.device = select_nir_torch_device()

        # UI Control States
        # Legacy sessions may persist other Mamba3 variants (e.g. T=7); the
        # plugin now only ships T=3, so normalize any Mamba3 value to it.
        if (
            isinstance(active_model, str)
            and active_model.startswith("Mamba3 (T=")
            and active_model != MAMBA3_LABEL
        ):
            active_model = MAMBA3_LABEL
        self.active_model = active_model
        self.flip_vertically = flip_vertically
        self.flip_horizontally = flip_horizontally
        self._enable_calibration = enable_calibration

        # Preprocessing & Optimizations
        self._clahe = cv2.createCLAHE(clipLimit=CLIP_LIMIT, tileGridSize=(TILE_GRID_SIZE, TILE_GRID_SIZE))
        self._use_amp = (self.device.type == "cuda")

        # Temporal Smoothing State
        self._prev_center = None
        self._smooth_alpha = 0.4
        self._consecutive_jumps = 0

        # Load Models (Mamba3 T=3, RITnet as fallback)
        self.vivim_models = {}
        self._vivim_queues = {MAMBA3_T: collections.deque(maxlen=MAMBA3_T)}
        self._init_nnunet_models()
        self._init_ritnet_model()

        # Async Mamba3 inference worker. Mamba3 inference is moved off the
        # eye event loop so its (GPU-bound, multi-frame) latency no longer
        # stalls frame capture/detection. detect() dispatches the frame to
        # this worker and returns the newest completed result, which still
        # carries its original frame timestamp.
        self._postprocess_lock = threading.Lock()
        self._mamba3_queue = queue.Queue(maxsize=8)
        self._mamba3_results = {}
        self._mamba3_results_lock = threading.Lock()
        self._mamba3_last_fetched = None
        self._mamba3_worker = None
        if self.vivim_models:
            self._mamba3_worker = threading.Thread(
                target=self._mamba3_worker_loop,
                name=f"mamba3-worker-eye{getattr(g_pool, 'eye_id', -1)}",
                daemon=True,
            )
            self._mamba3_worker.start()
            logger.info(f"Mamba3 (T={MAMBA3_T}) async inference worker started.")

    @property
    def enable_calibration(self) -> bool:
        gazer = getattr(self.g_pool, "active_gaze_mapping_plugin", None)
        if gazer is not None:
            return getattr(gazer, "enable_calibration", True)
        return getattr(self.g_pool, "enable_calibration", True)

    @enable_calibration.setter
    def enable_calibration(self, value: bool):
        val = bool(value)
        if hasattr(self, "g_pool") and self.g_pool is not None:
            self.g_pool.enable_calibration = val
            gazer = getattr(self.g_pool, "active_gaze_mapping_plugin", None)
            if gazer is not None:
                gazer.enable_calibration = val
        self.notify_all({"subject": "calibration.set_enabled", "enabled": val})

    def _init_nnunet_models(self):
        try:
            logger.info(f"Initializing Vivim Mamba3 T={MAMBA3_T} model...")
            current_dir = os.path.dirname(os.path.abspath(__file__))
            pupil_src_dir = os.path.abspath(os.path.join(current_dir, "..", ".."))

            # T=3 model = base Vivim trainer fold (temporal_window=3 default).
            candidate_ckpts = [
                os.path.join(current_dir, f"best_checkpoint_t{MAMBA3_T}.pth"),
                os.path.join(pupil_src_dir, f"best_checkpoint_t{MAMBA3_T}.pth"),
                os.path.join(
                    NNUNET_DIR,
                    "nnUNet_results",
                    "Dataset600_OpenEDS2019",
                    "nnUNetTrainer_Vivim__nnUNetPlans__2d",
                    "fold_1",
                    "checkpoint_best.pth",
                ),
            ]

            ckpt_path = None
            for p in candidate_ckpts:
                if os.path.exists(p):
                    ckpt_path = p
                    break

            if ckpt_path is not None:
                try:
                    from .vivim import VivimBackbone
                except Exception:
                    from vivim import VivimBackbone

                model = VivimBackbone(
                    in_channels=1,
                    num_classes=4,
                    base_channels=32,
                    d_state=16,
                    d_conv=4,
                    expand=2,
                    use_mamba=True,
                ).to(self.device)

                ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
                state_dict = ckpt.get("network_weights", ckpt.get("model_state_dict", ckpt))
                new_state_dict = {}
                for k, v in state_dict.items():
                    if k.startswith("backbone."):
                        new_state_dict[k[len("backbone."):]] = v
                    else:
                        new_state_dict[k] = v
                model.load_state_dict(new_state_dict, strict=False)
                model.eval()
                self.vivim_models[MAMBA3_T] = model
                logger.info(f"✅ Vivim Mamba3 T={MAMBA3_T} model initialized successfully from {ckpt_path}")
            else:
                logger.warning(f"Vivim Mamba3 T={MAMBA3_T} checkpoint not found.")
        except Exception as e:
            logger.error(f"Failed to initialize Vivim Mamba3 model: {e}")

    def _init_ritnet_model(self):
        try:
            model_name = "densenet"
            current_dir = os.path.dirname(os.path.abspath(__file__))
            pupil_src_dir = os.path.abspath(os.path.join(current_dir, "..", ".."))
            model_path = os.path.join(pupil_src_dir, "best_model.pkl")
            alt_path = os.path.join(current_dir, "best_model.pkl")

            path_to_load = model_path if os.path.exists(model_path) else alt_path

            if model_name in model_dict and os.path.exists(path_to_load):
                self.ritnet_model = model_dict[model_name]().to(self.device)
                try:
                    self.ritnet_model.load_state_dict(torch.load(path_to_load, map_location=self.device, weights_only=False))
                except Exception:
                    if os.path.exists(alt_path):
                        self.ritnet_model.load_state_dict(torch.load(alt_path, map_location=self.device, weights_only=False))
                self.ritnet_model.eval()
                logger.info("RITnet model initialized successfully as fallback.")
            else:
                self.ritnet_model = None
        except Exception as e:
            logger.warning(f"RITnet model init skipped/failed: {e}")
            self.ritnet_model = None

        self.transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize([0.5], [0.5]),
            ]
        )

    def get_init_dict(self):
        init_dict = super().get_init_dict()
        init_dict["properties"] = self.detector_2d.get_properties()
        return init_dict

    def detect(self, frame, **kwargs):
        active = getattr(self, "active_model", MAMBA3_LABEL)
        if active.startswith("Mamba3 (T="):
            try:
                t_val = int(active.split("T=")[1].replace(")", ""))
            except Exception:
                t_val = MAMBA3_T
            model = self.vivim_models.get(t_val)
            if model is None:
                if not getattr(self, f"_logged_missing_mamba3_t{t_val}", False):
                    logger.error(f"❌ Active model '{active}' is not loaded or failed initialization!")
                    setattr(self, f"_logged_missing_mamba3_t{t_val}", True)
                return self._empty_datum(frame)
            setattr(self, f"_logged_missing_mamba3_t{t_val}", False)
            if self._mamba3_worker is None:
                return self._empty_datum(frame)

            gray = frame.gray
            if gray is None:
                return self._empty_datum(frame)

            # Dispatch to the async worker. Copy the pixels: the capture layer
            # may reuse or replace the underlying buffer between frames.
            item = (
                np.array(gray, dtype=np.uint8, copy=True),
                frame.timestamp,
                int(frame.width),
                int(frame.height),
            )
            if self._mamba3_queue.full():
                # Drop the oldest pending frame so the freshest one is processed.
                try:
                    self._mamba3_queue.get_nowait()
                except queue.Empty:
                    pass
                if not getattr(self, "_logged_mamba3_queue_drop", False):
                    logger.warning(
                        f"Mamba3 (T={MAMBA3_T}) inference queue saturated; "
                        "dropping oldest pending frame to stay fresh."
                    )
                    self._logged_mamba3_queue_drop = True
            self._mamba3_queue.put_nowait(item)

            datum = self._fetch_latest_mamba3_result()
            if datum is None:
                return self._empty_datum(frame)
            return datum

        elif active == "RITnet":
            if self.ritnet_model is None:
                if not getattr(self, "_logged_missing_ritnet", False):
                    logger.error("❌ Active model 'RITnet' is not loaded or failed initialization!")
                    self._logged_missing_ritnet = True
                return self._empty_datum(frame)
            self._logged_missing_ritnet = False
            return self._detect_ritnet(frame, **kwargs)

        elif active == "2D C++":
            # Only execute C++ detector when explicitly selected by the user in UI
            roi = Roi(*self.g_pool.roi.bounds)
            debug_img = frame.bgr if self.g_pool.display_mode == "algorithm" else None
            result = self.detector_2d.detect(
                gray_img=frame.gray,
                color_img=debug_img,
                roi=roi,
            )
            norm_pos = normalize(
                result["location"], (frame.width, frame.height), flip_y=True
            )
            datum = self.create_pupil_datum(
                norm_pos=norm_pos,
                diameter=result["diameter"],
                confidence=result["confidence"],
                timestamp=frame.timestamp,
            )
            datum["method"] = "2D C++"
            datum["ellipse"] = {
                "axes": result["ellipse"]["axes"],
                "angle": result["ellipse"]["angle"],
                "center": result["ellipse"]["center"],
            }
            return datum

        else:
            logger.error(f"❌ Unknown or uninitialized active model: {active}")
            return self._empty_datum(frame)

    def _empty_datum(self, frame):
        datum = self.create_pupil_datum(
            norm_pos=(0.0, 0.0),
            diameter=0.0,
            confidence=0.0,
            timestamp=frame.timestamp,
        )
        datum["method"] = getattr(self, "active_model", "2d c++")
        datum["ellipse"] = {
            "axes": (0.0, 0.0),
            "angle": -90.0,
            "center": (0.0, 0.0),
        }
        return datum

    def _preprocess_nnunet_frame(self, frame):
        gray = frame.gray
        if gray is None:
            return None, 0, 0, False, False

        gray = gray.astype(np.uint8)
        orig_h, orig_w = gray.shape[:2]

        flip_v = getattr(self, "flip_vertically", False)
        if flip_v:
            gray = cv2.flip(gray, 0)

        flip_h = getattr(self, "flip_horizontally", False)
        if flip_h:
            gray = cv2.flip(gray, 1)

        gray_float = gray.astype(np.float32)
        mean_val = float(gray_float.mean())
        std_val = float(gray_float.std()) + 1e-8
        img_norm = (gray_float - mean_val) / std_val

        t_tensor = torch.from_numpy(img_norm).unsqueeze(0).unsqueeze(0).to(self.device)
        return t_tensor, orig_h, orig_w, flip_v, flip_h

    def _detect_temporal_unet(self, frame, **kwargs):
        t_tensor, orig_h, orig_w, flip_v, flip_h = self._preprocess_nnunet_frame(frame)
        if t_tensor is None:
            return self._empty_datum(frame)

        with torch.inference_mode():
            if self._use_amp:
                with torch.cuda.amp.autocast():
                    logits = self.temporal_model.forward_streaming(t_tensor)
            else:
                logits = self.temporal_model.forward_streaming(t_tensor)

            if isinstance(logits, (list, tuple)):
                logits = logits[0]
            
            pred_mask = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
            if pred_mask.shape[:2] != (orig_h, orig_w):
                pred_mask = cv2.resize(pred_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

        return self._postprocess_mask_to_datum(pred_mask, frame, orig_h, orig_w, flip_v, flip_h)

    def _detect_nnunet_2d(self, frame, **kwargs):
        t_tensor, orig_h, orig_w, flip_v, flip_h = self._preprocess_nnunet_frame(frame)
        if t_tensor is None:
            return self._empty_datum(frame)

        with torch.inference_mode():
            if self._use_amp:
                with torch.cuda.amp.autocast():
                    logits = self.vanilla_2d_model(t_tensor)
            else:
                logits = self.vanilla_2d_model(t_tensor)

            if isinstance(logits, (list, tuple)):
                logits = logits[0]

            pred_mask = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
            if pred_mask.shape[:2] != (orig_h, orig_w):
                pred_mask = cv2.resize(pred_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

        return self._postprocess_mask_to_datum(pred_mask, frame, orig_h, orig_w, flip_v, flip_h)

    def _mamba3_worker_loop(self):
        """Dedicated inference thread: preproc -> T-window stack -> model ->
        postproc. Owns the temporal queue and the EMA smoothing state for the
        Mamba3 path so the eye event loop never blocks on it."""
        self._mamba3_warmup()
        while True:
            item = self._mamba3_queue.get()
            if item is None:
                break
            gray, timestamp, width, height = item
            try:
                datum = self._infer_mamba3(gray, timestamp, width, height)
            except Exception as e:
                logger.error(f"Mamba3 worker inference failed: {e}")
                continue
            with self._mamba3_results_lock:
                self._mamba3_results[timestamp] = datum
                while len(self._mamba3_results) > 16:
                    oldest_key = next(iter(self._mamba3_results))
                    del self._mamba3_results[oldest_key]

    def _mamba3_warmup(self):
        """Run one dummy forward pass on this thread so CUDA context init and
        Mamba3 kernel (JIT) warmup happen at startup instead of stalling the
        first real frame (or racing interpreter shutdown on a slow first call)."""
        model = self.vivim_models.get(MAMBA3_T)
        if model is None:
            return
        try:
            with torch.inference_mode():
                dummy = torch.zeros(1, MAMBA3_T, 1, 448, 448, device=self.device)
                model(dummy)
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            logger.info("Mamba3 worker warmup complete.")
        except Exception as e:
            logger.error(f"Mamba3 worker warmup failed: {e}")

    def _fetch_latest_mamba3_result(self):
        """Return the newest completed Mamba3 datum (original frame timestamp
        intact), or None before the first result is ready. A datum that was
        already returned for a previous frame is flagged so the eye process
        does not publish it twice."""
        with self._mamba3_results_lock:
            if not self._mamba3_results:
                return None
            latest_key = next(reversed(self._mamba3_results))
            datum = self._mamba3_results[latest_key]
        if datum is self._mamba3_last_fetched:
            datum["_published_externally"] = True
        else:
            self._mamba3_last_fetched = datum
        return datum

    def _infer_mamba3(self, gray, timestamp, width, height):
        frame = types.SimpleNamespace(timestamp=timestamp, width=width, height=height)
        model = self.vivim_models.get(MAMBA3_T)
        if gray is None or model is None:
            return self._empty_datum(frame)

        gray = gray.astype(np.uint8)
        orig_h, orig_w = gray.shape[:2]

        flip_v = getattr(self, "flip_vertically", False)
        if flip_v:
            gray = cv2.flip(gray, 0)

        flip_h = getattr(self, "flip_horizontally", False)
        if flip_h:
            gray = cv2.flip(gray, 1)

        # Dynamic Z-Score normalization (matches OpenEDS Dataset600 training)
        img_float = cv2.resize(gray, (400, 400)).astype(np.float32)
        mean_val = float(img_float.mean())
        std_val = float(img_float.std()) + 1e-8
        img_norm = (img_float - mean_val) / std_val

        canvas = np.zeros((448, 448), dtype=np.float32)
        canvas[24:424, 24:424] = img_norm
        t_tensor = torch.from_numpy(canvas).unsqueeze(0).to(self.device)  # [1, 448, 448]

        seq_queue = self._vivim_queues[MAMBA3_T]
        seq_queue.append(t_tensor)
        while len(seq_queue) < MAMBA3_T:
            seq_queue.append(t_tensor)

        seq_list = list(seq_queue)
        seq_tensor = torch.stack(seq_list, dim=1).unsqueeze(2)  # [1, T, 1, 448, 448]

        with torch.inference_mode():
            logits = model(seq_tensor.float())

            if isinstance(logits, (list, tuple)):
                logits = logits[0]

            pred_mask_448 = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

        pred_mask_400 = pred_mask_448[24:424, 24:424]
        pred_mask = cv2.resize(pred_mask_400, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

        return self._postprocess_mask_to_datum(
            pred_mask,
            frame,
            orig_h,
            orig_w,
            flip_v,
            flip_h,
        )

    def _postprocess_mask_to_datum(self, pred_mask, frame, orig_h, orig_w, flip_v, flip_h):
        # Shared EMA/jump state is mutated from the Mamba3 worker thread and
        # the main thread (RITnet path); serialize the whole postprocess.
        with self._postprocess_lock:
            return self._postprocess_mask_to_datum_locked(pred_mask, frame, orig_h, orig_w, flip_v, flip_h)

    def _postprocess_mask_to_datum_locked(self, pred_mask, frame, orig_h, orig_w, flip_v, flip_h):
        pupil_mask = np.zeros_like(pred_mask, dtype=np.uint8)
        pupil_mask[pred_mask == PUPIL_CLASS_ID] = 255

        # Anti-aliasing Gaussian blur & thresholding
        pupil_mask = cv2.GaussianBlur(pupil_mask, (5, 5), 0)
        _, pupil_mask = cv2.threshold(pupil_mask, 127, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(pupil_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        if not contours:
            self._prev_center = None
            return self._empty_datum(frame)

        best_contour = max(contours, key=cv2.contourArea)
        if len(best_contour) < 5:
            self._prev_center = None
            return self._empty_datum(frame)

        ellipse = cv2.fitEllipse(best_contour)
        (cx, cy), (d1, d2), angle_deg = ellipse

        # Guarantee axes[0] is minor_diameter and axes[1] is major_diameter (axes[0] <= axes[1])
        if d1 > d2:
            minor_d = float(d2)
            major_d = float(d1)
            angle_deg = (angle_deg + 90.0) % 180.0
        else:
            minor_d = float(d1)
            major_d = float(d2)
            angle_deg = angle_deg % 180.0

        if flip_v:
            cy = float(orig_h - 1.0 - cy)
            angle_deg = (180.0 - angle_deg) % 180.0
        if flip_h:
            cx = float(orig_w - 1.0 - cx)
            angle_deg = (180.0 - angle_deg) % 180.0

        area = cv2.contourArea(best_contour)
        ellipse_area = np.pi * (minor_d / 2.0) * (major_d / 2.0)
        area_diff_ratio = min(area, ellipse_area) / (max(area, ellipse_area) + 1e-6)
        aspect_ratio = minor_d / (major_d + 1e-6)

        # Blink rejection filter (only reject when eye is almost completely closed or tiny noise)
        if aspect_ratio < 0.20 or area < 15.0:
            self._prev_center = None
            self._prev_ellipse = None
            return self._empty_datum(frame)

        # High confidence (1.0) on valid detection to ensure Pye3D (0.98 threshold) updates the 3D eyeball
        confidence = 1.0

        # Temporal EMA smoothing across all ellipse parameters (center, axes, angle)
        if getattr(self, "_prev_ellipse", None) is not None:
            p_c, p_ax, p_ang = self._prev_ellipse
            dist = np.sqrt((cx - p_c[0]) ** 2 + (cy - p_c[1]) ** 2)
            if dist > 40.0:
                self._consecutive_jumps += 1
                if self._consecutive_jumps < 5:
                    confidence = 0.0
                    cx, cy = p_c
                    minor_d, major_d = p_ax
                    angle_deg = p_ang
                else:
                    self._consecutive_jumps = 0
            else:
                self._consecutive_jumps = 0

            a = self._smooth_alpha
            cx = a * cx + (1.0 - a) * p_c[0]
            cy = a * cy + (1.0 - a) * p_c[1]
            minor_d = a * minor_d + (1.0 - a) * p_ax[0]
            major_d = a * major_d + (1.0 - a) * p_ax[1]

            # Continuous circular angle smoothing (mod 180 deg)
            diff_ang = (angle_deg - p_ang + 90.0) % 180.0 - 90.0
            angle_deg = (p_ang + a * diff_ang) % 180.0
        else:
            self._consecutive_jumps = 0

        self._prev_ellipse = ((cx, cy), (minor_d, major_d), angle_deg)
        self._prev_center = (cx, cy)

        result = {
            "location": (float(cx), float(cy)),
            "diameter": float(major_d),
            "confidence": confidence,
            "ellipse": {
                "axes": (float(minor_d), float(major_d)),
                "angle": float(angle_deg),
                "center": (float(cx), float(cy)),
            },
        }

        norm_pos = normalize(result["location"], (frame.width, frame.height), flip_y=True)
        datum = self.create_pupil_datum(
            norm_pos=norm_pos,
            diameter=result["diameter"],
            confidence=result["confidence"],
            timestamp=frame.timestamp,
        )
        datum["method"] = getattr(self, "active_model", "2d c++")
        datum["ellipse"] = result["ellipse"]
        return datum

    def detect_RITnet(self, frame, **kwargs):
        """Backward compatibility alias for _detect_ritnet."""
        return self._detect_ritnet(frame, **kwargs)

    def _detect_ritnet(self, frame, **kwargs):
        gray = frame.gray
        if gray is None or self.ritnet_model is None:
            return self._empty_datum(frame)

        gray = gray.astype(np.uint8)
        orig_h, orig_w = gray.shape[:2]

        flip_v = getattr(self, "flip_vertically", False)
        if flip_v:
            gray = cv2.flip(gray, 0)

        flip_h = getattr(self, "flip_horizontally", False)
        if flip_h:
            gray = cv2.flip(gray, 1)

        img_tensor = self.get_img(gray)
        data = img_tensor.unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.ritnet_model(data)

        predict = get_predictions(output)
        predict_2d = predict[0].cpu().numpy()

        return self._postprocess_mask_to_datum(predict_2d, frame, orig_h, orig_w, flip_v, flip_h)

    def get_img(self, img: np.ndarray) -> torch.Tensor:
        table = float(COLOR_MAX) * (np.linspace(0, 1, COLOR_CAP) ** 0.8)
        img_gamma = cv2.LUT(img.astype(np.uint8), table.astype(np.uint8))
        img_clahe = self._clahe.apply(img_gamma)
        pil_img = PIL.Image.fromarray(img_clahe)
        return self.transform(pil_img)

    def _empty_datum(self, frame):
        norm_pos = (0.0, 0.0)
        datum = self.create_pupil_datum(
            norm_pos=norm_pos,
            diameter=0.0,
            confidence=0.0,
            timestamp=frame.timestamp,
        )
        datum["ellipse"] = {
            "axes": (0.0, 0.0),
            "angle": 0.0,
            "center": (0.0, 0.0),
        }
        return datum

    def cleanup(self):
        if self._mamba3_worker is not None and self._mamba3_worker.is_alive():
            # Drain pending frames, then send the stop sentinel. The worker
            # finishes its in-flight inference before exiting; wait long
            # enough for the (possibly JIT-compiling) first inference so the
            # thread is fully dead before interpreter shutdown.
            try:
                while self._mamba3_queue.full():
                    self._mamba3_queue.get_nowait()
                self._mamba3_queue.put_nowait(None)
            except queue.Empty:
                pass
            self._mamba3_worker.join(timeout=15.0)
            if self._mamba3_worker.is_alive():
                logger.warning("Mamba3 worker did not stop within timeout.")
        self._mamba3_worker = None
        super().cleanup()

    def init_ui(self):
        super().init_ui()
        self.menu.label = self.pretty_class_name
        self.menu_icon.label_font = "pupil_icons"
        info = ui.Info_Text(
            "Switch to the algorithm display mode to see a visualization of pupil detection parameters overlaid on the eye video."
        )
        self.menu.append(info)
        self.menu.append(
            ui.Selector(
                "active_model",
                self,
                label="Active Model",
                selection=[MAMBA3_LABEL, "2D C++", "RITnet"],
            )
        )
        self.menu.append(ui.Switch("enable_calibration", self, label="Enable Calibration Mapping"))
        self.menu.append(ui.Switch("flip_vertically", self, label="Flip Vertically (Eye 0)"))
        self.menu.append(ui.Switch("flip_horizontally", self, label="Flip Horizontally (Eye 0)"))
        self.menu.append(
            ui.Slider(
                "intensity_range",
                self.pupil_detector_properties,
                label="Pupil intensity range",
                min=0,
                max=60,
                step=1,
            )
        )
        self.menu.append(
            ui.Slider(
                "pupil_size_min",
                self.pupil_detector_properties,
                label="Pupil min",
                min=1,
                max=250,
                step=1,
            )
        )
        self.menu.append(
            ui.Slider(
                "pupil_size_max",
                self.pupil_detector_properties,
                label="Pupil max",
                min=50,
                max=400,
                step=1,
            )
        )
        self.menu.append(
            ui.Slider(
                "canny_treshold",
                self.pupil_detector_properties,
                label="Canny Threshold",
                min=0,
                max=1000,
                step=1,
            )
        )
        self.menu.append(ui.Info_Text("Color Legend"))
        self.menu.append(
            ui.Color_Legend(color_scheme.PUPIL_ELLIPSE_2D.as_float, "2D pupil ellipse")
        )

    def gl_display(self):
        if self._recent_detection_result:
            draw_pupil_outline(
                self._recent_detection_result,
                color_rgb=color_scheme.PUPIL_ELLIPSE_2D.as_float,
            )

    def on_resolution_change(self, old_size, new_size):
        properties = self.pupil_detector.get_properties()
        properties["pupil_size_max"] *= new_size[0] / old_size[0]
        properties["pupil_size_min"] *= new_size[0] / old_size[0]
        self.pupil_detector.update_properties(properties)

    def on_notify(self, notification):
        subj = notification.get("subject", "").lower()
        

        if subj == "calibration.set_enabled":
            val = bool(notification.get("enabled", True))
            if hasattr(self, "g_pool") and self.g_pool is not None:
                self.g_pool.enable_calibration = val
                gazer = getattr(self.g_pool, "active_gaze_mapping_plugin", None)
                if gazer is not None:
                    gazer.enable_calibration = val

        if "calibration" in subj and (subj.endswith(".should_start") or subj.endswith(".started")):
            if hasattr(self, "temporal_model") and self.temporal_model is not None:
                if hasattr(self.temporal_model, "reset_temporal_state"):
                    self.temporal_model.reset_temporal_state()
                    logger.info(f"Reset TemporalUNet ConvLSTM state for new Calibration ({subj}).")
