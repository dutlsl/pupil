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
import sys
import time
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

PUPIL_CLASS_ID = 3  # OpenEDS labels: background=0, sclera=1, iris=2, pupil=3
MEAN_VAL = 86.45
STD_VAL = 39.94
COLOR_MAX = 255
COLOR_CAP = 256
CLIP_LIMIT = 1.5
TILE_GRID_SIZE = 8

logger = logging.getLogger(__name__)


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
    ):
        super().__init__(g_pool=g_pool)
        self.detector_2d = detector_2d or Detector2D(properties or {})

        device_str = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device_str)

        # UI Control States
        self.active_model = "Mamba3 (T=5)"
        self.flip_vertically = False
        self.flip_horizontally = False

        # Preprocessing & Optimizations
        self._clahe = cv2.createCLAHE(clipLimit=CLIP_LIMIT, tileGridSize=(TILE_GRID_SIZE, TILE_GRID_SIZE))
        self._use_amp = (self.device.type == "cuda")

        # Temporal Smoothing State
        self._prev_center = None
        self._smooth_alpha = 0.4
        self._consecutive_jumps = 0

        # Load Models (Mamba3 T=3,5,7,9,11 models, RITnet as fallback)
        self.vivim_models = {}
        self._vivim_queues = {t: collections.deque(maxlen=t) for t in [3, 5, 7, 9, 11]}
        self._init_nnunet_models()
        self._init_ritnet_model()

    def _init_nnunet_models(self):
        try:
            logger.info("Initializing Vivim Mamba3 models (T=3, 5, 7, 9, 11)...")

            self.vivim_ckpts = {
                3: os.path.join(NNUNET_DIR, "nnUNet_results", "Dataset600_OpenEDS2019", "nnUNetTrainer_Vivim__nnUNetPlans__2d", "fold_1", "checkpoint_best.pth"),
                5: os.path.join(NNUNET_DIR, "nnUNet_results", "Dataset600_OpenEDS2019", "nnUNetTrainer_Vivim_T5__nnUNetPlans__2d", "fold_1", "checkpoint_best.pth"),
                7: os.path.join(NNUNET_DIR, "nnUNet_results", "Dataset600_OpenEDS2019", "nnUNetTrainer_Vivim__nnUNetPlans__2d", "fold_1_T7", "checkpoint_best.pth"),
                9: os.path.join(NNUNET_DIR, "nnUNet_results", "Dataset600_OpenEDS2019", "nnUNetTrainer_Vivim__nnUNetPlans__2d", "fold_1_T9", "checkpoint_best.pth"),
                11: os.path.join(NNUNET_DIR, "nnUNet_results", "Dataset600_OpenEDS2019", "nnUNetTrainer_Vivim__nnUNetPlans__2d", "fold_1_T11", "checkpoint_best.pth"),
            }

            from models.vivim_backbone import VivimBackbone
            self.vivim_models = {}
            for t, ckpt_path in self.vivim_ckpts.items():
                if os.path.exists(ckpt_path):
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
                    self.vivim_models[t] = model
                    logger.info(f"✅ nnUNet Vivim Mamba3 T={t} model initialized successfully.")
                else:
                    logger.warning(f"Vivim T={t} checkpoint not found at {ckpt_path}")
        except Exception as e:
            logger.error(f"Failed to initialize nnUNet models: {e}")

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
        active = getattr(self, "active_model", "Mamba3 (T=5)")
        if active.startswith("Mamba3 (T="):
            try:
                t_val = int(active.split("T=")[1].replace(")", ""))
            except Exception:
                t_val = 5
            model = self.vivim_models.get(t_val)
            if model is None:
                if not getattr(self, f"_logged_missing_mamba3_t{t_val}", False):
                    logger.error(f"❌ Active model '{active}' is not loaded or failed initialization!")
                    setattr(self, f"_logged_missing_mamba3_t{t_val}", True)
                return self._empty_datum(frame)
            setattr(self, f"_logged_missing_mamba3_t{t_val}", False)
            return self._detect_vivim_mamba_by_t(frame, model, t_val, **kwargs)

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
            return None, 0, 0, False, False, False

        gray = gray.astype(np.uint8)
        orig_h, orig_w = gray.shape[:2]

        flip_v = getattr(self, "flip_vertically", False)
        if flip_v:
            gray = cv2.flip(gray, 0)

        flip_h = getattr(self, "flip_horizontally", False)
        if flip_h:
            gray = cv2.flip(gray, 1)

        # Dynamic Z-Score normalization (adapts to camera IR domain shifts)
        gray_float = gray.astype(np.float32)
        mean_val = float(gray_float.mean())
        std_val = float(gray_float.std()) + 1e-8
        img_norm = (gray_float - mean_val) / std_val

        # Letterboxing: Aspect Ratio Preserving Padding to 640x400
        # Prevents Pupil Core (192x192) horizontal stretching distortion
        if (orig_h, orig_w) == (400, 640):
            t_tensor = torch.from_numpy(img_norm).unsqueeze(0).unsqueeze(0).to(self.device)
            is_letterboxed = False
        else:
            img_400 = cv2.resize(img_norm, (400, 400), interpolation=cv2.INTER_LINEAR)
            canvas = np.zeros((400, 640), dtype=np.float32)
            canvas[:, 120:520] = img_400
            t_tensor = torch.from_numpy(canvas).unsqueeze(0).unsqueeze(0).to(self.device)
            is_letterboxed = True

        return t_tensor, orig_h, orig_w, flip_v, flip_h, is_letterboxed

    def _detect_temporal_unet(self, frame, **kwargs):
        t_tensor, orig_h, orig_w, flip_v, flip_h, is_letterboxed = self._preprocess_nnunet_frame(frame)
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
            
            # Squeeze to (400, 640) for postprocessing
            pred_mask = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

        return self._postprocess_mask_to_datum(pred_mask, frame, orig_h, orig_w, flip_v, flip_h, is_letterboxed)

    def _detect_nnunet_2d(self, frame, **kwargs):
        t_tensor, orig_h, orig_w, flip_v, flip_h, is_letterboxed = self._preprocess_nnunet_frame(frame)
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

        return self._postprocess_mask_to_datum(pred_mask, frame, orig_h, orig_w, flip_v, flip_h, is_letterboxed)

    def _detect_vivim_mamba_by_t(self, frame, model, t_val, **kwargs):
        gray = frame.gray
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

        gray_float = gray.astype(np.float32)
        mean_val = float(gray_float.mean())
        std_val = float(gray_float.std()) + 1e-8
        img_norm = (gray_float - mean_val) / std_val

        if (orig_h, orig_w) == (400, 640):
            img_400 = img_norm[:, 120:520]
            is_openeds_400 = True
        else:
            img_400 = cv2.resize(img_norm, (400, 400), interpolation=cv2.INTER_LINEAR)
            is_openeds_400 = False

        canvas = np.zeros((448, 448), dtype=np.float32)
        canvas[24:424, 24:424] = img_400
        t_tensor = torch.from_numpy(canvas).unsqueeze(0).to(self.device)  # [1, 448, 448]

        queue = self._vivim_queues[t_val]
        queue.append(t_tensor)
        while len(queue) < t_val:
            queue.append(t_tensor)

        seq_list = list(queue)
        seq_tensor = torch.stack(seq_list, dim=1).unsqueeze(2)  # [1, T, 1, 448, 448]

        with torch.inference_mode():
            logits = model(seq_tensor.float())

            if isinstance(logits, (list, tuple)):
                logits = logits[0]

            pred_mask_448 = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

        # Unpad 24px -> 400x400
        pred_mask_400 = pred_mask_448[24:424, 24:424]

        full_canvas = np.zeros((400, 640), dtype=np.uint8)
        full_canvas[:, 120:520] = pred_mask_400

        return self._postprocess_mask_to_datum(
            full_canvas,
            frame,
            orig_h,
            orig_w,
            flip_v,
            flip_h,
            is_letterboxed=not is_openeds_400
        )

    def _postprocess_mask_to_datum(self, raw_pred_mask, frame, orig_h, orig_w, flip_v, flip_h, is_letterboxed=False):
        if is_letterboxed:
            # Crop 120px left/right padding -> 400x400 -> resize back to orig_w x orig_h (192x192)
            mask_400 = raw_pred_mask[:, 120:520]
            pred_mask = cv2.resize(mask_400, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        else:
            pred_mask = raw_pred_mask[:orig_h, :orig_w]

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
        (cx, cy), (MA, ma), angle_deg = ellipse

        if flip_v:
            cy = orig_h - 1 - cy
            angle_deg = 180.0 - angle_deg
        if flip_h:
            cx = orig_w - 1 - cx
            angle_deg = 180.0 - angle_deg

        area = cv2.contourArea(best_contour)
        ellipse_area = np.pi * (MA / 2.0) * (ma / 2.0)
        area_ratio = min(area, ellipse_area) / (max(area, ellipse_area) + 1e-6)
        aspect_ratio = min(MA, ma) / (max(MA, ma) + 1e-6)

        # Half-blink rejection filter
        if aspect_ratio < 0.65:
            self._prev_center = None
            return self._empty_datum(frame)

        raw_conf = float(np.sqrt(max(0.0, area_ratio * aspect_ratio)))
        confidence = float(np.clip(raw_conf, 0.0, 1.0))
        if np.isnan(confidence):
            confidence = 0.0

        # Temporal EMA smoothing & jump rejection
        if self._prev_center is not None:
            dist = np.sqrt((cx - self._prev_center[0]) ** 2 + (cy - self._prev_center[1]) ** 2)
            if dist > 40.0:
                self._consecutive_jumps += 1
                if self._consecutive_jumps < 5:
                    confidence = 0.0
                    cx, cy = self._prev_center
                else:
                    self._consecutive_jumps = 0
            else:
                self._consecutive_jumps = 0

            a = self._smooth_alpha
            cx = a * cx + (1.0 - a) * self._prev_center[0]
            cy = a * cy + (1.0 - a) * self._prev_center[1]
        else:
            self._consecutive_jumps = 0

        self._prev_center = (cx, cy)

        result = {
            "location": (float(cx), float(cy)),
            "diameter": float(MA),
            "confidence": confidence,
            "ellipse": {
                "axes": (float(MA), float(ma)),
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
                selection=["RITnet", "2D C++", "Mamba3 (T=3)", "Mamba3 (T=5)", "Mamba3 (T=7)", "Mamba3 (T=9)", "Mamba3 (T=11)"],
            )
        )
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
        
        # --- Handle Accuracy Logging ---
        if getattr(self.g_pool, "eye_id", 0) == 0:
            if subj == "calibration.successful":
                rmse_val = "unknown"
                try:
                    log_path = os.path.expanduser("~/PycharmProjects/pupil/pupil_capture.log")
                    if os.path.exists(log_path):
                        with open(log_path, "r", encoding="utf-8") as lf:
                            lines = lf.readlines()
                        for line in reversed(lines):
                            if "Fitting. RMSE =" in line:
                                parts = line.split("RMSE =")
                                if len(parts) > 1:
                                    rmse_val = parts[1].strip().split("px")[0].strip() + " px"
                                    break
                except Exception as ex:
                    logger.error(f"Failed to parse RMSE from log: {ex}")
                
                from .experiment_logger import save_accuracy_log
                active_model = getattr(self, "active_model", "TemporalUNet")
                save_accuracy_log(self.g_pool, active_model, "calibration", rmse_val)
                
            elif subj == "accuracy_visualizer.data":
                accuracy = notification.get("accuracy")
                precision = notification.get("precision")
                if accuracy is not None:
                    from .experiment_logger import save_accuracy_log
                    active_model = getattr(self, "active_model", "TemporalUNet")
                    save_accuracy_log(self.g_pool, active_model, "test", accuracy, precision_value=precision)
        
        if "calibration" in subj and (subj.endswith(".should_start") or subj.endswith(".started")):
            if hasattr(self, "temporal_model") and self.temporal_model is not None:
                if hasattr(self.temporal_model, "reset_temporal_state"):
                    self.temporal_model.reset_temporal_state()
                    logger.info(f"Reset TemporalUNet ConvLSTM state for new Calibration ({subj}).")