"""
Pupil - eye tracking platform
Copyright (C) Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
"""
import logging
import numpy as np
import os
import sys
import time
import cv2
import torch
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
        self.active_model = "TemporalUNet"
        self.flip_vertically = False
        self.flip_horizontally = False

        # Preprocessing & Optimizations
        self._clahe = cv2.createCLAHE(clipLimit=CLIP_LIMIT, tileGridSize=(TILE_GRID_SIZE, TILE_GRID_SIZE))
        self._use_amp = (self.device.type == "cuda")

        # Temporal Smoothing State
        self._prev_center = None
        self._smooth_alpha = 0.4
        self._consecutive_jumps = 0

        # Load Models (TemporalUNet as primary, nnUNet 2D vanilla as secondary, RITnet as fallback)
        self.temporal_model = None
        self.vanilla_2d_model = None
        self._init_nnunet_models()
        self._init_ritnet_model()

    def _init_nnunet_models(self):
        try:
            logger.info("Initializing TemporalUNet and 2D nnUNet models...")
            from temporal_unet import TemporalUNet

            model_dir = os.path.join(
                NNUNET_DIR,
                "nnUNet_results",
                "Dataset600_OpenEDS2019",
                "nnUNetTrainer_ImageNetPretrained__nnUNetPlans__2d"
            )
            temporal_ckpt = os.path.join(
                NNUNET_DIR,
                "nnUNet_results",
                "TemporalUNet_v1",
                "checkpoint_best.pth"
            )

            if os.path.exists(model_dir) and os.path.exists(temporal_ckpt):
                self.temporal_model = TemporalUNet.from_pretrained(
                    model_folder=model_dir,
                    checkpoint_name='checkpoint_best.pth',
                    num_classes=4,
                    deep_supervision=False,
                    device=self.device,
                )
                ckpt = torch.load(temporal_ckpt, map_location=self.device, weights_only=False)
                if 'model_state_dict' in ckpt:
                    self.temporal_model.load_state_dict(ckpt['model_state_dict'])
                else:
                    self.temporal_model.load_state_dict(ckpt)

                self.temporal_model.eval()
                self.temporal_model.to(self.device)
                self.temporal_model.reset_temporal_state()

                # Extract vanilla 2D model from encoder/pretrained network
                if hasattr(self.temporal_model, 'pretrained_unet'):
                    self.vanilla_2d_model = self.temporal_model.pretrained_unet
                else:
                    self.vanilla_2d_model = self.temporal_model.encoder

                logger.info("✅ TemporalUNet & 2D nnUNet initialized successfully.")
            else:
                logger.warning(f"nnUNet paths not found: model_dir={model_dir}, temporal_ckpt={temporal_ckpt}")
        except Exception as e:
            logger.error(f"Failed to initialize nnUNet models: {e}")

    def _init_ritnet_model(self):
        try:
            model_name = "densenet"
            current_dir = os.path.dirname(os.path.abspath(__file__))
            pupil_src_dir = os.path.abspath(os.path.join(current_dir, "..", ".."))
            model_path = os.path.join(pupil_src_dir, "best_model.pkl")

            if model_name in model_dict and os.path.exists(model_path):
                self.ritnet_model = model_dict[model_name]().to(self.device)
                self.ritnet_model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=False))
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
        active = getattr(self, "active_model", "TemporalUNet")
        if active == "TemporalUNet" and self.temporal_model is not None:
            return self._detect_temporal_unet(frame, **kwargs)
        elif active == "nnUNet 2D" and self.vanilla_2d_model is not None:
            return self._detect_nnunet_2d(frame, **kwargs)
        elif active == "RITnet" and self.ritnet_model is not None:
            return self._detect_ritnet(frame, **kwargs)

        # Default C++ 2d detector fallback
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
        datum["ellipse"] = {
            "axes": result["ellipse"]["axes"],
            "angle": result["ellipse"]["angle"],
            "center": result["ellipse"]["center"],
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

        confidence = float(np.clip(np.sqrt(area_ratio * aspect_ratio), 0.0, 1.0))

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
        datum["ellipse"] = result["ellipse"]
        return datum

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
                selection=["TemporalUNet", "nnUNet 2D", "RITnet", "2D C++"],
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