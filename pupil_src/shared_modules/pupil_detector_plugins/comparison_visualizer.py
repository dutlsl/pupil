"""
comparison_visualizer.py
========================
RITnet vs U-Mamba 실시간 비교 시각화 모듈.

메인 detect 로직에 간섭하지 않는 독립 모듈.
Detector2DPlugin.__init__에서 이 모듈을 초기화하고,
detect_umamba에서 한 줄(compare())만 호출하면 됨.

사용법 (detector_2d_plugin.py):
    # __init__ 끝에:
    from pupil_detector_plugins.comparison_visualizer import ComparisonVisualizer
    self.comparator = ComparisonVisualizer(device=self.device)

    # detect_umamba 안에서, pred_mask 구한 직후:
    self.comparator.compare(gray, pred_mask)
"""

import logging
import os
import numpy as np
import cv2
import torch
import PIL.Image
import torchvision

logger = logging.getLogger(__name__)

# ─── RITnet 전처리 상수 ───
COLOR_MAX = 255
COLOR_CAP = 256
CLIP_LIMIT = 1.5
TILE_GRID_SIZE = 8
PUPIL_CLASS_ID = 3   # OpenEDS: bg=0, sclera=1, iris=2, pupil=3

# ─── 시각화 색상 (BGR) ───
COLOR_SCLERA = (200, 180, 100)   # 연한 하늘색
COLOR_IRIS   = (0,   200, 0)     # 초록
COLOR_PUPIL  = (0,   0,   255)   # 빨강
COLOR_ELLIPSE_RITNET = (0, 255, 255)  # 노랑
COLOR_ELLIPSE_UMAMBA = (0, 255, 0)    # 초록


class ComparisonVisualizer:
    """RITnet 모델을 자체 로드하여, 매 프레임 U-Mamba 결과와 나란히 비교."""

    def __init__(self, device=None, window_name="RITnet vs U-Mamba"):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.window_name = window_name
        self._window_created = False

        # ─── RITnet 모델 로드 ───
        from pupil_detector_plugins.models import model_dict
        from pupil_detector_plugins.utils import get_predictions
        self.get_predictions = get_predictions

        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, "best_model.pkl")

        if not os.path.exists(model_path):
            # fallback
            pupil_src_dir = os.path.abspath(os.path.join(current_dir, "..", ".."))
            model_path = os.path.join(pupil_src_dir, "best_model.pkl")

        self.model = model_dict["densenet"]().to(self.device)
        self.model.load_state_dict(torch.load(model_path, weights_only=False))
        self.model.eval()
        logger.info(f"[ComparisonVisualizer] RITnet loaded from {model_path}")

        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.5], [0.5]),
        ])
        self.clahe = cv2.createCLAHE(clipLimit=CLIP_LIMIT, tileGridSize=(TILE_GRID_SIZE, TILE_GRID_SIZE))

    # ──────────────────── 공개 API ────────────────────
    def compare(self, gray: np.ndarray, umamba_mask: np.ndarray):
        """
        매 프레임 호출. 메인 로직에 영향 없음.

        Args:
            gray:        원본 그레이스케일 (H, W), uint8
            umamba_mask: U-Mamba 예측 라벨 맵 (H, W), 0~3
        """
        try:
            ritnet_mask = self._run_ritnet(gray)
            vis = self._build_comparison(gray, ritnet_mask, umamba_mask)
            if not self._window_created:
                cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
                self._window_created = True
            cv2.imshow(self.window_name, vis)
            cv2.waitKey(1)
        except Exception as e:
            logger.debug(f"[ComparisonVisualizer] error: {e}")

    def cleanup(self):
        """플러그인 종료 또는 토글 오프 시 호출."""
        if self._window_created:
            cv2.destroyWindow(self.window_name)
            cv2.waitKey(1)
            self._window_created = False

    # ──────────────────── 내부 구현 ────────────────────
    def _preprocess_ritnet(self, gray: np.ndarray) -> torch.Tensor:
        """RITnet 전처리: gamma -> CLAHE -> ToTensor -> Normalize."""
        table = float(COLOR_MAX) * (np.linspace(0, 1, COLOR_CAP) ** 0.8)
        img_gamma = cv2.LUT(gray.astype(np.uint8), table.astype(np.uint8))
        img_clahe = self.clahe.apply(img_gamma)
        pil_img = PIL.Image.fromarray(img_clahe)
        tensor_img = self.transform(pil_img)
        return tensor_img

    def _run_ritnet(self, gray: np.ndarray) -> np.ndarray:
        """RITnet 추론 → 라벨 맵 (H, W), 0~3."""
        tensor = self._preprocess_ritnet(gray).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.model(tensor)
        pred = self.get_predictions(output)  # [1, H, W]
        return pred[0].cpu().numpy().astype(np.uint8)

    def _fit_ellipse(self, mask_label: np.ndarray):
        """라벨 맵에서 동공(3) 컨투어 → 타원 피팅. 실패 시 None."""
        pupil_bin = np.zeros_like(mask_label, dtype=np.uint8)
        pupil_bin[mask_label == PUPIL_CLASS_ID] = 255
        contours, _ = cv2.findContours(pupil_bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        if not contours:
            return None
        best = max(contours, key=cv2.contourArea)
        if len(best) < 5:
            return None
        return cv2.fitEllipse(best)

    def _overlay_mask(self, bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """세그멘테이션 마스크를 반투명 컬러로 오버레이."""
        overlay = bgr.copy()
        overlay[mask == 1] = COLOR_SCLERA
        overlay[mask == 2] = COLOR_IRIS
        overlay[mask == 3] = COLOR_PUPIL
        return cv2.addWeighted(bgr, 0.5, overlay, 0.5, 0)

    def _build_comparison(self, gray: np.ndarray, ritnet_mask: np.ndarray, umamba_mask: np.ndarray) -> np.ndarray:
        """좌: RITnet, 우: U-Mamba 나란히 비교 이미지 생성."""
        h, w = gray.shape[:2]
        bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        # --- 좌측: RITnet ---
        left = self._overlay_mask(bgr.copy(), ritnet_mask)
        ell_r = self._fit_ellipse(ritnet_mask)
        if ell_r is not None:
            cv2.ellipse(left, ell_r, COLOR_ELLIPSE_RITNET, 2)
            cx, cy = int(ell_r[0][0]), int(ell_r[0][1])
            cv2.circle(left, (cx, cy), 3, COLOR_ELLIPSE_RITNET, -1)

        # --- 우측: U-Mamba ---
        right = self._overlay_mask(bgr.copy(), umamba_mask)
        ell_u = self._fit_ellipse(umamba_mask)
        if ell_u is not None:
            cv2.ellipse(right, ell_u, COLOR_ELLIPSE_UMAMBA, 2)
            cx, cy = int(ell_u[0][0]), int(ell_u[0][1])
            cv2.circle(right, (cx, cy), 3, COLOR_ELLIPSE_UMAMBA, -1)

        # --- 라벨 ---
        cv2.putText(left, "RITnet", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(right, "U-Mamba", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # --- 범례 (하단) ---
        legend_h = 25
        legend = np.zeros((legend_h, w * 2, 3), dtype=np.uint8)
        cv2.putText(legend, "Sclera", (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_SCLERA, 1)
        cv2.putText(legend, "Iris", (100, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_IRIS, 1)
        cv2.putText(legend, "Pupil", (160, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_PUPIL, 1)

        comparison = np.hstack([left, right])
        comparison = np.vstack([comparison, legend])
        return comparison
