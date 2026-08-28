#!/usr/bin/env python3
"""Calibrate Eye0 (NIR) and a DAVIS event camera from checkerboard samples.

This tool deliberately does *not* render a pattern and does not open either
camera.  An external monitor is the pattern source; Eye0 and the event camera
only observe it.  Save matching observations with the same filename stem::

    samples/eye0/0001.png
    samples/event/0001.npz

For an event sample, ``.npz`` may contain either a reconstructed image under
``image``/``frame`` or raw arrays named ``x``, ``y`` and ``p`` (or
``polarity``).  Raw arrays must cover one checkerboard inversion only.  The
polarity image is reconstructed at the DAVIS sensor resolution, so the
TDTracker 80x60 representation is never used for geometry calibration.

The result convention is explicit: ``event_from_eye0`` maps a 3-D point from
the Eye0 camera coordinate system into the event camera coordinate system.
Translation is in the same unit supplied via ``--square-size-mm`` (millimetres
by default).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np


# DAVIS346's event and APS frame resolution.  Direct image inputs override this
# automatically; this fallback is used only when rebuilding an image from raw
# x/y/p event arrays.
DAVIS346_EVENT_SIZE: Tuple[int, int] = (346, 260)  # (width, height)


@dataclass(frozen=True)
class CheckerboardSpec:
    """Geometry of the checkerboard's *inner* corners."""

    inner_corners: Tuple[int, int] = (9, 6)  # (columns, rows)
    square_size_mm: float = 10.0

    def __post_init__(self) -> None:
        columns, rows = self.inner_corners
        if columns < 2 or rows < 2:
            raise ValueError("inner_corners must contain at least 2 by 2 corners")
        if self.square_size_mm <= 0.0:
            raise ValueError("square_size_mm must be positive")

    def object_points(self) -> np.ndarray:
        """Return OpenCV object points ordered like a checkerboard detection."""

        columns, rows = self.inner_corners
        grid = np.zeros((columns * rows, 3), dtype=np.float32)
        grid[:, :2] = np.mgrid[0:columns, 0:rows].T.reshape(-1, 2)
        grid[:, :2] *= float(self.square_size_mm)
        return grid


@dataclass(frozen=True)
class StereoObservation:
    """One paired, already-detected checkerboard observation."""

    name: str
    eye0_corners: np.ndarray
    event_corners: np.ndarray


@dataclass(frozen=True)
class Intrinsics:
    camera_matrix: np.ndarray
    distortion_coefficients: np.ndarray

    def __post_init__(self) -> None:
        camera_matrix = np.asarray(self.camera_matrix, dtype=np.float64)
        distortion = np.asarray(self.distortion_coefficients, dtype=np.float64)
        if camera_matrix.shape != (3, 3):
            raise ValueError("camera_matrix must have shape (3, 3)")
        if distortion.ndim not in (1, 2):
            raise ValueError("distortion_coefficients must be a vector")
        object.__setattr__(self, "camera_matrix", camera_matrix)
        object.__setattr__(self, "distortion_coefficients", distortion.reshape(-1, 1))


@dataclass(frozen=True)
class StereoCalibration:
    eye0_intrinsics: Intrinsics
    event_intrinsics: Intrinsics
    rotation_event_from_eye0: np.ndarray
    translation_event_from_eye0: np.ndarray
    eye0_rms: Optional[float]
    event_rms: Optional[float]
    stereo_rms: float
    epipolar_error_px: float
    pair_count: int

    def as_dict(
        self,
        *,
        checkerboard: CheckerboardSpec,
        eye0_image_size: Tuple[int, int],
        event_image_size: Tuple[int, int],
    ) -> Dict[str, Any]:
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = self.rotation_event_from_eye0
        transform[:3, 3] = self.translation_event_from_eye0.reshape(3)

        return {
            "format": "nir-event-checkerboard-calibration-v1",
            "checkerboard": {
                "inner_corners": list(checkerboard.inner_corners),
                "square_size_mm": checkerboard.square_size_mm,
            },
            "eye0": {
                "image_size_px": list(eye0_image_size),
                "camera_matrix": self.eye0_intrinsics.camera_matrix.tolist(),
                "distortion_coefficients": self.eye0_intrinsics.distortion_coefficients.reshape(-1).tolist(),
            },
            "event": {
                "image_size_px": list(event_image_size),
                "camera_matrix": self.event_intrinsics.camera_matrix.tolist(),
                "distortion_coefficients": self.event_intrinsics.distortion_coefficients.reshape(-1).tolist(),
            },
            "event_from_eye0": {
                "rotation_matrix": self.rotation_event_from_eye0.tolist(),
                "translation_mm": self.translation_event_from_eye0.reshape(3).tolist(),
                "transform_4x4": transform.tolist(),
            },
            "quality": {
                "pair_count": self.pair_count,
                "eye0_intrinsics_rms_px": self.eye0_rms,
                "event_intrinsics_rms_px": self.event_rms,
                "stereo_rms_px": self.stereo_rms,
                "mean_epipolar_error_px": self.epipolar_error_px,
            },
        }


def _as_gray_u8(image: np.ndarray) -> np.ndarray:
    """Convert a camera image into a normalised uint8 grayscale image."""

    image = np.asarray(image)
    if image.ndim == 3:
        if image.shape[2] == 1:
            image = image[:, :, 0]
        elif image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        else:
            raise ValueError(f"Unsupported image channel count: {image.shape[2]}")
    if image.ndim != 2:
        raise ValueError(f"Expected a grayscale or colour image, got {image.shape}")
    if image.dtype == np.uint8:
        return image

    image = image.astype(np.float32, copy=False)
    low = float(np.min(image))
    high = float(np.max(image))
    if not np.isfinite(low) or not np.isfinite(high):
        raise ValueError("Image contains non-finite values")
    if high <= low:
        return np.zeros(image.shape, dtype=np.uint8)
    return cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def event_frame_from_events(
    x: np.ndarray,
    y: np.ndarray,
    polarity: np.ndarray,
    *,
    image_size: Tuple[int, int] = DAVIS346_EVENT_SIZE,
    blur_kernel: int = 3,
) -> np.ndarray:
    """Reconstruct a signed contrast image from one DVS inversion interval.

    Positive polarity is rendered bright and negative polarity dark.  A full
    monitor inversion therefore yields a checkerboard in either polarity;
    do not mix two consecutive inversions in a single call because they cancel.
    """

    width, height = (int(image_size[0]), int(image_size[1]))
    if width <= 0 or height <= 0:
        raise ValueError("image_size must be positive")
    if blur_kernel < 0 or blur_kernel % 2 == 0:
        raise ValueError("blur_kernel must be an odd non-negative integer")

    x = np.asarray(x).reshape(-1)
    y = np.asarray(y).reshape(-1)
    polarity = np.asarray(polarity).reshape(-1)
    if not (x.size == y.size == polarity.size):
        raise ValueError("x, y and polarity must have the same number of entries")

    image = np.zeros((height, width), dtype=np.float32)
    if x.size == 0:
        return np.full(image.shape, 127, dtype=np.uint8)

    x_int = x.astype(np.int64, copy=False)
    y_int = y.astype(np.int64, copy=False)
    valid = (x_int >= 0) & (x_int < width) & (y_int >= 0) & (y_int < height)
    if not np.any(valid):
        return np.full(image.shape, 127, dtype=np.uint8)

    values = np.where(polarity[valid] != 0, 1.0, -1.0).astype(np.float32)
    np.add.at(image, (y_int[valid], x_int[valid]), values)
    peak = float(np.max(np.abs(image)))
    if peak <= 0.0:
        return np.full(image.shape, 127, dtype=np.uint8)

    image = np.clip(127.5 + 127.5 * image / peak, 0, 255).astype(np.uint8)
    if blur_kernel >= 3:
        image = cv2.GaussianBlur(image, (blur_kernel, blur_kernel), 0)
    return image


def find_checkerboard_corners(
    image: np.ndarray, checkerboard: CheckerboardSpec
) -> Optional[np.ndarray]:
    """Find sub-pixel checkerboard corners in normal or inverse polarity."""

    gray = _as_gray_u8(image)
    sb_flags = (
        cv2.CALIB_CB_NORMALIZE_IMAGE
        | cv2.CALIB_CB_EXHAUSTIVE
        | cv2.CALIB_CB_ACCURACY
    )
    classic_flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE

    # Inverted tries matter for a polarity-only event reconstruction.  They are
    # harmless for normal NIR frames and avoid a phase-specific code path.
    for candidate in (gray, cv2.bitwise_not(gray)):
        found = False
        corners: Optional[np.ndarray] = None
        finder_sb = getattr(cv2, "findChessboardCornersSB", None)
        if finder_sb is not None:
            found, corners = finder_sb(candidate, checkerboard.inner_corners, sb_flags)
        if not found:
            found, corners = cv2.findChessboardCorners(
                candidate, checkerboard.inner_corners, classic_flags
            )
            if found:
                cv2.cornerSubPix(
                    candidate,
                    corners,
                    (5, 5),
                    (-1, -1),
                    (
                        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER,
                        30,
                        1e-3,
                    ),
                )
        if found and corners is not None:
            expected = checkerboard.inner_corners[0] * checkerboard.inner_corners[1]
            if len(corners) == expected:
                return np.asarray(corners, dtype=np.float32).reshape(-1, 1, 2)
    return None


def _validate_corners(corners: np.ndarray, checkerboard: CheckerboardSpec) -> np.ndarray:
    corners = np.asarray(corners, dtype=np.float32).reshape(-1, 1, 2)
    expected = checkerboard.inner_corners[0] * checkerboard.inner_corners[1]
    if len(corners) != expected:
        raise ValueError(f"Expected {expected} checkerboard corners, got {len(corners)}")
    return corners


def detect_observation(
    name: str,
    eye0_image: np.ndarray,
    event_image: np.ndarray,
    checkerboard: CheckerboardSpec,
) -> Optional[StereoObservation]:
    """Return a paired observation only when both camera images detect it."""

    eye0_corners = find_checkerboard_corners(eye0_image, checkerboard)
    event_corners = find_checkerboard_corners(event_image, checkerboard)
    if eye0_corners is None or event_corners is None:
        return None
    return StereoObservation(
        name=name,
        eye0_corners=_validate_corners(eye0_corners, checkerboard),
        event_corners=_validate_corners(event_corners, checkerboard),
    )


class CheckerboardStereoCollector:
    """Reader-only collector for use from the live Eye0/DVS callbacks.

    The caller owns both cameras and decides which Eye0 frame matches an event
    inversion interval.  This class never opens a device, changes monitor
    output, or waits in the inference path.  Feed one complete inversion's raw
    DVS events through ``event_x/event_y/event_polarity``; alternatively pass a
    pre-reconstructed ``event_image``.
    """

    def __init__(
        self,
        checkerboard: CheckerboardSpec,
        *,
        event_size: Tuple[int, int] = DAVIS346_EVENT_SIZE,
    ) -> None:
        self.checkerboard = checkerboard
        self.event_size = tuple(int(value) for value in event_size)
        self.observations: List[StereoObservation] = []
        self.rejected: List[Dict[str, str]] = []
        self.eye0_image_size: Optional[Tuple[int, int]] = None
        self.event_image_size: Optional[Tuple[int, int]] = None

    def add_pair(
        self,
        name: str,
        eye0_image: np.ndarray,
        *,
        event_image: Optional[np.ndarray] = None,
        event_x: Optional[np.ndarray] = None,
        event_y: Optional[np.ndarray] = None,
        event_polarity: Optional[np.ndarray] = None,
    ) -> bool:
        """Try to add one paired board pose and return whether it was accepted."""

        raw_values = (event_x, event_y, event_polarity)
        has_raw_events = any(value is not None for value in raw_values)
        if event_image is not None and has_raw_events:
            raise ValueError("Pass event_image or raw events, not both")
        if event_image is None:
            if not all(value is not None for value in raw_values):
                raise ValueError(
                    "Raw event input requires event_x, event_y and event_polarity"
                )
            event_image = event_frame_from_events(
                event_x, event_y, event_polarity, image_size=self.event_size
            )

        current_eye0_size = (int(eye0_image.shape[1]), int(eye0_image.shape[0]))
        current_event_size = (int(event_image.shape[1]), int(event_image.shape[0]))
        if self.eye0_image_size is None:
            self.eye0_image_size = current_eye0_size
        if self.event_image_size is None:
            self.event_image_size = current_event_size
        if current_eye0_size != self.eye0_image_size:
            raise ValueError("Eye0 image size changed during calibration")
        if current_event_size != self.event_image_size:
            raise ValueError("Event image size changed during calibration")

        observation = detect_observation(
            str(name), eye0_image, event_image, self.checkerboard
        )
        if observation is None:
            self.rejected.append(
                {"name": str(name), "reason": "checkerboard not found in both images"}
            )
            return False
        self.observations.append(observation)
        return True

    def calibrate(
        self,
        *,
        eye0_intrinsics: Optional[Intrinsics] = None,
        event_intrinsics: Optional[Intrinsics] = None,
    ) -> StereoCalibration:
        """Fit a stereo model from all accepted observations."""

        if self.eye0_image_size is None or self.event_image_size is None:
            raise ValueError("No readable samples have been submitted")
        return calibrate_stereo(
            self.observations,
            checkerboard=self.checkerboard,
            eye0_image_size=self.eye0_image_size,
            event_image_size=self.event_image_size,
            eye0_intrinsics=eye0_intrinsics,
            event_intrinsics=event_intrinsics,
        )


def _calibrate_intrinsics(
    observations: Sequence[StereoObservation],
    *,
    checkerboard: CheckerboardSpec,
    image_size: Tuple[int, int],
    camera: str,
) -> Tuple[float, Intrinsics]:
    object_points = [checkerboard.object_points() for _ in observations]
    if camera == "eye0":
        image_points = [item.eye0_corners for item in observations]
    elif camera == "event":
        image_points = [item.event_corners for item in observations]
    else:
        raise ValueError(f"Unknown camera: {camera}")
    rms, camera_matrix, distortion, _, _ = cv2.calibrateCamera(
        object_points, image_points, image_size, None, None
    )
    return float(rms), Intrinsics(camera_matrix, distortion)


def _mean_epipolar_error(
    observations: Sequence[StereoObservation], fundamental: np.ndarray
) -> float:
    errors: List[np.ndarray] = []
    for observation in observations:
        points_eye0 = observation.eye0_corners.reshape(-1, 1, 2)
        points_event = observation.event_corners.reshape(-1, 2)
        lines_event = cv2.computeCorrespondEpilines(points_eye0, 1, fundamental).reshape(-1, 3)
        denominator = np.hypot(lines_event[:, 0], lines_event[:, 1])
        distance = np.abs(
            lines_event[:, 0] * points_event[:, 0]
            + lines_event[:, 1] * points_event[:, 1]
            + lines_event[:, 2]
        ) / np.maximum(denominator, np.finfo(np.float64).eps)
        errors.append(distance)
    if not errors:
        return float("nan")
    return float(np.mean(np.concatenate(errors)))


def calibrate_stereo(
    observations: Sequence[StereoObservation],
    *,
    checkerboard: CheckerboardSpec,
    eye0_image_size: Tuple[int, int],
    event_image_size: Tuple[int, int],
    eye0_intrinsics: Optional[Intrinsics] = None,
    event_intrinsics: Optional[Intrinsics] = None,
) -> StereoCalibration:
    """Estimate Eye0/event intrinsics and the rigid event-from-Eye0 transform.

    Pass both intrinsic parameter sets to preserve previously calibrated camera
    models.  Passing only one is intentionally rejected: mixed fixed/free
    calibration makes output quality ambiguous.
    """

    observations = list(observations)
    if len(observations) < 3:
        raise ValueError("At least three paired checkerboard observations are required")
    for observation in observations:
        _validate_corners(observation.eye0_corners, checkerboard)
        _validate_corners(observation.event_corners, checkerboard)

    fixed_intrinsics = eye0_intrinsics is not None or event_intrinsics is not None
    if fixed_intrinsics and (eye0_intrinsics is None or event_intrinsics is None):
        raise ValueError("Provide both Eye0 and event intrinsics, or neither")

    eye0_rms: Optional[float]
    event_rms: Optional[float]
    if fixed_intrinsics:
        eye0_rms = None
        event_rms = None
        assert eye0_intrinsics is not None and event_intrinsics is not None
    else:
        eye0_rms, eye0_intrinsics = _calibrate_intrinsics(
            observations,
            checkerboard=checkerboard,
            image_size=eye0_image_size,
            camera="eye0",
        )
        event_rms, event_intrinsics = _calibrate_intrinsics(
            observations,
            checkerboard=checkerboard,
            image_size=event_image_size,
            camera="event",
        )

    assert eye0_intrinsics is not None and event_intrinsics is not None
    object_points = [checkerboard.object_points() for _ in observations]
    eye0_points = [item.eye0_corners for item in observations]
    event_points = [item.event_corners for item in observations]
    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER,
        100,
        1e-7,
    )
    (
        stereo_rms,
        eye0_matrix,
        eye0_distortion,
        event_matrix,
        event_distortion,
        rotation,
        translation,
        _essential,
        fundamental,
    ) = cv2.stereoCalibrate(
        object_points,
        eye0_points,
        event_points,
        eye0_intrinsics.camera_matrix.copy(),
        eye0_intrinsics.distortion_coefficients.copy(),
        event_intrinsics.camera_matrix.copy(),
        event_intrinsics.distortion_coefficients.copy(),
        eye0_image_size,
        criteria=criteria,
        flags=cv2.CALIB_FIX_INTRINSIC,
    )
    return StereoCalibration(
        eye0_intrinsics=Intrinsics(eye0_matrix, eye0_distortion),
        event_intrinsics=Intrinsics(event_matrix, event_distortion),
        rotation_event_from_eye0=np.asarray(rotation, dtype=np.float64),
        translation_event_from_eye0=np.asarray(translation, dtype=np.float64),
        eye0_rms=eye0_rms,
        event_rms=event_rms,
        stereo_rms=float(stereo_rms),
        epipolar_error_px=_mean_epipolar_error(observations, fundamental),
        pair_count=len(observations),
    )


def _first_array(bundle: Mapping[str, np.ndarray], names: Iterable[str]) -> np.ndarray:
    for name in names:
        if name in bundle:
            return np.asarray(bundle[name])
    raise KeyError(f"Expected one of {', '.join(names)} in event .npz sample")


def load_eye0_image(path: Path) -> np.ndarray:
    """Load an Eye0 image from a standard image file or a simple .npy/.npz file."""

    suffix = path.suffix.lower()
    if suffix == ".npy":
        return np.load(path, allow_pickle=False)
    if suffix == ".npz":
        with np.load(path, allow_pickle=False) as bundle:
            return _first_array(bundle, ("image", "frame", "gray"))
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Could not load Eye0 image: {path}")
    return image


def load_event_image(
    path: Path, *, event_size: Tuple[int, int] = DAVIS346_EVENT_SIZE
) -> np.ndarray:
    """Load an event reconstruction or make one from a raw-event ``.npz``."""

    suffix = path.suffix.lower()
    if suffix == ".npy":
        return np.load(path, allow_pickle=False)
    if suffix == ".npz":
        with np.load(path, allow_pickle=False) as bundle:
            names = set(bundle.files)
            if names.intersection({"image", "frame", "event_image"}):
                return _first_array(bundle, ("image", "frame", "event_image"))
            x = _first_array(bundle, ("x", "xs"))
            y = _first_array(bundle, ("y", "ys"))
            polarity = _first_array(bundle, ("p", "polarity", "polarities", "ps"))
        return event_frame_from_events(x, y, polarity, image_size=event_size)
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Could not load event image: {path}")
    return image


_SAMPLE_SUFFIXES = frozenset({".bmp", ".jpeg", ".jpg", ".npy", ".npz", ".png", ".tif", ".tiff"})


def _samples_by_stem(directory: Path) -> Dict[str, Path]:
    if not directory.is_dir():
        raise ValueError(f"Sample directory does not exist: {directory}")
    samples: Dict[str, Path] = {}
    for path in sorted(directory.iterdir()):
        if not path.is_file() or path.suffix.lower() not in _SAMPLE_SUFFIXES:
            continue
        if path.stem in samples:
            raise ValueError(
                f"More than one sample has stem {path.stem!r} in {directory}"
            )
        samples[path.stem] = path
    if not samples:
        raise ValueError(f"No supported samples found in {directory}")
    return samples


def load_paired_observations(
    *,
    eye0_dir: Path,
    event_dir: Path,
    checkerboard: CheckerboardSpec,
    event_size: Tuple[int, int] = DAVIS346_EVENT_SIZE,
) -> Tuple[List[StereoObservation], List[Dict[str, str]], Tuple[int, int], Tuple[int, int]]:
    """Load same-stem samples and retain only pairs with both detections."""

    eye0_samples = _samples_by_stem(eye0_dir)
    event_samples = _samples_by_stem(event_dir)
    names = sorted(set(eye0_samples).intersection(event_samples))
    if not names:
        raise ValueError("No matching filename stems between Eye0 and event samples")

    observations: List[StereoObservation] = []
    rejected: List[Dict[str, str]] = []
    eye0_size: Optional[Tuple[int, int]] = None
    event_image_size: Optional[Tuple[int, int]] = None
    for name in names:
        try:
            eye0_image = load_eye0_image(eye0_samples[name])
            event_image = load_event_image(event_samples[name], event_size=event_size)
            current_eye0_size = (int(eye0_image.shape[1]), int(eye0_image.shape[0]))
            current_event_size = (int(event_image.shape[1]), int(event_image.shape[0]))
            if eye0_size is None:
                eye0_size = current_eye0_size
            if event_image_size is None:
                event_image_size = current_event_size
            if current_eye0_size != eye0_size or current_event_size != event_image_size:
                raise ValueError("All samples for a camera must have the same image size")
            observation = detect_observation(name, eye0_image, event_image, checkerboard)
            if observation is None:
                rejected.append({"name": name, "reason": "checkerboard not found in both images"})
            else:
                observations.append(observation)
        except Exception as error:
            rejected.append({"name": name, "reason": f"{type(error).__name__}: {error}"})

    if eye0_size is None or event_image_size is None:
        raise ValueError("No readable paired samples")
    return observations, rejected, eye0_size, event_image_size


def _load_intrinsics(path: Path) -> Intrinsics:
    with path.open("r", encoding="utf8") as file:
        payload = json.load(file)
    if not isinstance(payload, Mapping):
        raise ValueError(f"Intrinsics JSON must be an object: {path}")
    if "camera_matrix" not in payload or "distortion_coefficients" not in payload:
        raise ValueError(
            f"{path} must contain camera_matrix and distortion_coefficients"
        )
    return Intrinsics(payload["camera_matrix"], payload["distortion_coefficients"])


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Offline Eye0 NIR / DAVIS checkerboard stereo calibration. "
            "This reads samples only; it never displays the checkerboard."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--eye0-dir", required=True, type=Path)
    parser.add_argument("--event-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--inner-corners",
        metavar=("COLUMNS", "ROWS"),
        type=int,
        nargs=2,
        default=(9, 6),
        help="Checkerboard inner-corner count; a 10x7-square board is 9 6",
    )
    parser.add_argument("--square-size-mm", type=float, required=True)
    parser.add_argument(
        "--event-size",
        metavar=("WIDTH", "HEIGHT"),
        type=int,
        nargs=2,
        default=DAVIS346_EVENT_SIZE,
        help="Used only for raw x/y/p event .npz samples",
    )
    parser.add_argument(
        "--eye0-intrinsics",
        type=Path,
        help="Optional JSON with camera_matrix and distortion_coefficients",
    )
    parser.add_argument(
        "--event-intrinsics",
        type=Path,
        help="Optional JSON with camera_matrix and distortion_coefficients",
    )
    parser.add_argument(
        "--min-pairs",
        type=int,
        default=12,
        help="Refuse output unless at least this many valid paired poses exist",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    if args.min_pairs < 3:
        print("--min-pairs must be at least 3", file=sys.stderr)
        return 2
    if bool(args.eye0_intrinsics) != bool(args.event_intrinsics):
        print(
            "Provide both --eye0-intrinsics and --event-intrinsics, or neither.",
            file=sys.stderr,
        )
        return 2
    try:
        checkerboard = CheckerboardSpec(
            inner_corners=tuple(args.inner_corners), square_size_mm=args.square_size_mm
        )
        observations, rejected, eye0_size, event_size = load_paired_observations(
            eye0_dir=args.eye0_dir,
            event_dir=args.event_dir,
            checkerboard=checkerboard,
            event_size=tuple(args.event_size),
        )
        print(
            f"Checkerboard detected in {len(observations)} paired samples; "
            f"rejected {len(rejected)}."
        )
        if len(observations) < args.min_pairs:
            print(
                f"Need at least {args.min_pairs} valid pairs; no calibration was written.",
                file=sys.stderr,
            )
            return 2
        eye0_intrinsics = (
            _load_intrinsics(args.eye0_intrinsics) if args.eye0_intrinsics else None
        )
        event_intrinsics = (
            _load_intrinsics(args.event_intrinsics) if args.event_intrinsics else None
        )
        calibration = calibrate_stereo(
            observations,
            checkerboard=checkerboard,
            eye0_image_size=eye0_size,
            event_image_size=event_size,
            eye0_intrinsics=eye0_intrinsics,
            event_intrinsics=event_intrinsics,
        )
        result = calibration.as_dict(
            checkerboard=checkerboard,
            eye0_image_size=eye0_size,
            event_image_size=event_size,
        )
        result["accepted_sample_names"] = [item.name for item in observations]
        result["rejected_samples"] = rejected
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf8") as file:
            json.dump(result, file, indent=2)
            file.write("\n")
        print(
            f"Wrote {args.output} | stereo RMS {calibration.stereo_rms:.4f}px | "
            f"mean epipolar error {calibration.epipolar_error_px:.4f}px"
        )
        return 0
    except Exception as error:
        print(f"Calibration failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
