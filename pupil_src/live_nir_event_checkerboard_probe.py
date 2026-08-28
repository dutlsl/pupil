#!/usr/bin/env python3
"""Capture one Eye0/DAVIS checkerboard probe without owning Pupil's runtime.

This is a setup check, not the final collector: it opens the selected Eye0 UVC
device and DAVIS briefly, saves diagnostic images, and reports whether the
same checkerboard can be detected in each source.  The DVS image uses the most
active raw-event batch observed during the interval, which should coincide
with one black/white monitor swap.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import cv2
import numpy as np
from dv_processing.io import CameraCapture

from nir_event_checkerboard_calibration import (
    CheckerboardSpec,
    event_frame_from_events,
    find_checkerboard_corners,
)


def _capture_eye0_frame(device: str, deadline: float) -> Optional[np.ndarray]:
    capture = cv2.VideoCapture(device, cv2.CAP_V4L2)
    if not capture.isOpened():
        return None
    try:
        capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, 400)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 400)
        while time.monotonic() < deadline:
            ok, frame = capture.read()
            if ok and frame is not None:
                return frame
            time.sleep(0.005)
        return None
    finally:
        capture.release()


def _capture_davis(
    deadline: float,
) -> Tuple[Tuple[int, int], Optional[np.ndarray], Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    capture = CameraCapture()
    event_size = tuple(int(value) for value in capture.getEventResolution())
    aps_image: Optional[np.ndarray] = None
    best_events: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None
    best_event_count = 0

    while time.monotonic() < deadline:
        frame = capture.getNextFrame()
        if frame is not None:
            aps_image = np.asarray(frame.image).copy()

        events = capture.getNextEventBatch()
        if events is not None:
            coordinates = np.asarray(events.coordinates())
            count = int(coordinates.shape[0])
            if count > best_event_count:
                best_event_count = count
                best_events = (
                    coordinates[:, 0].copy(),
                    coordinates[:, 1].copy(),
                    np.asarray(events.polarities()).copy(),
                )
        time.sleep(0.001)
    return event_size, aps_image, best_events


def _annotate(image: np.ndarray, corners: Optional[np.ndarray], spec: CheckerboardSpec) -> np.ndarray:
    if image.ndim == 2:
        annotated = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        annotated = image.copy()
    cv2.drawChessboardCorners(annotated, spec.inner_corners, corners, corners is not None)
    return annotated


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check whether Eye0 and DAVIS can see the displayed checkerboard."
    )
    parser.add_argument("--eye0-device", default="/dev/video0")
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/nir_event_checkerboard_probe"))
    parser.add_argument("--duration-s", type=float, default=4.0)
    parser.add_argument("--inner-corners", type=int, nargs=2, default=(9, 6))
    parser.add_argument("--square-size-mm", type=float, default=10.0)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    if args.duration_s <= 0.0:
        raise SystemExit("--duration-s must be positive")
    spec = CheckerboardSpec(tuple(args.inner_corners), args.square_size_mm)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    deadline = time.monotonic() + args.duration_s
    eye0_image = _capture_eye0_frame(args.eye0_device, deadline)
    event_size, aps_image, raw_events = _capture_davis(deadline)
    event_image = (
        event_frame_from_events(*raw_events, image_size=event_size)
        if raw_events is not None
        else None
    )

    results: Dict[str, Any] = {
        "eye0_device": args.eye0_device,
        "davis_event_size_px": list(event_size),
        "inner_corners": list(spec.inner_corners),
        "eye0": {"received": eye0_image is not None, "corners": 0},
        "davis_aps": {"received": aps_image is not None, "corners": 0},
        "davis_events": {
            "received": raw_events is not None,
            "event_count": 0 if raw_events is None else int(raw_events[0].size),
            "corners": 0,
        },
    }
    for name, image in (
        ("eye0", eye0_image),
        ("davis_aps", aps_image),
        ("davis_events", event_image),
    ):
        if image is None:
            continue
        corners = find_checkerboard_corners(image, spec)
        results[name]["corners"] = 0 if corners is None else int(len(corners))
        cv2.imwrite(str(args.output_dir / f"{name}.png"), image)
        cv2.imwrite(
            str(args.output_dir / f"{name}_corners.png"),
            _annotate(image, corners, spec),
        )

    with (args.output_dir / "result.json").open("w", encoding="utf8") as file:
        json.dump(results, file, indent=2)
        file.write("\n")
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
