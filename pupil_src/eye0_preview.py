#!/usr/bin/env python3
"""Small live Eye0 preview with optional checkerboard-corner overlay."""

from __future__ import annotations

import argparse
import time
from typing import Optional, Sequence

import cv2

from nir_event_checkerboard_calibration import (
    CheckerboardSpec,
    find_checkerboard_corners,
)


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Show the live Eye0 image.")
    parser.add_argument("--device", default="/dev/video0")
    parser.add_argument("--width", type=int, default=400)
    parser.add_argument("--height", type=int, default=400)
    parser.add_argument("--inner-corners", type=int, nargs=2, default=(9, 6))
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    checkerboard = CheckerboardSpec(tuple(args.inner_corners), 10.0)
    capture = cv2.VideoCapture(args.device, cv2.CAP_V4L2)
    if not capture.isOpened():
        raise SystemExit(f"Could not open Eye0 device: {args.device}")
    capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    title = "Eye0 preview — Esc/q: close"
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(title, args.width, args.height)
    try:
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                time.sleep(0.005)
                continue
            corners = find_checkerboard_corners(frame, checkerboard)
            if corners is not None:
                cv2.drawChessboardCorners(
                    frame, checkerboard.inner_corners, corners, True
                )
                status = "checkerboard: 54/54"
                colour = (0, 255, 0)
            else:
                status = "checkerboard: not found"
                colour = (0, 0, 255)
            cv2.putText(
                frame,
                status,
                (8, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                colour,
                2,
                cv2.LINE_AA,
            )
            cv2.imshow(title, frame)
            if (cv2.waitKey(1) & 0xFF) in (27, ord("q")):
                break
    finally:
        capture.release()
        cv2.destroyWindow(title)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
