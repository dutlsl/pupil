#!/usr/bin/env python3
"""Show a live DAVIS event stream without NIR, TDTracker, or Pupil.

ON events are red and OFF events are blue.  The display integrates only a
short time window so it remains a live event view rather than a tracker input.
Press Escape or q to close.
"""

from __future__ import annotations

import argparse
import time
from typing import Optional, Sequence

import cv2
import numpy as np
from dv_processing.io import CameraCapture


WINDOW_NAME = "DAVIS event preview — Esc/q: close"


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Open only the DAVIS event-camera preview."
    )
    parser.add_argument(
        "--accumulate-ms",
        type=float,
        default=33.0,
        help="Event accumulation window for display only (default: 33 ms).",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=2.0,
        help="OpenCV window scale factor (default: 2).",
    )
    return parser.parse_args(argv)


def _render(on: np.ndarray, off: np.ndarray, event_count: int, window_ms: float) -> np.ndarray:
    """Render the current raw-event accumulation; no TDTracker data is used."""
    image = np.zeros((*on.shape, 3), dtype=np.uint8)
    on_limited = np.minimum(on, 16).astype(np.uint8)
    off_limited = np.minimum(off, 16).astype(np.uint8)
    image[..., 2] = on_limited * 16
    image[..., 0] = off_limited * 16
    image[..., 1] = np.minimum(on_limited + off_limited, 12) * 10
    cv2.putText(
        image,
        f"events {event_count} / {window_ms:.1f} ms",
        (7, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (230, 230, 230),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        "ON red / OFF blue",
        (7, image.shape[0] - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (230, 230, 230),
        1,
        cv2.LINE_AA,
    )
    return image


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    if args.accumulate_ms <= 0.0 or args.scale <= 0.0:
        raise SystemExit("--accumulate-ms and --scale must be positive")

    capture = CameraCapture()
    width, height = (int(value) for value in capture.getEventResolution())
    accumulate_us = round(args.accumulate_ms * 1000.0)
    on = np.zeros((height, width), dtype=np.uint16)
    off = np.zeros_like(on)
    window_start_us: Optional[int] = None
    event_count = 0

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, round(width * args.scale), round(height * args.scale))
    print(
        f"DAVIS event preview opened: {width}x{height}. Press Esc or q to close.",
        flush=True,
    )
    try:
        while capture.isRunning():
            events = capture.getNextEventBatch()
            if events is not None:
                timestamps = np.asarray(events.timestamps(), dtype=np.int64)
                coordinates = np.asarray(events.coordinates(), dtype=np.int64)
                polarities = np.asarray(events.polarities(), dtype=np.uint8)
                if timestamps.size:
                    if window_start_us is None:
                        window_start_us = int(timestamps[0])
                    x = coordinates[:, 0].astype(np.intp, copy=False)
                    y = coordinates[:, 1].astype(np.intp, copy=False)
                    valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
                    if np.any(valid):
                        x = x[valid]
                        y = y[valid]
                        positive = polarities[valid] != 0
                        np.add.at(on, (y[positive], x[positive]), 1)
                        np.add.at(off, (y[~positive], x[~positive]), 1)
                        event_count += int(np.count_nonzero(valid))

                    elapsed_us = int(timestamps[-1]) - window_start_us
                    if elapsed_us >= accumulate_us:
                        cv2.imshow(
                            WINDOW_NAME,
                            _render(on, off, event_count, elapsed_us / 1000.0),
                        )
                        on.fill(0)
                        off.fill(0)
                        event_count = 0
                        window_start_us = None

            if (cv2.waitKey(1) & 0xFF) in (27, ord("q")):
                break
            # Keep DAVIS read pacing predictable without starving the desktop.
            time.sleep(0.001)
    finally:
        del capture
        cv2.destroyWindow(WINDOW_NAME)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
