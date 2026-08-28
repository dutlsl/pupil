#!/usr/bin/env python3
"""Fullscreen black/white checkerboard stimulus for an event-camera calibration.

This is intentionally separate from ``nir_event_checkerboard_calibration.py``:
the latter reads Eye0/DVS observations, while this small tool only displays a
pattern.  Every phase transition swaps black and white squares in place, which
creates a clean ON/OFF event response without moving the checkerboard corners.

Keys: ``Esc``/``q`` close, ``space`` pauses/resumes, and ``s`` swaps once while
paused.
"""

from __future__ import annotations

import argparse
import time
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np


def make_checkerboard(
    image_size: Tuple[int, int],
    *,
    squares_x: int = 10,
    squares_y: int = 7,
    inverted: bool = False,
) -> np.ndarray:
    """Return a centred checkerboard with fixed pixel geometry."""

    width, height = (int(image_size[0]), int(image_size[1]))
    if width <= 0 or height <= 0:
        raise ValueError("image_size must be positive")
    if squares_x < 2 or squares_y < 2:
        raise ValueError("A checkerboard needs at least 2 by 2 squares")

    square = min(width // squares_x, height // squares_y)
    if square < 1:
        raise ValueError("The requested checkerboard does not fit the display")
    board_width = squares_x * square
    board_height = squares_y * square
    left = (width - board_width) // 2
    top = (height - board_height) // 2

    # A neutral fixed border prevents the border itself from dominating the
    # event stream; only checkerboard squares change at a phase transition.
    image = np.full((height, width), 127, dtype=np.uint8)
    inverse_offset = 1 if inverted else 0
    for row in range(squares_y):
        for column in range(squares_x):
            bright = (row + column + inverse_offset) % 2 == 0
            colour = 255 if bright else 0
            x0 = left + column * square
            y0 = top + row * square
            image[y0 : y0 + square, x0 : x0 + square] = colour
    return image


def _screen_size(fallback: Tuple[int, int]) -> Tuple[int, int]:
    """Ask the current desktop for its size without making it a dependency."""

    try:
        import tkinter

        root = tkinter.Tk()
        root.withdraw()
        size = (root.winfo_screenwidth(), root.winfo_screenheight())
        root.destroy()
        if size[0] > 0 and size[1] > 0:
            return size
    except Exception:
        pass
    return fallback


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Display a checkerboard that swaps black and white squares."
    )
    parser.add_argument("--squares-x", type=int, default=10)
    parser.add_argument("--squares-y", type=int, default=7)
    parser.add_argument(
        "--toggle-hz",
        type=float,
        default=5.0,
        help="Number of full black/white swaps per second; 0 shows a static board.",
    )
    parser.add_argument("--width", type=int, help="Override current display width")
    parser.add_argument("--height", type=int, help="Override current display height")
    parser.add_argument("--windowed", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    if args.toggle_hz < 0.0:
        raise SystemExit("--toggle-hz must be non-negative")
    if (args.width is None) != (args.height is None):
        raise SystemExit("Specify --width and --height together")

    image_size = (
        (args.width, args.height)
        if args.width is not None
        else _screen_size((1920, 1080))
    )
    normal = make_checkerboard(
        image_size,
        squares_x=args.squares_x,
        squares_y=args.squares_y,
        inverted=False,
    )
    inverse = make_checkerboard(
        image_size,
        squares_x=args.squares_x,
        squares_y=args.squares_y,
        inverted=True,
    )

    title = "DVS checkerboard stimulus — Esc/q: close, space: pause"
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    if not args.windowed:
        cv2.setWindowProperty(title, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    paused = False
    manual_inverse = False
    last_phase: Optional[bool] = None
    start = time.monotonic()
    try:
        while True:
            if paused or args.toggle_hz == 0.0:
                inverted = manual_inverse
            else:
                inverted = (int((time.monotonic() - start) * args.toggle_hz) % 2) == 1
            if inverted != last_phase:
                cv2.imshow(title, inverse if inverted else normal)
                last_phase = inverted

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord(" "):
                paused = not paused
                if paused:
                    manual_inverse = bool(last_phase)
            elif key == ord("s") and paused:
                manual_inverse = not manual_inverse
            time.sleep(0.001)
    finally:
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
