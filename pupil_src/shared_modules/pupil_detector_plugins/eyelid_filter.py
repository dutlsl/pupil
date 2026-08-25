"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
import cv2
import numpy as np


def fit_eyelid_robust_ellipse(best_contour: np.ndarray, enabled: bool = True):
    """
    Fits an ellipse to pupil contour with optional eyelid occlusion compensation.

    Parameters:
        best_contour: Nx1x2 or Nx2 array of contour coordinates.
        enabled: If True, identifies and removes flat horizontal chords created by
                 upper/lower eyelids, fitting the ellipse strictly on the unoccluded
                 circular/elliptical perimeter. If False, performs standard cv2.fitEllipse.

    Returns:
        ((cx, cy), (MA, ma), angle_deg)
    """
    if not enabled:
        return cv2.fitEllipse(best_contour)

    pts = best_contour.reshape(-1, 2)
    if len(pts) < 5:
        return cv2.fitEllipse(best_contour)

    y_min, y_max = pts[:, 1].min(), pts[:, 1].max()
    h = y_max - y_min
    if h < 8:
        return cv2.fitEllipse(best_contour)

    # Check for upper eyelid chord (dense horizontal line at top)
    top_band = pts[pts[:, 1] <= y_min + max(2, int(h * 0.08))]
    # Check for lower eyelid chord (dense horizontal line at bottom)
    bot_band = pts[pts[:, 1] >= y_max - max(2, int(h * 0.08))]

    mask_valid = np.ones(len(pts), dtype=bool)

    # If top chord is flat (occluding upper eyelid)
    if len(top_band) >= 5 and (top_band[:, 0].max() - top_band[:, 0].min()) > (h * 0.2):
        mask_valid[pts[:, 1] <= y_min + max(2, int(h * 0.06))] = False

    # If bottom chord is flat (occluding lower eyelid)
    if len(bot_band) >= 5 and (bot_band[:, 0].max() - bot_band[:, 0].min()) > (h * 0.2):
        mask_valid[pts[:, 1] >= y_max - max(2, int(h * 0.06))] = False

    valid_pts = pts[mask_valid]
    if len(valid_pts) >= 5:
        try:
            return cv2.fitEllipse(valid_pts)
        except Exception:
            pass

    return cv2.fitEllipse(best_contour)
