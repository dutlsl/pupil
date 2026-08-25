"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
import math
import numpy as np


class OneEuroFilter:
    """
    Speed-adaptive low-pass filter for noisy biometric / HCI signals.
    Based on:
      Casiez, G., Roussel, N., and Vogel, D. (2012).
      1€ Filter: A Simple Speed-based Low-pass Filter for Noisy Input in Human-Computer Interaction.
      Proceedings of the SIGCHI Conference on Human Factors in Computing Systems (CHI '12), 2527-2530.
    """

    def __init__(
        self,
        min_cutoff: float = 1.0,
        beta: float = 0.007,
        d_cutoff: float = 1.0,
    ):
        """
        Parameters:
            min_cutoff: Minimum cutoff frequency in Hz. Lower values reduce jitter during fixation.
            beta: Speed coefficient. Higher values reduce lag during rapid saccades.
            d_cutoff: Cutoff frequency for the derivative (speed) filter in Hz.
        """
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)

        self.x_prev = None
        self.dx_prev = None
        self.t_prev = None

    def reset(self):
        self.x_prev = None
        self.dx_prev = None
        self.t_prev = None

    @staticmethod
    def _smoothing_factor(dt: float, cutoff: float) -> float:
        r = 2.0 * math.pi * cutoff * dt
        return r / (r + 1.0)

    def filter(self, t: float, x: np.ndarray) -> np.ndarray:
        """
        Filter an input vector x at timestamp t.
        Parameters:
            t: Current timestamp in seconds.
            x: 1D numpy array of values (e.g. [cx, cy] or [cx, cy, MA, ma]).
        Returns:
            Filtered numpy array.
        """
        x = np.asarray(x, dtype=np.float64)

        if self.x_prev is None or self.t_prev is None:
            self.x_prev = x.copy()
            self.dx_prev = np.zeros_like(x)
            self.t_prev = float(t)
            return x

        dt = float(t) - self.t_prev
        if dt <= 1e-5:
            # Timestamp too close or duplicate; return previous smoothed value
            return self.x_prev.copy()

        # 1. Estimate derivative (speed)
        dx = (x - self.x_prev) / dt
        alpha_d = self._smoothing_factor(dt, self.d_cutoff)
        dx_hat = alpha_d * dx + (1.0 - alpha_d) * self.dx_prev

        # 2. Dynamic cutoff frequency based on magnitude of velocity
        speed = np.linalg.norm(dx_hat)
        cutoff = self.min_cutoff + self.beta * speed

        # 3. Dynamic smoothing of the signal
        alpha = self._smoothing_factor(dt, cutoff)
        x_hat = alpha * x + (1.0 - alpha) * self.x_prev

        # 4. Update historical states
        self.x_prev = x_hat.copy()
        self.dx_prev = dx_hat.copy()
        self.t_prev = float(t)

        return x_hat


class PupilGeometryFilter:
    """
    Dedicated filter for pupil ellipse geometry:
    Filters (cx, cy), (MA, ma), and angle theta with angle wrap-around handling.
    """

    def __init__(
        self,
        min_cutoff: float = 1.0,
        beta: float = 0.007,
        d_cutoff: float = 1.0,
    ):
        self.pos_filter = OneEuroFilter(min_cutoff=min_cutoff, beta=beta, d_cutoff=d_cutoff)
        self.axes_filter = OneEuroFilter(min_cutoff=min_cutoff, beta=beta, d_cutoff=d_cutoff)
        self.angle_prev = None
        self.t_prev = None

    def reset(self):
        self.pos_filter.reset()
        self.axes_filter.reset()
        self.angle_prev = None
        self.t_prev = None

    def filter(
        self,
        t: float,
        cx: float,
        cy: float,
        MA: float,
        ma: float,
        angle_deg: float,
    ):
        # 1. Filter position (cx, cy)
        filtered_pos = self.pos_filter.filter(t, np.array([cx, cy]))
        out_cx, out_cy = float(filtered_pos[0]), float(filtered_pos[1])

        # 2. Filter axes (MA, ma)
        filtered_axes = self.axes_filter.filter(t, np.array([MA, ma]))
        out_MA, out_ma = float(filtered_axes[0]), float(filtered_axes[1])

        # 3. Angle wrap-around filtering (modulo 180 degrees)
        if self.angle_prev is None or self.t_prev is None:
            out_angle = float(angle_deg)
        else:
            diff = (angle_deg - self.angle_prev + 90.0) % 180.0 - 90.0
            unwrapped_angle = self.angle_prev + diff
            # Moderate smoothing on angle
            out_angle = float(0.3 * unwrapped_angle + 0.7 * self.angle_prev) % 180.0

        self.angle_prev = out_angle
        self.t_prev = float(t)

        return out_cx, out_cy, out_MA, out_ma, out_angle
