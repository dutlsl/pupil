"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
import logging
import os
import typing as T
from collections import deque

import numpy as np


logger = logging.getLogger(__name__)


class RealtimeMatcher:
    def __init__(self):
        self.min_pupil_confidence = 0.6
        self._caches = (deque(), deque())
        self.recently_estimated_framerate = 1 / 120
        self.framerate_estimation_smoothing_factor = 1 / 50
        self.sample_cutoff = 10

    def is_cache_valid(self, cache):
        return len(cache) >= 2

    def estimate_frame_rate_raw(self, cache):
        return np.mean(np.diff([d["timestamp"] for d in cache]))

    def estimate_framerate_smoothed(self, eye0_cache, eye1_cache):
        if self.is_cache_valid(eye0_cache) and self.is_cache_valid(eye1_cache):
            eye0_framerate_raw = self.estimate_frame_rate_raw(eye0_cache)
            eye1_framerate_raw = self.estimate_frame_rate_raw(eye1_cache)
            estimated_framerate_raw = max(eye0_framerate_raw, eye1_framerate_raw)
        elif self.is_cache_valid(eye0_cache):
            estimated_framerate_raw = self.estimate_frame_rate_raw(eye0_cache)
        elif self.is_cache_valid(eye1_cache):
            estimated_framerate_raw = self.estimate_frame_rate_raw(eye1_cache)
        else:
            return self.recently_estimated_framerate

        self.recently_estimated_framerate += (
            estimated_framerate_raw - self.recently_estimated_framerate
        ) * self.framerate_estimation_smoothing_factor
        return self.recently_estimated_framerate

    def map_batch(self, pupil_list):
        current_caches = self._caches
        self._caches = (deque(), deque())
        results = []
        for p in pupil_list:
            results.extend(self.on_pupil_datum(p))

        self._caches = current_caches
        return results

    def on_pupil_datum(self, p) -> T.Iterator:
        """Returns a list with either zero, one or two pupil datums.
        - zero: not enough data in queue
        - one: no binocular match possible
        - two: binocular match
        """
        self._caches[p["id"]].append(p)
        temporal_cutoff = 2 * self.estimate_framerate_smoothed(*self._caches)

        # map low confidence pupil data monocularly
        if (
            self._caches[0]
            and self._caches[0][0]["confidence"] < self.min_pupil_confidence
        ):
            p = self._caches[0].popleft()
            yield [p]
        elif (
            self._caches[1]
            and self._caches[1][0]["confidence"] < self.min_pupil_confidence
        ):
            p = self._caches[1].popleft()
            yield [p]
        # map high confidence data binocularly if available
        elif self._caches[0] and self._caches[1]:
            # we have binocular data
            if self._caches[0][0]["timestamp"] < self._caches[1][0]["timestamp"]:
                p0 = self._caches[0].popleft()
                p1 = self._caches[1][0]
                older_pt = p0
            else:
                p0 = self._caches[0][0]
                p1 = self._caches[1].popleft()
                older_pt = p1

            if abs(p0["timestamp"] - p1["timestamp"]) < temporal_cutoff:
                yield [p0, p1]
            else:
                yield [older_pt]

        elif len(self._caches[0]) > self.sample_cutoff:
            p = self._caches[0].popleft()
            yield [p]
        elif len(self._caches[1]) > self.sample_cutoff:
            p = self._caches[1].popleft()
            yield [p]


class HybridBinocularMatcher:
    """Match every Eye0 sample against the newest usable Eye1 sample.

    The integrated NIR/DVS pipeline produces Eye0 data at the event-camera
    rate, while Eye1 remains limited by its NIR/Pye3D frame rate.  A regular
    queue matcher is unsuitable here: consuming Eye1 samples would either
    suppress most Eye0 output or cause a previously consumed Eye1 sample to be
    paired unpredictably.  Instead, Eye1 is a single, read-only snapshot for
    every subsequent Eye0 datum.

    Eye1 arrivals intentionally yield no match.  Only an Eye0 arrival can
    create gaze output, which preserves the fresh-only TDTracker contract.
    """

    DEFAULT_EYE1_MAX_AGE_MS = 50.0
    _FALSE_ENV_VALUES = frozenset(("", "0", "false", "no", "off"))

    def __init__(
        self,
        eye1_max_age_ms: float = DEFAULT_EYE1_MAX_AGE_MS,
        min_pupil_confidence: float = 0.6,
    ):
        eye1_max_age_seconds = float(eye1_max_age_ms) / 1000.0
        if not np.isfinite(eye1_max_age_seconds) or eye1_max_age_seconds < 0.0:
            raise ValueError(
                "eye1_max_age_ms must be a finite, non-negative number, got "
                f"{eye1_max_age_ms!r}"
            )

        self.eye1_max_age_seconds = eye1_max_age_seconds
        self.min_pupil_confidence = float(min_pupil_confidence)
        self._latest_eye1 = None
        self._latest_eye1_observation_timestamp = None

    @classmethod
    def from_environment(cls) -> "HybridBinocularMatcher":
        """Build a matcher from the integration-specific age setting.

        An invalid optional setting must not take down the regular gaze path;
        retain the documented default and leave a diagnostic in the log.
        """

        raw_value = os.getenv(
            "PUPIL_HYBRID_EYE1_MAX_AGE_MS", str(cls.DEFAULT_EYE1_MAX_AGE_MS)
        )
        try:
            return cls(eye1_max_age_ms=float(raw_value))
        except (TypeError, ValueError):
            logger.warning(
                "Invalid PUPIL_HYBRID_EYE1_MAX_AGE_MS=%r; using %.1f ms",
                raw_value,
                cls.DEFAULT_EYE1_MAX_AGE_MS,
            )
            return cls(eye1_max_age_ms=cls.DEFAULT_EYE1_MAX_AGE_MS)

    @classmethod
    def integrated_mode_enabled(cls) -> bool:
        """Return whether the dedicated asymmetric matcher is requested."""

        value = os.getenv("PUPIL_HYBRID_INTEGRATED", "")
        return value.strip().lower() not in cls._FALSE_ENV_VALUES

    def map_batch(self, pupil_list):
        """Map a finite batch without changing the live Eye1 snapshot.

        This mirrors :class:`RealtimeMatcher`'s isolation semantics for
        calibration/offline callers while still exercising the asymmetric
        matching policy inside the batch.
        """

        live_eye1 = self._latest_eye1
        live_eye1_observation_timestamp = self._latest_eye1_observation_timestamp
        self._latest_eye1 = None
        self._latest_eye1_observation_timestamp = None
        try:
            results = []
            for pupil_datum in pupil_list:
                results.extend(self.on_pupil_datum(pupil_datum))
            return results
        finally:
            self._latest_eye1 = live_eye1
            self._latest_eye1_observation_timestamp = live_eye1_observation_timestamp

    def on_pupil_datum(self, pupil_datum) -> T.Iterator:
        """Yield one Eye0 mono/binocular match, or nothing for Eye1 updates."""

        eye_id = pupil_datum["id"]
        if eye_id == 1:
            self._update_eye1_snapshot(pupil_datum)
            return
        if eye_id != 0:
            raise ValueError(f"Expected eye id 0 or 1, got {eye_id!r}")

        if not self._is_valid(pupil_datum):
            # Keep conventional monocular handling for low-confidence Eye0
            # data. GazerBase normally filters these before they reach us.
            yield [pupil_datum]
            return

        eye1 = self._latest_eye1
        if eye1 is not None and self._is_eye1_fresh_for(eye1, pupil_datum):
            # Do not consume Eye1: it is deliberately reused for all fresh
            # high-rate Eye0/DVS results until the next valid Eye1 snapshot.
            yield [pupil_datum, eye1]
        else:
            yield [pupil_datum]

    def _update_eye1_snapshot(self, pupil_datum) -> None:
        """Store the newest Eye1 datum, or invalidate a newer bad snapshot."""

        timestamp = pupil_datum["timestamp"]
        if (
            self._latest_eye1_observation_timestamp is not None
            and timestamp < self._latest_eye1_observation_timestamp
        ):
            # Batches are normally timestamp-sorted. If a delayed Eye1 packet
            # arrives nevertheless, never revive or regress state that a newer
            # valid *or invalid* Eye1 observation has already replaced.
            return

        self._latest_eye1_observation_timestamp = timestamp

        if self._is_valid(pupil_datum):
            self._latest_eye1 = pupil_datum
        else:
            # A current low-confidence Eye1 observation means binocular output
            # is unsafe; wait for its next valid NIR/Pye3D update.
            self._latest_eye1 = None

    def _is_eye1_fresh_for(self, eye1, eye0) -> bool:
        age_seconds = eye0["timestamp"] - eye1["timestamp"]
        return 0.0 <= age_seconds <= self.eye1_max_age_seconds

    def _is_valid(self, pupil_datum) -> bool:
        return pupil_datum["confidence"] >= self.min_pupil_confidence
