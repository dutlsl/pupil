"""Low-latency NIR Pye3D / DVS pupil fusion primitives.

This module deliberately has no Pupil UI, ZeroMQ, CUDA, or ``pye3d`` imports.
It is safe to import in the spawned TDTracker process.  The NIR eye process
publishes a small numeric snapshot of the most recent Pye3D eye model; the DVS
process reads that snapshot whenever a *new* TDTracker result is available and
creates a normal Pupil 3D pupil datum.

Coordinate convention
---------------------
``mapped_nir_norm_pos`` is already the result of the existing DVS
``scale/offset/flip`` mapping.  It is normalized in **NIR image coordinates**:
``(0, 0)`` is the top-left and ``(1, 1)`` is the bottom-right.  This module
does not apply scale, offset, flip, or a cross-camera calibration a second
time.  The emitted Pupil ``norm_pos`` flips Y as Pupil's public datum format
expects.

The geometric approximation follows Pye3D's single-camera model: unproject
the mapped NIR pixel into an eye-camera ray, intersect that ray with the
cached eyeball sphere, and use the sphere-surface direction as
``circle_3d.normal``.  The cached Pye3D pupil radius is retained.  It is an
integration bridge, not a replacement for a future proper eye0--event-camera
calibration.

The shared store is intentionally a fixed-size ``multiprocessing`` shared
array guarded by a process lock.  Unlike a Manager dict it performs no IPC
round trip for a read, and unlike a lock-free array it cannot return a torn
multi-field eye-model state.  Create it with the same ``spawn`` context that
will start the DVS process, then pass the store object directly as a process
argument.
"""

from __future__ import annotations

import ctypes
import math
import multiprocessing as mp
import time
from dataclasses import dataclass, replace
from typing import Any, Mapping, Optional, Sequence, Tuple


Vec2 = Tuple[float, float]
Vec3 = Tuple[float, float, float]


class SnapshotError(ValueError):
    """Raised when a snapshot cannot represent a usable numeric eye model."""


class FusionGeometryError(ValueError):
    """Raised when a DVS/NIR coordinate cannot be converted into a camera ray."""


@dataclass(frozen=True)
class DVSActivity:
    """The last successfully published fused DVS 3D event."""

    seq_id: int
    monotonic_ns: int

    def age_ms(self, now_monotonic_ns: Optional[int] = None) -> float:
        if now_monotonic_ns is None:
            now_monotonic_ns = time.monotonic_ns()
        return max(0, int(now_monotonic_ns) - self.monotonic_ns) / 1e6


def _finite_float(value: Any, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise SnapshotError(f"{field} must be numeric") from exc
    if not math.isfinite(result):
        raise SnapshotError(f"{field} must be finite")
    return result


def _vector(value: Any, length: int, field: str) -> Tuple[float, ...]:
    if isinstance(value, (str, bytes)):
        raise SnapshotError(f"{field} must be a {length}-vector")
    try:
        values = tuple(value)
    except TypeError as exc:
        raise SnapshotError(f"{field} must be a {length}-vector") from exc
    if len(values) != length:
        raise SnapshotError(f"{field} must have {length} values")
    return tuple(_finite_float(item, f"{field}[{index}]") for index, item in enumerate(values))


def _clamp_confidence(value: Any, field: str) -> float:
    return min(max(_finite_float(value, field), 0.0), 1.0)


def _default_projected_sphere(
    sphere_center: Vec3,
    sphere_radius: float,
    focal_length: float,
    frame_size: Tuple[int, int],
) -> Tuple[Vec2, Vec2, float]:
    """Project a sphere with Pye3D's pinhole convention.

    Pye3D itself normally supplies this field.  This fallback is kept here so
    the fusion datum stays renderer/exporter compatible if an older Pye3D
    result does not contain ``projected_sphere``.
    """

    width, height = frame_size
    z = sphere_center[2]
    if z <= 0.0:
        raise SnapshotError("sphere.center[2] must be positive")
    scale = focal_length / z
    center = (
        scale * sphere_center[0] + width / 2.0,
        scale * sphere_center[1] + height / 2.0,
    )
    diameter = 2.0 * scale * sphere_radius
    return center, (diameter, diameter), 0.0


@dataclass(frozen=True)
class Pye3DEyeModelSnapshot:
    """The numeric Pye3D state needed by a high-rate DVS fusion worker.

    ``sphere_center`` is Pye3D's corrected eye-camera-space center, so the
    resulting ``sphere``/``circle_3d`` fields have the same coordinate system
    expected by Pupil's existing 3D gaze mapper.
    """

    timestamp: float
    monotonic_ns: int
    pupil_confidence: float
    model_confidence: float
    sphere_center: Vec3
    sphere_radius: float
    pupil_radius: float
    focal_length: float
    frame_width: int
    frame_height: int
    ellipse_axes: Vec2
    ellipse_angle: float
    projected_sphere_center: Vec2
    projected_sphere_axes: Vec2
    projected_sphere_angle: float
    generation: int = 0

    @property
    def frame_size(self) -> Tuple[int, int]:
        return self.frame_width, self.frame_height

    @property
    def geometry_is_valid(self) -> bool:
        numeric_values = (
            self.timestamp,
            self.pupil_confidence,
            self.model_confidence,
            *self.sphere_center,
            self.sphere_radius,
            self.pupil_radius,
            self.focal_length,
            *self.ellipse_axes,
            self.ellipse_angle,
            *self.projected_sphere_center,
            *self.projected_sphere_axes,
            self.projected_sphere_angle,
        )
        return (
            all(math.isfinite(float(value)) for value in numeric_values)
            and self.monotonic_ns > 0
            and self.frame_width > 0
            and self.frame_height > 0
            and self.sphere_center[2] > 0.0
            and self.sphere_radius > 0.0
            and self.pupil_radius > 0.0
            and self.focal_length > 0.0
            and self.ellipse_axes[0] >= 0.0
            and self.ellipse_axes[1] >= 0.0
            and self.projected_sphere_axes[0] >= 0.0
            and self.projected_sphere_axes[1] >= 0.0
        )

    def is_usable(self, min_model_confidence: float = 0.6) -> bool:
        """Whether this snapshot may seed a fused event datum."""

        return self.geometry_is_valid and self.model_confidence >= float(
            min_model_confidence
        )

    def age_ms(self, now_monotonic_ns: Optional[int] = None) -> float:
        if now_monotonic_ns is None:
            now_monotonic_ns = time.monotonic_ns()
        return max(0, int(now_monotonic_ns) - self.monotonic_ns) / 1e6

    @classmethod
    def from_pye3d_datum(
        cls,
        datum: Mapping[str, Any],
        *,
        focal_length: float,
        frame_size: Sequence[int],
        monotonic_ns: Optional[int] = None,
    ) -> Optional["Pye3DEyeModelSnapshot"]:
        """Extract a numeric snapshot from a regular Pye3D result.

        ``None`` means that the datum contains no valid 3D eye model (for
        example Pye3D's zero-filled startup/default datum).  This makes it
        convenient for the NIR writer to invalidate the shared state instead
        of accidentally letting the DVS worker use an old model forever.
        """

        try:
            frame_width, frame_height = _vector(frame_size, 2, "frame_size")
            if not frame_width.is_integer() or not frame_height.is_integer():
                raise SnapshotError("frame_size must contain integer dimensions")
            width, height = int(frame_width), int(frame_height)
            if width <= 0 or height <= 0:
                raise SnapshotError("frame_size must be positive")

            focal = _finite_float(focal_length, "focal_length")
            sphere = datum["sphere"]
            circle = datum["circle_3d"]
            sphere_center = _vector(sphere["center"], 3, "sphere.center")
            sphere_radius = _finite_float(sphere["radius"], "sphere.radius")
            pupil_radius = _finite_float(circle["radius"], "circle_3d.radius")
            timestamp = _finite_float(datum["timestamp"], "timestamp")
            pupil_confidence = _clamp_confidence(
                datum.get("confidence", 0.0), "confidence"
            )
            model_confidence = _clamp_confidence(
                datum.get("model_confidence", 0.0), "model_confidence"
            )

            ellipse = datum.get("ellipse", {})
            diameter = _finite_float(datum.get("diameter", 0.0), "diameter")
            ellipse_axes = _vector(
                ellipse.get("axes", (diameter, diameter)), 2, "ellipse.axes"
            )
            ellipse_angle = _finite_float(ellipse.get("angle", 0.0), "ellipse.angle")

            projected = datum.get("projected_sphere")
            if projected is None:
                projected_center, projected_axes, projected_angle = (
                    _default_projected_sphere(
                        sphere_center,
                        sphere_radius,
                        focal,
                        (width, height),
                    )
                )
            else:
                projected_center = _vector(
                    projected["center"], 2, "projected_sphere.center"
                )
                projected_axes = _vector(
                    projected["axes"], 2, "projected_sphere.axes"
                )
                projected_angle = _finite_float(
                    projected.get("angle", 0.0), "projected_sphere.angle"
                )
            source_monotonic_ns = (
                time.monotonic_ns() if monotonic_ns is None else int(monotonic_ns)
            )
            if source_monotonic_ns <= 0:
                raise SnapshotError("monotonic_ns must be positive")

            snapshot = cls(
                timestamp=timestamp,
                monotonic_ns=source_monotonic_ns,
                pupil_confidence=pupil_confidence,
                model_confidence=model_confidence,
                sphere_center=sphere_center,  # type: ignore[arg-type]
                sphere_radius=sphere_radius,
                pupil_radius=pupil_radius,
                focal_length=focal,
                frame_width=width,
                frame_height=height,
                ellipse_axes=ellipse_axes,  # type: ignore[arg-type]
                ellipse_angle=ellipse_angle,
                projected_sphere_center=projected_center,  # type: ignore[arg-type]
                projected_sphere_axes=projected_axes,  # type: ignore[arg-type]
                projected_sphere_angle=projected_angle,
            )
        except (KeyError, SnapshotError, TypeError, ValueError):
            return None

        return snapshot if snapshot.geometry_is_valid else None


# ``EyeModelSnapshot`` is intentionally short for call sites in the eye and
# DVS processes.  Keep the explicit Pye3D name as the canonical public type.
EyeModelSnapshot = Pye3DEyeModelSnapshot


class Pye3DSnapshotStore:
    """Single-writer / multi-reader shared-memory store for eye-model snapshots.

    Use ``Pye3DSnapshotStore.create(mp.get_context("spawn"))`` in the eye
    process before it starts the DVS process.  The resulting instance is
    directly passable in ``Process(..., args=(..., snapshot_store, ...))``.
    Every read and write holds the same tiny process lock, so a reader sees an
    all-old or all-new model, never a mixture of both.
    """

    _TIMESTAMP = 0
    _PUPIL_CONFIDENCE = 1
    _MODEL_CONFIDENCE = 2
    _SPHERE_X = 3
    _SPHERE_Y = 4
    _SPHERE_Z = 5
    _SPHERE_RADIUS = 6
    _PUPIL_RADIUS = 7
    _FOCAL_LENGTH = 8
    _FRAME_WIDTH = 9
    _FRAME_HEIGHT = 10
    _ELLIPSE_AXIS_0 = 11
    _ELLIPSE_AXIS_1 = 12
    _ELLIPSE_ANGLE = 13
    _PROJECTED_CENTER_X = 14
    _PROJECTED_CENTER_Y = 15
    _PROJECTED_AXIS_0 = 16
    _PROJECTED_AXIS_1 = 17
    _PROJECTED_ANGLE = 18
    _FIELD_COUNT = 19

    def __init__(
        self,
        values: Any,
        generation: Any,
        valid: Any,
        source_ns: Any,
        dvs_seq_id: Any,
        dvs_result_ns: Any,
        lock: Any,
    ):
        self._values = values
        self._generation = generation
        self._valid = valid
        self._source_ns = source_ns
        self._dvs_seq_id = dvs_seq_id
        self._dvs_result_ns = dvs_result_ns
        self._lock = lock

    @classmethod
    def create(cls, context: Optional[Any] = None) -> "Pye3DSnapshotStore":
        """Allocate a store using a spawn-safe multiprocessing context.

        Supplying a context is useful when a caller already owns one.  The
        default is ``spawn`` rather than the platform default because the DVS
        worker is explicitly spawned and CUDA must not be forked.
        """

        context = context or mp.get_context("spawn")
        return cls(
            context.RawArray(ctypes.c_double, cls._FIELD_COUNT),
            context.RawValue(ctypes.c_ulonglong, 0),
            context.RawValue(ctypes.c_bool, False),
            context.RawValue(ctypes.c_ulonglong, 0),
            context.RawValue(ctypes.c_ulonglong, 0),
            context.RawValue(ctypes.c_ulonglong, 0),
            context.Lock(),
        )

    def write(self, snapshot: Pye3DEyeModelSnapshot) -> Pye3DEyeModelSnapshot:
        """Atomically replace the current model and return its new generation."""

        if not snapshot.geometry_is_valid:
            raise SnapshotError("cannot write an invalid Pye3D eye-model snapshot")
        if snapshot.monotonic_ns > (2**64 - 1):
            raise SnapshotError("monotonic_ns exceeds shared-memory storage")

        values = (
            snapshot.timestamp,
            snapshot.pupil_confidence,
            snapshot.model_confidence,
            *snapshot.sphere_center,
            snapshot.sphere_radius,
            snapshot.pupil_radius,
            snapshot.focal_length,
            float(snapshot.frame_width),
            float(snapshot.frame_height),
            *snapshot.ellipse_axes,
            snapshot.ellipse_angle,
            *snapshot.projected_sphere_center,
            *snapshot.projected_sphere_axes,
            snapshot.projected_sphere_angle,
        )
        assert len(values) == self._FIELD_COUNT
        with self._lock:
            for index, value in enumerate(values):
                self._values[index] = float(value)
            self._source_ns.value = int(snapshot.monotonic_ns)
            self._generation.value += 1
            self._valid.value = True
            generation = int(self._generation.value)
        return replace(snapshot, generation=generation)

    def update_from_pye3d_datum(
        self,
        datum: Mapping[str, Any],
        *,
        focal_length: float,
        frame_size: Sequence[int],
        monotonic_ns: Optional[int] = None,
    ) -> Optional[Pye3DEyeModelSnapshot]:
        """Parse and publish a Pye3D datum, invalidating stale state on failure."""

        snapshot = Pye3DEyeModelSnapshot.from_pye3d_datum(
            datum,
            focal_length=focal_length,
            frame_size=frame_size,
            monotonic_ns=monotonic_ns,
        )
        if snapshot is None:
            self.invalidate()
            return None
        return self.write(snapshot)

    def invalidate(self) -> int:
        """Atomically mark the store unavailable and return its generation."""

        with self._lock:
            self._valid.value = False
            self._generation.value += 1
            return int(self._generation.value)

    def mark_dvs_fused_result(
        self, seq_id: int, monotonic_ns: Optional[int] = None
    ) -> bool:
        """Record one successfully generated fused DVS datum.

        This is both an atomic fresh-only gate and an activity heartbeat for
        the NIR Pye3D plugin.  Call it *after* ``fuse_dvs_pupil`` returned a
        datum (and immediately before publishing that datum).  Raw TDTracker
        outputs that could not be fused must not suppress NIR fallback output.

        Returns ``True`` only when ``seq_id`` is newer than the last accepted
        fused event.  A stale/replayed completion therefore cannot be
        published as a fresh 3D event.
        """

        try:
            sequence = int(seq_id)
        except (TypeError, ValueError) as exc:
            raise ValueError("seq_id must be an integer") from exc
        if sequence <= 0 or sequence > (2**64 - 1):
            raise ValueError("seq_id is outside shared-memory storage")
        result_ns = time.monotonic_ns() if monotonic_ns is None else int(monotonic_ns)
        if result_ns <= 0 or result_ns > (2**64 - 1):
            raise ValueError("monotonic_ns is outside shared-memory storage")

        with self._lock:
            if sequence <= int(self._dvs_seq_id.value):
                return False
            self._dvs_seq_id.value = sequence
            self._dvs_result_ns.value = result_ns
            return True

    # Short alias for the integration path.  It has the same "successfully
    # fused, not merely inferred" semantics as the explicit method above.
    mark_dvs_result = mark_dvs_fused_result

    def latest_dvs_activity(self) -> Optional[DVSActivity]:
        """Return the latest accepted fused DVS event, if any."""

        with self._lock:
            sequence = int(self._dvs_seq_id.value)
            result_ns = int(self._dvs_result_ns.value)
        if sequence <= 0 or result_ns <= 0:
            return None
        return DVSActivity(seq_id=sequence, monotonic_ns=result_ns)

    def dvs_is_fresh(
        self,
        max_age_ms: float,
        *,
        now_monotonic_ns: Optional[int] = None,
    ) -> bool:
        """Whether a recently fused DVS 3D stream should suppress NIR output."""

        if float(max_age_ms) < 0.0:
            raise ValueError("max_age_ms must be non-negative")
        activity = self.latest_dvs_activity()
        if activity is None:
            return False
        return activity.age_ms(now_monotonic_ns) <= float(max_age_ms)

    def read(
        self,
        *,
        now_monotonic_ns: Optional[int] = None,
        max_age_ms: Optional[float] = None,
        min_model_confidence: Optional[float] = None,
    ) -> Optional[Pye3DEyeModelSnapshot]:
        """Return an atomic snapshot, or ``None`` when invalid/stale/unusable.

        The DVS worker should call this immediately after receiving a fresh
        TDTracker output.  It never waits for a new NIR frame.
        """

        if max_age_ms is not None and float(max_age_ms) < 0.0:
            raise ValueError("max_age_ms must be non-negative")
        with self._lock:
            if not self._valid.value:
                return None
            values = tuple(float(value) for value in self._values)
            generation = int(self._generation.value)
            source_ns = int(self._source_ns.value)

        try:
            snapshot = Pye3DEyeModelSnapshot(
                timestamp=values[self._TIMESTAMP],
                monotonic_ns=source_ns,
                pupil_confidence=values[self._PUPIL_CONFIDENCE],
                model_confidence=values[self._MODEL_CONFIDENCE],
                sphere_center=(
                    values[self._SPHERE_X],
                    values[self._SPHERE_Y],
                    values[self._SPHERE_Z],
                ),
                sphere_radius=values[self._SPHERE_RADIUS],
                pupil_radius=values[self._PUPIL_RADIUS],
                focal_length=values[self._FOCAL_LENGTH],
                frame_width=int(round(values[self._FRAME_WIDTH])),
                frame_height=int(round(values[self._FRAME_HEIGHT])),
                ellipse_axes=(
                    values[self._ELLIPSE_AXIS_0],
                    values[self._ELLIPSE_AXIS_1],
                ),
                ellipse_angle=values[self._ELLIPSE_ANGLE],
                projected_sphere_center=(
                    values[self._PROJECTED_CENTER_X],
                    values[self._PROJECTED_CENTER_Y],
                ),
                projected_sphere_axes=(
                    values[self._PROJECTED_AXIS_0],
                    values[self._PROJECTED_AXIS_1],
                ),
                projected_sphere_angle=values[self._PROJECTED_ANGLE],
                generation=generation,
            )
        except (IndexError, OverflowError, ValueError):
            return None

        if not snapshot.geometry_is_valid:
            return None
        now_ns = time.monotonic_ns() if now_monotonic_ns is None else int(now_monotonic_ns)
        if max_age_ms is not None and snapshot.age_ms(now_ns) > float(max_age_ms):
            return None
        if (
            min_model_confidence is not None
            and not snapshot.is_usable(float(min_model_confidence))
        ):
            return None
        return snapshot


def normalized_nir_position_to_ray(
    mapped_nir_norm_pos: Sequence[float],
    *,
    frame_size: Sequence[int],
    focal_length: float,
) -> Vec3:
    """Unproject a top-left-origin NIR normalized point into a unit ray.

    This intentionally mirrors Pye3D's pinhole convention
    ``(pixel_x - width/2, pixel_y - height/2, focal_length)``.  The caller has
    already applied the DVS scale/offset/flip transform.
    """

    try:
        x, y = _vector(mapped_nir_norm_pos, 2, "mapped_nir_norm_pos")
        width_raw, height_raw = _vector(frame_size, 2, "frame_size")
        focal = _finite_float(focal_length, "focal_length")
    except SnapshotError as exc:
        raise FusionGeometryError(str(exc)) from exc
    width, height = int(width_raw), int(height_raw)
    if width <= 0 or height <= 0 or focal <= 0.0:
        raise FusionGeometryError("frame_size and focal_length must be positive")

    ray = (x * width - width / 2.0, y * height - height / 2.0, focal)
    norm = math.sqrt(sum(component * component for component in ray))
    if norm <= 0.0 or not math.isfinite(norm):
        raise FusionGeometryError("cannot normalize NIR camera ray")
    return tuple(component / norm for component in ray)  # type: ignore[return-value]


def _point_on_sphere_from_ray(
    sphere_center: Vec3,
    sphere_radius: float,
    ray: Vec3,
    *,
    allow_nearest: bool,
) -> Optional[Tuple[Vec3, str]]:
    """Find the visible sphere point for a camera-origin ray.

    Pye3D uses the near ray/sphere intersection.  For a miss, its historical
    model code falls back to a nearest point on the sphere; retaining that
    behavior here helps tolerate the existing approximate DVS→NIR mapping.
    The caller can disable it while validating a stricter calibration.
    """

    projection = sum(component * center for component, center in zip(ray, sphere_center))
    center_norm_sq = sum(component * component for component in sphere_center)
    perpendicular_sq = max(center_norm_sq - projection * projection, 0.0)
    discriminant = sphere_radius * sphere_radius - perpendicular_sq

    if discriminant >= 0.0:
        root = math.sqrt(discriminant)
        near = projection - root
        far = projection + root
        positive_intersections = [distance for distance in (near, far) if distance > 0.0]
        if positive_intersections:
            distance = min(positive_intersections)
            return (
                tuple(distance * component for component in ray),  # type: ignore[return-value]
                "direct",
            )

    if not allow_nearest or projection <= 0.0:
        return None

    closest_on_ray = tuple(projection * component for component in ray)
    outward = tuple(
        coordinate - center
        for coordinate, center in zip(closest_on_ray, sphere_center)
    )
    outward_norm = math.sqrt(sum(component * component for component in outward))
    if outward_norm <= 0.0 or not math.isfinite(outward_norm):
        return None
    return (
        tuple(
            center + sphere_radius * component / outward_norm
            for center, component in zip(sphere_center, outward)
        ),  # type: ignore[return-value]
        "nearest",
    )


def fuse_dvs_pupil(
    snapshot: Optional[Pye3DEyeModelSnapshot],
    *,
    mapped_nir_norm_pos: Sequence[float],
    dvs_confidence: float,
    timestamp: float,
    eye_id: int = 0,
    seq_id: Optional[int] = None,
    now_monotonic_ns: Optional[int] = None,
    max_snapshot_age_ms: Optional[float] = None,
    min_model_confidence: float = 0.6,
    allow_nearest_sphere_point: bool = True,
) -> Optional[dict]:
    """Build a Pye3D-compatible 3D pupil datum from a fresh DVS center.

    Return ``None`` rather than publishing fabricated 3D data when the NIR
    model is absent, stale, low-confidence, or cannot be connected to the
    DVS-derived ray.  The caller is responsible for the *fresh-only* rule:
    call this only when TDTracker's ``seq_id`` has advanced.
    """

    if snapshot is None:
        return None
    if max_snapshot_age_ms is not None and float(max_snapshot_age_ms) < 0.0:
        raise ValueError("max_snapshot_age_ms must be non-negative")
    if not snapshot.is_usable(min_model_confidence):
        return None
    now_ns = time.monotonic_ns() if now_monotonic_ns is None else int(now_monotonic_ns)
    snapshot_age_ms = snapshot.age_ms(now_ns)
    if (
        max_snapshot_age_ms is not None
        and snapshot_age_ms > float(max_snapshot_age_ms)
    ):
        return None

    try:
        x, y = _vector(mapped_nir_norm_pos, 2, "mapped_nir_norm_pos")
        event_confidence = _clamp_confidence(dvs_confidence, "dvs_confidence")
        event_timestamp = _finite_float(timestamp, "timestamp")
        ray = normalized_nir_position_to_ray(
            (x, y),
            frame_size=snapshot.frame_size,
            focal_length=snapshot.focal_length,
        )
    except SnapshotError as exc:
        raise FusionGeometryError(str(exc)) from exc

    intersection = _point_on_sphere_from_ray(
        snapshot.sphere_center,
        snapshot.sphere_radius,
        ray,
        allow_nearest=allow_nearest_sphere_point,
    )
    if intersection is None:
        return None
    pupil_center, intersection_kind = intersection
    normal = tuple(
        (coordinate - center) / snapshot.sphere_radius
        for coordinate, center in zip(pupil_center, snapshot.sphere_center)
    )
    normal_norm = math.sqrt(sum(component * component for component in normal))
    if normal_norm <= 0.0 or not math.isfinite(normal_norm):
        return None
    normal = tuple(component / normal_norm for component in normal)
    phi = math.atan2(normal[2], normal[0])
    theta = math.acos(min(max(normal[1], -1.0), 1.0))

    pixel_center = (x * snapshot.frame_width, y * snapshot.frame_height)
    datum = {
        "id": int(eye_id),
        "topic": f"pupil.{int(eye_id)}.3d",
        # Gazer3D filters on the literal substring ``3d``.
        "method": "hybrid pye3d+dvs 3d",
        "timestamp": event_timestamp,
        # Pupil's public normalized datum coordinates have a bottom-left origin.
        "norm_pos": (x, 1.0 - y),
        "confidence": event_confidence,
        "model_confidence": float(snapshot.model_confidence),
        "diameter": float(max(snapshot.ellipse_axes)),
        "diameter_3d": float(2.0 * snapshot.pupil_radius),
        "location": pixel_center,
        "ellipse": {
            "center": pixel_center,
            "axes": tuple(float(value) for value in snapshot.ellipse_axes),
            "angle": float(snapshot.ellipse_angle),
        },
        "projected_sphere": {
            "center": tuple(
                float(value) for value in snapshot.projected_sphere_center
            ),
            "axes": tuple(float(value) for value in snapshot.projected_sphere_axes),
            "angle": float(snapshot.projected_sphere_angle),
        },
        "sphere": {
            "center": tuple(float(value) for value in snapshot.sphere_center),
            "radius": float(snapshot.sphere_radius),
        },
        "circle_3d": {
            "center": tuple(float(value) for value in pupil_center),
            "normal": tuple(float(value) for value in normal),
            "radius": float(snapshot.pupil_radius),
        },
        "phi": float(phi),
        "theta": float(theta),
        # Explicit tags make it possible for the relay to suppress only the
        # duplicate NIR Pye3D stream while retaining this fused DVS stream.
        "source": "dvs_pye3d_fused",
        "source_2d": "dvs",
        "source_3d_model": "pye3d",
        "tdtracker_confidence": event_confidence,
        "pye3d_snapshot_confidence": float(snapshot.pupil_confidence),
        "pye3d_snapshot_generation": int(snapshot.generation),
        "pye3d_snapshot_timestamp": float(snapshot.timestamp),
        "pye3d_snapshot_age_ms": float(snapshot_age_ms),
        "ray_sphere_intersection": intersection_kind,
        "fresh": True,
    }
    if seq_id is not None:
        datum["seq_id"] = int(seq_id)
    return datum


__all__ = [
    "DVSActivity",
    "EyeModelSnapshot",
    "FusionGeometryError",
    "Pye3DEyeModelSnapshot",
    "Pye3DSnapshotStore",
    "SnapshotError",
    "fuse_dvs_pupil",
    "normalized_nir_position_to_ray",
]
