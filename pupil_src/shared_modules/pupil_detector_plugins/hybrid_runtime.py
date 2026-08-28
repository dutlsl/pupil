"""High-rate DVS runtime used by the Hybrid pupil detector.

The hot path deliberately has no Pupil UI dependencies.  It can therefore run
in a spawned process and keep DAVIS reads, 1 ms slicing, BinaRep construction,
TDTracker inference, and CUDA result readback in one scheduling domain.
"""

from __future__ import annotations

import ctypes
import logging
import os
import queue
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)

BINAREP_CHANNELS = 2
BINAREP_HEIGHT = 60
BINAREP_WIDTH = 80
BINAREP_TIME_BINS = 4
SEQUENCE_LENGTH = 8


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"", "0", "false", "no", "off"}


def _fill_binarep_numpy_uint8(
    xs: np.ndarray,
    ys: np.ndarray,
    timestamps_us: np.ndarray,
    polarities: np.ndarray,
    *,
    slice_start_us: int,
    time_window_us: int,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Fill an OFF/ON BinaRep using one bit per temporal bin."""
    if out is None:
        out = np.zeros(
            (BINAREP_CHANNELS, BINAREP_HEIGHT, BINAREP_WIDTH), dtype=np.uint8
        )
    else:
        if out.shape != (BINAREP_CHANNELS, BINAREP_HEIGHT, BINAREP_WIDTH):
            raise ValueError(f"Unexpected BinaRep output shape: {out.shape}")
        if out.dtype != np.uint8:
            raise TypeError("Internal BinaRep output must use uint8")
        out.fill(0)

    xs = np.asarray(xs)
    ys = np.asarray(ys)
    timestamps_us = np.asarray(timestamps_us)
    polarities = np.asarray(polarities)
    event_count = len(xs)
    if not (len(ys) == len(timestamps_us) == len(polarities) == event_count):
        raise ValueError("DVS event arrays must have equal lengths")
    if event_count == 0:
        return out
    if time_window_us <= 0:
        raise ValueError("time_window_us must be positive")

    valid = (
        (xs >= 0)
        & (xs < BINAREP_WIDTH)
        & (ys >= 0)
        & (ys < BINAREP_HEIGHT)
    )
    if not np.any(valid):
        return out

    x_valid = xs[valid].astype(np.intp, copy=False)
    y_valid = ys[valid].astype(np.intp, copy=False)
    p_valid = (polarities[valid] != 0).astype(np.intp, copy=False)
    relative_t = timestamps_us[valid].astype(np.int64, copy=False) - int(
        slice_start_us
    )
    bins = np.floor_divide(
        np.clip(relative_t, 0, time_window_us - 1) * BINAREP_TIME_BINS,
        time_window_us,
    ).astype(np.uint8, copy=False)
    values = np.left_shift(np.uint8(1), bins)

    # Repeated advanced-index assignments do not accumulate. np.bitwise_or.at is
    # required for event-by-event equivalence when many events hit one pixel.
    np.bitwise_or.at(out, (p_valid, y_valid, x_valid), values)
    return out


def fill_binarep_numpy(
    xs: np.ndarray,
    ys: np.ndarray,
    timestamps_us: np.ndarray,
    polarities: np.ndarray,
    *,
    slice_start_us: int,
    time_window_us: int,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Return float32 BinaRep values in the unnormalised range 0..15."""
    uint8_out = _fill_binarep_numpy_uint8(
        xs,
        ys,
        timestamps_us,
        polarities,
        slice_start_us=slice_start_us,
        time_window_us=time_window_us,
    )
    if out is None:
        return uint8_out.astype(np.float32)
    if out.shape != uint8_out.shape or out.dtype != np.float32:
        raise TypeError("BinaRep output must be float32 with shape (2, 60, 80)")
    np.copyto(out, uint8_out, casting="unsafe")
    return out


def _fill_legacy_binarep_numpy_uint8(
    xs: np.ndarray,
    ys: np.ndarray,
    timestamps_us: np.ndarray,
    polarities: np.ndarray,
    *,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Build the exact uint8 form used by the original tonic pipeline.

    The original ``DVSDetectorPlugin`` used ``ToFrame(n_time_bins=4)`` followed
    by ``ToBinaRep(n_frames=1, n_bits=4)``.  That representation has three
    material properties that differ from the older fast helper above:

    * bins span the *actual* first-to-last event duration,
    * the earliest bin is the most-significant bit, and
    * values are later normalised to ``0..1`` by division by 15.

    Keeping this implementation here avoids pulling tonic into the hot worker
    while preserving its output exactly for sorted event packets.
    """
    if out is None:
        out = np.zeros(
            (BINAREP_CHANNELS, BINAREP_HEIGHT, BINAREP_WIDTH), dtype=np.uint8
        )
    else:
        if out.shape != (BINAREP_CHANNELS, BINAREP_HEIGHT, BINAREP_WIDTH):
            raise ValueError(f"Unexpected BinaRep output shape: {out.shape}")
        if out.dtype != np.uint8:
            raise TypeError("Internal BinaRep output must use uint8")
        out.fill(0)

    xs = np.asarray(xs)
    ys = np.asarray(ys)
    timestamps_us = np.asarray(timestamps_us)
    polarities = np.asarray(polarities)
    event_count = len(xs)
    if not (len(ys) == len(timestamps_us) == len(polarities) == event_count):
        raise ValueError("DVS event arrays must have equal lengths")
    if event_count == 0:
        return out

    first_timestamp = int(timestamps_us[0])
    duration_us = int(timestamps_us[-1]) - first_timestamp
    bin_width_us = duration_us // BINAREP_TIME_BINS
    # tonic's SliceByTimeBins creates empty slices when all timestamps are equal.
    if bin_width_us <= 0:
        return out

    valid = (
        (xs >= 0)
        & (xs < BINAREP_WIDTH)
        & (ys >= 0)
        & (ys < BINAREP_HEIGHT)
    )
    if not np.any(valid):
        return out

    x_valid = xs[valid].astype(np.intp, copy=False)
    y_valid = ys[valid].astype(np.intp, copy=False)
    p_valid = (polarities[valid] != 0).astype(np.intp, copy=False)
    relative_t = timestamps_us[valid].astype(np.int64, copy=False) - first_timestamp
    bins = np.floor_divide(relative_t, bin_width_us)
    in_time_range = (bins >= 0) & (bins < BINAREP_TIME_BINS)
    if not np.any(in_time_range):
        return out
    # tonic assigns weights [8, 4, 2, 1] to bins [0, 1, 2, 3].
    values = np.left_shift(
        np.uint8(1), (BINAREP_TIME_BINS - 1 - bins[in_time_range]).astype(np.uint8)
    )
    np.bitwise_or.at(
        out,
        (
            p_valid[in_time_range],
            y_valid[in_time_range],
            x_valid[in_time_range],
        ),
        values,
    )
    return out


def fill_legacy_binarep_numpy(
    xs: np.ndarray,
    ys: np.ndarray,
    timestamps_us: np.ndarray,
    polarities: np.ndarray,
    *,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Return the original tonic-compatible float32 BinaRep in ``0..1``."""
    uint8_out = _fill_legacy_binarep_numpy_uint8(
        xs, ys, timestamps_us, polarities
    )
    if out is None:
        return uint8_out.astype(np.float32) / np.float32(15.0)
    if out.shape != uint8_out.shape or out.dtype != np.float32:
        raise TypeError("BinaRep output must be float32 with shape (2, 60, 80)")
    np.multiply(uint8_out, np.float32(1.0 / 15.0), out=out, casting="unsafe")
    return out


def _normalise_binarep_mode(mode: str) -> str:
    value = str(mode).strip().lower()
    aliases = {
        "legacy": "legacy",
        "original": "legacy",
        "tonic": "legacy",
        "fast": "fast",
        "optimized": "fast",
    }
    try:
        return aliases[value]
    except KeyError as error:
        raise ValueError(
            "Unsupported PUPIL_HYBRID_BINAREP_MODE: "
            f"{mode!r}; use 'legacy' or 'fast'"
        ) from error


def _resize_event_coordinates(
    coordinates: np.ndarray,
    *,
    original_width: int,
    original_height: int,
    binarep_mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Match either legacy tonic coordinate conversion or the fast integer path."""
    coordinates = np.asarray(coordinates)
    if coordinates.ndim != 2 or coordinates.shape[1] < 2:
        raise ValueError("DVS coordinates must have shape (N, 2)")
    mode = _normalise_binarep_mode(binarep_mode)
    if mode == "legacy":
        # This deliberately mirrors the existing plugin's float32 normalise then
        # cast sequence instead of algebraically simplifying it.
        resized_x = (
            coordinates[:, 0].astype(np.float32)
            / float(original_width)
            * float(BINAREP_WIDTH)
        ).astype(np.int64)
        resized_y = (
            coordinates[:, 1].astype(np.float32)
            / float(original_height)
            * float(BINAREP_HEIGHT)
        ).astype(np.int64)
        return resized_x, resized_y
    return (
        np.floor_divide(coordinates[:, 0] * BINAREP_WIDTH, original_width),
        np.floor_divide(coordinates[:, 1] * BINAREP_HEIGHT, original_height),
    )


class NativeBinaRep:
    """ctypes wrapper for the optional native BinaRep implementation."""

    def __init__(self, library_path: Optional[str] = None):
        default_path = Path(__file__).with_name("_native_binarep.so")
        path = Path(
            library_path
            or os.getenv("PUPIL_HYBRID_CPP_BINAREP_PATH", str(default_path))
        )
        self.path = path
        self._library = ctypes.CDLL(str(path))
        self._fill = self._library.pupil_fill_binarep
        pointer = ctypes.c_void_p
        self._fill.argtypes = [
            pointer,
            pointer,
            pointer,
            pointer,
            ctypes.c_size_t,
            ctypes.c_int64,
            ctypes.c_int64,
            pointer,
        ]
        self._fill.restype = ctypes.c_int
        try:
            self._fill_legacy = self._library.pupil_fill_binarep_legacy
        except AttributeError:
            # An older locally built .so can still serve the fast path.  The
            # worker falls back to the NumPy-compatible legacy encoder until
            # the library is rebuilt.
            self._fill_legacy = None
        else:
            self._fill_legacy.argtypes = [
                pointer,
                pointer,
                pointer,
                pointer,
                ctypes.c_size_t,
                pointer,
            ]
            self._fill_legacy.restype = ctypes.c_int

    @property
    def supports_legacy(self) -> bool:
        return self._fill_legacy is not None

    def fill_uint8(
        self,
        xs: np.ndarray,
        ys: np.ndarray,
        timestamps_us: np.ndarray,
        polarities: np.ndarray,
        *,
        slice_start_us: int,
        time_window_us: int,
        out: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        xs = np.ascontiguousarray(xs, dtype=np.int64)
        ys = np.ascontiguousarray(ys, dtype=np.int64)
        timestamps_us = np.ascontiguousarray(timestamps_us, dtype=np.int64)
        polarities = np.ascontiguousarray(polarities, dtype=np.uint8)
        if not (len(xs) == len(ys) == len(timestamps_us) == len(polarities)):
            raise ValueError("DVS event arrays must have equal lengths")

        if out is None:
            out = np.empty(
                (BINAREP_CHANNELS, BINAREP_HEIGHT, BINAREP_WIDTH), dtype=np.uint8
            )
        elif out.shape != (BINAREP_CHANNELS, BINAREP_HEIGHT, BINAREP_WIDTH):
            raise ValueError(f"Unexpected BinaRep output shape: {out.shape}")
        elif out.dtype != np.uint8 or not out.flags.c_contiguous:
            raise TypeError("Native BinaRep output must be contiguous uint8")
        result = self._fill(
            xs.ctypes.data,
            ys.ctypes.data,
            timestamps_us.ctypes.data,
            polarities.ctypes.data,
            len(xs),
            int(slice_start_us),
            int(time_window_us),
            out.ctypes.data,
        )
        if result != 0:
            raise RuntimeError(f"Native BinaRep failed with status {result}")
        return out

    def fill_legacy_uint8(
        self,
        xs: np.ndarray,
        ys: np.ndarray,
        timestamps_us: np.ndarray,
        polarities: np.ndarray,
        *,
        out: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Run the tonic-compatible encoder in the optional native library."""
        if self._fill_legacy is None:
            raise RuntimeError(
                "Native BinaRep library does not provide the legacy encoder; rebuild it"
            )
        xs = np.ascontiguousarray(xs, dtype=np.int64)
        ys = np.ascontiguousarray(ys, dtype=np.int64)
        timestamps_us = np.ascontiguousarray(timestamps_us, dtype=np.int64)
        polarities = np.ascontiguousarray(polarities, dtype=np.uint8)
        if not (len(xs) == len(ys) == len(timestamps_us) == len(polarities)):
            raise ValueError("DVS event arrays must have equal lengths")
        if out is None:
            out = np.empty(
                (BINAREP_CHANNELS, BINAREP_HEIGHT, BINAREP_WIDTH), dtype=np.uint8
            )
        elif out.shape != (BINAREP_CHANNELS, BINAREP_HEIGHT, BINAREP_WIDTH):
            raise ValueError(f"Unexpected BinaRep output shape: {out.shape}")
        elif out.dtype != np.uint8 or not out.flags.c_contiguous:
            raise TypeError("Native BinaRep output must be contiguous uint8")
        result = self._fill_legacy(
            xs.ctypes.data,
            ys.ctypes.data,
            timestamps_us.ctypes.data,
            polarities.ctypes.data,
            len(xs),
            out.ctypes.data,
        )
        if result != 0:
            raise RuntimeError(f"Native legacy BinaRep failed with status {result}")
        return out

    def fill(
        self,
        xs: np.ndarray,
        ys: np.ndarray,
        timestamps_us: np.ndarray,
        polarities: np.ndarray,
        *,
        slice_start_us: int,
        time_window_us: int,
        out: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        native_out = self.fill_uint8(
            xs,
            ys,
            timestamps_us,
            polarities,
            slice_start_us=slice_start_us,
            time_window_us=time_window_us,
        )
        if out is None:
            return native_out.astype(np.float32)
        if out.shape != native_out.shape or out.dtype != np.float32:
            raise TypeError("BinaRep output must be float32 with shape (2, 60, 80)")
        np.copyto(out, native_out, casting="unsafe")
        return out


def encode_tdtracker_binarep(
    coordinates: np.ndarray,
    timestamps_us: np.ndarray,
    polarities: np.ndarray,
    *,
    original_width: int,
    original_height: int,
    time_window_us: int,
    binarep_mode: str,
    native: Optional[NativeBinaRep] = None,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Encode one TDTracker frame using the selected compatible representation.

    ``legacy`` is the default and is bit-for-bit compatible with the original
    tonic preprocessing used by :class:`DVSDetectorPlugin`.  ``fast`` retains
    the previous fixed-window representation for explicit experiments only.
    """
    mode = _normalise_binarep_mode(binarep_mode)
    timestamps_us = np.asarray(timestamps_us)
    if timestamps_us.size == 0:
        if out is None:
            return np.zeros(
                (BINAREP_CHANNELS, BINAREP_HEIGHT, BINAREP_WIDTH), dtype=np.uint8
            )
        out.fill(0)
        return out
    resized_x, resized_y = _resize_event_coordinates(
        coordinates,
        original_width=original_width,
        original_height=original_height,
        binarep_mode=mode,
    )
    if mode == "legacy":
        if native is not None and native.supports_legacy:
            return native.fill_legacy_uint8(
                resized_x, resized_y, timestamps_us, polarities, out=out
            )
        return _fill_legacy_binarep_numpy_uint8(
            resized_x, resized_y, timestamps_us, polarities, out=out
        )

    slice_start = int(timestamps_us[-1]) - int(time_window_us) + 1
    if native is not None:
        return native.fill_uint8(
            resized_x,
            resized_y,
            timestamps_us,
            polarities,
            slice_start_us=slice_start,
            time_window_us=time_window_us,
            out=out,
        )
    return _fill_binarep_numpy_uint8(
        resized_x,
        resized_y,
        timestamps_us,
        polarities,
        slice_start_us=slice_start,
        time_window_us=time_window_us,
        out=out,
    )


def copy_ordered_tdtracker_stack(
    frame_ring: np.ndarray,
    frame_count: int,
    *,
    binarep_mode: str,
    out: np.ndarray,
) -> None:
    """Copy the chronological ring into model input scale for TDTracker."""
    if frame_count < SEQUENCE_LENGTH:
        raise ValueError("Not enough BinaRep frames for a TDTracker sequence")
    if frame_ring.shape != (
        SEQUENCE_LENGTH,
        BINAREP_CHANNELS,
        BINAREP_HEIGHT,
        BINAREP_WIDTH,
    ):
        raise ValueError(f"Unexpected TDTracker ring shape: {frame_ring.shape}")
    if out.shape != frame_ring.shape or out.dtype != np.float32:
        raise TypeError("TDTracker stack output must be float32 with ring shape")
    mode = _normalise_binarep_mode(binarep_mode)
    oldest = frame_count % SEQUENCE_LENGTH
    first_count = SEQUENCE_LENGTH - oldest
    source_parts = (
        (frame_ring[oldest:], out[:first_count]),
        (frame_ring[:oldest], out[first_count:]),
    )
    for source, target in source_parts:
        if not len(source):
            continue
        if mode == "legacy":
            np.multiply(
                source,
                np.float32(1.0 / 15.0),
                out=target,
                casting="unsafe",
            )
        else:
            np.copyto(target, source, casting="unsafe")


def _decode_last_on_device(torch: Any, output_x: Any, output_y: Any) -> Any:
    """Decode the last TDTracker timestep without allocating a CPU tensor."""
    functional = torch.nn.functional
    x_index = output_x.argmax(dim=2)
    y_index = output_y.argmax(dim=2)
    pooled_x = functional.avg_pool1d(output_x, kernel_size=10, stride=1, padding=1)
    pooled_y = functional.avg_pool1d(output_y, kernel_size=10, stride=1, padding=1)
    x_conf = torch.softmax(pooled_x, dim=2).amax(dim=2)
    y_conf = torch.softmax(pooled_y, dim=2).amax(dim=2)
    return torch.stack(
        (
            x_index[:, -1].float() / float(BINAREP_WIDTH),
            y_index[:, -1].float() / float(BINAREP_HEIGHT),
            (x_conf[:, -1] + y_conf[:, -1]).float(),
        ),
        dim=1,
    )[0]


def _load_tdtracker(torch: Any, checkpoint_path: str, device: Any) -> Any:
    from pupil_detector_plugins.dvs_models.TDTracker import Model

    args = type("TDTrackerArgs", (), {})()
    args.sensor_width = BINAREP_WIDTH
    args.sensor_height = BINAREP_HEIGHT
    args.spatial_factor = 1
    args.pixel_tolerances = [1, 3, 5, 10]
    model = Model(args).to(device).eval()
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(state, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            if key in state and isinstance(state[key], dict):
                state = state[key]
                break
    if isinstance(state, dict) and any(key.startswith("module.") for key in state):
        state = {key.removeprefix("module."): value for key, value in state.items()}
    model.load_state_dict(state, strict=False)
    return model


class _TDTrackerEagerRunner:
    def __init__(self, checkpoint_path: str, device_index: int = 0):
        import torch

        self.torch = torch
        self.device = torch.device(f"cuda:{device_index}")
        self.model = _load_tdtracker(torch, checkpoint_path, self.device)

    def submit(self, stack: np.ndarray, seq_id: int, metadata: Dict[str, Any]):
        torch = self.torch
        started_ns = time.monotonic_ns()
        with torch.inference_mode():
            input_tensor = torch.as_tensor(
                stack, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            result = _decode_last_on_device(torch, *self.model(input_tensor))
            values = result.cpu().tolist()
        ready_ns = time.monotonic_ns()
        metadata = dict(metadata)
        metadata["infer_ms"] = (ready_ns - started_ns) / 1e6
        return [_result_record(values, seq_id, metadata, ready_ns)]

    def close(self):
        return None


class _TDTrackerCompileRunner(_TDTrackerEagerRunner):
    def __init__(self, checkpoint_path: str, device_index: int = 0):
        super().__init__(checkpoint_path, device_index)
        self.model = self.torch.compile(self.model, mode="reduce-overhead")
        warmup = self.torch.zeros(
            (1, SEQUENCE_LENGTH, BINAREP_CHANNELS, BINAREP_HEIGHT, BINAREP_WIDTH),
            dtype=self.torch.float32,
            device=self.device,
        )
        with self.torch.inference_mode():
            for _ in range(3):
                self.model(warmup)
        self.torch.cuda.synchronize(self.device)


class _TDTrackerGraphRunner:
    """Fixed-address CUDA Graph runner with non-blocking result readback."""

    def __init__(
        self,
        checkpoint_path: str,
        device_index: int = 0,
        *,
        async_result: bool = True,
        result_slots: int = 8,
        profile_sync: bool = False,
    ):
        import torch

        if result_slots < 2 and async_result:
            raise ValueError("Async CUDA results require at least two slots")
        self.torch = torch
        self.device = torch.device(f"cuda:{device_index}")
        torch.cuda.set_device(self.device)
        self.model = _load_tdtracker(torch, checkpoint_path, self.device)
        self.async_result = async_result
        self.profile_sync = profile_sync

        input_shape = (
            1,
            SEQUENCE_LENGTH,
            BINAREP_CHANNELS,
            BINAREP_HEIGHT,
            BINAREP_WIDTH,
        )
        self.static_input = torch.zeros(
            input_shape, dtype=torch.float32, device=self.device
        )
        self.host_input = torch.empty(
            input_shape, dtype=torch.float32, pin_memory=True
        )

        warmup_stream = torch.cuda.Stream(device=self.device)
        warmup_stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(warmup_stream), torch.inference_mode():
            for _ in range(3):
                self.static_result = _decode_last_on_device(
                    torch, *self.model(self.static_input)
                )
        torch.cuda.current_stream(self.device).wait_stream(warmup_stream)
        torch.cuda.synchronize(self.device)

        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph), torch.inference_mode():
            self.static_result = _decode_last_on_device(
                torch, *self.model(self.static_input)
            )

        self.slots = []
        if async_result:
            for _ in range(result_slots):
                self.slots.append(
                    {
                        "cpu": torch.empty(3, dtype=torch.float32, pin_memory=True),
                        "event": torch.cuda.Event(blocking=False),
                        "pending": False,
                        "seq_id": -1,
                        "metadata": None,
                    }
                )
        self.next_slot = 0
        self.last_consumed_seq = -1
        self.drop_count = 0
        self.completion_count = 0

    def _poll_completed(self):
        completed = []
        for slot in self.slots:
            if not slot["pending"] or not slot["event"].query():
                continue
            slot["pending"] = False
            if slot["seq_id"] <= self.last_consumed_seq:
                continue
            completed.append(slot)
        if not completed:
            return []
        self.completion_count += len(completed)
        latest = max(completed, key=lambda item: item["seq_id"])
        self.last_consumed_seq = latest["seq_id"]
        return [
            _result_record(
                latest["cpu"].tolist(),
                latest["seq_id"],
                latest["metadata"],
                time.monotonic_ns(),
            )
        ]

    def submit(self, stack: np.ndarray, seq_id: int, metadata: Dict[str, Any]):
        torch = self.torch
        started_ns = time.monotonic_ns()
        completed = self._poll_completed() if self.async_result else []
        self.host_input[0].copy_(torch.from_numpy(stack))
        self.static_input.copy_(self.host_input, non_blocking=True)
        self.graph.replay()
        if self.profile_sync:
            torch.cuda.synchronize(self.device)
            metadata = dict(metadata)
            metadata["infer_ms"] = (time.monotonic_ns() - started_ns) / 1e6

        if not self.async_result:
            values = self.static_result.cpu().tolist()
            self.completion_count += 1
            return [
                _result_record(values, seq_id, metadata, time.monotonic_ns())
            ]

        selected = None
        for offset in range(len(self.slots)):
            index = (self.next_slot + offset) % len(self.slots)
            slot = self.slots[index]
            if not slot["pending"]:
                selected = slot
                self.next_slot = (index + 1) % len(self.slots)
                break
        if selected is None:
            # Never wait for readback. A full ring means this result is dropped while
            # the GPU keeps progressing; the latest-only contract prevents backlog.
            self.drop_count += 1
            return completed

        selected["cpu"].copy_(self.static_result, non_blocking=True)
        selected["event"].record(torch.cuda.current_stream(self.device))
        selected["pending"] = True
        selected["seq_id"] = seq_id
        selected["metadata"] = dict(metadata)
        return completed

    def close(self):
        if self.async_result:
            self._poll_completed()


def _result_record(
    values: Any,
    seq_id: int,
    metadata: Dict[str, Any],
    ready_monotonic_ns: int,
) -> Dict[str, Any]:
    result = dict(metadata)
    result.update(
        {
            "seq_id": int(seq_id),
            "x": float(values[0]),
            "y": float(values[1]),
            "confidence": float(values[2]),
            "cuda_ready_monotonic_ns": int(ready_monotonic_ns),
        }
    )
    return result


def create_tdtracker_runner(config: "DVSWorkerConfig"):
    mode = config.tdtracker_mode.lower()
    if mode not in {"auto", "graph", "compile", "eager"}:
        raise ValueError(f"Unsupported TDTracker mode: {mode}")
    attempts = ["graph", "compile", "eager"] if mode == "auto" else [mode]
    last_error = None
    for candidate in attempts:
        try:
            if candidate == "graph":
                return _TDTrackerGraphRunner(
                    config.checkpoint_path,
                    config.gpu_id,
                    async_result=config.async_result,
                    result_slots=config.async_result_slots,
                    profile_sync=config.profile_sync,
                )
            if candidate == "compile":
                return _TDTrackerCompileRunner(
                    config.checkpoint_path, config.gpu_id
                )
            return _TDTrackerEagerRunner(config.checkpoint_path, config.gpu_id)
        except Exception as error:
            last_error = error
            logger.exception(
                f"TDTracker {candidate} runner initialization failed"
            )
            if mode != "auto":
                raise
    raise RuntimeError("No TDTracker runner could be initialized") from last_error


@dataclass
class DVSWorkerConfig:
    checkpoint_path: str
    gpu_id: int = 0
    original_width: int = 346
    original_height: int = 260
    time_window_us: int = 1000
    # Default to the existing plugin's tonic-compatible representation.  The
    # previous fast encoder remains available only as an explicit experiment.
    binarep_mode: str = "legacy"
    tdtracker_mode: str = "graph"
    async_result: bool = True
    async_result_slots: int = 8
    cpp_binarep: bool = True
    native_library_path: Optional[str] = None
    profile_sync: bool = False
    latency_log: bool = False

    @classmethod
    def from_environment(cls) -> "DVSWorkerConfig":
        plugin_dir = Path(__file__).resolve().parent
        checkpoint = os.getenv(
            "PUPIL_HYBRID_TDTRACKER_CKPT",
            str(plugin_dir / "best_checkpoint.pth"),
        )
        return cls(
            checkpoint_path=checkpoint,
            gpu_id=int(os.getenv("PUPIL_HYBRID_GPU_ID", "0")),
            original_width=int(
                os.getenv("PUPIL_HYBRID_DVS_WIDTH", "346")
            ),
            original_height=int(
                os.getenv("PUPIL_HYBRID_DVS_HEIGHT", "260")
            ),
            time_window_us=int(os.getenv("PUPIL_HYBRID_TIME_WINDOW_US", "1000")),
            binarep_mode=os.getenv("PUPIL_HYBRID_BINAREP_MODE", "legacy"),
            tdtracker_mode=os.getenv(
                "PUPIL_HYBRID_TDTRACKER_MODE", "graph"
            ),
            async_result=_env_bool("PUPIL_HYBRID_ASYNC_RESULT", True),
            async_result_slots=int(
                os.getenv("PUPIL_HYBRID_ASYNC_RESULT_SLOTS", "8")
            ),
            cpp_binarep=_env_bool("PUPIL_HYBRID_CPP_BINAREP", True),
            native_library_path=os.getenv("PUPIL_HYBRID_CPP_BINAREP_PATH"),
            profile_sync=_env_bool("PUPIL_HYBRID_PROFILE_SYNC", False),
            latency_log=_env_bool("PUPIL_HYBRID_LATENCY_LOG", False),
        )


def put_latest(result_queue: Any, item: Dict[str, Any]) -> None:
    try:
        result_queue.put_nowait(item)
        return
    except queue.Full:
        pass
    try:
        result_queue.get_nowait()
    except queue.Empty:
        pass
    try:
        result_queue.put_nowait(item)
    except queue.Full:
        pass


def dvs_worker_process_main(
    config_values: Dict[str, Any], result_queue: Any, stop_event: Any
) -> None:
    """Spawn target that owns the complete DAVIS-to-TDTracker hot path."""
    config = DVSWorkerConfig(**config_values)
    runner = None
    try:
        from process_affinity import apply_process_affinity

        apply_process_affinity("dvs")
        from datetime import timedelta

        from dv_processing import EventStreamSlicer
        from dv_processing.io import CameraCapture

        config.binarep_mode = _normalise_binarep_mode(config.binarep_mode)
        runner = create_tdtracker_runner(config)
        native = None
        if config.cpp_binarep:
            try:
                native = NativeBinaRep(config.native_library_path)
            except OSError:
                logger.exception("Native BinaRep unavailable; falling back to NumPy")
        if (
            config.binarep_mode == "legacy"
            and native is not None
            and not native.supports_legacy
        ):
            logger.warning(
                "Native BinaRep lacks the legacy encoder; falling back to NumPy. "
                "Rebuild _native_binarep.so to restore the native hot path."
            )

        capture = CameraCapture()
        slicer = EventStreamSlicer()
        frame_ring = np.zeros(
            (
                SEQUENCE_LENGTH,
                BINAREP_CHANNELS,
                BINAREP_HEIGHT,
                BINAREP_WIDTH,
            ),
            dtype=np.uint8,
        )
        ordered_stack = np.empty(frame_ring.shape, dtype=np.float32)
        frame_count = 0
        seq_id = 0
        counters = {"slice": 0, "submit": 0, "infer": 0, "drop": 0}
        pending_metrics = None
        metric_started_ns = time.monotonic_ns()
        legacy_coordinate_parts = []
        legacy_timestamp_parts = []
        legacy_polarity_parts = []
        legacy_start_timestamp = None

        def on_slice(events):
            nonlocal counters, frame_count, metric_started_ns, pending_metrics, seq_id
            nonlocal legacy_start_timestamp
            counters["slice"] += 1
            timestamps = events.timestamps().astype(np.int64)
            if timestamps.size == 0:
                return
            coordinates = events.coordinates().astype(np.int64)
            polarities = events.polarities().astype(np.uint8)

            if config.binarep_mode == "legacy":
                # Match the original plugin: gather slicer packets until a full
                # window has elapsed, then build one tonic-compatible BinaRep.
                if legacy_start_timestamp is None:
                    legacy_start_timestamp = int(timestamps[0])
                legacy_coordinate_parts.append(coordinates)
                legacy_timestamp_parts.append(timestamps)
                legacy_polarity_parts.append(polarities)
                if (
                    int(timestamps[-1]) - legacy_start_timestamp
                    < config.time_window_us
                ):
                    return
                encode_coordinates = np.concatenate(legacy_coordinate_parts)
                encode_timestamps = np.concatenate(legacy_timestamp_parts)
                encode_polarities = np.concatenate(legacy_polarity_parts)
                legacy_coordinate_parts.clear()
                legacy_timestamp_parts.clear()
                legacy_polarity_parts.clear()
                legacy_start_timestamp = None
            else:
                encode_coordinates = coordinates
                encode_timestamps = timestamps
                encode_polarities = polarities

            hardware_ts = int(encode_timestamps[-1])
            slot_index = frame_count % SEQUENCE_LENGTH
            encode_tdtracker_binarep(
                encode_coordinates,
                encode_timestamps,
                encode_polarities,
                original_width=config.original_width,
                original_height=config.original_height,
                time_window_us=config.time_window_us,
                binarep_mode=config.binarep_mode,
                native=native,
                out=frame_ring[slot_index],
            )
            frame_count += 1
            if frame_count < SEQUENCE_LENGTH:
                return

            copy_ordered_tdtracker_stack(
                frame_ring,
                frame_count,
                binarep_mode=config.binarep_mode,
                out=ordered_stack,
            )

            seq_id += 1
            submit_ns = time.monotonic_ns()
            counters["submit"] += 1
            drop_count_before = getattr(runner, "drop_count", 0)
            completion_count_before = getattr(runner, "completion_count", 0)
            results = runner.submit(
                ordered_stack,
                seq_id,
                {
                    "dvs_hardware_timestamp_us": hardware_ts,
                    "stack_submit_monotonic_ns": submit_ns,
                },
            )
            counters["drop"] += (
                getattr(runner, "drop_count", 0) - drop_count_before
            )
            if hasattr(runner, "completion_count"):
                counters["infer"] += (
                    runner.completion_count - completion_count_before
                )
            else:
                counters["infer"] += len(results)
            for result in results:
                if pending_metrics is not None:
                    result["worker_metrics"] = pending_metrics
                    pending_metrics = None
                put_latest(result_queue, result)

            now_ns = time.monotonic_ns()
            elapsed = (now_ns - metric_started_ns) / 1e9
            if elapsed >= 1.0:
                pending_metrics = {"elapsed_s": elapsed, **counters}
                counters = {"slice": 0, "submit": 0, "infer": 0, "drop": 0}
                metric_started_ns = now_ns

        slicer.doEveryTimeInterval(
            timedelta(microseconds=config.time_window_us), on_slice
        )
        put_latest(result_queue, {"kind": "worker_started", "config": asdict(config)})
        while not stop_event.is_set() and capture.isRunning():
            batch = capture.getNextEventBatch()
            if batch is not None:
                slicer.accept(batch)
        put_latest(result_queue, {"kind": "worker_stopped"})
    except Exception as error:
        logger.exception("Hybrid DVS worker failed")
        put_latest(
            result_queue,
            {
                "kind": "worker_error",
                "error": f"{type(error).__name__}: {error}",
            },
        )
    finally:
        if runner is not None:
            runner.close()
