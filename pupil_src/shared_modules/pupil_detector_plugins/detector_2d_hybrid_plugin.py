"""RITnet anchor plus high-rate DAVIS/TDTracker pupil detector."""

from __future__ import annotations

import json
import logging
import math
import multiprocessing as mp
import os
import queue
import threading
import time
from copy import deepcopy

from pyglui import ui

from .detector_2d_plugin import Detector2DPlugin
from .hybrid_pye3d_fusion import (
    FusionGeometryError,
    Pye3DSnapshotStore,
    fuse_dvs_pupil,
)
from .hybrid_runtime import DVSWorkerConfig, dvs_worker_process_main

logger = logging.getLogger(__name__)


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"", "0", "false", "no", "off"}


def _eye_uses_dvs(eye_id: int) -> bool:
    value = os.getenv("PUPIL_HYBRID_DVS_EYE_ID", "0").strip().lower()
    if value in {"all", "both", "*"}:
        return True
    try:
        return eye_id in {int(item.strip()) for item in value.split(",")}
    except ValueError:
        logger.warning(
            f"Invalid PUPIL_HYBRID_DVS_EYE_ID={value!r}; using eye 0"
        )
        return eye_id == 0


def _env_nonnegative_float(name: str, default: float) -> float:
    """Read a latency/configuration value without making the eye process fragile."""

    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        value = float(raw_value)
    except ValueError:
        logger.warning("Invalid %s=%r; using %s", name, raw_value, default)
        return default
    if not math.isfinite(value) or value < 0.0:
        logger.warning("Invalid %s=%r; using %s", name, raw_value, default)
        return default
    return value


class HybridDetector2DPlugin(Detector2DPlugin):
    """Use accurate NIR ellipses as anchors and DVS for fresh center updates."""

    pupil_detection_identifier = "2d"
    pupil_detection_method = "hybrid ritnet+dvs"
    label = "Hybrid RITnet + DVS"

    @property
    def pretty_class_name(self):
        return "Hybrid Pupil Detector"

    def _init_nnunet_models(self):
        self.temporal_model = None
        self.vanilla_2d_model = None

    def _init_vivim_model(self):
        # Hybrid operation intentionally keeps the NIR side to RITnet.
        self.vivim_models = {}

    def __init__(self, g_pool=None, properties=None, detector_2d=None):
        super().__init__(
            g_pool=g_pool, properties=properties, detector_2d=detector_2d
        )
        self.active_model = "RITnet"
        self.confidence_threshold = float(
            os.getenv("PUPIL_HYBRID_CONF_THRESHOLD", "0.3")
        )
        self.publish_hz = float(os.getenv("PUPIL_HYBRID_PUBLISH_HZ", "1000"))
        self._state_lock = threading.Lock()
        self._anchor = None
        self._latest_dvs = {
            "seq_id": 0,
            "x": 0.5,
            "y": 0.5,
            "confidence": 0.0,
            "parent_receive_monotonic_ns": 0,
        }
        self._frame_size = (192, 192)
        self._stop_thread = threading.Event()
        self._worker = None
        self._worker_stop = None
        self._result_queue = None
        self._dvs_enabled_for_eye = _eye_uses_dvs(self.g_pool.eye_id)
        self._integrated_mode = self._dvs_enabled_for_eye and _env_bool(
            "PUPIL_HYBRID_INTEGRATED", False
        )
        self._mp_context = mp.get_context("spawn")
        self._pye3d_snapshot_store = None
        self._nir_fallback_queue = None
        self._publisher_thread = None
        self._last_received_seq = 0
        self._fresh_received_interval = 0
        self._missing_sequence_interval = 0
        self.snapshot_max_age_ms = _env_nonnegative_float(
            "PUPIL_HYBRID_SNAPSHOT_MAX_AGE_MS", 250.0
        )
        self.dvs_fallback_ms = _env_nonnegative_float(
            "PUPIL_HYBRID_DVS_FALLBACK_MS", 50.0
        )
        self.min_model_confidence = _env_nonnegative_float(
            "PUPIL_HYBRID_PYE3D_MODEL_CONFIDENCE", 0.6
        )

        if self._integrated_mode:
            # This numeric store is the only cross-process dependency between
            # NIR/Pye3D and TDTracker.  It deliberately contains no frames,
            # model objects, or queues that would couple the two inference loops.
            self._pye3d_snapshot_store = Pye3DSnapshotStore.create(
                self._mp_context
            )
            self._nir_fallback_queue = queue.Queue(maxsize=1)
            self.g_pool.hybrid_pye3d_snapshot_store = self._pye3d_snapshot_store
            # HybridPye3DPlugin submits fallback candidates here.  This plugin's
            # broker thread is the *only* owner of the Eye0 final-3D PUB socket.
            self.g_pool.hybrid_pye3d_broker = self

        if self._dvs_enabled_for_eye and _env_bool(
            "PUPIL_HYBRID_DVS_PROCESS", True
        ):
            self._start_worker()
        if self._integrated_mode:
            self._publisher_thread = threading.Thread(
                target=self._integrated_3d_broker_loop,
                name="hybrid-eye0-3d-broker",
                daemon=True,
            )
            self._publisher_thread.start()
        elif self._dvs_enabled_for_eye:
            self._publisher_thread = threading.Thread(
                target=self._parent_publish_loop,
                name="hybrid-dvs-publisher",
                daemon=True,
            )
            self._publisher_thread.start()

    def _start_worker(self):
        config = DVSWorkerConfig.from_environment()
        if not os.path.isfile(config.checkpoint_path):
            logger.error(
                f"TDTracker checkpoint does not exist: {config.checkpoint_path}"
            )
            return
        self._result_queue = self._mp_context.Queue(maxsize=1)
        self._worker_stop = self._mp_context.Event()
        self._worker = self._mp_context.Process(
            target=dvs_worker_process_main,
            args=(config.__dict__, self._result_queue, self._worker_stop),
            name="hybrid-dvs-worker",
            daemon=True,
        )
        self._worker.start()

    def _drain_worker_results(self):
        """Drain the latest TDTracker result and return it only once per seq_id."""

        if self._result_queue is None:
            return None
        newest = None
        while True:
            try:
                item = self._result_queue.get_nowait()
            except queue.Empty:
                break
            kind = item.get("kind")
            if kind == "metrics":
                elapsed = max(float(item["elapsed_s"]), 1e-9)
                logger.info(
                    f"Hybrid DVS slice={item['slice'] / elapsed:.1f}Hz "
                    f"submit={item['submit'] / elapsed:.1f}Hz "
                    f"infer={item['infer'] / elapsed:.1f}Hz "
                    f"drop={item['drop']}"
                )
            elif kind == "worker_error":
                logger.error(f"Hybrid DVS worker: {item['error']}")
            elif kind == "worker_started":
                logger.info(f"Hybrid DVS worker started: {item['config']}")
            elif kind == "worker_stopped":
                logger.info("Hybrid DVS worker stopped")
            elif "seq_id" in item:
                metrics = item.pop("worker_metrics", None)
                if metrics is not None:
                    elapsed = max(float(metrics["elapsed_s"]), 1e-9)
                    logger.info(
                        f"Hybrid DVS slice={metrics['slice'] / elapsed:.1f}Hz "
                        f"submit={metrics['submit'] / elapsed:.1f}Hz "
                        f"infer={metrics['infer'] / elapsed:.1f}Hz "
                        f"drop={metrics['drop']}"
                    )
                if newest is None or item["seq_id"] > newest["seq_id"]:
                    newest = item
        fresh_result = None
        if newest is not None:
            newest["parent_receive_monotonic_ns"] = time.monotonic_ns()
            seq_id = int(newest["seq_id"])
            if self._last_received_seq > 0 and seq_id > self._last_received_seq:
                self._missing_sequence_interval += max(
                    seq_id - self._last_received_seq - 1, 0
                )
            if seq_id > self._last_received_seq:
                self._fresh_received_interval += 1
                self._last_received_seq = seq_id
                fresh_result = newest
            with self._state_lock:
                self._latest_dvs = newest
        return fresh_result

    def _map_dvs(self, x, y):
        x = x * float(os.getenv("PUPIL_HYBRID_DVS_SCALE_X", "1.0"))
        y = y * float(os.getenv("PUPIL_HYBRID_DVS_SCALE_Y", "1.0"))
        x += float(os.getenv("PUPIL_HYBRID_DVS_OFFSET_X", "0.0"))
        y += float(os.getenv("PUPIL_HYBRID_DVS_OFFSET_Y", "0.0"))
        if _env_bool("PUPIL_HYBRID_DVS_FLIP_X", False):
            x = 1.0 - x
        if _env_bool("PUPIL_HYBRID_DVS_FLIP_Y", False):
            y = 1.0 - y
        return min(max(float(x), 0.0), 1.0), min(max(float(y), 0.0), 1.0)

    def _compose_datum(self, timestamp):
        now_ns = time.monotonic_ns()
        with self._state_lock:
            anchor = deepcopy(self._anchor)
            dvs = dict(self._latest_dvs)
            width, height = self._frame_size

        use_dvs = (
            dvs["seq_id"] > 0
            and dvs["confidence"] >= self.confidence_threshold
        )
        if anchor is None:
            anchor = self.create_pupil_datum(
                norm_pos=(0.5, 0.5),
                diameter=0.0,
                confidence=0.0,
                timestamp=timestamp,
            )
            anchor["ellipse"] = {
                "axes": (0.0, 0.0),
                "angle": 0.0,
                "center": (width * 0.5, height * 0.5),
            }

        datum = anchor
        datum["timestamp"] = timestamp
        datum["method"] = self.pupil_detection_method
        datum["topic"] = (
            f"pupil.{self.g_pool.eye_id}.{self.pupil_detection_identifier}"
        )
        if use_dvs:
            x, y = self._map_dvs(dvs["x"], dvs["y"])
            datum["norm_pos"] = (x, 1.0 - y)
            datum["tdtracker_confidence"] = float(dvs["confidence"])
            datum["confidence"] = min(max(float(dvs["confidence"]), 0.0), 1.0)
            ellipse = deepcopy(datum.get("ellipse", {}))
            ellipse["center"] = (x * width, y * height)
            ellipse.setdefault("axes", (datum.get("diameter", 0.0),) * 2)
            ellipse.setdefault("angle", 0.0)
            datum["ellipse"] = ellipse
            source = "dvs"
        else:
            source = "ritnet"

        receive_ns = int(dvs.get("parent_receive_monotonic_ns", 0))
        state_age_ms = (
            (now_ns - receive_ns) / 1e6 if receive_ns > 0 else float("inf")
        )
        submit_ns = dvs.get("stack_submit_monotonic_ns")
        ready_ns = dvs.get("cuda_ready_monotonic_ns")
        datum.update(
            {
                "source": source,
                "seq_id": int(dvs["seq_id"]),
                "dvs_hardware_timestamp_us": dvs.get(
                    "dvs_hardware_timestamp_us"
                ),
                "stack_submit_monotonic_ns": dvs.get(
                    "stack_submit_monotonic_ns"
                ),
                "cuda_ready_monotonic_ns": dvs.get(
                    "cuda_ready_monotonic_ns"
                ),
                "parent_receive_monotonic_ns": receive_ns or None,
                "publish_monotonic_ns": now_ns,
                "state_age_ms": state_age_ms,
                "submit_to_ready_ms": (
                    (ready_ns - submit_ns) / 1e6
                    if submit_ns is not None and ready_ns is not None
                    else None
                ),
                "submit_to_parent_ms": (
                    (receive_ns - submit_ns) / 1e6
                    if submit_ns is not None and receive_ns > 0
                    else None
                ),
                "submit_to_publish_ms": (
                    (now_ns - submit_ns) / 1e6
                    if submit_ns is not None
                    else None
                ),
            }
        )
        return datum

    def _parent_publish_loop(self):
        # The socket is created and exclusively used in this thread; ZeroMQ sockets
        # themselves are not thread-safe.
        from zmq_tools import Msg_Streamer

        publisher = Msg_Streamer(
            self.g_pool.zmq_ctx,
            self.g_pool.ipc_pub_url,
        )
        interval_ns = max(int(1e9 / max(self.publish_hz, 1.0)), 1)
        deadline_ns = time.monotonic_ns()
        last_seq = -1
        metric_start_ns = deadline_ns
        metric_start_seq = 0
        publish_count = 0
        metrics_stream = None
        metrics_path = os.getenv("PUPIL_HYBRID_METRICS_PATH")
        if metrics_path:
            try:
                metrics_stream = open(metrics_path, "a", buffering=1)
            except OSError:
                logger.exception(
                    f"Could not open Hybrid metrics file: {metrics_path}"
                )
        try:
            while not self._stop_thread.is_set():
                self._drain_worker_results()
                now_ns = time.monotonic_ns()
                if now_ns < deadline_ns:
                    self._stop_thread.wait((deadline_ns - now_ns) / 1e9)
                    continue
                timestamp = self.g_pool.get_timestamp()
                datum = self._compose_datum(timestamp)
                datum["fresh"] = datum["seq_id"] != last_seq
                if datum["fresh"]:
                    last_seq = datum["seq_id"]
                publisher.send(datum)
                publish_count += 1
                deadline_ns += interval_ns
                if deadline_ns < now_ns - interval_ns:
                    deadline_ns = now_ns + interval_ns

                if now_ns - metric_start_ns >= 1_000_000_000:
                    elapsed = (now_ns - metric_start_ns) / 1e9
                    current_seq = self._last_received_seq
                    graph_replay_hz = (
                        max(current_seq - metric_start_seq, 0) / elapsed
                    )
                    fresh_result_hz = self._fresh_received_interval / elapsed
                    missing_sequences = self._missing_sequence_interval
                    publish_hz = publish_count / elapsed
                    logger.info(
                        f"Hybrid runtime replay={graph_replay_hz:.1f}Hz "
                        f"fresh={fresh_result_hz:.1f}Hz "
                        f"publish={publish_hz:.1f}Hz "
                        f"missing={missing_sequences} "
                        f"seq={datum['seq_id']} "
                        f"state_age_ms={datum['state_age_ms']:.3f} "
                        f"src={datum['source']}"
                    )
                    if metrics_stream is not None:
                        metrics_stream.write(
                            json.dumps(
                                {
                                    "epoch_s": time.time(),
                                    "elapsed_s": elapsed,
                                    "graph_replay_hz": graph_replay_hz,
                                    "fresh_result_hz": fresh_result_hz,
                                    "publish_hz": publish_hz,
                                    "missing_sequences": missing_sequences,
                                    "seq_id": current_seq,
                                },
                                allow_nan=False,
                            )
                            + "\n"
                        )
                    publish_count = 0
                    metric_start_ns = now_ns
                    metric_start_seq = current_seq
                    self._fresh_received_interval = 0
                    self._missing_sequence_interval = 0
        finally:
            if metrics_stream is not None:
                metrics_stream.close()
            publisher.socket.close(linger=0)

    def submit_nir_fallback(self, datum):
        """Queue one Eye0 NIR/Pye3D fallback candidate for the 3D broker.

        This method is called from the normal eye plugin loop.  The broker owns
        the ZeroMQ socket on a different thread, therefore the hand-off is a
        latest-only in-process queue instead of sharing a socket between
        threads.  Keeping only the latest NIR candidate also prevents a slow
        broker from accumulating frame-rate work while DVS is active.
        """

        if not self._integrated_mode or self._nir_fallback_queue is None:
            return False

        candidate = deepcopy(datum)
        candidate.pop("_published_externally", None)
        candidate.update(
            {
                "source": "nir_pye3d_fallback",
                "source_2d": "nir_ritnet",
                "hybrid_output_owner": "eye0_3d_broker",
            }
        )
        try:
            self._nir_fallback_queue.put_nowait(candidate)
            return True
        except queue.Full:
            pass
        try:
            self._nir_fallback_queue.get_nowait()
        except queue.Empty:
            pass
        try:
            self._nir_fallback_queue.put_nowait(candidate)
            return True
        except queue.Full:
            # A concurrent broker read may temporarily make the queue appear
            # full. Dropping this NIR frame is safe; the next NIR frame is the
            # preferred fallback candidate anyway.
            return False

    def _integrated_3d_broker_loop(self):
        """Publish the one authoritative Eye0 3D stream for integrated mode.

        NIR/Pye3D and TDTracker remain independent producers.  Their only
        meeting point is the shared numeric Pye3D snapshot and this output
        broker.  Consequently Pye3D fallback and DVS-fused results cannot race
        on two separate PUB sockets or create duplicate Eye0 3D messages.
        """

        from zmq_tools import Msg_Streamer

        publisher = Msg_Streamer(self.g_pool.zmq_ctx, self.g_pool.ipc_pub_url)
        try:
            while not self._stop_thread.is_set():
                # Always process a completed event result before a pending NIR
                # fallback. Once the fresh fused datum is marked, the fallback
                # gate below sees DVS as active in the same broker iteration.
                fresh_dvs = self._drain_worker_results()
                if fresh_dvs is not None:
                    self._publish_fused_dvs_result(publisher, fresh_dvs)
                self._publish_pending_nir_fallback(publisher)
                # TDTracker slices are nominally 1 ms. Event.wait keeps shutdown
                # responsive without a busy spinning CPU core in the eye process.
                self._stop_thread.wait(0.001)
        except Exception:
            logger.exception("Hybrid Eye0 3D broker stopped unexpectedly")
        finally:
            publisher.socket.close(linger=0)

    def _publish_fused_dvs_result(self, publisher, dvs_result):
        """Fuse exactly one fresh TDTracker completion into a Pye3D 3D datum."""

        store = self._pye3d_snapshot_store
        if store is None:
            return False
        try:
            seq_id = int(dvs_result["seq_id"])
            confidence = float(dvs_result["confidence"])
        except (KeyError, TypeError, ValueError):
            logger.warning("Ignoring malformed TDTracker completion: %r", dvs_result)
            return False
        if not math.isfinite(confidence) or confidence < self.confidence_threshold:
            return False

        now_ns = time.monotonic_ns()
        snapshot = store.read(
            now_monotonic_ns=now_ns,
            max_age_ms=self.snapshot_max_age_ms,
            min_model_confidence=self.min_model_confidence,
        )
        if snapshot is None:
            return False

        try:
            mapped_x, mapped_y = self._map_dvs(dvs_result["x"], dvs_result["y"])
            fused = fuse_dvs_pupil(
                snapshot,
                mapped_nir_norm_pos=(mapped_x, mapped_y),
                dvs_confidence=confidence,
                timestamp=self.g_pool.get_timestamp(),
                eye_id=self.g_pool.eye_id,
                seq_id=seq_id,
                now_monotonic_ns=now_ns,
                max_snapshot_age_ms=self.snapshot_max_age_ms,
                min_model_confidence=self.min_model_confidence,
            )
        except (FusionGeometryError, KeyError, TypeError, ValueError):
            logger.exception("Could not fuse fresh TDTracker result seq=%s", seq_id)
            return False

        if fused is None:
            return False
        # This atomic generation gate gives the fresh-only contract a second
        # layer of protection if the worker ever delivers an old sequence after
        # a restart/reordering event.  Activity is marked only for a datum that
        # passed every fusion check, so invalid DVS data never blocks NIR fallback.
        if not store.mark_dvs_fused_result(seq_id, monotonic_ns=now_ns):
            return False

        submit_ns = dvs_result.get("stack_submit_monotonic_ns")
        ready_ns = dvs_result.get("cuda_ready_monotonic_ns")
        receive_ns = dvs_result.get("parent_receive_monotonic_ns")
        fused.update(
            {
                "dvs_hardware_timestamp_us": dvs_result.get(
                    "dvs_hardware_timestamp_us"
                ),
                "stack_submit_monotonic_ns": submit_ns,
                "cuda_ready_monotonic_ns": ready_ns,
                "parent_receive_monotonic_ns": receive_ns,
                "publish_monotonic_ns": now_ns,
                "submit_to_ready_ms": (
                    (ready_ns - submit_ns) / 1e6
                    if submit_ns is not None and ready_ns is not None
                    else None
                ),
                "submit_to_parent_ms": (
                    (receive_ns - submit_ns) / 1e6
                    if submit_ns is not None and receive_ns is not None
                    else None
                ),
                "submit_to_publish_ms": (
                    (now_ns - submit_ns) / 1e6 if submit_ns is not None else None
                ),
                "hybrid_output_owner": "eye0_3d_broker",
            }
        )
        publisher.send(fused)
        return True

    def _publish_pending_nir_fallback(self, publisher):
        """Emit only the newest NIR Pye3D fallback when fused DVS is stale."""

        if self._nir_fallback_queue is None:
            return False
        newest = None
        while True:
            try:
                newest = self._nir_fallback_queue.get_nowait()
            except queue.Empty:
                break
        if newest is None:
            return False

        store = self._pye3d_snapshot_store
        if store is not None and store.dvs_is_fresh(self.dvs_fallback_ms):
            return False
        newest["hybrid_dvs_active"] = False
        publisher.send(newest)
        return True

    def detect(self, frame, **kwargs):
        self._frame_size = (frame.width, frame.height)
        anchor = self._detect_ritnet(frame, **kwargs)
        with self._state_lock:
            self._anchor = anchor
        if self._integrated_mode:
            # Pye3D must model the eyeball from the independent NIR ellipse.
            # Do not replace its center with an older/latest DVS point here.
            # The final Eye0 3D output is instead owned by the broker above.
            anchor["source"] = "nir_ritnet_anchor"
            anchor["_published_externally"] = True
            return anchor
        if not self._dvs_enabled_for_eye:
            anchor["method"] = self.pupil_detection_method
            return anchor
        datum = self._compose_datum(frame.timestamp)
        # pye3d still consumes this result in the same eye-frame event, but eye.py
        # must not publish it again because the dedicated 1000 Hz timer already does.
        datum["_published_externally"] = True
        return datum

    def init_ui(self):
        # Keep controls focused on runtime behavior; nnUNet selectors from the parent
        # plugin do not apply to this detector.
        super(Detector2DPlugin, self).init_ui()
        self.menu.label = self.pretty_class_name
        self.menu.append(
            ui.Info_Text(
                "RITnet supplies the ellipse anchor. DAVIS/TDTracker updates the "
                "center through an external CUDA Graph worker."
            )
        )
        self.menu.append(
            ui.Slider(
                "confidence_threshold",
                self,
                label="DVS confidence threshold",
                min=0.0,
                max=1.0,
                step=0.01,
            )
        )

    def cleanup(self):
        self._stop_thread.set()
        if self._publisher_thread is not None:
            self._publisher_thread.join(timeout=2.0)
        if self._worker_stop is not None:
            self._worker_stop.set()
        if self._worker is not None:
            self._worker.join(timeout=5.0)
            if self._worker.is_alive():
                logger.warning("Terminating unresponsive Hybrid DVS worker")
                self._worker.terminate()
                self._worker.join(timeout=2.0)
        if self._result_queue is not None:
            self._result_queue.close()
            self._result_queue.join_thread()
        if getattr(self.g_pool, "hybrid_pye3d_broker", None) is self:
            self.g_pool.hybrid_pye3d_broker = None
        if (
            getattr(self.g_pool, "hybrid_pye3d_snapshot_store", None)
            is self._pye3d_snapshot_store
        ):
            self.g_pool.hybrid_pye3d_snapshot_store = None
        return 0
