"""Focused tests for the single-owner Eye0 integrated 3D broker."""

import multiprocessing as mp
import queue
import sys
import time
import unittest
from pathlib import Path
from types import SimpleNamespace


SHARED_MODULES = Path(__file__).parents[1] / "shared_modules"
if str(SHARED_MODULES) not in sys.path:
    sys.path.insert(0, str(SHARED_MODULES))

from pupil_detector_plugins.detector_2d_hybrid_plugin import (  # noqa: E402
    HybridDetector2DPlugin,
)
from pupil_detector_plugins.hybrid_pye3d_fusion import (  # noqa: E402
    Pye3DEyeModelSnapshot,
    Pye3DSnapshotStore,
)


class _Publisher:
    def __init__(self):
        self.sent = []

    def send(self, datum):
        self.sent.append(dict(datum))


def _snapshot():
    return Pye3DEyeModelSnapshot(
        timestamp=10.0,
        monotonic_ns=time.monotonic_ns(),
        pupil_confidence=0.9,
        model_confidence=0.95,
        sphere_center=(0.0, 0.0, 50.0),
        sphere_radius=12.0,
        pupil_radius=2.0,
        focal_length=400.0,
        frame_width=400,
        frame_height=400,
        ellipse_axes=(12.0, 14.0),
        ellipse_angle=0.0,
        projected_sphere_center=(200.0, 200.0),
        projected_sphere_axes=(190.0, 190.0),
        projected_sphere_angle=0.0,
    )


def _broker_plugin(store):
    """Build the small state surface the broker methods actually require."""

    plugin = HybridDetector2DPlugin.__new__(HybridDetector2DPlugin)
    plugin._pye3d_snapshot_store = store
    plugin._nir_fallback_queue = queue.Queue(maxsize=1)
    plugin._integrated_mode = True
    plugin.confidence_threshold = 0.3
    plugin.snapshot_max_age_ms = 1000.0
    plugin.dvs_fallback_ms = 50.0
    plugin.min_model_confidence = 0.6
    plugin.g_pool = SimpleNamespace(eye_id=0, get_timestamp=lambda: 11.0)
    return plugin


class Hybrid3DBrokerTests(unittest.TestCase):
    def test_fresh_tdtracker_sequence_publishes_once_and_marks_activity(self):
        store = Pye3DSnapshotStore.create(mp.get_context("spawn"))
        store.write(_snapshot())
        plugin = _broker_plugin(store)
        publisher = _Publisher()
        result = {
            "seq_id": 4,
            "x": 0.5,
            "y": 0.5,
            "confidence": 0.8,
            "stack_submit_monotonic_ns": time.monotonic_ns(),
        }

        self.assertTrue(plugin._publish_fused_dvs_result(publisher, result))
        self.assertFalse(plugin._publish_fused_dvs_result(publisher, result))
        self.assertEqual(len(publisher.sent), 1)
        self.assertEqual(publisher.sent[0]["topic"], "pupil.0.3d")
        self.assertEqual(publisher.sent[0]["source"], "dvs_pye3d_fused")
        self.assertEqual(store.latest_dvs_activity().seq_id, 4)

    def test_invalid_event_does_not_block_nir_fallback(self):
        store = Pye3DSnapshotStore.create(mp.get_context("spawn"))
        store.write(_snapshot())
        plugin = _broker_plugin(store)
        publisher = _Publisher()

        self.assertFalse(
            plugin._publish_fused_dvs_result(
                publisher,
                {"seq_id": 1, "x": 0.5, "y": 0.5, "confidence": 0.1},
            )
        )
        self.assertIsNone(store.latest_dvs_activity())

        plugin.submit_nir_fallback(
            {"topic": "pupil.0.3d", "source": "pye3d", "timestamp": 11.0}
        )
        self.assertTrue(plugin._publish_pending_nir_fallback(publisher))
        self.assertEqual(len(publisher.sent), 1)
        self.assertEqual(publisher.sent[0]["source"], "nir_pye3d_fallback")

    def test_recent_fused_event_suppresses_nir_fallback_in_same_broker(self):
        store = Pye3DSnapshotStore.create(mp.get_context("spawn"))
        plugin = _broker_plugin(store)
        publisher = _Publisher()
        self.assertTrue(store.mark_dvs_fused_result(9))

        plugin.submit_nir_fallback(
            {"topic": "pupil.0.3d", "source": "pye3d", "timestamp": 11.0}
        )
        self.assertFalse(plugin._publish_pending_nir_fallback(publisher))
        self.assertEqual(publisher.sent, [])


if __name__ == "__main__":
    unittest.main()
