import multiprocessing as mp
import sys
import unittest
from pathlib import Path


SHARED_MODULES = Path(__file__).parents[1] / "shared_modules"
if str(SHARED_MODULES) not in sys.path:
    sys.path.insert(0, str(SHARED_MODULES))

from pupil_detector_plugins.hybrid_pye3d_fusion import (  # noqa: E402
    Pye3DEyeModelSnapshot,
    Pye3DSnapshotStore,
    fuse_dvs_pupil,
)


def _read_store_from_spawn_child(store, output_queue, now_monotonic_ns):
    """Top-level function required by multiprocessing's spawn pickler."""

    snapshot = store.read(
        now_monotonic_ns=now_monotonic_ns,
        max_age_ms=20.0,
        min_model_confidence=0.6,
    )
    activity = store.latest_dvs_activity()
    output_queue.put(
        {
            "generation": None if snapshot is None else snapshot.generation,
            "sphere_center": None if snapshot is None else snapshot.sphere_center,
            "activity_seq": None if activity is None else activity.seq_id,
        }
    )


def _pye3d_datum(**overrides):
    datum = {
        "timestamp": 42.0,
        "confidence": 0.91,
        "model_confidence": 0.95,
        "sphere": {"center": (0.0, 0.0, 50.0), "radius": 12.0},
        "circle_3d": {
            "center": (0.0, 0.0, 38.0),
            "normal": (0.0, 0.0, -1.0),
            "radius": 2.1,
        },
        "diameter": 15.0,
        "ellipse": {
            "center": (200.0, 200.0),
            "axes": (12.0, 15.0),
            "angle": 7.0,
        },
        "projected_sphere": {
            "center": (200.0, 200.0),
            "axes": (192.0, 192.0),
            "angle": 0.0,
        },
    }
    datum.update(overrides)
    return datum


def _snapshot(monotonic_ns=1_000_000_000):
    snapshot = Pye3DEyeModelSnapshot.from_pye3d_datum(
        _pye3d_datum(),
        focal_length=400.0,
        frame_size=(400, 400),
        monotonic_ns=monotonic_ns,
    )
    assert snapshot is not None
    return snapshot


class HybridPye3DFusionTests(unittest.TestCase):
    def assertTupleAlmostEqual(self, actual, expected, places=7):
        self.assertEqual(len(actual), len(expected))
        for actual_value, expected_value in zip(actual, expected):
            self.assertAlmostEqual(actual_value, expected_value, places=places)

    def test_fused_datum_has_pye3d_fields_and_center_ray_geometry(self):
        snapshot = _snapshot()

        fused = fuse_dvs_pupil(
            snapshot,
            mapped_nir_norm_pos=(0.5, 0.5),
            dvs_confidence=0.87,
            timestamp=43.0,
            eye_id=0,
            seq_id=12,
            now_monotonic_ns=1_003_000_000,
            max_snapshot_age_ms=10.0,
        )

        self.assertIsNotNone(fused)
        assert fused is not None
        self.assertEqual(fused["topic"], "pupil.0.3d")
        self.assertIn("3d", fused["method"])
        self.assertEqual(fused["source"], "dvs_pye3d_fused")
        self.assertEqual(fused["seq_id"], 12)
        self.assertTrue(fused["fresh"])
        self.assertEqual(fused["norm_pos"], (0.5, 0.5))
        self.assertEqual(fused["ellipse"]["center"], (200.0, 200.0))
        self.assertEqual(fused["projected_sphere"], _pye3d_datum()["projected_sphere"])
        self.assertEqual(fused["sphere"], _pye3d_datum()["sphere"])
        self.assertTupleAlmostEqual(fused["circle_3d"]["center"], (0.0, 0.0, 38.0))
        self.assertTupleAlmostEqual(fused["circle_3d"]["normal"], (0.0, 0.0, -1.0))
        self.assertEqual(fused["circle_3d"]["radius"], 2.1)
        self.assertEqual(fused["diameter_3d"], 4.2)
        self.assertEqual(fused["ray_sphere_intersection"], "direct")

    def test_mapped_nir_image_y_is_flipped_only_in_public_norm_pos(self):
        snapshot = _snapshot()

        fused = fuse_dvs_pupil(
            snapshot,
            # This is already after DVS scale/offset/flip mapping.  It is an
            # NIR image coordinate, not an already Pupil-flipped coordinate.
            mapped_nir_norm_pos=(0.75, 0.25),
            dvs_confidence=0.9,
            timestamp=43.0,
            now_monotonic_ns=1_001_000_000,
        )

        self.assertIsNotNone(fused)
        assert fused is not None
        self.assertEqual(fused["ellipse"]["center"], (300.0, 100.0))
        self.assertEqual(fused["norm_pos"], (0.75, 0.75))
        center = fused["circle_3d"]["center"]
        normal = fused["circle_3d"]["normal"]
        self.assertAlmostEqual(sum(value * value for value in normal), 1.0)
        self.assertAlmostEqual(
            sum((value - center_value) ** 2 for value, center_value in zip(center, (0.0, 0.0, 50.0))),
            12.0**2,
        )

    def test_missing_or_invalid_pye3d_data_never_becomes_a_snapshot(self):
        self.assertIsNone(
            Pye3DEyeModelSnapshot.from_pye3d_datum(
                {"timestamp": 1.0}, focal_length=400.0, frame_size=(400, 400)
            )
        )
        self.assertIsNone(
            Pye3DEyeModelSnapshot.from_pye3d_datum(
                _pye3d_datum(circle_3d={"radius": 0.0}),
                focal_length=400.0,
                frame_size=(400, 400),
            )
        )
        self.assertIsNone(
            Pye3DEyeModelSnapshot.from_pye3d_datum(
                _pye3d_datum(sphere={"center": (0.0, 0.0, float("nan")), "radius": 12.0}),
                focal_length=400.0,
                frame_size=(400, 400),
            )
        )

    def test_snapshot_store_rejects_stale_or_low_confidence_models(self):
        context = mp.get_context("spawn")
        store = Pye3DSnapshotStore.create(context)
        written = store.write(_snapshot())
        self.assertEqual(written.generation, 1)

        current = store.read(
            now_monotonic_ns=1_004_000_000,
            max_age_ms=5.0,
            min_model_confidence=0.6,
        )
        self.assertIsNotNone(current)
        assert current is not None
        self.assertEqual(current.generation, 1)
        self.assertEqual(current.sphere_center, (0.0, 0.0, 50.0))
        self.assertIsNone(
            store.read(now_monotonic_ns=1_006_000_000, max_age_ms=5.0)
        )
        self.assertIsNone(
            store.read(
                now_monotonic_ns=1_004_000_000,
                max_age_ms=5.0,
                min_model_confidence=0.99,
            )
        )

        self.assertIsNone(
            store.update_from_pye3d_datum(
                {"timestamp": 1.0}, focal_length=400.0, frame_size=(400, 400)
            )
        )
        self.assertIsNone(store.read(now_monotonic_ns=1_004_000_000))

    def test_dvs_activity_is_atomic_fresh_gate_and_nir_fallback_heartbeat(self):
        store = Pye3DSnapshotStore.create(mp.get_context("spawn"))
        self.assertFalse(store.dvs_is_fresh(5.0, now_monotonic_ns=1_000_000_000))
        self.assertTrue(store.mark_dvs_fused_result(7, monotonic_ns=1_000_000_000))
        self.assertFalse(store.mark_dvs_fused_result(7, monotonic_ns=1_001_000_000))
        self.assertFalse(store.mark_dvs_fused_result(6, monotonic_ns=1_001_000_000))
        self.assertTrue(store.dvs_is_fresh(5.0, now_monotonic_ns=1_005_000_000))
        self.assertFalse(store.dvs_is_fresh(5.0, now_monotonic_ns=1_005_000_001))
        activity = store.latest_dvs_activity()
        self.assertIsNotNone(activity)
        assert activity is not None
        self.assertEqual(activity.seq_id, 7)

    def test_spawn_worker_receives_the_same_numeric_snapshot_and_dvs_activity(self):
        context = mp.get_context("spawn")
        store = Pye3DSnapshotStore.create(context)
        store.write(_snapshot())
        self.assertTrue(store.mark_dvs_fused_result(4, monotonic_ns=1_001_000_000))
        output_queue = context.Queue()
        process = context.Process(
            target=_read_store_from_spawn_child,
            args=(store, output_queue, 1_002_000_000),
        )
        process.start()
        result = output_queue.get(timeout=10.0)
        process.join(timeout=10.0)
        try:
            self.assertEqual(process.exitcode, 0)
            self.assertEqual(result["generation"], 1)
            self.assertEqual(result["sphere_center"], (0.0, 0.0, 50.0))
            self.assertEqual(result["activity_seq"], 4)
        finally:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5.0)
            output_queue.close()
            output_queue.join_thread()

    def test_ray_miss_can_fallback_to_nearest_sphere_point_or_be_rejected(self):
        snapshot = _snapshot()
        exact_only = fuse_dvs_pupil(
            snapshot,
            mapped_nir_norm_pos=(1.0, 0.5),
            dvs_confidence=0.9,
            timestamp=43.0,
            now_monotonic_ns=1_001_000_000,
            allow_nearest_sphere_point=False,
        )
        self.assertIsNone(exact_only)
        nearest = fuse_dvs_pupil(
            snapshot,
            mapped_nir_norm_pos=(1.0, 0.5),
            dvs_confidence=0.9,
            timestamp=43.0,
            now_monotonic_ns=1_001_000_000,
            allow_nearest_sphere_point=True,
        )
        self.assertIsNotNone(nearest)
        assert nearest is not None
        self.assertEqual(nearest["ray_sphere_intersection"], "nearest")
        center = nearest["circle_3d"]["center"]
        self.assertAlmostEqual(
            sum((value - center_value) ** 2 for value, center_value in zip(center, (0.0, 0.0, 50.0))),
            12.0**2,
        )


if __name__ == "__main__":
    unittest.main()
