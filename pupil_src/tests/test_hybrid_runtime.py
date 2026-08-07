import importlib.util
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


MODULE_PATH = (
    Path(__file__).parents[1]
    / "shared_modules"
    / "pupil_detector_plugins"
    / "hybrid_runtime.py"
)
NATIVE_SOURCE = MODULE_PATH.with_name("_native_binarep.c")

spec = importlib.util.spec_from_file_location("_hybrid_runtime_test", MODULE_PATH)
runtime = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = runtime
spec.loader.exec_module(runtime)


class HybridRuntimeTests(unittest.TestCase):
    def test_numpy_binarep_is_event_by_event_equivalent(self):
        xs = np.array([0, 0, 0, 79, -1, 80])
        ys = np.array([0, 0, 0, 59, 2, 2])
        timestamps = np.array([0, 250, 999, 750, 500, 500])
        polarities = np.array([0, 0, 0, 1, 1, 1])

        result = runtime.fill_binarep_numpy(
            xs,
            ys,
            timestamps,
            polarities,
            slice_start_us=0,
            time_window_us=1000,
        )

        self.assertEqual(result.shape, (2, 60, 80))
        self.assertEqual(result.dtype, np.float32)
        self.assertEqual(result[0, 0, 0], 0b1011)
        self.assertEqual(result[1, 59, 79], 0b1000)
        self.assertEqual(np.count_nonzero(result), 2)

    @unittest.skipUnless(shutil.which("cc"), "No C compiler available")
    def test_native_binarep_matches_numpy_bit_exactly(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            library = Path(temp_dir) / "_native_binarep.so"
            subprocess.run(
                [
                    shutil.which("cc"),
                    "-O3",
                    "-shared",
                    "-fPIC",
                    str(NATIVE_SOURCE),
                    "-o",
                    str(library),
                ],
                check=True,
            )
            random = np.random.default_rng(7)
            xs = random.integers(-5, 85, size=20_000, dtype=np.int64)
            ys = random.integers(-5, 65, size=20_000, dtype=np.int64)
            timestamps = random.integers(-100, 1100, size=20_000, dtype=np.int64)
            polarities = random.integers(0, 2, size=20_000, dtype=np.uint8)
            expected = runtime.fill_binarep_numpy(
                xs,
                ys,
                timestamps,
                polarities,
                slice_start_us=0,
                time_window_us=1000,
            )
            actual = runtime.NativeBinaRep(str(library)).fill(
                xs,
                ys,
                timestamps,
                polarities,
                slice_start_us=0,
                time_window_us=1000,
            )
            np.testing.assert_array_equal(actual, expected)


if __name__ == "__main__":
    unittest.main()
