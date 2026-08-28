#!/usr/bin/env python3
"""Strict launcher for the NIR + event-camera hybrid integration.

This entry point deliberately keeps :mod:`main` untouched.  It establishes the
runtime contract required by the hybrid pipeline and then hands control to the
normal Pupil launcher.  Consequently all existing ``main.py`` arguments keep
their usual meaning::

    python main_int.py capture
    python main_int.py --hide-ui

The integration-specific options are removed before the hand-off:

``--hybrid-dvs-gpu ID``
    CUDA device used by the TDTracker/DVS worker (default: 0).
``--hybrid-nir-gpu ID``
    CUDA device used by the Eye0/Eye1 NIR models (default: 0).
``--hybrid-snapshot-max-age-ms MS``
    Maximum age of an Eye0 Pye3D snapshot that may be fused into an event
    result (default: 250 ms).
``--hybrid-dvs-fallback-ms MS``
    Silence interval after the last valid fused event before Eye0 NIR/Pye3D
    becomes the final 3D fallback again (default: 50 ms).
``--hybrid-eye1-max-age-ms MS``
    Maximum Eye1 NIR/Pye3D age used to form an Eye0-driven binocular gaze
    sample (default: 50 ms).
``--hybrid-print-config``
    Validate and print the resolved integration environment without launching
    Pupil.

The TDTracker path is intentionally *CUDA Graph only*.  ``compile``, ``eager``
and ``auto`` are rejected here instead of silently weakening the real-time
contract.  Event output is also fresh-only: a TDTracker sequence is published
only when its ``seq_id`` advances; it is never timer-replayed as a new gaze
sample.

TDTracker input also remains compatible with the existing event plugin's
``tonic.ToFrame → tonic.ToBinaRep`` preprocessing.  The launcher rejects the
experimental fast representation so a CUDA-Graph optimization cannot change
the trained model's input semantics.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, MutableMapping, Optional, Sequence, Tuple


PROJECT_DIR = Path(__file__).resolve().parent
MAIN_LAUNCHER = PROJECT_DIR / "main.py"
DEFAULT_TDTRACKER_CHECKPOINT = (
    PROJECT_DIR / "shared_modules" / "pupil_detector_plugins" / "best_checkpoint.pth"
)


class HybridConfigurationError(ValueError):
    """Raised when an integration-only invariant is not satisfied."""


_FALSE_VALUES = frozenset({"", "0", "false", "no", "off"})
_CONFIG_KEYS = (
    "PUPIL_HYBRID_ENABLED",
    "PUPIL_HYBRID_INTEGRATED",
    "PUPIL_HYBRID_DVS_PROCESS",
    "PUPIL_HYBRID_DVS_EYE_ID",
    "PUPIL_HYBRID_FRESH_ONLY",
    "PUPIL_HYBRID_TDTRACKER_MODE",
    "PUPIL_HYBRID_BINAREP_MODE",
    "PUPIL_HYBRID_TDTRACKER_CKPT",
    "PUPIL_HYBRID_GPU_ID",
    "PUPIL_HYBRID_NIR_GPU_ID",
    "PUPIL_HYBRID_SNAPSHOT_MAX_AGE_MS",
    "PUPIL_HYBRID_DVS_FALLBACK_MS",
    "PUPIL_HYBRID_EYE1_MAX_AGE_MS",
)


def _is_enabled(value: str) -> bool:
    return value.strip().lower() not in _FALSE_VALUES


def _require_enabled(environment: MutableMapping[str, str], name: str) -> None:
    if not _is_enabled(environment[name]):
        raise HybridConfigurationError(
            f"{name}=1 is required by main_int.py; disable hybrid mode by "
            "launching main.py instead."
        )
    environment[name] = "1"


def _require_nonnegative_integer(
    environment: MutableMapping[str, str], name: str
) -> int:
    raw_value = environment[name].strip()
    try:
        value = int(raw_value)
    except ValueError as error:
        raise HybridConfigurationError(
            f"{name} must be a non-negative CUDA device index, got {raw_value!r}."
        ) from error
    if value < 0:
        raise HybridConfigurationError(
            f"{name} must be a non-negative CUDA device index, got {raw_value!r}."
        )
    environment[name] = str(value)
    return value


def _require_positive_float(
    environment: MutableMapping[str, str], name: str
) -> float:
    raw_value = environment[name].strip()
    try:
        value = float(raw_value)
    except ValueError as error:
        raise HybridConfigurationError(
            f"{name} must be a positive number of milliseconds, got {raw_value!r}."
        ) from error
    if not math.isfinite(value) or not value > 0.0:
        raise HybridConfigurationError(
            f"{name} must be a positive number of milliseconds, got {raw_value!r}."
        )
    environment[name] = format(value, "g")
    return value


def configure_hybrid_environment(
    environment: Optional[MutableMapping[str, str]] = None,
    *,
    check_checkpoint: bool = True,
) -> Dict[str, str]:
    """Apply and validate the non-negotiable hybrid runtime configuration.

    ``environment`` is injectable to keep this function independently testable.
    When omitted, the current process environment is updated for the subsequent
    hand-off to ``main.py``.
    """

    if environment is None:
        environment = os.environ

    defaults = {
        "PUPIL_HYBRID_ENABLED": "1",
        "PUPIL_HYBRID_INTEGRATED": "1",
        "PUPIL_HYBRID_DVS_PROCESS": "1",
        "PUPIL_HYBRID_DVS_EYE_ID": "0",
        "PUPIL_HYBRID_FRESH_ONLY": "1",
        "PUPIL_HYBRID_TDTRACKER_MODE": "graph",
        "PUPIL_HYBRID_BINAREP_MODE": "legacy",
        "PUPIL_HYBRID_TDTRACKER_CKPT": str(DEFAULT_TDTRACKER_CHECKPOINT),
        "PUPIL_HYBRID_GPU_ID": "0",
        "PUPIL_HYBRID_NIR_GPU_ID": "0",
        "PUPIL_HYBRID_SNAPSHOT_MAX_AGE_MS": "250",
        "PUPIL_HYBRID_DVS_FALLBACK_MS": "50",
        "PUPIL_HYBRID_EYE1_MAX_AGE_MS": "50",
    }
    for name, default_value in defaults.items():
        environment.setdefault(name, default_value)

    # This launcher always enables the separate DVS process.  The NIR eye
    # process owns its own RITnet/Pye3D work and only exchanges compact state
    # with the DVS side; neither inference loop waits for the other.
    _require_enabled(environment, "PUPIL_HYBRID_ENABLED")
    _require_enabled(environment, "PUPIL_HYBRID_INTEGRATED")
    _require_enabled(environment, "PUPIL_HYBRID_DVS_PROCESS")
    _require_enabled(environment, "PUPIL_HYBRID_FRESH_ONLY")

    dvs_eye_id = environment["PUPIL_HYBRID_DVS_EYE_ID"].strip()
    if dvs_eye_id != "0":
        raise HybridConfigurationError(
            "main_int.py supports one event camera attached to Eye0 only: "
            f"PUPIL_HYBRID_DVS_EYE_ID must be '0', got {dvs_eye_id!r}."
        )
    environment["PUPIL_HYBRID_DVS_EYE_ID"] = "0"

    tdtracker_mode = environment["PUPIL_HYBRID_TDTRACKER_MODE"].strip().lower()
    if tdtracker_mode != "graph":
        raise HybridConfigurationError(
            "main_int.py requires PUPIL_HYBRID_TDTRACKER_MODE=graph. "
            "CUDA Graph is mandatory for this integration; use main.py for "
            "compile/eager experiments."
        )
    environment["PUPIL_HYBRID_TDTRACKER_MODE"] = "graph"

    binarep_mode = environment["PUPIL_HYBRID_BINAREP_MODE"].strip().lower()
    if binarep_mode not in {"legacy", "original", "tonic"}:
        raise HybridConfigurationError(
            "main_int.py requires the original TDTracker preprocessing: "
            "PUPIL_HYBRID_BINAREP_MODE must be 'legacy'."
        )
    environment["PUPIL_HYBRID_BINAREP_MODE"] = "legacy"

    _require_nonnegative_integer(environment, "PUPIL_HYBRID_GPU_ID")
    _require_nonnegative_integer(environment, "PUPIL_HYBRID_NIR_GPU_ID")
    _require_positive_float(environment, "PUPIL_HYBRID_SNAPSHOT_MAX_AGE_MS")
    _require_positive_float(environment, "PUPIL_HYBRID_DVS_FALLBACK_MS")
    _require_positive_float(environment, "PUPIL_HYBRID_EYE1_MAX_AGE_MS")

    checkpoint = Path(environment["PUPIL_HYBRID_TDTRACKER_CKPT"]).expanduser()
    if check_checkpoint and not checkpoint.is_file():
        raise HybridConfigurationError(
            "TDTracker checkpoint does not exist: "
            f"{checkpoint}. Set PUPIL_HYBRID_TDTRACKER_CKPT to a valid file."
        )
    environment["PUPIL_HYBRID_TDTRACKER_CKPT"] = str(checkpoint)

    return {name: environment[name] for name in _CONFIG_KEYS}


def _parse_integration_arguments(
    argv: Sequence[str], environment: MutableMapping[str, str]
) -> Tuple[List[str], bool]:
    """Consume only main_int-specific flags and preserve every Pupil argument."""

    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--hybrid-dvs-gpu", type=int, metavar="ID")
    parser.add_argument("--hybrid-nir-gpu", type=int, metavar="ID")
    parser.add_argument("--hybrid-snapshot-max-age-ms", type=float, metavar="MS")
    parser.add_argument("--hybrid-dvs-fallback-ms", type=float, metavar="MS")
    parser.add_argument("--hybrid-eye1-max-age-ms", type=float, metavar="MS")
    parser.add_argument("--hybrid-print-config", action="store_true")
    options, launcher_args = parser.parse_known_args(list(argv))

    if options.hybrid_dvs_gpu is not None:
        environment["PUPIL_HYBRID_GPU_ID"] = str(options.hybrid_dvs_gpu)
    if options.hybrid_nir_gpu is not None:
        environment["PUPIL_HYBRID_NIR_GPU_ID"] = str(options.hybrid_nir_gpu)
    if options.hybrid_snapshot_max_age_ms is not None:
        environment["PUPIL_HYBRID_SNAPSHOT_MAX_AGE_MS"] = str(
            options.hybrid_snapshot_max_age_ms
        )
    if options.hybrid_dvs_fallback_ms is not None:
        environment["PUPIL_HYBRID_DVS_FALLBACK_MS"] = str(
            options.hybrid_dvs_fallback_ms
        )
    if options.hybrid_eye1_max_age_ms is not None:
        environment["PUPIL_HYBRID_EYE1_MAX_AGE_MS"] = str(
            options.hybrid_eye1_max_age_ms
        )
    return launcher_args, bool(options.hybrid_print_config)


def _delegate_to_main(launcher_args: Sequence[str]) -> None:
    """Replace this process with the unmodified standard Pupil launcher."""

    if not MAIN_LAUNCHER.is_file():
        raise HybridConfigurationError(
            f"Could not find standard launcher: {MAIN_LAUNCHER}"
        )

    command = [sys.executable, str(MAIN_LAUNCHER), *launcher_args]
    if os.name == "nt":
        # ``execv`` is unavailable on Windows.  Keeping the fallback here makes
        # the source launcher usable there without changing main.py.
        import subprocess

        completed = subprocess.run(command, check=False)
        raise SystemExit(completed.returncode)
    os.execv(sys.executable, command)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Validate integration settings and run the existing launcher."""

    launcher_args, print_config = _parse_integration_arguments(
        sys.argv[1:] if argv is None else argv, os.environ
    )
    try:
        resolved = configure_hybrid_environment()
    except HybridConfigurationError as error:
        print(f"Hybrid integration configuration error: {error}", file=sys.stderr)
        return 2

    if print_config:
        print(json.dumps(resolved, indent=2, sort_keys=True))
        return 0

    _delegate_to_main(launcher_args)
    return 0  # ``execv`` does not return; retained for type checkers.


if __name__ == "__main__":
    raise SystemExit(main())
