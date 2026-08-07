"""Optional CPU-affinity configuration for Hybrid Pupil processes."""

from __future__ import annotations

import logging
import os
from typing import Optional, Set

logger = logging.getLogger(__name__)


def _parse_cpu_list(value: str) -> Set[int]:
    cpus = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_text, end_text = part.split("-", 1)
            start, end = int(start_text), int(end_text)
            if start < 0 or end < start:
                raise ValueError(f"Invalid CPU range: {part}")
            cpus.update(range(start, end + 1))
        else:
            cpu = int(part)
            if cpu < 0:
                raise ValueError(f"Invalid CPU index: {cpu}")
            cpus.add(cpu)
    if not cpus:
        raise ValueError("CPU list is empty")
    return cpus


def _affinity_value(role: str, eye_id: Optional[int]) -> Optional[str]:
    candidates = []
    if role == "eye" and eye_id is not None:
        candidates.append(f"PUPIL_CPU_AFFINITY_EYE{eye_id}")
    candidates.append(f"PUPIL_CPU_AFFINITY_{role.upper()}")
    for name in candidates:
        value = os.getenv(name)
        if value:
            return value
    return None


def apply_process_affinity(role: str, eye_id: Optional[int] = None) -> bool:
    """Apply a role-specific affinity if its environment variable is set."""
    value = _affinity_value(role, eye_id)
    if value is None:
        return False
    try:
        cpus = _parse_cpu_list(value)
        if hasattr(os, "sched_setaffinity"):
            os.sched_setaffinity(0, cpus)
        else:
            import psutil

            psutil.Process().cpu_affinity(sorted(cpus))
    except (OSError, ValueError) as error:
        logger.warning(
            f"Could not set {role} CPU affinity ({value}): {error}"
        )
        return False
    logger.info(f"Set {role} CPU affinity to {sorted(cpus)}")
    return True
