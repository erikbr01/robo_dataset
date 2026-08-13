"""Optional-dependency bookkeeping, so a missing package is never silent.

Most of this suite degrades to skips when an optional package is absent: the
whole LeRobot module disappears without PyAV, the RLDS cases without tfrecord,
and so on. A green "N passed, M skipped" then looks identical whether the code
works or was never exercised.

Two things prevent that:
  * every run prints which optional packages are present and which are missing,
    plus a summary of how many tests each absence skipped;
  * ROBO_DATASET_STRICT_DEPS makes the absence a hard error, so CI cannot go
    green on a suite that quietly stopped running.

Set ROBO_DATASET_STRICT_DEPS=1 to require all of them, or to a comma-separated
list ("pyarrow,av") to require only those.
"""

import importlib
import os
from typing import Dict, List

# package name → what stops being tested without it
OPTIONAL_DEPS: Dict[str, str] = {
    "pyarrow": "LeRobot backend (parquet episode metadata and frame columns)",
    "av": "PyAV encode/decode: VideoStore, and the whole LeRobot video path",
    "torchcodec": "torchcodec decoder path",
    "tfrecord": "RLDS backend",
    "cv2": "image resize transforms",
}

_ENV_VAR = "ROBO_DATASET_STRICT_DEPS"


def missing_deps() -> List[str]:
    missing = []
    for name in OPTIONAL_DEPS:
        try:
            importlib.import_module(name)
        except ImportError:
            missing.append(name)
    return missing


def strictly_required() -> List[str]:
    """Packages this run must not be missing, from ROBO_DATASET_STRICT_DEPS."""
    raw = os.environ.get(_ENV_VAR, "").strip()
    if not raw or raw.lower() in ("0", "false", "no"):
        return []
    if raw.lower() in ("1", "true", "yes", "all"):
        return list(OPTIONAL_DEPS)
    requested = [p.strip() for p in raw.split(",") if p.strip()]
    unknown = [p for p in requested if p not in OPTIONAL_DEPS]
    if unknown:
        raise ValueError(
            f"{_ENV_VAR} names unknown packages {unknown}; "
            f"known: {sorted(OPTIONAL_DEPS)}"
        )
    return requested
