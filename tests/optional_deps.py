"""Optional-dependency bookkeeping, so a missing capability is never silent.

Most of this suite degrades to skips when something optional is absent: the
whole LeRobot module disappears without PyAV, the RLDS cases without tfrecord,
the sharded-zarr backend when zarr resolves to 2.x. A green "N passed, M
skipped" then looks identical whether the code works or was never exercised.

Two things prevent that:
  * every run prints which capabilities are present and which are missing, plus
    how many tests each absence skipped;
  * ROBO_DATASET_STRICT_DEPS makes the absence a hard error, so CI cannot go
    green on a suite that quietly stopped running.

Set ROBO_DATASET_STRICT_DEPS=1 to require all of them, or to a comma-separated
list ("pyarrow,av") to require only those.

Entries are capabilities, not just imports: "zarr3" is present as an installed
zarr whose major version is at least 3, because zarr 2.x silently disables an
entire backend rather than failing to import.
"""

import importlib
import os
from typing import Dict, List, NamedTuple, Optional

_ENV_VAR = "ROBO_DATASET_STRICT_DEPS"


class Capability(NamedTuple):
    module: str
    min_major: Optional[int]  # None = any version
    description: str          # what stops being tested without it
    skip_token: str           # substring identifying its skip reasons


OPTIONAL_DEPS: Dict[str, Capability] = {
    "pyarrow": Capability(
        "pyarrow", None,
        "LeRobot backend (parquet episode metadata and frame columns)",
        "pyarrow"),
    "av": Capability(
        "av", None,
        "PyAV encode/decode: VideoStore, and the whole LeRobot video path",
        "av"),
    "torchcodec": Capability(
        "torchcodec", None, "torchcodec decoder path", "torchcodec"),
    "tfrecord": Capability(
        "tfrecord", None, "RLDS backend", "tfrecord"),
    "cv2": Capability(
        "cv2", None, "image resize transforms", "cv2"),
    "zarr3": Capability(
        "zarr", 3, "sharded zarr-v3 backend (ShardingCodec)", "zarr >= 3.0"),
}


def _version_major(module) -> Optional[int]:
    raw = getattr(module, "__version__", "")
    try:
        return int(str(raw).split(".")[0])
    except (ValueError, IndexError):
        return None


def describe(name: str) -> str:
    """Human-readable state of one capability, e.g. 'zarr3 (zarr 2.18.2)'."""
    cap = OPTIONAL_DEPS[name]
    try:
        module = importlib.import_module(cap.module)
    except ImportError:
        return f"{name} (not installed)"
    version = getattr(module, "__version__", "unknown")
    if cap.min_major is not None:
        return f"{name} ({cap.module} {version}, need >= {cap.min_major})"
    return f"{name} ({version})"


def missing_deps() -> List[str]:
    missing = []
    for name, cap in OPTIONAL_DEPS.items():
        try:
            module = importlib.import_module(cap.module)
        except ImportError:
            missing.append(name)
            continue
        if cap.min_major is not None:
            major = _version_major(module)
            if major is None or major < cap.min_major:
                missing.append(name)
    return missing


def strictly_required() -> List[str]:
    """Capabilities this run must not be missing, from ROBO_DATASET_STRICT_DEPS."""
    raw = os.environ.get(_ENV_VAR, "").strip()
    if not raw or raw.lower() in ("0", "false", "no"):
        return []
    if raw.lower() in ("1", "true", "yes", "all"):
        return list(OPTIONAL_DEPS)
    requested = [p.strip() for p in raw.split(",") if p.strip()]
    unknown = [p for p in requested if p not in OPTIONAL_DEPS]
    if unknown:
        raise ValueError(
            f"{_ENV_VAR} names unknown capabilities {unknown}; "
            f"known: {sorted(OPTIONAL_DEPS)}"
        )
    return requested
