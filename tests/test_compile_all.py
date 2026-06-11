"""Every module must at least compile, regardless of optional deps.

Modules behind optional dependencies (zarr v3, tfrecord, pyzed) are skipped
by their functional tests in envs without those deps — this sweep ensures a
syntax error can never hide there.
"""

import pathlib
import py_compile

import pytest

_PKG = pathlib.Path(__file__).resolve().parents[1] / "robo_dataset"
_MODULES = sorted(_PKG.rglob("*.py"))


@pytest.mark.parametrize("module", _MODULES, ids=lambda p: str(p.relative_to(_PKG)))
def test_module_compiles(module):
    py_compile.compile(str(module), doraise=True)
