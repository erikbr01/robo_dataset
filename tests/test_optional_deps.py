"""Tests for the optional-capability guard itself.

This machinery exists to stop the suite going green on tests that never ran, so
it is worth checking that it actually detects absence — including the awkward
case where a package is installed but too old to enable its feature.
"""

import sys
import types

import pytest

from .optional_deps import (
    OPTIONAL_DEPS,
    Capability,
    describe,
    missing_deps,
    strictly_required,
)


class TestMissingDetection:
    def test_absent_module_is_missing(self, monkeypatch):
        monkeypatch.setitem(
            OPTIONAL_DEPS, "definitely_not_installed",
            Capability("robo_dataset_no_such_module", None, "nothing", "nope"))
        assert "definitely_not_installed" in missing_deps()

    def test_installed_but_too_old_is_missing(self, monkeypatch):
        """The zarr 2.x case: imports fine, silently disables a backend."""
        stub = types.ModuleType("robo_dataset_fake_dep")
        stub.__version__ = "2.18.2"
        monkeypatch.setitem(sys.modules, "robo_dataset_fake_dep", stub)
        monkeypatch.setitem(
            OPTIONAL_DEPS, "fake3",
            Capability("robo_dataset_fake_dep", 3, "fake backend", "fake >= 3"))

        assert "fake3" in missing_deps()
        assert "2.18.2" in describe("fake3") and "need >= 3" in describe("fake3")

    def test_installed_and_new_enough_is_present(self, monkeypatch):
        stub = types.ModuleType("robo_dataset_fake_dep")
        stub.__version__ = "3.3.0"
        monkeypatch.setitem(sys.modules, "robo_dataset_fake_dep", stub)
        monkeypatch.setitem(
            OPTIONAL_DEPS, "fake3",
            Capability("robo_dataset_fake_dep", 3, "fake backend", "fake >= 3"))
        assert "fake3" not in missing_deps()

    def test_unparseable_version_counts_as_missing(self, monkeypatch):
        """Better to over-report than to claim coverage we cannot confirm."""
        stub = types.ModuleType("robo_dataset_fake_dep")
        stub.__version__ = "not-a-version"
        monkeypatch.setitem(sys.modules, "robo_dataset_fake_dep", stub)
        monkeypatch.setitem(
            OPTIONAL_DEPS, "fake3",
            Capability("robo_dataset_fake_dep", 3, "fake backend", "fake >= 3"))
        assert "fake3" in missing_deps()

    def test_real_capabilities_are_all_declared_consistently(self):
        for name, cap in OPTIONAL_DEPS.items():
            assert cap.module and cap.description and cap.skip_token, name
            assert cap.min_major is None or cap.min_major > 0, name


class TestStrictParsing:
    def test_unset_requires_nothing(self, monkeypatch):
        monkeypatch.delenv("ROBO_DATASET_STRICT_DEPS", raising=False)
        assert strictly_required() == []

    @pytest.mark.parametrize("value", ["0", "false", "no", ""])
    def test_falsey_requires_nothing(self, monkeypatch, value):
        monkeypatch.setenv("ROBO_DATASET_STRICT_DEPS", value)
        assert strictly_required() == []

    @pytest.mark.parametrize("value", ["1", "true", "yes", "all", "ALL"])
    def test_truthy_requires_everything(self, monkeypatch, value):
        monkeypatch.setenv("ROBO_DATASET_STRICT_DEPS", value)
        assert set(strictly_required()) == set(OPTIONAL_DEPS)

    def test_subset_is_honoured(self, monkeypatch):
        monkeypatch.setenv("ROBO_DATASET_STRICT_DEPS", "pyarrow, av")
        assert strictly_required() == ["pyarrow", "av"]

    def test_typo_is_rejected_rather_than_ignored(self, monkeypatch):
        """Silently ignoring an unknown name would disable the guard."""
        monkeypatch.setenv("ROBO_DATASET_STRICT_DEPS", "pyarow")
        with pytest.raises(ValueError, match="unknown capabilities"):
            strictly_required()
