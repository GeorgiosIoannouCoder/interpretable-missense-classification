"""Minimal smoke tests for the package skeleton."""

from __future__ import annotations

import importlib


def test_package_imports() -> None:
    """Top-level package and all subpackages should import cleanly."""
    for name in [
        "imc",
        "imc.data",
        "imc.features",
        "imc.models",
        "imc.training",
        "imc.eval",
        "imc.viz",
        "imc.utils",
        "imc.utils.seed",
        "imc.utils.logging",
        "imc.utils.io",
    ]:
        module = importlib.import_module(name)
        assert module is not None


def test_set_seed_runs() -> None:
    """``set_seed`` should not raise on the default seed."""
    from imc.utils.seed import set_seed

    set_seed(42)


def test_get_logger_runs(tmp_path) -> None:
    """``get_logger`` should produce a logger that can write to a file."""
    from imc.utils.logging import get_logger

    log_file = tmp_path / "test.log"
    logger = get_logger("imc.tests.test_smoke", log_file=log_file)
    logger.info("hello")
    assert log_file.exists()
