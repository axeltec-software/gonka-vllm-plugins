"""Plugin INFO logs must reach vLLM's log handler.

Why they did not: see ``_adopt_vllm_logging``. CPU-only — builds a
vLLM-shaped logging tree with the stdlib; no vllm import.
"""

from __future__ import annotations

import logging

import pytest

from gonka_poc.plugin import _adopt_vllm_logging


@pytest.fixture()
def clean_loggers():
    """Snapshot and restore the two loggers the adoption touches."""
    saved = {}
    for name in ("vllm", "gonka_poc"):
        lg = logging.getLogger(name)
        saved[name] = (lg.handlers[:], lg.level, lg.propagate)
    yield
    for name, (handlers, level, propagate) in saved.items():
        lg = logging.getLogger(name)
        lg.handlers = handlers
        lg.level = level
        lg.propagate = propagate


def _configure_vllm_style_logging(records: list[logging.LogRecord]) -> None:
    """The shape DEFAULT_LOGGING_CONFIG produces: one handler, INFO, no propagation."""
    handler = logging.Handler()
    handler.emit = records.append
    vllm_logger = logging.getLogger("vllm")
    vllm_logger.handlers = [handler]
    vllm_logger.setLevel(logging.INFO)
    vllm_logger.propagate = False


def test_plugin_info_reaches_vllm_handler(clean_loggers) -> None:
    records: list[logging.LogRecord] = []
    _configure_vllm_style_logging(records)

    _adopt_vllm_logging()
    logging.getLogger("gonka_poc.poc.routes").info("Generated: 5000 nonces (1500/min)")

    assert [r.getMessage() for r in records] == ["Generated: 5000 nonces (1500/min)"]


def test_no_double_emission_when_root_has_a_handler(clean_loggers) -> None:
    """Adopted records must not also propagate to root once it gains a handler."""
    vllm_records: list[logging.LogRecord] = []
    _configure_vllm_style_logging(vllm_records)
    root_records: list[logging.LogRecord] = []
    root_handler = logging.Handler()
    root_handler.emit = root_records.append
    logging.getLogger().addHandler(root_handler)
    try:
        _adopt_vllm_logging()
        logging.getLogger("gonka_poc.poc.routes").info("once")
    finally:
        logging.getLogger().removeHandler(root_handler)

    assert len(vllm_records) == 1
    assert root_records == []


def test_unconfigured_vllm_logging_is_left_alone(clean_loggers) -> None:
    """VLLM_CONFIGURE_LOGGING=0: the operator owns the tree; do not detach us from it."""
    vllm_logger = logging.getLogger("vllm")
    vllm_logger.handlers = []
    before = logging.getLogger("gonka_poc")
    handlers, level, propagate = before.handlers[:], before.level, before.propagate

    _adopt_vllm_logging()

    after = logging.getLogger("gonka_poc")
    assert (after.handlers, after.level, after.propagate) == (handlers, level, propagate)

