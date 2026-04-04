"""Tests for structured logging — Gap 35."""

import json
import logging
import re

import pytest

from src.utils import logging as vip_log


@pytest.fixture(autouse=True)
def _reset_logging():
    """Reset module-level state between tests."""
    vip_log._configured = False
    vip_log._run_id.set("")
    # Remove all handlers from root logger added by configure_logging
    root = logging.getLogger()
    original_handlers = root.handlers[:]
    yield
    root.handlers = original_handlers
    vip_log._configured = False
    vip_log._run_id.set("")


def test_run_id_format():
    rid = vip_log._new_run_id()
    assert re.match(r"\d{8}-\d{6}-[0-9a-f]{4}$", rid)


def test_set_and_get_run_id():
    rid = vip_log.set_run_id("test-1234")
    assert rid == "test-1234"
    assert vip_log.get_run_id() == "test-1234"


def test_set_run_id_auto_generates():
    rid = vip_log.set_run_id()
    assert len(rid) > 0
    assert re.match(r"\d{8}-\d{6}-[0-9a-f]{4}$", rid)


def test_configure_logging_json(capfd):
    """JSON format should produce valid JSON with required keys."""
    vip_log.configure_logging(json_format=True, run_id="json-test")
    logger = logging.getLogger("test.json")
    logger.info("hello world")

    captured = capfd.readouterr()
    line = captured.out.strip().split("\n")[-1]
    obj = json.loads(line)
    assert obj["level"] == "INFO"
    assert obj["msg"] == "hello world"
    assert obj["run_id"] == "json-test"
    assert "ts" in obj
    assert "logger" in obj


def test_configure_logging_pretty(capfd):
    """Pretty format should include run_id bracket and human-readable layout."""
    vip_log.configure_logging(json_format=False, run_id="pretty-test")
    logger = logging.getLogger("test.pretty")
    logger.info("hello world")

    captured = capfd.readouterr()
    assert "[pretty-test]" in captured.out
    assert "hello world" in captured.out


def test_configure_logging_idempotent():
    """Second call should still set run_id but not duplicate handlers."""
    vip_log.configure_logging(run_id="first")
    handler_count = len(logging.getLogger().handlers)
    vip_log.configure_logging(run_id="second")
    assert len(logging.getLogger().handlers) == handler_count
    assert vip_log.get_run_id() == "second"
