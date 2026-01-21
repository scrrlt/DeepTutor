# -*- coding: utf-8 -*-
"""Tests for suppressed_logging context manager."""

import logging
from src.logging.logger import suppressed_logging


def test_suppressed_logging_filters_non_critical(caplog):
    logger_name = "test.suppress"
    logger = logging.getLogger(logger_name)
    # Ensure logger propagates to root so caplog can capture
    logger.propagate = True
    # Ensure caplog captures INFO/ERROR level logs
    caplog.set_level(logging.INFO)

    # Emit an INFO log outside context
    logger.info("info-outside")

    with suppressed_logging([logger_name], level=logging.CRITICAL):
        logger.info("info-inside")
        logger.error("error-inside")
        logger.critical("crit-inside")

    # Emit another INFO outside to ensure normal behavior restored
    logger.info("info-after")

    messages = [rec.getMessage() for rec in caplog.records]

    # info-inside and error-inside should be suppressed; crit-inside should appear
    assert not any("info-inside" in m for m in messages)
    assert not any("error-inside" in m for m in messages)
    assert any("crit-inside" in m for m in messages)
