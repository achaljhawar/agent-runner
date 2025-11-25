"""Unit tests for structured JSON logging system."""

import json
import logging
import time
from pathlib import Path

import pytest

from agentrunner.core.logger import AgentRunnerLogger, JSONFormatter


@pytest.fixture
def temp_log_dir(tmp_path):
    """Create temporary log directory."""
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    return log_dir


@pytest.fixture
def logger(temp_log_dir):
    """Create logger with temporary directory."""
    return AgentRunnerLogger(log_dir=str(temp_log_dir), level="DEBUG")


def read_log_lines(log_file):
    """Read and parse JSON log lines."""
    if not log_file.exists():
        return []

    lines = []
    with log_file.open() as f:
        for line in f:
            if line.strip():
                lines.append(json.loads(line))
    return lines


def test_logger_initialization(temp_log_dir):
    """Test logger initialization creates log directory and file."""
    logger = AgentRunnerLogger(log_dir=str(temp_log_dir))

    assert temp_log_dir.exists()
    assert logger.log_file.parent == temp_log_dir


def test_logger_default_directory(tmp_path, monkeypatch):
    """Test logger uses default ~/.agentrunner/logs directory."""
    # Mock HOME environment variable to use tmp_path for isolation
    fake_home = tmp_path / "fake_home"
    fake_home.mkdir()
    monkeypatch.setenv("HOME", str(fake_home))

    logger = AgentRunnerLogger()

    # Use expanduser() to match the logger's behavior
    expected_dir = Path("~/.agentrunner/logs").expanduser()
    assert logger.log_dir == expected_dir
    assert expected_dir.exists()  # Verify directory was created


def test_info_logging(logger, temp_log_dir):
    """Test info level logging."""
    logger.info("Test message")

    # Flush handlers to ensure log is written
    for handler in logger._logger.handlers:
        handler.flush()

    lines = read_log_lines(logger.log_file)
    assert len(lines) == 1
    assert lines[0]["level"] == "INFO"
    assert lines[0]["message"] == "Test message"
    assert "timestamp" in lines[0]


def test_warn_logging(logger):
    """Test warning level logging."""
    logger.warn("Warning message")

    lines = read_log_lines(logger.log_file)
    assert len(lines) == 1
    assert lines[0]["level"] == "WARNING"
    assert lines[0]["message"] == "Warning message"


def test_error_logging(logger):
    """Test error level logging."""
    logger.error("Error message")

    lines = read_log_lines(logger.log_file)
    assert len(lines) == 1
    assert lines[0]["level"] == "ERROR"
    assert lines[0]["message"] == "Error message"


def test_debug_logging(logger):
    """Test debug level logging (should be visible by default)."""
    logger.set_level("DEBUG")
    logger.debug("Debug message")

    lines = read_log_lines(logger.log_file)
    assert len(lines) == 1
    assert lines[0]["level"] == "DEBUG"
    assert lines[0]["message"] == "Debug message"


def test_structured_logging_with_kv_pairs(logger):
    """Test structured logging with key-value pairs."""
    logger.info("Tool executed", tool_name="read_file", duration_ms=123)

    # Flush handlers to ensure log is written
    for handler in logger._logger.handlers:
        handler.flush()

    lines = read_log_lines(logger.log_file)
    assert len(lines) == 1
    assert lines[0]["message"] == "Tool executed"
    assert lines[0]["tool_name"] == "read_file"
    assert lines[0]["duration_ms"] == 123


def test_log_level_filtering(logger):
    """Test log level filtering."""
    logger.set_level("ERROR")

    logger.debug("Debug message")
    logger.info("Info message")
    logger.warn("Warning message")
    logger.error("Error message")

    lines = read_log_lines(logger.log_file)
    assert len(lines) == 1
    assert lines[0]["level"] == "ERROR"


def test_log_level_from_env(temp_log_dir, monkeypatch):
    """Test log level configuration from AGENTRUNNER_LOG_LEVEL environment variable."""
    monkeypatch.setenv("AGENTRUNNER_LOG_LEVEL", "ERROR")

    logger = AgentRunnerLogger(log_dir=str(temp_log_dir))
    logger.info("Info message")
    logger.error("Error message")

    lines = read_log_lines(logger.log_file)
    assert len(lines) == 1
    assert lines[0]["level"] == "ERROR"


def test_operation_context_manager(logger):
    """Test operation context manager with automatic timing."""
    with logger.operation("tool_execution", tool_name="read_file"):
        time.sleep(0.01)  # Simulate work

    # Flush handlers to ensure log is written
    for handler in logger._logger.handlers:
        handler.flush()

    lines = read_log_lines(logger.log_file)
    assert len(lines) == 2

    # Check start message
    assert lines[0]["message"] == "tool_execution_start"
    assert lines[0]["tool_name"] == "read_file"

    # Check end message
    assert lines[1]["message"] == "tool_execution_end"
    assert lines[1]["tool_name"] == "read_file"
    assert "duration_ms" in lines[1]
    assert lines[1]["duration_ms"] > 0


def test_operation_context_manager_with_exception(logger):
    """Test operation context manager logs end even on exception."""
    try:
        with logger.operation("failing_operation"):
            raise ValueError("Test error")
    except ValueError:
        pass

    # Flush handlers to ensure log is written
    for handler in logger._logger.handlers:
        handler.flush()

    lines = read_log_lines(logger.log_file)
    assert len(lines) == 2
    assert lines[0]["message"] == "failing_operation_start"
    assert lines[1]["message"] == "failing_operation_end"
    assert "duration_ms" in lines[1]


def test_start_timer_and_end_timer(logger):
    """Test manual timer functionality."""
    logger.set_level("DEBUG")

    logger.start_timer("test_operation")
    time.sleep(0.01)
    duration = logger.end_timer("test_operation", context="test")

    assert duration > 0

    lines = read_log_lines(logger.log_file)
    assert len(lines) == 1
    assert lines[0]["message"] == "timer_test_operation"
    assert lines[0]["duration_ms"] == duration
    assert lines[0]["context"] == "test"


def test_end_timer_without_start_raises_error(logger):
    """Test that ending a timer that wasn't started raises KeyError."""
    with pytest.raises(KeyError, match="Timer 'nonexistent' not started"):
        logger.end_timer("nonexistent")


def test_multiple_timers(logger):
    """Test multiple concurrent timers."""
    logger.set_level("DEBUG")

    logger.start_timer("timer1")
    logger.start_timer("timer2")
    time.sleep(0.01)

    duration1 = logger.end_timer("timer1")
    duration2 = logger.end_timer("timer2")

    assert duration1 > 0
    assert duration2 > 0


def test_json_formatter():
    """Test JSON formatter formats records correctly."""
    formatter = JSONFormatter()

    # Create a mock log record
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname="",
        lineno=0,
        msg="Test message",
        args=(),
        exc_info=None,
    )
    record.kv = {"key1": "value1", "key2": 42}

    result = formatter.format(record)
    data = json.loads(result)

    assert data["level"] == "INFO"
    assert data["message"] == "Test message"
    assert data["key1"] == "value1"
    assert data["key2"] == 42
    assert "timestamp" in data


def test_log_rotation_creates_backup(temp_log_dir):
    """Test log rotation creates backup files."""
    # Create logger with very small max_bytes to trigger rotation
    logger = AgentRunnerLogger(
        log_dir=str(temp_log_dir),
        max_bytes=100,  # Very small to trigger rotation
        backup_count=2,
        level="DEBUG",
    )

    # Write enough logs to trigger rotation
    for i in range(50):
        logger.info(f"Message {i}" * 10)  # Long messages

    # Flush handlers to ensure log is written
    for handler in logger._logger.handlers:
        handler.flush()

    # Check that backup file(s) were created
    log_files = list(temp_log_dir.glob("agentrunner.log*"))
    assert len(log_files) > 1  # Should have main log + at least one backup


def test_set_level_with_warn_alias(logger):
    """Test that 'WARN' is accepted as alias for 'WARNING'."""
    logger.set_level("WARN")

    logger.debug("Debug")
    logger.info("Info")
    logger.warn("Warning")

    lines = read_log_lines(logger.log_file)
    assert len(lines) == 1
    assert lines[0]["level"] == "WARNING"


def test_multiple_log_calls(logger):
    """Test multiple log calls are all written."""
    logger.info("Message 1", id=1)
    logger.info("Message 2", id=2)
    logger.warn("Message 3", id=3)
    logger.error("Message 4", id=4)

    # Flush handlers to ensure log is written
    for handler in logger._logger.handlers:
        handler.flush()

    lines = read_log_lines(logger.log_file)
    assert len(lines) == 4
    assert lines[0]["id"] == 1
    assert lines[1]["id"] == 2
    assert lines[2]["id"] == 3
    assert lines[3]["id"] == 4


def test_timestamp_format(logger):
    """Test that timestamp is in correct ISO format."""
    logger.info("Test")

    # Flush handlers to ensure log is written
    for handler in logger._logger.handlers:
        handler.flush()

    lines = read_log_lines(logger.log_file)
    timestamp = lines[0]["timestamp"]

    # Check format: YYYY-MM-DDTHH:MM:SS.MMMZ
    assert timestamp.endswith("Z")
    assert "T" in timestamp
    assert len(timestamp) == 24  # Fixed length format


def test_logger_with_empty_kv(logger):
    """Test logging without key-value pairs works correctly."""
    logger.info("Simple message")

    # Flush handlers to ensure log is written
    for handler in logger._logger.handlers:
        handler.flush()

    lines = read_log_lines(logger.log_file)
    assert len(lines) == 1
    assert lines[0]["message"] == "Simple message"
    assert lines[0]["level"] == "INFO"
    assert "timestamp" in lines[0]


def test_disable_file_logging_via_env(monkeypatch):
    """Test that AGENTRUNNER_DISABLE_FILE_LOGGING environment variable disables file logging."""
    monkeypatch.setenv("AGENTRUNNER_DISABLE_FILE_LOGGING", "1")

    logger = AgentRunnerLogger()

    # Should have None for log_dir and log_file when file logging is disabled
    assert logger.log_dir is None
    assert logger.log_file is None

    # Should still be able to log (to console)
    logger.info("Test message")


def test_file_logging_enabled_by_default(temp_log_dir, monkeypatch):
    """Test that file logging is enabled by default when not disabled via env."""
    # Ensure the env var is not set
    monkeypatch.delenv("AGENTRUNNER_DISABLE_FILE_LOGGING", raising=False)

    logger = AgentRunnerLogger(log_dir=str(temp_log_dir), level="INFO")

    # Should have valid log_dir and log_file when file logging is enabled
    assert logger.log_dir is not None
    assert logger.log_file is not None
    assert logger.log_dir == temp_log_dir

    # Should be able to write to file
    logger.info("Test message")
    for handler in logger._logger.handlers:
        handler.flush()

    lines = read_log_lines(logger.log_file)
    assert len(lines) == 1
