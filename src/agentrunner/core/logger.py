"""Structured JSON logging system.

Implements AgentRunnerLogger per INTERFACES/CONFIG_LOG_ERRORS.md.
"""

import json
import logging
import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any


class AgentRunnerLogger:
    """Structured JSON logger with rotation and timing utilities.

    Outputs JSON lines to ~/.agentrunner/logs/agentrunner.log with automatic rotation.
    Supports structured key-value logging and operation timing.
    """

    log_dir: Path | None
    log_file: Path | None

    def __init__(
        self,
        log_dir: str | None = None,
        max_bytes: int = 10 * 1024 * 1024,  # 10MB
        backup_count: int = 5,
        level: str | None = None,
    ) -> None:
        """Initialize logger with rotation.

        Args:
            log_dir: Directory for log files (defaults to ~/.agentrunner/logs/)
            max_bytes: Maximum size before rotation (default 10MB)
            backup_count: Number of backup files to keep (default 5)
            level: Log level (DEBUG/INFO/WARN/ERROR), reads from AGENTRUNNER_LOG_LEVEL env if not provided
        """
        # Check if file logging is disabled via environment variable
        disable_file_logging = os.environ.get("AGENTRUNNER_DISABLE_FILE_LOGGING", "").lower() in (
            "1",
            "true",
            "yes",
        )

        # Configure logger
        self._logger = logging.getLogger("agentrunner")
        self._logger.setLevel(logging.DEBUG)
        self._logger.propagate = False

        for handler in self._logger.handlers[:]:
            handler.close()
            self._logger.removeHandler(handler)

        # Only set up file logging if not disabled
        if not disable_file_logging:
            if log_dir is None:
                self.log_dir = Path("~/.agentrunner/logs").expanduser()
            else:
                self.log_dir = Path(log_dir)

            self.log_dir.mkdir(parents=True, exist_ok=True)
            self.log_file = self.log_dir / "agentrunner.log"

            file_handler = RotatingFileHandler(
                self.log_file,
                maxBytes=max_bytes,
                backupCount=backup_count,
            )
            file_handler.setFormatter(JSONFormatter())
            self._logger.addHandler(file_handler)
        else:
            # No file logging - set attributes to None
            self.log_dir = None
            self.log_file = None

        # Always add console handler (for CloudWatch, stdout, etc.)
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(JSONFormatter())
        self._logger.addHandler(console_handler)

        # Set log level from env or parameter (default to WARNING to reduce verbosity)
        log_level = level or os.environ.get("AGENTRUNNER_LOG_LEVEL", "WARNING")
        self.set_level(log_level)

        # Timers for performance tracking
        self._timers: dict[str, float] = {}

    def set_level(self, level: str) -> None:
        """Set logging level.

        Args:
            level: One of DEBUG, INFO, WARN/WARNING, ERROR
        """
        level_upper = level.upper()
        if level_upper == "WARN":
            level_upper = "WARNING"

        numeric_level = getattr(logging, level_upper, logging.INFO)
        self._logger.setLevel(numeric_level)

    def debug(self, msg: str, **kv: Any) -> None:
        """Log debug message with optional key-value pairs.

        Args:
            msg: Log message
            **kv: Additional key-value pairs to include
        """
        self._logger.debug(msg, extra={"kv": kv})

    def info(self, msg: str, **kv: Any) -> None:
        """Log info message with optional key-value pairs.

        Args:
            msg: Log message
            **kv: Additional key-value pairs to include
        """
        self._logger.info(msg, extra={"kv": kv})

    def warn(self, msg: str, **kv: Any) -> None:
        """Log warning message with optional key-value pairs.

        Args:
            msg: Log message
            **kv: Additional key-value pairs to include
        """
        self._logger.warning(msg, extra={"kv": kv})

    def error(self, msg: str, **kv: Any) -> None:
        """Log error message with optional key-value pairs.

        Args:
            msg: Log message
            **kv: Additional key-value pairs to include
        """
        self._logger.error(msg, extra={"kv": kv})

    @contextmanager
    def operation(self, operation_name: str, **kv: Any) -> Iterator[None]:
        """Context manager for operation timing.

        Automatically logs operation start and end with duration.

        Args:
            operation_name: Name of the operation
            **kv: Additional key-value pairs to include

        Example:
            with logger.operation("tool_execution", tool_name="read_file"):
                # ... operation code ...
                pass
        """
        start_time = time.time()
        self.info(f"{operation_name}_start", **kv)

        try:
            yield
        finally:
            duration_ms = (time.time() - start_time) * 1000
            self.info(f"{operation_name}_end", duration_ms=duration_ms, **kv)

    def start_timer(self, label: str) -> None:
        """Start a named timer for performance tracking.

        Args:
            label: Timer label/name
        """
        self._timers[label] = time.time()

    def end_timer(self, label: str, **kv: Any) -> float:
        """End a named timer and log the duration.

        Args:
            label: Timer label/name
            **kv: Additional key-value pairs to include in log

        Returns:
            Duration in milliseconds

        Raises:
            KeyError: If timer was not started
        """
        if label not in self._timers:
            raise KeyError(f"Timer '{label}' not started")

        start_time = self._timers.pop(label)
        duration_ms = (time.time() - start_time) * 1000

        self.debug(f"timer_{label}", duration_ms=duration_ms, **kv)

        return duration_ms


class JSONFormatter(logging.Formatter):
    """Formats log records as JSON lines."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON.

        Args:
            record: Log record to format

        Returns:
            JSON string
        """
        # Create timestamp with milliseconds
        dt = datetime.fromtimestamp(record.created, tz=UTC)
        timestamp = dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"

        log_data = {
            "timestamp": timestamp,
            "level": record.levelname,
            "message": record.getMessage(),
        }

        # Add key-value pairs if present
        if hasattr(record, "kv") and record.kv:
            log_data.update(record.kv)

        return json.dumps(log_data)
