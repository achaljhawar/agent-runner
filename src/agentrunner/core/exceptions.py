"""Exception hierarchy with error codes for Agent Runner.

Implements structured error handling with error codes that map to tool protocol standards.
See INTERFACES/CONFIG_LOG_ERRORS.md and INTERFACES/TOOL_PROTOCOL.md for specifications.
"""

from dataclasses import dataclass, field
from typing import Any

# Standard error codes from Tool Protocol
E_NOT_FOUND = "E_NOT_FOUND"
E_NOT_UNIQUE = "E_NOT_UNIQUE"
E_VALIDATION = "E_VALIDATION"
E_PERMISSIONS = "E_PERMISSIONS"
E_TIMEOUT = "E_TIMEOUT"
E_UNSAFE = "E_UNSAFE"
E_CONFIRMED_DENY = "E_CONFIRMED_DENY"
E_TOOL_UNKNOWN = "E_TOOL_UNKNOWN"


@dataclass
class AgentRunnerException(Exception):  # noqa: N818
    """Base exception for all Agent Runner-specific errors.

    Provides structured error handling with error codes and metadata for
    consistent error reporting across the system.
    """

    message: str
    error_code: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        """Return human-readable error message."""
        return self.message

    def __post_init__(self) -> None:
        """Initialize the exception with the message."""
        super().__init__(self.message)


@dataclass
class ToolExecutionError(AgentRunnerException):
    """Error during tool execution.

    Raised when a tool fails to execute properly, with specific error code
    and tool information for debugging.
    """

    tool_name: str = ""
    details: str | None = None

    def __post_init__(self) -> None:
        """Initialize with tool-specific metadata."""
        if self.tool_name:
            self.metadata["tool_name"] = self.tool_name
        if self.details:
            self.metadata["details"] = self.details
        super().__post_init__()


@dataclass
class WorkspaceSecurityError(AgentRunnerException):
    """Error when workspace security constraints are violated.

    Raised for path traversal attempts, access outside workspace root,
    and other security violations.
    """

    path: str = ""
    reason: str = ""

    def __post_init__(self) -> None:
        """Initialize with security-specific metadata."""
        if not self.error_code:
            self.error_code = E_PERMISSIONS
        if self.path:
            self.metadata["path"] = self.path
        if self.reason:
            self.metadata["reason"] = self.reason
        super().__post_init__()


@dataclass
class TokenLimitExceededError(AgentRunnerException):
    """Error when token limits are exceeded.

    Raised when operations would exceed model context limits or
    configured token budgets.
    """

    current: int = 0
    limit: int = 0

    def __post_init__(self) -> None:
        """Initialize with token count metadata."""
        if not self.error_code:
            self.error_code = E_VALIDATION
        if self.current > 0:
            self.metadata["current_tokens"] = self.current
        if self.limit > 0:
            self.metadata["token_limit"] = self.limit
        super().__post_init__()


@dataclass
class ModelResponseError(AgentRunnerException):
    """Error from LLM provider responses.

    Raised when providers return errors, timeouts, or invalid responses.
    """

    provider: str = ""
    status_code: int | None = None

    def __post_init__(self) -> None:
        """Initialize with provider-specific metadata."""
        if not self.error_code:
            self.error_code = E_TIMEOUT
        if self.provider:
            self.metadata["provider"] = self.provider
        if self.status_code is not None:
            self.metadata["status_code"] = self.status_code
        super().__post_init__()


@dataclass
class ConfigurationError(AgentRunnerException):
    """Error in system configuration.

    Raised for invalid config values, missing required settings,
    or configuration file problems.
    """

    key: str = ""
    reason: str = ""

    def __post_init__(self) -> None:
        """Initialize with configuration-specific metadata."""
        if not self.error_code:
            self.error_code = E_VALIDATION
        if self.key:
            self.metadata["config_key"] = self.key
        if self.reason:
            self.metadata["reason"] = self.reason
        super().__post_init__()


def format_error_for_user(exception: AgentRunnerException) -> str:
    """Format exception for user-friendly display.

    Args:
        exception: The agentrunner exception to format

    Returns:
        Human-readable error message without internal details
    """
    if isinstance(exception, ToolExecutionError):
        if exception.tool_name:
            return f"Tool '{exception.tool_name}' failed: {exception.message}"
        return f"Tool execution failed: {exception.message}"

    if isinstance(exception, WorkspaceSecurityError):
        if exception.path:
            return f"Security error with path '{exception.path}': {exception.message}"
        return f"Workspace security error: {exception.message}"

    if isinstance(exception, TokenLimitExceededError):
        if exception.current and exception.limit:
            return (
                f"Token limit exceeded ({exception.current}/{exception.limit}): "
                f"{exception.message}"
            )
        return f"Token limit exceeded: {exception.message}"

    if isinstance(exception, ModelResponseError):
        if exception.provider:
            return f"Provider '{exception.provider}' error: {exception.message}"
        return f"Model response error: {exception.message}"

    if isinstance(exception, ConfigurationError):
        if exception.key:
            return f"Configuration error '{exception.key}': {exception.message}"
        return f"Configuration error: {exception.message}"

    return str(exception.message)


def format_error_for_log(exception: AgentRunnerException) -> dict[str, Any]:
    """Format exception for structured logging.

    Args:
        exception: The agentrunner exception to format

    Returns:
        Dictionary with structured error information for logs
    """
    log_data: dict[str, Any] = {
        "error_type": type(exception).__name__,
        "message": exception.message,
        "error_code": exception.error_code,
    }

    # Include all metadata
    if exception.metadata:
        log_data["metadata"] = exception.metadata

    # Add exception-specific fields
    if isinstance(exception, ToolExecutionError):
        if exception.tool_name:
            log_data["tool_name"] = exception.tool_name
        if exception.details:
            log_data["details"] = exception.details

    elif isinstance(exception, WorkspaceSecurityError):
        if exception.path:
            log_data["path"] = exception.path
        if exception.reason:
            log_data["reason"] = exception.reason

    elif isinstance(exception, TokenLimitExceededError):
        if exception.current:
            log_data["current_tokens"] = exception.current
        if exception.limit:
            log_data["token_limit"] = exception.limit

    elif isinstance(exception, ModelResponseError):
        if exception.provider:
            log_data["provider"] = exception.provider
        if exception.status_code is not None:
            log_data["status_code"] = exception.status_code

    elif isinstance(exception, ConfigurationError):
        if exception.key:
            log_data["config_key"] = exception.key
        if exception.reason:
            log_data["reason"] = exception.reason

    return log_data
