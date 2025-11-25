"""Unit tests for exception hierarchy and error handling.

Tests all exception types, error codes, formatting utilities, and edge cases.
Targets >90% coverage as specified in TASK_PROMPTS.md.
"""

import pytest

from agentrunner.core.exceptions import (
    E_CONFIRMED_DENY,
    E_NOT_FOUND,
    E_NOT_UNIQUE,
    E_PERMISSIONS,
    E_TIMEOUT,
    E_TOOL_UNKNOWN,
    E_UNSAFE,
    E_VALIDATION,
    AgentRunnerException,
    ConfigurationError,
    ModelResponseError,
    TokenLimitExceededError,
    ToolExecutionError,
    WorkspaceSecurityError,
    format_error_for_log,
    format_error_for_user,
)


class TestAgentRunnerException:
    """Test the base AgentRunnerException class."""

    def test_basic_creation(self):
        """Test basic exception creation with message."""
        exc = AgentRunnerException("Test error")
        assert exc.message == "Test error"
        assert exc.error_code is None
        assert exc.metadata == {}
        assert str(exc) == "Test error"

    def test_with_error_code(self):
        """Test exception with error code."""
        exc = AgentRunnerException("Test error", error_code=E_VALIDATION)
        assert exc.message == "Test error"
        assert exc.error_code == E_VALIDATION
        assert exc.metadata == {}

    def test_with_metadata(self):
        """Test exception with metadata."""
        metadata = {"key": "value", "count": 42}
        exc = AgentRunnerException("Test error", metadata=metadata)
        assert exc.message == "Test error"
        assert exc.metadata == metadata

    def test_full_initialization(self):
        """Test exception with all fields."""
        metadata = {"detail": "extra info"}
        exc = AgentRunnerException("Test error", error_code=E_NOT_FOUND, metadata=metadata)
        assert exc.message == "Test error"
        assert exc.error_code == E_NOT_FOUND
        assert exc.metadata == metadata

    def test_exception_behavior(self):
        """Test that it behaves like a standard exception."""
        exc = AgentRunnerException("Test error")
        with pytest.raises(AgentRunnerException) as exc_info:
            raise exc
        assert str(exc_info.value) == "Test error"


class TestToolExecutionError:
    """Test the ToolExecutionError class."""

    def test_basic_creation(self):
        """Test basic tool error creation."""
        exc = ToolExecutionError("Tool failed")
        assert exc.message == "Tool failed"
        assert exc.tool_name == ""
        assert exc.details is None
        assert exc.error_code is None

    def test_with_tool_name(self):
        """Test tool error with tool name."""
        exc = ToolExecutionError("Tool failed", tool_name="read_file")
        assert exc.message == "Tool failed"
        assert exc.tool_name == "read_file"
        assert exc.metadata["tool_name"] == "read_file"

    def test_with_details(self):
        """Test tool error with details."""
        exc = ToolExecutionError("Tool failed", details="File not found")
        assert exc.message == "Tool failed"
        assert exc.details == "File not found"
        assert exc.metadata["details"] == "File not found"

    def test_with_all_fields(self):
        """Test tool error with all fields."""
        exc = ToolExecutionError(
            "Tool failed", error_code=E_NOT_FOUND, tool_name="read_file", details="File not found"
        )
        assert exc.message == "Tool failed"
        assert exc.error_code == E_NOT_FOUND
        assert exc.tool_name == "read_file"
        assert exc.details == "File not found"
        assert exc.metadata["tool_name"] == "read_file"
        assert exc.metadata["details"] == "File not found"


class TestWorkspaceSecurityError:
    """Test the WorkspaceSecurityError class."""

    def test_basic_creation(self):
        """Test basic security error creation."""
        exc = WorkspaceSecurityError("Security violation")
        assert exc.message == "Security violation"
        assert exc.path == ""
        assert exc.reason == ""
        assert exc.error_code == E_PERMISSIONS  # Auto-assigned

    def test_with_path(self):
        """Test security error with path."""
        exc = WorkspaceSecurityError("Path outside workspace", path="/etc/passwd")
        assert exc.message == "Path outside workspace"
        assert exc.path == "/etc/passwd"
        assert exc.metadata["path"] == "/etc/passwd"
        assert exc.error_code == E_PERMISSIONS

    def test_with_reason(self):
        """Test security error with reason."""
        exc = WorkspaceSecurityError("Security violation", reason="Path traversal attempt")
        assert exc.message == "Security violation"
        assert exc.reason == "Path traversal attempt"
        assert exc.metadata["reason"] == "Path traversal attempt"

    def test_with_custom_error_code(self):
        """Test security error can override default error code."""
        exc = WorkspaceSecurityError("Security violation", error_code=E_UNSAFE)
        assert exc.error_code == E_UNSAFE


class TestTokenLimitExceededError:
    """Test the TokenLimitExceededError class."""

    def test_basic_creation(self):
        """Test basic token limit error creation."""
        exc = TokenLimitExceededError("Token limit exceeded")
        assert exc.message == "Token limit exceeded"
        assert exc.current == 0
        assert exc.limit == 0
        assert exc.error_code == E_VALIDATION  # Auto-assigned

    def test_with_token_counts(self):
        """Test token limit error with counts."""
        exc = TokenLimitExceededError("Too many tokens", current=5000, limit=4096)
        assert exc.message == "Too many tokens"
        assert exc.current == 5000
        assert exc.limit == 4096
        assert exc.metadata["current_tokens"] == 5000
        assert exc.metadata["token_limit"] == 4096
        assert exc.error_code == E_VALIDATION

    def test_with_custom_error_code(self):
        """Test token limit error can override default error code."""
        exc = TokenLimitExceededError("Token limit exceeded", error_code=E_TIMEOUT)
        assert exc.error_code == E_TIMEOUT


class TestModelResponseError:
    """Test the ModelResponseError class."""

    def test_basic_creation(self):
        """Test basic model response error creation."""
        exc = ModelResponseError("Provider failed")
        assert exc.message == "Provider failed"
        assert exc.provider == ""
        assert exc.status_code is None
        assert exc.error_code == E_TIMEOUT  # Auto-assigned

    def test_with_provider(self):
        """Test model error with provider."""
        exc = ModelResponseError("API failed", provider="openai")
        assert exc.message == "API failed"
        assert exc.provider == "openai"
        assert exc.metadata["provider"] == "openai"

    def test_with_status_code(self):
        """Test model error with status code."""
        exc = ModelResponseError("HTTP error", status_code=429)
        assert exc.message == "HTTP error"
        assert exc.status_code == 429
        assert exc.metadata["status_code"] == 429

    def test_with_all_fields(self):
        """Test model error with all fields."""
        exc = ModelResponseError(
            "Rate limited", error_code=E_TIMEOUT, provider="openai", status_code=429
        )
        assert exc.message == "Rate limited"
        assert exc.error_code == E_TIMEOUT
        assert exc.provider == "openai"
        assert exc.status_code == 429
        assert exc.metadata["provider"] == "openai"
        assert exc.metadata["status_code"] == 429


class TestConfigurationError:
    """Test the ConfigurationError class."""

    def test_basic_creation(self):
        """Test basic configuration error creation."""
        exc = ConfigurationError("Invalid config")
        assert exc.message == "Invalid config"
        assert exc.key == ""
        assert exc.reason == ""
        assert exc.error_code == E_VALIDATION  # Auto-assigned

    def test_with_key(self):
        """Test config error with key."""
        exc = ConfigurationError("Invalid value", key="model.temperature")
        assert exc.message == "Invalid value"
        assert exc.key == "model.temperature"
        assert exc.metadata["config_key"] == "model.temperature"

    def test_with_reason(self):
        """Test config error with reason."""
        exc = ConfigurationError("Invalid config", reason="Value out of range")
        assert exc.message == "Invalid config"
        assert exc.reason == "Value out of range"
        assert exc.metadata["reason"] == "Value out of range"

    def test_with_all_fields(self):
        """Test config error with all fields."""
        exc = ConfigurationError(
            "Invalid temperature",
            error_code=E_VALIDATION,
            key="model.temperature",
            reason="Must be between 0 and 2",
        )
        assert exc.message == "Invalid temperature"
        assert exc.error_code == E_VALIDATION
        assert exc.key == "model.temperature"
        assert exc.reason == "Must be between 0 and 2"
        assert exc.metadata["config_key"] == "model.temperature"
        assert exc.metadata["reason"] == "Must be between 0 and 2"


class TestErrorFormattingForUser:
    """Test the format_error_for_user function."""

    def test_tool_execution_error(self):
        """Test formatting ToolExecutionError for users."""
        exc = ToolExecutionError("Failed to read", tool_name="read_file")
        result = format_error_for_user(exc)
        assert result == "Tool 'read_file' failed: Failed to read"

    def test_tool_execution_error_no_name(self):
        """Test formatting ToolExecutionError without tool name."""
        exc = ToolExecutionError("Failed to execute")
        result = format_error_for_user(exc)
        assert result == "Tool execution failed: Failed to execute"

    def test_workspace_security_error(self):
        """Test formatting WorkspaceSecurityError for users."""
        exc = WorkspaceSecurityError("Path not allowed", path="/etc/passwd")
        result = format_error_for_user(exc)
        assert result == "Security error with path '/etc/passwd': Path not allowed"

    def test_workspace_security_error_no_path(self):
        """Test formatting WorkspaceSecurityError without path."""
        exc = WorkspaceSecurityError("Security violation")
        result = format_error_for_user(exc)
        assert result == "Workspace security error: Security violation"

    def test_token_limit_exceeded_error(self):
        """Test formatting TokenLimitExceededError for users."""
        exc = TokenLimitExceededError("Too many tokens", current=5000, limit=4096)
        result = format_error_for_user(exc)
        assert result == "Token limit exceeded (5000/4096): Too many tokens"

    def test_token_limit_exceeded_error_no_counts(self):
        """Test formatting TokenLimitExceededError without counts."""
        exc = TokenLimitExceededError("Too many tokens")
        result = format_error_for_user(exc)
        assert result == "Token limit exceeded: Too many tokens"

    def test_model_response_error(self):
        """Test formatting ModelResponseError for users."""
        exc = ModelResponseError("API failed", provider="openai")
        result = format_error_for_user(exc)
        assert result == "Provider 'openai' error: API failed"

    def test_model_response_error_no_provider(self):
        """Test formatting ModelResponseError without provider."""
        exc = ModelResponseError("API failed")
        result = format_error_for_user(exc)
        assert result == "Model response error: API failed"

    def test_configuration_error(self):
        """Test formatting ConfigurationError for users."""
        exc = ConfigurationError("Invalid value", key="temperature")
        result = format_error_for_user(exc)
        assert result == "Configuration error 'temperature': Invalid value"

    def test_configuration_error_no_key(self):
        """Test formatting ConfigurationError without key."""
        exc = ConfigurationError("Invalid config")
        result = format_error_for_user(exc)
        assert result == "Configuration error: Invalid config"

    def test_base_agentrunner_exception(self):
        """Test formatting base AgentRunnerException for users."""
        exc = AgentRunnerException("Generic error")
        result = format_error_for_user(exc)
        assert result == "Generic error"


class TestErrorFormattingForLog:
    """Test the format_error_for_log function."""

    def test_base_agentrunner_exception(self):
        """Test logging format for base AgentRunnerException."""
        exc = AgentRunnerException("Test error", error_code=E_VALIDATION, metadata={"key": "value"})
        result = format_error_for_log(exc)

        expected = {
            "error_type": "AgentRunnerException",
            "message": "Test error",
            "error_code": E_VALIDATION,
            "metadata": {"key": "value"},
        }
        assert result == expected

    def test_tool_execution_error(self):
        """Test logging format for ToolExecutionError."""
        exc = ToolExecutionError("Tool failed", tool_name="read_file", details="File not found")
        result = format_error_for_log(exc)

        assert result["error_type"] == "ToolExecutionError"
        assert result["message"] == "Tool failed"
        assert result["tool_name"] == "read_file"
        assert result["details"] == "File not found"
        assert "tool_name" in result["metadata"]
        assert "details" in result["metadata"]

    def test_workspace_security_error(self):
        """Test logging format for WorkspaceSecurityError."""
        exc = WorkspaceSecurityError(
            "Path violation", path="/etc/passwd", reason="Outside workspace"
        )
        result = format_error_for_log(exc)

        assert result["error_type"] == "WorkspaceSecurityError"
        assert result["message"] == "Path violation"
        assert result["path"] == "/etc/passwd"
        assert result["reason"] == "Outside workspace"
        assert result["error_code"] == E_PERMISSIONS

    def test_token_limit_exceeded_error(self):
        """Test logging format for TokenLimitExceededError."""
        exc = TokenLimitExceededError("Too many tokens", current=5000, limit=4096)
        result = format_error_for_log(exc)

        assert result["error_type"] == "TokenLimitExceededError"
        assert result["message"] == "Too many tokens"
        assert result["current_tokens"] == 5000
        assert result["token_limit"] == 4096
        assert result["error_code"] == E_VALIDATION

    def test_model_response_error(self):
        """Test logging format for ModelResponseError."""
        exc = ModelResponseError("API failed", provider="openai", status_code=429)
        result = format_error_for_log(exc)

        assert result["error_type"] == "ModelResponseError"
        assert result["message"] == "API failed"
        assert result["provider"] == "openai"
        assert result["status_code"] == 429
        assert result["error_code"] == E_TIMEOUT

    def test_configuration_error(self):
        """Test logging format for ConfigurationError."""
        exc = ConfigurationError("Invalid value", key="temperature", reason="Out of range")
        result = format_error_for_log(exc)

        assert result["error_type"] == "ConfigurationError"
        assert result["message"] == "Invalid value"
        assert result["config_key"] == "temperature"
        assert result["reason"] == "Out of range"
        assert result["error_code"] == E_VALIDATION

    def test_empty_optional_fields(self):
        """Test logging format with empty optional fields."""
        exc = ToolExecutionError("Tool failed")  # No tool_name or details
        result = format_error_for_log(exc)

        assert result["error_type"] == "ToolExecutionError"
        assert result["message"] == "Tool failed"
        assert "tool_name" not in result  # Should not include empty fields
        assert "details" not in result


class TestErrorCodes:
    """Test that all error codes are properly defined."""

    def test_error_code_constants(self):
        """Test that all error code constants are defined."""
        assert E_NOT_FOUND == "E_NOT_FOUND"
        assert E_NOT_UNIQUE == "E_NOT_UNIQUE"
        assert E_VALIDATION == "E_VALIDATION"
        assert E_PERMISSIONS == "E_PERMISSIONS"
        assert E_TIMEOUT == "E_TIMEOUT"
        assert E_UNSAFE == "E_UNSAFE"
        assert E_CONFIRMED_DENY == "E_CONFIRMED_DENY"
        assert E_TOOL_UNKNOWN == "E_TOOL_UNKNOWN"
