"""Unit tests for tool_protocol module.

Tests ToolCall, ToolResult, ToolDefinition dataclasses and all utility functions.
Uses real code with no mocks since this module does pure data transformations.
"""

import pytest

from agentrunner.core.tool_protocol import (
    E_CONFIRMED_DENY,
    # Constants
    E_NOT_FOUND,
    E_NOT_UNIQUE,
    E_PERMISSIONS,
    E_TIMEOUT,
    E_TOOL_UNKNOWN,
    E_UNSAFE,
    E_VALIDATION,
    # Dataclasses
    ToolCall,
    ToolDefinition,
    ToolResult,
    # Functions (generic only)
    format_tool_result,
    validate_tool_schema,
)


class TestErrorCodes:
    """Test error code constants."""

    def test_error_codes_exist(self):
        """Test all standard error codes are defined."""
        assert E_NOT_FOUND == "E_NOT_FOUND"
        assert E_NOT_UNIQUE == "E_NOT_UNIQUE"
        assert E_VALIDATION == "E_VALIDATION"
        assert E_PERMISSIONS == "E_PERMISSIONS"
        assert E_TIMEOUT == "E_TIMEOUT"
        assert E_UNSAFE == "E_UNSAFE"
        assert E_CONFIRMED_DENY == "E_CONFIRMED_DENY"
        assert E_TOOL_UNKNOWN == "E_TOOL_UNKNOWN"


class TestToolCall:
    """Test ToolCall dataclass."""

    def test_tool_call_creation(self):
        """Test ToolCall creation with all fields."""
        tool_call = ToolCall(
            id="call_123", name="read_file", arguments={"file_path": "/test/file.py", "offset": 1}
        )

        assert tool_call.id == "call_123"
        assert tool_call.name == "read_file"
        assert tool_call.arguments == {"file_path": "/test/file.py", "offset": 1}

    def test_tool_call_empty_arguments(self):
        """Test ToolCall with empty arguments."""
        tool_call = ToolCall(id="call_456", name="list_files", arguments={})

        assert tool_call.id == "call_456"
        assert tool_call.name == "list_files"
        assert tool_call.arguments == {}


class TestToolResult:
    """Test ToolResult dataclass."""

    def test_tool_result_success(self):
        """Test successful ToolResult."""
        result = ToolResult(
            success=True,
            output="File read successfully",
            data={"lines": 42},
            files_changed=["/test/file.py"],
        )

        assert result.success is True
        assert result.output == "File read successfully"
        assert result.error is None
        assert result.error_code is None
        assert result.data == {"lines": 42}
        assert result.files_changed == ["/test/file.py"]

    def test_tool_result_failure(self):
        """Test failed ToolResult."""
        result = ToolResult(success=False, error="File not found", error_code=E_NOT_FOUND)

        assert result.success is False
        assert result.output is None
        assert result.error == "File not found"
        assert result.error_code == E_NOT_FOUND
        assert result.data is None
        assert result.diffs is None
        assert result.files_changed is None

    def test_tool_result_with_diffs(self):
        """Test ToolResult with diff information."""
        diffs = [
            {
                "file": "/test/file.py",
                "format": "unified",
                "content": "--- a/file.py\n+++ b/file.py",
            }
        ]

        result = ToolResult(
            success=True, output="File edited", diffs=diffs, files_changed=["/test/file.py"]
        )

        assert result.success is True
        assert result.diffs == diffs
        assert result.files_changed == ["/test/file.py"]


class TestToolDefinition:
    """Test ToolDefinition dataclass."""

    def test_tool_definition_creation(self):
        """Test ToolDefinition creation with all fields."""
        parameters = {
            "type": "object",
            "properties": {"file_path": {"type": "string"}, "offset": {"type": "integer"}},
            "required": ["file_path"],
        }

        tool_def = ToolDefinition(
            name="read_file",
            description="Read file contents",
            parameters=parameters,
            safety={"requires_read_first": False, "requires_confirmation": False},
        )

        assert tool_def.name == "read_file"
        assert tool_def.description == "Read file contents"
        assert tool_def.parameters == parameters
        assert tool_def.safety["requires_read_first"] is False
        assert tool_def.safety["requires_confirmation"] is False

    def test_tool_definition_default_safety(self):
        """Test ToolDefinition auto-fills default safety flags."""
        tool_def = ToolDefinition(
            name="test_tool",
            description="Test tool",
            parameters={"type": "object", "properties": {}},
        )

        assert tool_def.safety["requires_read_first"] is False
        assert tool_def.safety["requires_confirmation"] is False

    def test_tool_definition_partial_safety(self):
        """Test ToolDefinition fills missing safety flags."""
        tool_def = ToolDefinition(
            name="test_tool",
            description="Test tool",
            parameters={"type": "object", "properties": {}},
            safety={"requires_confirmation": True},
        )

        assert tool_def.safety["requires_read_first"] is False  # Default filled
        assert tool_def.safety["requires_confirmation"] is True  # User provided

    def test_tool_definition_invalid_parameters(self):
        """Test ToolDefinition validation for invalid parameters."""
        with pytest.raises(ValueError, match="Tool parameters must be a dictionary"):
            ToolDefinition(name="bad_tool", description="Bad tool", parameters="not a dict")

        with pytest.raises(ValueError, match="Tool parameters must specify 'type'"):
            ToolDefinition(
                name="bad_tool",
                description="Bad tool",
                parameters={"properties": {}},  # Missing "type"
            )


class TestToolCallDataclass:
    def test_tool_call_creation(self):
        tc = ToolCall(id="call_1", name="read_file", arguments={"file_path": "/x"})
        assert tc.id == "call_1"
        assert tc.name == "read_file"
        assert tc.arguments == {"file_path": "/x"}


class TestFormatToolResult:
    """Test format_tool_result function."""

    def test_format_successful_result(self):
        """Test formatting successful tool result."""
        result = ToolResult(success=True, output="File read successfully\nContent: print('hello')")

        message = format_tool_result("call_123", result)

        assert message["role"] == "tool"
        assert message["tool_call_id"] == "call_123"
        assert message["content"] == "File read successfully\nContent: print('hello')"
        assert "meta" not in message

    def test_format_error_result(self):
        """Test formatting error tool result."""
        result = ToolResult(
            success=False, error="File not found: /nonexistent.py", error_code=E_NOT_FOUND
        )

        message = format_tool_result("call_456", result)

        assert message["role"] == "tool"
        assert message["tool_call_id"] == "call_456"
        assert message["content"] == "File not found: /nonexistent.py"

    def test_format_result_with_metadata(self):
        """Test formatting result with data, diffs, and files_changed."""
        diffs = [
            {"file": "/test.py", "format": "unified", "content": "--- a/test.py\n+++ b/test.py"}
        ]

        result = ToolResult(
            success=True,
            output="File edited successfully",
            data={"lines_added": 3, "lines_removed": 1},
            diffs=diffs,
            files_changed=["/test.py"],
        )

        message = format_tool_result("call_789", result)

        assert message["role"] == "tool"
        assert message["tool_call_id"] == "call_789"
        assert message["content"] == "File edited successfully"
        assert message["meta"]["data"] == {"lines_added": 3, "lines_removed": 1}
        assert message["meta"]["diffs"] == diffs
        assert message["meta"]["files_changed"] == ["/test.py"]

    def test_format_result_empty_output(self):
        """Test formatting result with no output."""
        result = ToolResult(success=True)

        message = format_tool_result("call_000", result)

        assert message["role"] == "tool"
        assert message["tool_call_id"] == "call_000"
        assert message["content"] == ""


class TestValidateToolSchema:
    """Test validate_tool_schema function."""

    def test_validate_valid_object_schema(self):
        """Test validating a valid object schema."""
        tool_def = ToolDefinition(
            name="read_file",
            description="Read file",
            parameters={
                "type": "object",
                "properties": {"file_path": {"type": "string"}, "offset": {"type": "integer"}},
                "required": ["file_path"],
            },
        )

        assert validate_tool_schema(tool_def) is True

    def test_validate_simple_schema(self):
        """Test validating a simple non-object schema."""
        tool_def = ToolDefinition(
            name="simple_tool", description="Simple tool", parameters={"type": "string"}
        )

        assert validate_tool_schema(tool_def) is True

    def test_validate_missing_type(self):
        """Test ToolDefinition raises error when type is missing."""
        # __post_init__ should catch this during construction
        with pytest.raises(ValueError, match="Tool parameters must specify 'type'"):
            ToolDefinition(
                name="bad_tool",
                description="Bad tool",
                parameters={"properties": {"arg": {"type": "string"}}},  # Missing top-level type
            )

    def test_validate_invalid_properties(self):
        """Test validation fails for invalid properties."""
        tool_def = ToolDefinition(
            name="bad_tool",
            description="Bad tool",
            parameters={"type": "object", "properties": "not a dict"},  # Should be dict
        )

        assert validate_tool_schema(tool_def) is False

    def test_validate_invalid_required(self):
        """Test validation fails for invalid required field."""
        tool_def = ToolDefinition(
            name="bad_tool",
            description="Bad tool",
            parameters={
                "type": "object",
                "properties": {"arg": {"type": "string"}},
                "required": "not a list",  # Should be list
            },
        )

        assert validate_tool_schema(tool_def) is False

    def test_validate_invalid_property_definition(self):
        """Test validation fails for invalid property definitions."""
        tool_def = ToolDefinition(
            name="bad_tool",
            description="Bad tool",
            parameters={
                "type": "object",
                "properties": {"arg": "not a dict"},  # Should be dict with type
            },
        )

        assert validate_tool_schema(tool_def) is False

    def test_validate_property_missing_type(self):
        """Test validation fails when property is missing type."""
        tool_def = ToolDefinition(
            name="bad_tool",
            description="Bad tool",
            parameters={
                "type": "object",
                "properties": {"arg": {"description": "An argument"}},  # Missing type
            },
        )

        assert validate_tool_schema(tool_def) is False


class TestNormalizationMovedToProviders:
    def test_note(self):
        assert True


class TestFormatConversionsMovedToProviders:
    def test_note(self):
        assert True
