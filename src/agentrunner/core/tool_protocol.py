"""Tool protocol core types and helpers."""

from dataclasses import dataclass, field
from typing import Any

# Standard error codes for tool execution
E_NOT_FOUND = "E_NOT_FOUND"
E_NOT_UNIQUE = "E_NOT_UNIQUE"
E_VALIDATION = "E_VALIDATION"
E_PERMISSIONS = "E_PERMISSIONS"
E_TIMEOUT = "E_TIMEOUT"
E_UNSAFE = "E_UNSAFE"
E_CONFIRMED_DENY = "E_CONFIRMED_DENY"
E_TOOL_UNKNOWN = "E_TOOL_UNKNOWN"


@dataclass
class ToolCall:
    """Represents a tool call request from LLM."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ToolResult:
    """Result from tool execution."""

    success: bool
    output: str | None = None
    error: str | None = None
    error_code: str | None = None
    data: dict[str, Any] | None = None
    diffs: list[dict[str, Any]] | None = None
    files_changed: list[str] | None = None


@dataclass
class ToolDefinition:
    """Tool definition with JSON Schema and safety flags."""

    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema
    safety: dict[str, bool] = field(
        default_factory=dict
    )  # requires_read_first, requires_confirmation

    def __post_init__(self) -> None:
        """Validate tool definition structure."""
        if not isinstance(self.parameters, dict):
            raise ValueError("Tool parameters must be a dictionary")
        if "type" not in self.parameters:
            raise ValueError("Tool parameters must specify 'type'")

        # Ensure safety flags have defaults
        safety_defaults = {"requires_read_first": False, "requires_confirmation": False}
        for key, default in safety_defaults.items():
            if key not in self.safety:
                self.safety[key] = default


# Provider-specific parsing/normalization lives in provider adapters.


def format_tool_result(call_id: str, result: ToolResult) -> dict[str, Any]:
    """Format tool result as message for conversation history.

    Args:
        call_id: ID of the originating tool call
        result: ToolResult to format

    Returns:
        Message dictionary suitable for conversation history
    """
    # Base message structure
    message: dict[str, Any] = {
        "role": "tool",
        "tool_call_id": call_id,
        "content": result.output or (result.error if not result.success else ""),
    }

    # Add metadata if available
    if result.data or result.diffs or result.files_changed:
        message["meta"] = {}
        if result.data:
            message["meta"]["data"] = result.data
        if result.diffs:
            message["meta"]["diffs"] = result.diffs
        if result.files_changed:
            message["meta"]["files_changed"] = result.files_changed

    return message


def validate_tool_schema(tool_def: ToolDefinition) -> bool:
    """Validate tool definition JSON Schema.

    Args:
        tool_def: ToolDefinition to validate

    Returns:
        True if valid, False otherwise
    """
    try:
        # Basic structure validation
        params = tool_def.parameters

        # Must have type
        if not isinstance(params.get("type"), str):
            return False

        # If type is object, should have properties
        if params["type"] == "object":
            if "properties" not in params:
                return False
            if not isinstance(params["properties"], dict):
                return False

            # Required should be a list if present
            if "required" in params and not isinstance(params["required"], list):
                return False

        # Validate individual properties if they exist
        if "properties" in params:
            for _prop_name, prop_def in params["properties"].items():
                if not isinstance(prop_def, dict):
                    return False
                if "type" not in prop_def:
                    return False

        return True

    except (KeyError, TypeError, AttributeError):
        return False
