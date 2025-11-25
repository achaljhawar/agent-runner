"""Tests for non-tool-calling provider methods in base provider."""

import pytest

from agentrunner.core.prompts.sections import NonNativeToolFormat
from agentrunner.core.tool_protocol import ToolDefinition
from agentrunner.providers.base import BaseLLMProvider, ProviderConfig


class MockNonToolCallingProvider(BaseLLMProvider):
    """Mock provider without native tool calling for testing."""

    def __init__(self) -> None:
        config = ProviderConfig(model="mock-model")
        super().__init__(api_key="mock-key", config=config)

    @property
    def supports_native_tool_calling(self) -> bool:
        return False

    async def chat(self, messages, tools, config):
        """Not tested here."""
        raise NotImplementedError

    def chat_stream(self, messages, tools, config):
        """Not tested here."""
        raise NotImplementedError

    def get_model_info(self):
        """Not tested here."""
        raise NotImplementedError

    def count_tokens(self, text: str) -> int:
        """Not tested here."""
        return len(text) // 4

    def get_system_prompt(self, workspace_root: str, tools=None) -> str:
        """Not tested here."""
        return "Mock system prompt"


@pytest.fixture
def provider():
    """Create mock provider."""
    return MockNonToolCallingProvider()


@pytest.fixture
def tool_format():
    """Create tool format instance."""
    return NonNativeToolFormat()


@pytest.fixture
def sample_tools():
    """Create sample tool definitions."""
    return [
        ToolDefinition(
            name="read_file",
            description="Read a file from workspace",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path"},
                },
                "required": ["path"],
            },
        ),
        ToolDefinition(
            name="write_file",
            description="Write content to a file",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        ),
    ]


def test_supports_native_tool_calling(provider):
    """Test property override works."""
    assert provider.supports_native_tool_calling is False


def test_serialize_tools_to_prompt_empty(tool_format):
    """Test serializing empty tool list."""
    result = tool_format.to_prompt([])

    assert result == ""


def test_serialize_tools_to_prompt_basic(tool_format, sample_tools):
    """Test serializing tools to prompt format."""
    result = tool_format.to_prompt(sample_tools)

    # Check structure
    assert "# Available Tools" in result
    assert "tool_calls" in result
    assert "```json" in result

    # Check both tools listed
    assert "read_file" in result
    assert "write_file" in result
    assert "Read a file from workspace" in result
    assert "Write content to a file" in result


def test_parse_tool_calls_from_text_json_block(tool_format):
    """Test parsing tool calls from JSON code block."""
    text = """
Here's what I'll do:

```json
{
    "tool_calls": [
        {
            "id": "call_123",
            "name": "read_file",
            "arguments": {"path": "test.py"}
        }
    ]
}
```
    """

    result = tool_format.to_tools(text)

    assert result is not None
    assert len(result) == 1
    assert result[0]["id"] == "call_123"
    assert result[0]["name"] == "read_file"
    assert result[0]["arguments"] == {"path": "test.py"}


def test_parse_tool_calls_from_text_no_code_block(tool_format):
    """Test parsing tool calls from plain JSON."""
    text = """
    {"tool_calls": [{"name": "write_file", "arguments": {"path": "new.py", "content": "test"}}]}
    """

    result = tool_format.to_tools(text)

    assert result is not None
    assert len(result) == 1
    assert result[0]["name"] == "write_file"
    # Should auto-generate ID if missing
    assert "id" in result[0]
    assert result[0]["id"].startswith("call_")


def test_parse_tool_calls_from_text_no_tool_calls(tool_format):
    """Test parsing text without tool calls."""
    text = "Just a regular response with no tools."

    result = tool_format.to_tools(text)

    assert result is None


def test_parse_tool_calls_from_text_invalid_json(tool_format):
    """Test parsing invalid JSON."""
    text = """
```json
{invalid json here}
```
    """

    result = tool_format.to_tools(text)

    assert result is None


def test_parse_tool_calls_from_text_multiple_calls(tool_format):
    """Test parsing multiple tool calls."""
    text = """
```json
{
    "tool_calls": [
        {"id": "call_1", "name": "read_file", "arguments": {"path": "a.py"}},
        {"id": "call_2", "name": "read_file", "arguments": {"path": "b.py"}}
    ]
}
```
    """

    result = tool_format.to_tools(text)

    assert result is not None
    assert len(result) == 2
    assert result[0]["name"] == "read_file"
    assert result[1]["name"] == "read_file"


# Tests for _append_to_system_message removed - method moved to ToolFormat system
# The functionality is now tested via ToolFormat tests
