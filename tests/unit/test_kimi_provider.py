"""Tests for Kimi (Moonshot AI) provider."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentrunner.core.messages import Message
from agentrunner.core.tool_protocol import ToolDefinition
from agentrunner.providers.base import ProviderConfig, ProviderResponse
from agentrunner.providers.kimi_provider import KimiProvider


@pytest.fixture
def kimi_provider():
    """Create KimiProvider instance for testing."""
    config = ProviderConfig(model="kimi-k2-0905-preview", temperature=0.7)
    return KimiProvider(api_key="test-key", config=config)


@pytest.fixture
def sample_messages():
    """Sample message history."""
    return [
        Message(
            id="1",
            role="system",
            content="You are a helpful assistant.",
        ),
        Message(
            id="2",
            role="user",
            content="Hello!",
        ),
    ]


def test_kimi_provider_init():
    """Test KimiProvider initialization."""
    config = ProviderConfig(model="kimi-k2-0905-preview", temperature=0.5)
    provider = KimiProvider(api_key="test-key", config=config)

    assert provider.config.model == "kimi-k2-0905-preview"
    assert provider.config.temperature == 0.5
    assert provider.max_retries == 3


def test_kimi_provider_custom_base_url():
    """Test KimiProvider with custom base URL."""
    config = ProviderConfig(model="kimi-k2-0905-preview")
    custom_url = "https://custom.api.endpoint"
    provider = KimiProvider(api_key="test-key", config=config, base_url=custom_url)

    # The client should be initialized with custom URL
    assert provider.client.base_url == custom_url


def test_convert_messages_to_kimi(kimi_provider, sample_messages):
    """Test message conversion to Kimi format."""
    kimi_messages = kimi_provider._convert_messages_to_kimi(sample_messages)

    assert len(kimi_messages) == 2
    assert kimi_messages[0]["role"] == "system"
    assert kimi_messages[0]["content"] == "You are a helpful assistant."
    assert kimi_messages[1]["role"] == "user"
    assert kimi_messages[1]["content"] == "Hello!"


def test_convert_messages_with_tool_calls(kimi_provider):
    """Test message conversion with tool calls."""
    messages = [
        Message(
            id="1",
            role="assistant",
            content="",
            tool_calls=[
                {
                    "id": "call_123",
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "arguments": {"path": "test.py"},
                    },
                }
            ],
        ),
    ]

    kimi_messages = kimi_provider._convert_messages_to_kimi(messages)

    assert len(kimi_messages) == 1
    assert "tool_calls" in kimi_messages[0]
    assert len(kimi_messages[0]["tool_calls"]) == 1
    assert kimi_messages[0]["tool_calls"][0]["id"] == "call_123"
    assert kimi_messages[0]["tool_calls"][0]["function"]["name"] == "read_file"


def test_convert_tools_to_kimi(kimi_provider):
    """Test tool definition conversion to Kimi format."""
    tools = [
        ToolDefinition(
            name="read_file",
            description="Read file contents",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path"},
                },
                "required": ["path"],
            },
        ),
    ]

    kimi_tools = kimi_provider._convert_tools_to_kimi(tools)

    assert len(kimi_tools) == 1
    assert kimi_tools[0]["type"] == "function"
    assert kimi_tools[0]["function"]["name"] == "read_file"
    assert kimi_tools[0]["function"]["description"] == "Read file contents"
    assert "parameters" in kimi_tools[0]["function"]


def test_count_tokens(kimi_provider):
    """Test token counting."""
    text = "Hello, world!"
    token_count = kimi_provider.count_tokens(text)

    assert isinstance(token_count, int)
    assert token_count > 0


def test_get_model_info(kimi_provider):
    """Test getting model information."""
    model_info = kimi_provider.get_model_info()

    assert model_info.name == "kimi-k2-0905-preview"
    assert model_info.context_window == 128000
    assert "input_per_1k" in model_info.pricing
    assert "output_per_1k" in model_info.pricing


@pytest.mark.asyncio
async def test_chat_success(kimi_provider, sample_messages):
    """Test successful chat completion."""
    # Mock response
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "Hello! How can I help?"
    mock_response.choices[0].message.tool_calls = None
    mock_response.usage.prompt_tokens = 10
    mock_response.usage.completion_tokens = 5
    mock_response.usage.total_tokens = 15

    with patch.object(
        kimi_provider.async_client.chat.completions,
        "create",
        new_callable=AsyncMock,
        return_value=mock_response,
    ):
        response = await kimi_provider.chat(sample_messages, None)

        assert isinstance(response, ProviderResponse)
        assert len(response.messages) == 1
        assert response.messages[0].role == "assistant"
        assert response.messages[0].content == "Hello! How can I help?"
        assert response.usage["prompt_tokens"] == 10
        assert response.usage["completion_tokens"] == 5


@pytest.mark.asyncio
async def test_chat_with_tools(kimi_provider, sample_messages):
    """Test chat with tool calls."""
    tools = [
        ToolDefinition(
            name="read_file",
            description="Read file",
            parameters={"type": "object", "properties": {}},
        ),
    ]

    # Mock response with tool call
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = ""

    mock_tool_call = MagicMock()
    mock_tool_call.id = "call_123"
    mock_tool_call.function.name = "read_file"
    mock_tool_call.function.arguments = json.dumps({"path": "test.py"})

    mock_response.choices[0].message.tool_calls = [mock_tool_call]
    mock_response.usage.prompt_tokens = 10
    mock_response.usage.completion_tokens = 5
    mock_response.usage.total_tokens = 15

    with patch.object(
        kimi_provider.async_client.chat.completions,
        "create",
        new_callable=AsyncMock,
        return_value=mock_response,
    ):
        response = await kimi_provider.chat(sample_messages, tools)

        assert isinstance(response, ProviderResponse)
        assert response.messages[0].tool_calls is not None
        assert len(response.messages[0].tool_calls) == 1
        assert response.messages[0].tool_calls[0]["function"]["name"] == "read_file"
