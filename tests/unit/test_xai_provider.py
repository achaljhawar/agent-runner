"""Tests for xAI (Grok) provider."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentrunner.core.messages import Message
from agentrunner.core.tool_protocol import ToolDefinition
from agentrunner.providers.base import ProviderConfig, ProviderResponse
from agentrunner.providers.xai_provider import XAI_BASE_URL, XAIProvider


@pytest.fixture
def xai_provider():
    """Create XAIProvider instance for testing."""
    config = ProviderConfig(model="grok-code-fast-1", temperature=0.7)
    return XAIProvider(api_key="test-key", config=config)


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


def test_xai_provider_init():
    """Test XAIProvider initialization."""
    config = ProviderConfig(model="grok-code-fast-1", temperature=0.5)
    provider = XAIProvider(api_key="test-key", config=config)

    assert provider.config.model == "grok-code-fast-1"
    assert provider.config.temperature == 0.5
    assert provider.max_retries == 3


def test_xai_provider_default_base_url():
    """Test XAIProvider uses correct default base URL."""
    config = ProviderConfig(model="grok-code-fast-1")
    provider = XAIProvider(api_key="test-key", config=config)

    # The async client should use the xAI base URL
    assert XAI_BASE_URL in str(provider.async_client.base_url)


def test_xai_provider_custom_base_url():
    """Test XAIProvider with custom base URL."""
    config = ProviderConfig(model="grok-code-fast-1")
    custom_url = "https://custom.api.endpoint"
    provider = XAIProvider(api_key="test-key", config=config, base_url=custom_url)

    # The client should be initialized with custom URL
    assert custom_url in str(provider.client.base_url)


def test_xai_provider_custom_retries():
    """Test XAIProvider with custom max retries."""
    config = ProviderConfig(model="grok-code-fast-1")
    provider = XAIProvider(api_key="test-key", config=config, max_retries=5)

    assert provider.max_retries == 5


def test_convert_messages_to_xai(xai_provider, sample_messages):
    """Test message conversion to xAI format."""
    xai_messages = xai_provider._convert_messages_to_xai(sample_messages)

    assert len(xai_messages) == 2
    assert xai_messages[0]["role"] == "system"
    assert xai_messages[0]["content"] == "You are a helpful assistant."
    assert xai_messages[1]["role"] == "user"
    assert xai_messages[1]["content"] == "Hello!"


def test_convert_messages_with_tool_calls(xai_provider):
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

    xai_messages = xai_provider._convert_messages_to_xai(messages)

    assert len(xai_messages) == 1
    assert "tool_calls" in xai_messages[0]
    assert len(xai_messages[0]["tool_calls"]) == 1
    assert xai_messages[0]["tool_calls"][0]["id"] == "call_123"
    assert xai_messages[0]["tool_calls"][0]["function"]["name"] == "read_file"


def test_convert_messages_with_tool_result(xai_provider):
    """Test message conversion with tool result."""
    messages = [
        Message(
            id="1",
            role="tool",
            content="File contents here",
            tool_call_id="call_123",
            meta={"tool_name": "read_file"},
        ),
    ]

    xai_messages = xai_provider._convert_messages_to_xai(messages)

    assert len(xai_messages) == 1
    assert xai_messages[0]["role"] == "tool"
    assert xai_messages[0]["content"] == "File contents here"
    assert xai_messages[0]["tool_call_id"] == "call_123"
    assert xai_messages[0]["name"] == "read_file"


def test_convert_messages_with_legacy_tool_call_format(xai_provider):
    """Test message conversion handles legacy tool call format."""
    messages = [
        Message(
            id="1",
            role="assistant",
            content="",
            tool_calls=[
                {
                    "id": "call_456",
                    "name": "write_file",
                    "arguments": {"path": "test.py", "content": "print('hello')"},
                }
            ],
        ),
    ]

    xai_messages = xai_provider._convert_messages_to_xai(messages)

    assert len(xai_messages) == 1
    assert "tool_calls" in xai_messages[0]
    assert xai_messages[0]["tool_calls"][0]["id"] == "call_456"
    assert xai_messages[0]["tool_calls"][0]["function"]["name"] == "write_file"
    # Arguments should be JSON string
    assert isinstance(xai_messages[0]["tool_calls"][0]["function"]["arguments"], str)


def test_convert_tools_to_xai(xai_provider):
    """Test tool definition conversion to xAI format."""
    tools = [
        ToolDefinition(
            name="read_file",
            description="Read a file from the workspace",
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file",
                    },
                },
                "required": ["path"],
            },
        ),
    ]

    xai_tools = xai_provider._convert_tools_to_xai(tools)

    assert len(xai_tools) == 1
    assert xai_tools[0]["type"] == "function"
    assert xai_tools[0]["function"]["name"] == "read_file"
    assert xai_tools[0]["function"]["description"] == "Read a file from the workspace"
    assert "path" in xai_tools[0]["function"]["parameters"]["properties"]


def test_parse_response_message(xai_provider):
    """Test parsing xAI response to Message."""
    # Mock xAI response
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message = MagicMock()
    mock_response.choices[0].message.content = "Hello, how can I help?"
    mock_response.choices[0].message.tool_calls = None
    mock_response.usage = MagicMock()
    mock_response.usage.prompt_tokens = 10
    mock_response.usage.completion_tokens = 20
    mock_response.usage.total_tokens = 30

    message = xai_provider._parse_response_message(mock_response)

    assert message.role == "assistant"
    assert message.content == "Hello, how can I help?"
    assert message.tool_calls is None


def test_parse_response_message_with_tool_calls(xai_provider):
    """Test parsing xAI response with tool calls."""
    # Mock xAI response with tool calls
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message = MagicMock()
    mock_response.choices[0].message.content = ""

    mock_tool_call = MagicMock()
    mock_tool_call.id = "call_789"
    mock_tool_call.function = MagicMock()
    mock_tool_call.function.name = "read_file"
    mock_tool_call.function.arguments = '{"path": "test.py"}'

    mock_response.choices[0].message.tool_calls = [mock_tool_call]

    message = xai_provider._parse_response_message(mock_response)

    assert message.role == "assistant"
    assert message.tool_calls is not None
    assert len(message.tool_calls) == 1
    assert message.tool_calls[0]["id"] == "call_789"
    assert message.tool_calls[0]["function"]["name"] == "read_file"
    assert message.tool_calls[0]["function"]["arguments"]["path"] == "test.py"


def test_parse_response_message_invalid_json_arguments(xai_provider):
    """Test parsing response with invalid JSON arguments."""
    # Mock xAI response with invalid JSON
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message = MagicMock()
    mock_response.choices[0].message.content = ""

    mock_tool_call = MagicMock()
    mock_tool_call.id = "call_999"
    mock_tool_call.function = MagicMock()
    mock_tool_call.function.name = "test_tool"
    mock_tool_call.function.arguments = "not valid json{"

    mock_response.choices[0].message.tool_calls = [mock_tool_call]

    message = xai_provider._parse_response_message(mock_response)

    # Should fallback to empty dict for invalid JSON
    assert message.tool_calls is not None
    assert len(message.tool_calls) == 1
    assert message.tool_calls[0]["function"]["arguments"] == {}


def test_parse_tool_calls(xai_provider):
    """Test parsing tool calls from assistant message."""
    assistant_msg = Message(
        id="1",
        role="assistant",
        content="",
        tool_calls=[
            {
                "id": "call_abc",
                "type": "function",
                "function": {
                    "name": "read_file",
                    "arguments": {"path": "test.py"},
                },
            }
        ],
    )

    tool_calls = xai_provider.parse_tool_calls(assistant_msg)

    assert len(tool_calls) == 1
    assert tool_calls[0].id == "call_abc"
    assert tool_calls[0].name == "read_file"
    assert tool_calls[0].arguments == {"path": "test.py"}


def test_parse_tool_calls_with_string_arguments(xai_provider):
    """Test parsing tool calls with JSON string arguments."""
    assistant_msg = Message(
        id="1",
        role="assistant",
        content="",
        tool_calls=[
            {
                "id": "call_def",
                "type": "function",
                "function": {
                    "name": "write_file",
                    "arguments": '{"path": "test.py", "content": "hello"}',
                },
            }
        ],
    )

    tool_calls = xai_provider.parse_tool_calls(assistant_msg)

    assert len(tool_calls) == 1
    assert tool_calls[0].arguments["path"] == "test.py"
    assert tool_calls[0].arguments["content"] == "hello"


def test_parse_tool_calls_empty(xai_provider):
    """Test parsing message with no tool calls."""
    assistant_msg = Message(
        id="1",
        role="assistant",
        content="Just a regular response",
    )

    tool_calls = xai_provider.parse_tool_calls(assistant_msg)

    assert tool_calls == []


@pytest.mark.skip(reason="Skipping temporarily - CI environment issue")
def test_count_tokens(xai_provider):
    """Test token counting."""
    text = "Hello, world! This is a test."
    count = xai_provider.count_tokens(text)

    assert count > 0
    assert isinstance(count, int)


@pytest.mark.asyncio
async def test_chat_success(xai_provider, sample_messages):
    """Test successful chat completion."""
    # Mock the async client
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message = MagicMock()
    mock_response.choices[0].message.content = "Hello!"
    mock_response.choices[0].message.tool_calls = None
    mock_response.usage = MagicMock()
    mock_response.usage.prompt_tokens = 10
    mock_response.usage.completion_tokens = 5
    mock_response.usage.total_tokens = 15

    with patch.object(
        xai_provider.async_client.chat.completions,
        "create",
        new_callable=AsyncMock,
        return_value=mock_response,
    ):
        response = await xai_provider.chat(sample_messages, tools=None)

        assert isinstance(response, ProviderResponse)
        assert len(response.messages) == 1
        assert response.messages[0].content == "Hello!"
        assert response.usage["prompt_tokens"] == 10
        assert response.usage["completion_tokens"] == 5
        assert response.usage["total_tokens"] == 15


@pytest.mark.asyncio
async def test_chat_with_tools(xai_provider, sample_messages):
    """Test chat completion with tools."""
    tools = [
        ToolDefinition(
            name="read_file",
            description="Read a file",
            parameters={
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        ),
    ]

    # Mock response with tool call
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message = MagicMock()
    mock_response.choices[0].message.content = ""

    mock_tool_call = MagicMock()
    mock_tool_call.id = "call_123"
    mock_tool_call.function = MagicMock()
    mock_tool_call.function.name = "read_file"
    mock_tool_call.function.arguments = '{"path": "test.py"}'

    mock_response.choices[0].message.tool_calls = [mock_tool_call]
    mock_response.usage = MagicMock()
    mock_response.usage.prompt_tokens = 20
    mock_response.usage.completion_tokens = 10
    mock_response.usage.total_tokens = 30

    with patch.object(
        xai_provider.async_client.chat.completions,
        "create",
        new_callable=AsyncMock,
        return_value=mock_response,
    ):
        response = await xai_provider.chat(sample_messages, tools=tools)

        assert isinstance(response, ProviderResponse)
        assert len(response.messages) == 1
        assert response.messages[0].tool_calls is not None
        assert len(response.messages[0].tool_calls) == 1


@pytest.mark.asyncio
async def test_chat_rate_limit_retry(xai_provider, sample_messages):
    """Test chat handles rate limits with retry."""
    from openai import RateLimitError

    # Mock response for successful retry
    mock_success = MagicMock()
    mock_success.choices = [MagicMock()]
    mock_success.choices[0].message = MagicMock()
    mock_success.choices[0].message.content = "Success!"
    mock_success.choices[0].message.tool_calls = None
    mock_success.usage = MagicMock()
    mock_success.usage.prompt_tokens = 10
    mock_success.usage.completion_tokens = 5
    mock_success.usage.total_tokens = 15

    # First call raises RateLimitError, second succeeds
    mock_create = AsyncMock(
        side_effect=[
            RateLimitError("Rate limit exceeded", response=MagicMock(), body={}),
            mock_success,
        ]
    )

    with patch.object(
        xai_provider.async_client.chat.completions,
        "create",
        mock_create,
    ):
        response = await xai_provider.chat(sample_messages, tools=None)

        assert isinstance(response, ProviderResponse)
        assert response.messages[0].content == "Success!"
        assert mock_create.call_count == 2


@pytest.mark.asyncio
async def test_chat_rate_limit_max_retries(xai_provider, sample_messages):
    """Test chat fails after max retries."""
    from openai import RateLimitError

    from agentrunner.core.exceptions import ModelResponseError

    # All retries raise RateLimitError
    mock_create = AsyncMock(
        side_effect=RateLimitError("Rate limit exceeded", response=MagicMock(), body={})
    )

    with patch.object(
        xai_provider.async_client.chat.completions,
        "create",
        mock_create,
    ):
        with pytest.raises(ModelResponseError) as exc_info:
            await xai_provider.chat(sample_messages, tools=None)

        assert "Rate limit exceeded" in str(exc_info.value)
        assert exc_info.value.provider == "xai"


@pytest.mark.asyncio
async def test_chat_timeout_error(xai_provider, sample_messages):
    """Test chat handles timeout errors."""
    from openai import APITimeoutError

    from agentrunner.core.exceptions import ModelResponseError

    mock_create = AsyncMock(side_effect=APITimeoutError(request=MagicMock()))

    with patch.object(
        xai_provider.async_client.chat.completions,
        "create",
        mock_create,
    ):
        with pytest.raises(ModelResponseError) as exc_info:
            await xai_provider.chat(sample_messages, tools=None)

        assert exc_info.value.error_code == "E_TIMEOUT"
        assert exc_info.value.provider == "xai"


def test_get_model_info(xai_provider):
    """Test get_model_info returns correct information."""
    model_info = xai_provider.get_model_info()

    assert model_info.name == "grok-code-fast-1"
    assert model_info.context_window == 256000
    assert "input_per_1k" in model_info.pricing
    assert "output_per_1k" in model_info.pricing


def test_get_model_info_grok_code():
    """Test get_model_info for Grok Code model."""
    config = ProviderConfig(model="grok-code-fast-1")
    provider = XAIProvider(api_key="test-key", config=config)

    model_info = provider.get_model_info()

    assert model_info.name == "grok-code-fast-1"
    assert model_info.context_window == 256000
    # Grok Code should have lower pricing
    assert model_info.pricing["input_per_1k"] < 0.001


def test_get_system_prompt(xai_provider):
    """Test system prompt generation."""
    prompt = xai_provider.get_system_prompt("/tmp/workspace")

    assert isinstance(prompt, str)
    assert len(prompt) > 0
