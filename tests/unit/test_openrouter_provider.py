"""Tests for OpenRouter provider."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentrunner.core.messages import Message
from agentrunner.core.tool_protocol import ToolDefinition
from agentrunner.providers.base import ProviderConfig, ProviderResponse
from agentrunner.providers.openrouter_provider import OPENROUTER_BASE_URL, OpenRouterProvider


@pytest.fixture
def openrouter_provider():
    """Create OpenRouterProvider instance for testing."""
    config = ProviderConfig(model="xiaomi/mimo-v2-flash:free", temperature=0.7)
    return OpenRouterProvider(api_key="test-key", config=config)


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


def test_openrouter_provider_init():
    """Test OpenRouterProvider initialization."""
    config = ProviderConfig(model="xiaomi/mimo-v2-flash:free", temperature=0.5)
    provider = OpenRouterProvider(api_key="test-key", config=config)

    assert provider.config.model == "xiaomi/mimo-v2-flash:free"
    assert provider.config.temperature == 0.5
    assert provider.max_retries == 3


def test_openrouter_provider_default_base_url():
    """Test OpenRouterProvider uses correct default base URL."""
    config = ProviderConfig(model="xiaomi/mimo-v2-flash:free")
    provider = OpenRouterProvider(api_key="test-key", config=config)

    # The async client should use the OpenRouter base URL
    assert OPENROUTER_BASE_URL in str(provider.async_client.base_url)


def test_openrouter_provider_custom_base_url():
    """Test OpenRouterProvider with custom base URL."""
    config = ProviderConfig(model="xiaomi/mimo-v2-flash:free")
    custom_url = "https://custom.api.endpoint"
    provider = OpenRouterProvider(api_key="test-key", config=config, base_url=custom_url)

    # The client should be initialized with custom URL
    assert custom_url in str(provider.client.base_url)


def test_openrouter_provider_custom_retries():
    """Test OpenRouterProvider with custom max retries."""
    config = ProviderConfig(model="xiaomi/mimo-v2-flash:free")
    provider = OpenRouterProvider(api_key="test-key", config=config, max_retries=5)

    assert provider.max_retries == 5


def test_convert_messages_to_openrouter(openrouter_provider, sample_messages):
    """Test message conversion to OpenRouter format."""
    openrouter_messages = openrouter_provider._convert_messages_to_openrouter(sample_messages)

    assert len(openrouter_messages) == 2
    assert openrouter_messages[0]["role"] == "system"
    assert openrouter_messages[0]["content"] == "You are a helpful assistant."
    assert openrouter_messages[1]["role"] == "user"
    assert openrouter_messages[1]["content"] == "Hello!"


def test_convert_messages_with_tool_calls(openrouter_provider):
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

    openrouter_messages = openrouter_provider._convert_messages_to_openrouter(messages)

    assert len(openrouter_messages) == 1
    assert "tool_calls" in openrouter_messages[0]
    assert len(openrouter_messages[0]["tool_calls"]) == 1
    assert openrouter_messages[0]["tool_calls"][0]["id"] == "call_123"
    assert openrouter_messages[0]["tool_calls"][0]["function"]["name"] == "read_file"


def test_convert_messages_with_tool_result(openrouter_provider):
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

    openrouter_messages = openrouter_provider._convert_messages_to_openrouter(messages)

    assert len(openrouter_messages) == 1
    assert openrouter_messages[0]["role"] == "tool"
    assert openrouter_messages[0]["content"] == "File contents here"
    assert openrouter_messages[0]["tool_call_id"] == "call_123"
    assert openrouter_messages[0]["name"] == "read_file"


def test_convert_messages_with_legacy_tool_call_format(openrouter_provider):
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

    openrouter_messages = openrouter_provider._convert_messages_to_openrouter(messages)

    assert len(openrouter_messages) == 1
    assert "tool_calls" in openrouter_messages[0]
    assert openrouter_messages[0]["tool_calls"][0]["id"] == "call_456"
    assert openrouter_messages[0]["tool_calls"][0]["function"]["name"] == "write_file"
    # Arguments should be JSON string
    assert isinstance(openrouter_messages[0]["tool_calls"][0]["function"]["arguments"], str)


def test_convert_tools_to_openrouter(openrouter_provider):
    """Test tool definition conversion to OpenRouter format."""
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

    openrouter_tools = openrouter_provider._convert_tools_to_openrouter(tools)

    assert len(openrouter_tools) == 1
    assert openrouter_tools[0]["type"] == "function"
    assert openrouter_tools[0]["function"]["name"] == "read_file"
    assert openrouter_tools[0]["function"]["description"] == "Read a file from the workspace"
    assert "path" in openrouter_tools[0]["function"]["parameters"]["properties"]


def test_parse_response_message(openrouter_provider):
    """Test parsing OpenRouter response to Message."""
    # Mock OpenRouter response
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message = MagicMock()
    mock_response.choices[0].message.content = "Hello, how can I help?"
    mock_response.choices[0].message.tool_calls = None
    mock_response.usage = MagicMock()
    mock_response.usage.prompt_tokens = 10
    mock_response.usage.completion_tokens = 20
    mock_response.usage.total_tokens = 30

    message = openrouter_provider._parse_response_message(mock_response)

    assert message.role == "assistant"
    assert message.content == "Hello, how can I help?"
    assert message.tool_calls is None


def test_parse_response_message_with_tool_calls(openrouter_provider):
    """Test parsing OpenRouter response with tool calls."""
    # Mock OpenRouter response with tool calls
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

    message = openrouter_provider._parse_response_message(mock_response)

    assert message.role == "assistant"
    assert message.tool_calls is not None
    assert len(message.tool_calls) == 1
    assert message.tool_calls[0]["id"] == "call_789"
    assert message.tool_calls[0]["function"]["name"] == "read_file"
    assert message.tool_calls[0]["function"]["arguments"]["path"] == "test.py"


def test_parse_response_message_invalid_json_arguments(openrouter_provider):
    """Test parsing response with invalid JSON arguments."""
    # Mock OpenRouter response with invalid JSON
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

    message = openrouter_provider._parse_response_message(mock_response)

    # Should fallback to empty dict for invalid JSON
    assert message.tool_calls is not None
    assert len(message.tool_calls) == 1
    assert message.tool_calls[0]["function"]["arguments"] == {}


def test_parse_tool_calls(openrouter_provider):
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

    tool_calls = openrouter_provider.parse_tool_calls(assistant_msg)

    assert len(tool_calls) == 1
    assert tool_calls[0].id == "call_abc"
    assert tool_calls[0].name == "read_file"
    assert tool_calls[0].arguments == {"path": "test.py"}


def test_parse_tool_calls_with_string_arguments(openrouter_provider):
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

    tool_calls = openrouter_provider.parse_tool_calls(assistant_msg)

    assert len(tool_calls) == 1
    assert tool_calls[0].arguments["path"] == "test.py"
    assert tool_calls[0].arguments["content"] == "hello"


def test_parse_tool_calls_empty(openrouter_provider):
    """Test parsing message with no tool calls."""
    assistant_msg = Message(
        id="1",
        role="assistant",
        content="Just a regular response",
    )

    tool_calls = openrouter_provider.parse_tool_calls(assistant_msg)

    assert tool_calls == []


@pytest.mark.skip(reason="Skipping temporarily - CI environment issue")
def test_count_tokens(openrouter_provider):
    """Test token counting."""
    text = "Hello, world! This is a test."
    count = openrouter_provider.count_tokens(text)

    assert count > 0
    assert isinstance(count, int)


@pytest.mark.asyncio
async def test_chat_success(openrouter_provider, sample_messages):
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
        openrouter_provider.async_client.chat.completions,
        "create",
        new_callable=AsyncMock,
        return_value=mock_response,
    ):
        response = await openrouter_provider.chat(sample_messages, tools=None)

        assert isinstance(response, ProviderResponse)
        assert len(response.messages) == 1
        assert response.messages[0].content == "Hello!"
        assert response.usage["prompt_tokens"] == 10
        assert response.usage["completion_tokens"] == 5
        assert response.usage["total_tokens"] == 15


@pytest.mark.asyncio
async def test_chat_with_tools(openrouter_provider, sample_messages):
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
        openrouter_provider.async_client.chat.completions,
        "create",
        new_callable=AsyncMock,
        return_value=mock_response,
    ):
        response = await openrouter_provider.chat(sample_messages, tools=tools)

        assert isinstance(response, ProviderResponse)
        assert len(response.messages) == 1
        assert response.messages[0].tool_calls is not None
        assert len(response.messages[0].tool_calls) == 1


@pytest.mark.asyncio
async def test_chat_rate_limit_retry(openrouter_provider, sample_messages):
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
        openrouter_provider.async_client.chat.completions,
        "create",
        mock_create,
    ):
        response = await openrouter_provider.chat(sample_messages, tools=None)

        assert isinstance(response, ProviderResponse)
        assert response.messages[0].content == "Success!"
        assert mock_create.call_count == 2


@pytest.mark.asyncio
async def test_chat_rate_limit_max_retries(openrouter_provider, sample_messages):
    """Test chat fails after max retries."""
    from openai import RateLimitError

    from agentrunner.core.exceptions import ModelResponseError

    # All retries raise RateLimitError
    mock_create = AsyncMock(
        side_effect=RateLimitError("Rate limit exceeded", response=MagicMock(), body={})
    )

    with patch.object(
        openrouter_provider.async_client.chat.completions,
        "create",
        mock_create,
    ):
        with pytest.raises(ModelResponseError) as exc_info:
            await openrouter_provider.chat(sample_messages, tools=None)

        assert "Rate limit exceeded" in str(exc_info.value)
        assert exc_info.value.provider == "openrouter"


@pytest.mark.asyncio
async def test_chat_timeout_error(openrouter_provider, sample_messages):
    """Test chat handles timeout errors."""
    from openai import APITimeoutError

    from agentrunner.core.exceptions import ModelResponseError

    mock_create = AsyncMock(side_effect=APITimeoutError(request=MagicMock()))

    with patch.object(
        openrouter_provider.async_client.chat.completions,
        "create",
        mock_create,
    ):
        with pytest.raises(ModelResponseError) as exc_info:
            await openrouter_provider.chat(sample_messages, tools=None)

        assert exc_info.value.error_code == "E_TIMEOUT"
        assert exc_info.value.provider == "openrouter"


def test_get_model_info(openrouter_provider):
    """Test get_model_info returns correct information."""
    model_info = openrouter_provider.get_model_info()

    assert model_info.name == "xiaomi/mimo-v2-flash:free"
    assert model_info.context_window == 128000
    assert "input_per_1k" in model_info.pricing
    assert "output_per_1k" in model_info.pricing


def test_get_system_prompt(openrouter_provider):
    """Test system prompt generation."""
    prompt = openrouter_provider.get_system_prompt("/tmp/workspace")

    assert isinstance(prompt, str)
    assert len(prompt) > 0
