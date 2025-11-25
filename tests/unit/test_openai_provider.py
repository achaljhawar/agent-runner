"""Tests for OpenAI provider implementation."""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from agentrunner.core.config import AgentConfig
from agentrunner.core.exceptions import ConfigurationError, ModelResponseError
from agentrunner.core.messages import Message
from agentrunner.core.tool_protocol import ToolCall, ToolDefinition
from agentrunner.providers.base import ProviderConfig
from agentrunner.providers.openai_provider import OpenAIProvider


@pytest.fixture
def openai_provider():
    """Create OpenAI provider instance for testing."""
    with (
        patch("agentrunner.providers.openai_provider.OpenAI"),
        patch("agentrunner.providers.openai_provider.AsyncOpenAI"),
        patch("agentrunner.providers.openai_provider.tiktoken.encoding_for_model") as mock_tiktoken,
    ):
        # Mock tokenizer
        mock_encoder = Mock()
        mock_encoder.encode.return_value = [1, 2, 3, 4, 5]
        mock_tiktoken.return_value = mock_encoder

        # Use gpt-4-test to avoid Responses API routing (gpt-5.x uses Responses API)
        config = ProviderConfig(model="gpt-4-test")
        provider = OpenAIProvider(api_key="test-key", config=config)
        return provider


@pytest.fixture
def sample_messages():
    """Sample messages for testing."""
    return [
        Message(id="1", role="system", content="You are a helpful assistant."),
        Message(id="2", role="user", content="Hello!"),
    ]


@pytest.fixture
def sample_tools():
    """Sample tool definitions for testing."""
    return [
        ToolDefinition(
            name="read_file",
            description="Read contents of a file",
            parameters={
                "type": "object",
                "properties": {"path": {"type": "string", "description": "File path"}},
                "required": ["path"],
            },
        )
    ]


@pytest.fixture
def agent_config():
    """Sample agent configuration."""
    return AgentConfig()


class TestOpenAIProviderInit:
    """Tests for OpenAI provider initialization."""

    def test_provider_initialization(self):
        """Test basic provider initialization."""
        with (
            patch("agentrunner.providers.openai_provider.OpenAI"),
            patch("agentrunner.providers.openai_provider.AsyncOpenAI"),
            patch("agentrunner.providers.openai_provider.tiktoken.encoding_for_model"),
        ):
            config = ProviderConfig(model="gpt-5.1-2025-11-13")
            provider = OpenAIProvider(api_key="test-key", config=config)
            assert provider.config.model == "gpt-5.1-2025-11-13"
            assert provider.max_retries == 3

    def test_provider_with_custom_base_url(self):
        """Test provider initialization with custom base URL."""
        with (
            patch("agentrunner.providers.openai_provider.OpenAI") as mock_openai,
            patch("agentrunner.providers.openai_provider.AsyncOpenAI") as mock_async_openai,
            patch("agentrunner.providers.openai_provider.tiktoken.encoding_for_model"),
        ):
            config = ProviderConfig(model="gpt-5.1-2025-11-13")
            provider = OpenAIProvider(
                api_key="test-key", config=config, base_url="https://custom.api.com"
            )
            assert provider.config.model == "gpt-5.1-2025-11-13"
            # Verify clients were initialized with base_url
            mock_openai.assert_called_once()
            mock_async_openai.assert_called_once()

    def test_provider_with_custom_max_retries(self):
        """Test provider initialization with custom max retries."""
        with (
            patch("agentrunner.providers.openai_provider.OpenAI"),
            patch("agentrunner.providers.openai_provider.AsyncOpenAI"),
            patch("agentrunner.providers.openai_provider.tiktoken.encoding_for_model"),
        ):
            config = ProviderConfig(model="gpt-5.1-2025-11-13")
            provider = OpenAIProvider(api_key="test-key", config=config, max_retries=5)
            assert provider.max_retries == 5

    def test_tokenizer_fallback_for_unknown_model(self):
        """Test tokenizer falls back to cl100k_base for unknown models."""
        with (
            patch("agentrunner.providers.openai_provider.OpenAI"),
            patch("agentrunner.providers.openai_provider.AsyncOpenAI"),
            patch(
                "agentrunner.providers.openai_provider.tiktoken.encoding_for_model"
            ) as mock_encoding,
            patch(
                "agentrunner.providers.openai_provider.tiktoken.get_encoding"
            ) as mock_get_encoding,
        ):
            # Simulate unknown model
            mock_encoding.side_effect = KeyError("Unknown model")
            mock_get_encoding.return_value = Mock()

            config = ProviderConfig(model="custom-model")
            OpenAIProvider(api_key="test-key", config=config)
            mock_get_encoding.assert_called_once_with("cl100k_base")


class TestGetModelInfo:
    """Tests for get_model_info method."""

    def test_get_model_info_gpt51(self, openai_provider):
        """Test getting model info for GPT-5.1."""
        openai_provider.config.model = "gpt-5.1-2025-11-13"
        info = openai_provider.get_model_info()
        assert info.name == "gpt-5.1-2025-11-13"
        assert info.context_window == 400000
        assert info.pricing["input_per_1k"] == 0.00125
        assert info.pricing["output_per_1k"] == 0.01

    def test_get_model_info_gpt5_codex(self, openai_provider):
        """Test getting model info for GPT-5 Codex."""
        openai_provider.config.model = "gpt-5-codex"
        info = openai_provider.get_model_info()
        assert info.name == "gpt-5-codex"
        assert info.context_window == 256000
        assert info.pricing["input_per_1k"] == 0.015
        assert info.pricing["output_per_1k"] == 0.075

    def test_get_model_info_gpt51_codex(self, openai_provider):
        """Test getting model info for GPT-5.1 Codex."""
        openai_provider.config.model = "gpt-5.1-codex"
        info = openai_provider.get_model_info()
        assert info.name == "gpt-5.1-codex"
        assert info.context_window == 400000
        assert info.pricing["input_per_1k"] == 0.00125
        assert info.pricing["output_per_1k"] == 0.01

    def test_get_model_info_unknown_model(self, openai_provider):
        """Test that unknown models raise ConfigurationError."""
        openai_provider.config.model = "unknown-model"
        with pytest.raises(ConfigurationError, match="Unknown model: unknown-model"):
            openai_provider.get_model_info()


class TestCountTokens:
    """Tests for count_tokens method."""

    def test_count_tokens_basic(self, openai_provider):
        """Test basic token counting."""
        # Mock tokenizer returns list of 5 tokens
        count = openai_provider.count_tokens("Hello world")
        assert count == 5

    def test_count_tokens_empty_string(self, openai_provider):
        """Test counting tokens for empty string."""
        openai_provider.tokenizer.encode.return_value = []
        count = openai_provider.count_tokens("")
        assert count == 0

    def test_count_tokens_long_text(self, openai_provider):
        """Test counting tokens for longer text."""
        openai_provider.tokenizer.encode.return_value = list(range(100))
        count = openai_provider.count_tokens("A" * 100)
        assert count == 100


class TestConvertMessages:
    """Tests for message conversion to OpenAI format."""

    def test_convert_basic_messages(self, openai_provider, sample_messages):
        """Test converting basic messages to OpenAI format."""
        openai_messages = openai_provider._convert_messages_to_openai(sample_messages)
        assert len(openai_messages) == 2
        assert openai_messages[0]["role"] == "system"
        assert openai_messages[0]["content"] == "You are a helpful assistant."
        assert openai_messages[1]["role"] == "user"
        assert openai_messages[1]["content"] == "Hello!"

    def test_convert_message_with_tool_call_id(self, openai_provider):
        """Test converting tool message with tool_call_id."""
        messages = [Message(id="1", role="tool", content="File contents", tool_call_id="call_123")]
        openai_messages = openai_provider._convert_messages_to_openai(messages)
        assert openai_messages[0]["role"] == "tool"
        assert openai_messages[0]["tool_call_id"] == "call_123"

    def test_convert_message_with_tool_calls(self, openai_provider):
        """Test converting assistant message with tool_calls."""
        tool_calls = [
            {
                "id": "call_123",
                "type": "function",
                "function": {"name": "read_file", "arguments": '{"path": "test.txt"}'},
            }
        ]
        messages = [Message(id="1", role="assistant", content="", tool_calls=tool_calls)]
        openai_messages = openai_provider._convert_messages_to_openai(messages)
        assert openai_messages[0]["role"] == "assistant"
        assert openai_messages[0]["tool_calls"] == tool_calls


class TestConvertTools:
    """Tests for tool conversion to OpenAI format."""

    def test_convert_single_tool(self, openai_provider, sample_tools):
        """Test converting single tool to OpenAI format."""
        openai_tools = openai_provider._convert_tools_to_openai(sample_tools)
        assert len(openai_tools) == 1
        assert openai_tools[0]["type"] == "function"
        assert openai_tools[0]["function"]["name"] == "read_file"
        assert openai_tools[0]["function"]["description"] == "Read contents of a file"

    def test_convert_multiple_tools(self, openai_provider):
        """Test converting multiple tools to OpenAI format."""
        tools = [
            ToolDefinition(
                name="tool1",
                description="First tool",
                parameters={"type": "object", "properties": {}},
            ),
            ToolDefinition(
                name="tool2",
                description="Second tool",
                parameters={"type": "object", "properties": {}},
            ),
        ]
        openai_tools = openai_provider._convert_tools_to_openai(tools)
        assert len(openai_tools) == 2
        assert openai_tools[0]["function"]["name"] == "tool1"
        assert openai_tools[1]["function"]["name"] == "tool2"

    def test_convert_empty_tools(self, openai_provider):
        """Test converting empty tools list."""
        openai_tools = openai_provider._convert_tools_to_openai([])
        assert openai_tools == []


class TestParseResponseMessage:
    """Tests for parsing OpenAI response into AgentRunner message."""

    def test_parse_simple_text_response(self, openai_provider):
        """Test parsing simple text response."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = "Hello! How can I help?"
        mock_response.choices[0].message.tool_calls = None

        message = openai_provider._parse_response_message(mock_response)
        assert message.role == "assistant"
        assert message.content == "Hello! How can I help?"
        assert message.tool_calls is None

    def test_parse_response_with_tool_calls(self, openai_provider):
        """Test parsing response with tool calls."""
        mock_tool_call = Mock()
        mock_tool_call.id = "call_123"
        mock_tool_call.function.name = "read_file"
        mock_tool_call.function.arguments = '{"path": "test.txt"}'

        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = ""
        mock_response.choices[0].message.tool_calls = [mock_tool_call]

        message = openai_provider._parse_response_message(mock_response)
        assert message.role == "assistant"
        assert len(message.tool_calls) == 1
        assert message.tool_calls[0]["id"] == "call_123"
        assert message.tool_calls[0]["function"]["name"] == "read_file"

    def test_parse_response_empty_content(self, openai_provider):
        """Test parsing response with None content."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = None
        mock_response.choices[0].message.tool_calls = None

        message = openai_provider._parse_response_message(mock_response)
        assert message.content == ""


class TestParseToolCalls:
    """Tests for parsing tool calls from assistant message."""

    def test_parse_tool_calls_from_message(self, openai_provider):
        """Test parsing tool calls from assistant message."""
        tool_calls = [
            {
                "id": "call_123",
                "type": "function",
                "function": {"name": "read_file", "arguments": '{"path": "test.txt"}'},
            }
        ]
        message = Message(id="1", role="assistant", content="", tool_calls=tool_calls)

        parsed = openai_provider.parse_tool_calls(message)
        assert len(parsed) == 1
        assert isinstance(parsed[0], ToolCall)
        assert parsed[0].id == "call_123"
        assert parsed[0].name == "read_file"
        assert parsed[0].arguments == {"path": "test.txt"}

    def test_parse_multiple_tool_calls(self, openai_provider):
        """Test parsing multiple tool calls."""
        tool_calls = [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "tool1", "arguments": '{"arg1": "value1"}'},
            },
            {
                "id": "call_2",
                "type": "function",
                "function": {"name": "tool2", "arguments": '{"arg2": "value2"}'},
            },
        ]
        message = Message(id="1", role="assistant", content="", tool_calls=tool_calls)

        parsed = openai_provider.parse_tool_calls(message)
        assert len(parsed) == 2
        assert parsed[0].name == "tool1"
        assert parsed[1].name == "tool2"

    def test_parse_tool_calls_no_tool_calls(self, openai_provider):
        """Test parsing message with no tool calls."""
        message = Message(id="1", role="assistant", content="Just text")
        parsed = openai_provider.parse_tool_calls(message)
        assert parsed == []

    def test_parse_tool_calls_invalid_json(self, openai_provider):
        """Test parsing tool calls with invalid JSON arguments."""
        tool_calls = [
            {
                "id": "call_123",
                "type": "function",
                "function": {"name": "read_file", "arguments": "invalid json"},
            }
        ]
        message = Message(id="1", role="assistant", content="", tool_calls=tool_calls)

        parsed = openai_provider.parse_tool_calls(message)
        assert len(parsed) == 1
        assert parsed[0].arguments == {}  # Falls back to empty dict

    def test_parse_tool_calls_dict_arguments(self, openai_provider):
        """Test parsing tool calls with dict arguments (not string)."""
        tool_calls = [
            {
                "id": "call_123",
                "type": "function",
                "function": {"name": "read_file", "arguments": {"path": "test.txt"}},
            }
        ]
        message = Message(id="1", role="assistant", content="", tool_calls=tool_calls)

        parsed = openai_provider.parse_tool_calls(message)
        assert len(parsed) == 1
        assert parsed[0].arguments == {"path": "test.txt"}


class TestChatMethod:
    """Tests for chat method."""

    @pytest.mark.asyncio
    async def test_chat_basic_success(self, openai_provider, sample_messages, agent_config):
        """Test successful chat completion."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = "Hello! How can I help?"
        mock_response.choices[0].message.tool_calls = None
        mock_response.usage = Mock()
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5
        mock_response.usage.total_tokens = 15

        openai_provider.async_client.chat.completions.create = AsyncMock(return_value=mock_response)

        response = await openai_provider.chat(sample_messages, [], agent_config)

        assert len(response.messages) == 1
        assert response.messages[0].role == "assistant"
        assert response.messages[0].content == "Hello! How can I help?"
        assert response.usage["prompt_tokens"] == 10
        assert response.usage["completion_tokens"] == 5

    @pytest.mark.asyncio
    async def test_chat_with_tools(
        self, openai_provider, sample_messages, sample_tools, agent_config
    ):
        """Test chat with tools included."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = "Let me read that file."
        mock_response.choices[0].message.tool_calls = None
        mock_response.usage = Mock()
        mock_response.usage.prompt_tokens = 20
        mock_response.usage.completion_tokens = 10
        mock_response.usage.total_tokens = 30

        mock_create = AsyncMock(return_value=mock_response)
        openai_provider.async_client.chat.completions.create = mock_create

        await openai_provider.chat(sample_messages, sample_tools, agent_config)

        # Verify tools were passed to API
        call_kwargs = mock_create.call_args.kwargs
        assert "tools" in call_kwargs
        assert "tool_choice" in call_kwargs
        assert call_kwargs["tool_choice"] == "auto"

    @pytest.mark.asyncio
    async def test_chat_with_max_tokens(self, openai_provider, sample_messages, agent_config):
        """Test chat with max_tokens in config."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = "Response"
        mock_response.choices[0].message.tool_calls = None
        mock_response.usage = Mock()
        mock_response.usage.prompt_tokens = 5
        mock_response.usage.completion_tokens = 3
        mock_response.usage.total_tokens = 8

        mock_create = AsyncMock(return_value=mock_response)
        openai_provider.async_client.chat.completions.create = mock_create

        openai_provider.config.max_tokens = 500
        await openai_provider.chat(sample_messages, [], agent_config)

        # Verify max_tokens was passed
        call_kwargs = mock_create.call_args.kwargs
        assert call_kwargs["max_tokens"] == 500

    @pytest.mark.asyncio
    async def test_chat_rate_limit_retry(self, openai_provider, sample_messages, agent_config):
        """Test chat retries on rate limit error."""
        from openai import RateLimitError

        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = "Success after retry"
        mock_response.choices[0].message.tool_calls = None
        mock_response.usage = Mock()
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5
        mock_response.usage.total_tokens = 15

        # Create mock httpx response for RateLimitError
        mock_http_response = Mock()
        mock_http_response.status_code = 429

        # First call raises RateLimitError, second succeeds
        mock_create = AsyncMock(
            side_effect=[
                RateLimitError("Rate limit exceeded", response=mock_http_response, body=None),
                mock_response,
            ]
        )
        openai_provider.async_client.chat.completions.create = mock_create

        response = await openai_provider.chat(sample_messages, [], agent_config)

        # Should succeed after retry
        assert response.messages[0].content == "Success after retry"
        assert mock_create.call_count == 2

    @pytest.mark.asyncio
    async def test_chat_rate_limit_max_retries(
        self, openai_provider, sample_messages, agent_config
    ):
        """Test chat fails after max retries on rate limit."""
        from openai import RateLimitError

        # Create mock httpx response
        mock_http_response = Mock()
        mock_http_response.status_code = 429

        mock_create = AsyncMock(
            side_effect=RateLimitError(
                "Rate limit exceeded", response=mock_http_response, body=None
            )
        )
        openai_provider.async_client.chat.completions.create = mock_create

        with pytest.raises(ModelResponseError, match="Rate limit exceeded"):
            await openai_provider.chat(sample_messages, [], agent_config)

        assert mock_create.call_count == openai_provider.max_retries

    @pytest.mark.asyncio
    async def test_chat_timeout_error(self, openai_provider, sample_messages, agent_config):
        """Test chat handles timeout error."""
        from openai import APITimeoutError

        mock_create = AsyncMock(side_effect=APITimeoutError(request=Mock()))
        openai_provider.async_client.chat.completions.create = mock_create

        with pytest.raises(ModelResponseError, match="timeout"):
            await openai_provider.chat(sample_messages, [], agent_config)

    @pytest.mark.asyncio
    async def test_chat_api_error(self, openai_provider, sample_messages, agent_config):
        """Test chat handles general API error."""
        from openai import APIError

        # Create mock httpx request
        mock_request = Mock()
        error = APIError("Internal server error", request=mock_request, body=None)
        mock_create = AsyncMock(side_effect=error)
        openai_provider.async_client.chat.completions.create = mock_create

        with pytest.raises(ModelResponseError):
            await openai_provider.chat(sample_messages, [], agent_config)


class TestChatStreamMethod:
    """Tests for chat_stream method."""

    @pytest.mark.asyncio
    async def test_chat_stream_basic(self, openai_provider, sample_messages, agent_config):
        """Test basic streaming response."""

        async def mock_stream():
            # Simulate streaming chunks
            chunk1 = Mock()
            chunk1.choices = [Mock()]
            chunk1.choices[0].delta = Mock()
            chunk1.choices[0].delta.content = "Hello"
            chunk1.choices[0].delta.tool_calls = None
            chunk1.choices[0].finish_reason = None

            chunk2 = Mock()
            chunk2.choices = [Mock()]
            chunk2.choices[0].delta = Mock()
            chunk2.choices[0].delta.content = " world"
            chunk2.choices[0].delta.tool_calls = None
            chunk2.choices[0].finish_reason = None

            chunk3 = Mock()
            chunk3.choices = [Mock()]
            chunk3.choices[0].delta = Mock()
            chunk3.choices[0].delta.content = None
            chunk3.choices[0].delta.tool_calls = None
            chunk3.choices[0].finish_reason = "stop"

            for chunk in [chunk1, chunk2, chunk3]:
                yield chunk

        openai_provider.async_client.chat.completions.create = AsyncMock(return_value=mock_stream())

        chunks = []
        async for chunk in openai_provider.chat_stream(sample_messages, [], agent_config):
            chunks.append(chunk)

        assert len(chunks) == 3
        assert chunks[0].type == "token"
        assert chunks[0].payload["content"] == "Hello"
        assert chunks[1].type == "token"
        assert chunks[1].payload["content"] == " world"
        assert chunks[2].type == "status"

    @pytest.mark.asyncio
    async def test_chat_stream_with_tool_calls(
        self, openai_provider, sample_messages, agent_config
    ):
        """Test streaming with tool calls."""

        async def mock_stream():
            chunk = Mock()
            chunk.choices = [Mock()]
            chunk.choices[0].delta = Mock()
            chunk.choices[0].delta.content = None

            mock_tool_call = Mock()
            mock_tool_call.id = "call_123"
            mock_tool_call.function = Mock()
            mock_tool_call.function.name = "read_file"
            mock_tool_call.function.arguments = '{"path": "test.txt"}'

            chunk.choices[0].delta.tool_calls = [mock_tool_call]
            chunk.choices[0].finish_reason = None

            yield chunk

        openai_provider.async_client.chat.completions.create = AsyncMock(return_value=mock_stream())

        chunks = []
        async for chunk in openai_provider.chat_stream(sample_messages, [], agent_config):
            chunks.append(chunk)

        assert len(chunks) == 1
        assert chunks[0].type == "tool_call"
        assert chunks[0].payload["id"] == "call_123"

    @pytest.mark.asyncio
    async def test_chat_stream_empty_choices(self, openai_provider, sample_messages, agent_config):
        """Test streaming handles empty choices gracefully."""

        async def mock_stream():
            chunk = Mock()
            chunk.choices = []
            yield chunk

        openai_provider.async_client.chat.completions.create = AsyncMock(return_value=mock_stream())

        chunks = []
        async for chunk in openai_provider.chat_stream(sample_messages, [], agent_config):
            chunks.append(chunk)

        assert len(chunks) == 0

    @pytest.mark.asyncio
    async def test_chat_stream_rate_limit_error(
        self, openai_provider, sample_messages, agent_config
    ):
        """Test streaming handles rate limit error."""
        from openai import RateLimitError

        # Create mock httpx response
        mock_http_response = Mock()
        mock_http_response.status_code = 429

        async def mock_stream():
            if False:
                yield  # Make this a generator
            raise RateLimitError("Rate limit exceeded", response=mock_http_response, body=None)

        openai_provider.async_client.chat.completions.create = AsyncMock(return_value=mock_stream())

        chunks = []
        async for chunk in openai_provider.chat_stream(sample_messages, [], agent_config):
            chunks.append(chunk)

        assert len(chunks) == 1
        assert chunks[0].type == "error"
        assert chunks[0].payload["error"] == "rate_limit"

    @pytest.mark.asyncio
    async def test_chat_stream_timeout_error(self, openai_provider, sample_messages, agent_config):
        """Test streaming handles timeout error."""
        from openai import APITimeoutError

        async def mock_stream():
            if False:
                yield  # Make this a generator
            raise APITimeoutError(request=Mock())

        openai_provider.async_client.chat.completions.create = AsyncMock(return_value=mock_stream())

        chunks = []
        async for chunk in openai_provider.chat_stream(sample_messages, [], agent_config):
            chunks.append(chunk)

        assert len(chunks) == 1
        assert chunks[0].type == "error"
        assert chunks[0].payload["error"] == "timeout"

    @pytest.mark.asyncio
    async def test_chat_stream_api_error(self, openai_provider, sample_messages, agent_config):
        """Test streaming handles general API error."""
        from openai import APIError

        # Create mock httpx request
        mock_request = Mock()

        async def mock_stream():
            if False:
                yield  # Make this a generator
            raise APIError("Internal error", request=mock_request, body=None)

        openai_provider.async_client.chat.completions.create = AsyncMock(return_value=mock_stream())

        chunks = []
        async for chunk in openai_provider.chat_stream(sample_messages, [], agent_config):
            chunks.append(chunk)

        assert len(chunks) == 1
        assert chunks[0].type == "error"
        assert chunks[0].payload["error"] == "api_error"
