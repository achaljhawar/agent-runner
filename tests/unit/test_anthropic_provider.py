"""Unit tests for Anthropic provider implementation.

Tests the AnthropicProvider class with mocked Anthropic API calls,
focusing on tool call normalization, streaming, and error handling.
"""

from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import pytest

from agentrunner.core.config import AgentConfig
from agentrunner.core.exceptions import ModelResponseError
from agentrunner.core.messages import Message
from agentrunner.core.tool_protocol import ToolDefinition
from agentrunner.providers.anthropic_provider import AnthropicProvider
from agentrunner.providers.base import ProviderConfig, ProviderResponse


# Mock Anthropic types for testing
@dataclass
class MockTextBlock:
    """Mock Anthropic text block."""

    type: str = "text"
    text: str = ""


@dataclass
class MockToolUseBlock:
    """Mock Anthropic tool_use block."""

    type: str = "tool_use"
    id: str = ""
    name: str = ""
    input: dict[str, Any] = None

    def __post_init__(self):
        if self.input is None:
            self.input = {}


@dataclass
class MockUsage:
    """Mock usage statistics."""

    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class MockMessage:
    """Mock Anthropic message response."""

    content: list[Any] = None
    usage: MockUsage = None
    stop_reason: str = "end_turn"

    def __post_init__(self):
        if self.content is None:
            self.content = []
        if self.usage is None:
            self.usage = MockUsage()


@pytest.fixture
def provider():
    """Create AnthropicProvider instance for testing."""
    config = ProviderConfig(model="claude-sonnet-4-5-20250929")
    # Mock the Anthropic client to avoid actual API calls during initialization
    with patch("agentrunner.providers.anthropic_provider.AsyncAnthropic") as mock_client_class:
        # Create a mock client instance with async methods
        mock_client = AsyncMock()
        mock_client_class.return_value = mock_client

        provider_instance = AnthropicProvider(api_key="test-key", config=config)
        return provider_instance


@pytest.fixture
def config():
    """Create test configuration."""
    return AgentConfig()


@pytest.fixture
def tool_definition():
    """Create test tool definition."""
    return ToolDefinition(
        name="get_weather",
        description="Get weather for a location",
        parameters={
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City name"},
                "units": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["location"],
        },
    )


class TestAnthropicProviderInit:
    """Tests for provider initialization."""

    def test_init(self):
        """Test provider initialization."""
        config = ProviderConfig(model="claude-opus-4-1-20250805")
        provider = AnthropicProvider(api_key="test-key", config=config)
        assert provider.config.model == "claude-opus-4-1-20250805"
        assert provider.client is not None

    def test_get_model_info_known_model(self, provider):
        """Test getting model info for known model."""
        info = provider.get_model_info()
        assert info.name == "claude-sonnet-4-5-20250929"
        assert info.max_tokens == 200000
        assert "input_per_1k" in info.pricing
        assert "output_per_1k" in info.pricing

    def test_get_model_info_unknown_model(self):
        """Test that unknown models raise ConfigurationError."""
        from agentrunner.core.exceptions import ConfigurationError

        config = ProviderConfig(model="claude-unknown")
        provider = AnthropicProvider(api_key="test-key", config=config)

        with pytest.raises(ConfigurationError, match="Unknown model: claude-unknown"):
            provider.get_model_info()


class TestTokenCounting:
    """Tests for token counting."""

    def test_count_tokens(self, provider):
        """Test token counting approximation."""
        text = "Hello, world! This is a test."
        count = provider.count_tokens(text)
        # Should be roughly len(text) / 4
        assert count > 0
        assert count == len(text) // 4

    def test_count_tokens_empty(self, provider):
        """Test token counting with empty string."""
        count = provider.count_tokens("")
        assert count == 0


class TestMessageConversion:
    """Tests for message format conversion."""

    def test_convert_simple_messages(self, provider):
        """Test converting simple user/assistant messages."""
        messages = [
            Message(id="1", role="system", content="You are helpful."),
            Message(id="2", role="user", content="Hello!"),
            Message(id="3", role="assistant", content="Hi there!"),
        ]

        system_msg, anthropic_msgs = provider._convert_messages(messages)

        assert system_msg == "You are helpful."
        assert len(anthropic_msgs) == 2
        assert anthropic_msgs[0]["role"] == "user"
        assert anthropic_msgs[0]["content"] == "Hello!"
        assert anthropic_msgs[1]["role"] == "assistant"
        assert anthropic_msgs[1]["content"] == "Hi there!"

    def test_convert_messages_with_tool_calls(self, provider):
        """Test converting messages with tool calls."""
        messages = [
            Message(id="1", role="user", content="What's the weather?"),
            Message(
                id="2",
                role="assistant",
                content="Let me check.",
                tool_calls=[
                    {
                        "id": "call_123",
                        "name": "get_weather",
                        "arguments": {"location": "London"},
                    }
                ],
            ),
        ]

        _, anthropic_msgs = provider._convert_messages(messages)

        assert len(anthropic_msgs) == 2
        assistant_msg = anthropic_msgs[1]
        assert assistant_msg["role"] == "assistant"
        assert isinstance(assistant_msg["content"], list)

        content = assistant_msg["content"]
        assert len(content) == 2  # text + tool_use
        assert content[0]["type"] == "text"
        assert content[0]["text"] == "Let me check."
        assert content[1]["type"] == "tool_use"
        assert content[1]["id"] == "call_123"
        assert content[1]["name"] == "get_weather"
        assert content[1]["input"] == {"location": "London"}

    def test_convert_tool_result_message(self, provider):
        """Test converting tool result messages."""
        messages = [
            Message(
                id="1",
                role="tool",
                content="Temperature is 20°C",
                tool_call_id="call_123",
            ),
        ]

        _, anthropic_msgs = provider._convert_messages(messages)

        assert len(anthropic_msgs) == 1
        assert anthropic_msgs[0]["role"] == "user"
        content = anthropic_msgs[0]["content"]
        assert isinstance(content, list)
        assert content[0]["type"] == "tool_result"
        assert content[0]["tool_use_id"] == "call_123"
        assert content[0]["content"] == "Temperature is 20°C"

    def test_convert_multiple_system_messages(self, provider):
        """Test that only first system message is used."""
        messages = [
            Message(id="1", role="system", content="First system message."),
            Message(id="2", role="system", content="Second system message."),
            Message(id="3", role="user", content="Hello!"),
        ]

        system_msg, anthropic_msgs = provider._convert_messages(messages)

        assert system_msg == "First system message."
        assert len(anthropic_msgs) == 1


class TestToolConversion:
    """Tests for tool definition conversion."""

    def test_tool_to_anthropic_format(self, provider, tool_definition):
        """Test converting tool definition to Anthropic format."""
        anthropic_tool = provider._tool_to_anthropic_format(tool_definition)

        assert anthropic_tool["name"] == "get_weather"
        assert anthropic_tool["description"] == "Get weather for a location"
        assert "input_schema" in anthropic_tool
        assert anthropic_tool["input_schema"]["type"] == "object"
        assert "location" in anthropic_tool["input_schema"]["properties"]


class TestResponseParsing:
    """Tests for response parsing."""

    def test_parse_text_response(self, provider):
        """Test parsing simple text response."""
        response = MockMessage(
            content=[MockTextBlock(text="Hello, world!")],
            usage=MockUsage(input_tokens=10, output_tokens=5),
        )

        messages = provider._parse_response(response)

        assert len(messages) == 1
        assert messages[0].role == "assistant"
        assert messages[0].content == "Hello, world!"
        assert messages[0].tool_calls is None

    def test_parse_response_with_tool_use(self, provider):
        """Test parsing response with tool_use blocks."""
        response = MockMessage(
            content=[
                MockTextBlock(text="Let me check that."),
                MockToolUseBlock(
                    id="call_123",
                    name="get_weather",
                    input={"location": "Paris"},
                ),
            ],
            usage=MockUsage(input_tokens=20, output_tokens=10),
        )

        messages = provider._parse_response(response)

        assert len(messages) == 1
        msg = messages[0]
        assert msg.role == "assistant"
        assert msg.content == "Let me check that."
        assert msg.tool_calls is not None
        assert len(msg.tool_calls) == 1
        assert msg.tool_calls[0]["id"] == "call_123"
        assert msg.tool_calls[0]["name"] == "get_weather"
        assert msg.tool_calls[0]["arguments"] == {"location": "Paris"}

    def test_parse_response_multiple_tool_calls(self, provider):
        """Test parsing response with multiple tool calls."""
        response = MockMessage(
            content=[
                MockToolUseBlock(
                    id="call_1",
                    name="tool_a",
                    input={"arg": "value1"},
                ),
                MockToolUseBlock(
                    id="call_2",
                    name="tool_b",
                    input={"arg": "value2"},
                ),
            ],
            usage=MockUsage(input_tokens=15, output_tokens=8),
        )

        messages = provider._parse_response(response)

        assert len(messages) == 1
        assert len(messages[0].tool_calls) == 2


@pytest.mark.asyncio
class TestChatCompletion:
    """Tests for chat completion."""

    async def test_chat_success(self, provider, config):
        """Test successful chat completion."""
        messages = [Message(id="1", role="user", content="Hello!")]

        mock_response = MockMessage(
            content=[MockTextBlock(text="Hi there!")],
            usage=MockUsage(input_tokens=10, output_tokens=5),
        )

        with patch.object(
            provider.client.messages, "create", new_callable=AsyncMock
        ) as mock_create:
            mock_create.return_value = mock_response

            response = await provider.chat(messages, None, config)

            assert isinstance(response, ProviderResponse)
            assert len(response.messages) == 1
            assert response.messages[0].content == "Hi there!"
            assert response.usage["prompt_tokens"] == 10
            assert response.usage["completion_tokens"] == 5
            assert response.usage["total_tokens"] == 15

    async def test_chat_with_tools(self, provider, config, tool_definition):
        """Test chat completion with tools."""
        messages = [Message(id="1", role="user", content="What's the weather?")]
        tools = [tool_definition]

        mock_response = MockMessage(
            content=[
                MockTextBlock(text="Checking weather..."),
                MockToolUseBlock(
                    id="call_123",
                    name="get_weather",
                    input={"location": "London"},
                ),
            ],
            usage=MockUsage(input_tokens=20, output_tokens=15),
        )

        with patch.object(
            provider.client.messages, "create", new_callable=AsyncMock
        ) as mock_create:
            mock_create.return_value = mock_response

            response = await provider.chat(messages, tools, config)

            # Verify tools were passed to API
            call_args = mock_create.call_args
            assert "tools" in call_args[1]
            assert len(call_args[1]["tools"]) == 1

            # Verify response
            assert len(response.messages) == 1
            msg = response.messages[0]
            assert msg.tool_calls is not None
            assert len(msg.tool_calls) == 1

    async def test_chat_with_system_message(self, provider, config):
        """Test chat with system message extraction."""
        messages = [
            Message(id="1", role="system", content="Be helpful."),
            Message(id="2", role="user", content="Hello!"),
        ]

        mock_response = MockMessage(
            content=[MockTextBlock(text="Hi!")],
            usage=MockUsage(input_tokens=10, output_tokens=3),
        )

        with patch.object(
            provider.client.messages, "create", new_callable=AsyncMock
        ) as mock_create:
            mock_create.return_value = mock_response

            await provider.chat(messages, None, config)

            # Verify system message was passed as parameter
            call_args = mock_create.call_args
            assert "system" in call_args[1]
            assert call_args[1]["system"] == "Be helpful."
            # Verify only user message in messages array
            assert len(call_args[1]["messages"]) == 1

    async def test_chat_api_error(self, provider, config):
        """Test chat with API error."""
        messages = [Message(id="1", role="user", content="Hello!")]

        error = Exception("API Error")
        error.status_code = 500

        with patch.object(
            provider.client.messages, "create", new_callable=AsyncMock
        ) as mock_create:
            mock_create.side_effect = error

            with pytest.raises(ModelResponseError) as exc_info:
                await provider.chat(messages, None, config)

            assert exc_info.value.provider == "anthropic"
            assert exc_info.value.status_code == 500


@pytest.mark.asyncio
class TestStreaming:
    """Tests for streaming chat completion."""

    async def test_chat_stream_text(self, provider, config):
        """Test streaming text response."""
        messages = [Message(id="1", role="user", content="Hello!")]

        # Mock stream events
        @dataclass
        class TextDelta:
            type: str = "text_delta"
            text: str = ""

        @dataclass
        class ContentBlockDelta:
            type: str = "content_block_delta"
            delta: Any = None

        @dataclass
        class MessageStop:
            type: str = "message_stop"

        events = [
            ContentBlockDelta(delta=TextDelta(text="Hello")),
            ContentBlockDelta(delta=TextDelta(text=" there!")),
            MessageStop(),
        ]

        mock_stream = AsyncMock()
        mock_stream.__aenter__ = AsyncMock(return_value=mock_stream)
        mock_stream.__aexit__ = AsyncMock(return_value=None)

        async def async_iter(events):
            for event in events:
                yield event

        mock_stream.__aiter__ = lambda self: async_iter(events)

        # Make stream() a regular Mock that returns the async context manager
        with patch.object(provider.client.messages, "stream", Mock(return_value=mock_stream)):
            chunks = []
            async for chunk in provider.chat_stream(messages, None, config):
                chunks.append(chunk)

            # Should have token chunks and status
            token_chunks = [c for c in chunks if c.type == "token"]
            assert len(token_chunks) >= 2
            assert any("Hello" in c.payload.get("content", "") for c in token_chunks)

    async def test_chat_stream_with_tools(self, provider, config, tool_definition):
        """Test streaming with tool calls."""
        messages = [Message(id="1", role="user", content="What's the weather?")]
        tools = [tool_definition]

        @dataclass
        class ToolUseStart:
            type: str = "tool_use"
            id: str = ""
            name: str = ""

        @dataclass
        class ContentBlockStart:
            type: str = "content_block_start"
            content_block: Any = None

        @dataclass
        class MessageStop:
            type: str = "message_stop"

        events = [
            ContentBlockStart(content_block=ToolUseStart(id="call_123", name="get_weather")),
            MessageStop(),
        ]

        mock_stream = AsyncMock()
        mock_stream.__aenter__ = AsyncMock(return_value=mock_stream)
        mock_stream.__aexit__ = AsyncMock(return_value=None)

        async def async_iter(events):
            for event in events:
                yield event

        mock_stream.__aiter__ = lambda self: async_iter(events)

        # Make stream() a regular Mock that returns the async context manager
        with patch.object(provider.client.messages, "stream", Mock(return_value=mock_stream)):
            chunks = []
            async for chunk in provider.chat_stream(messages, tools, config):
                chunks.append(chunk)

            # Should have tool_call chunks
            tool_chunks = [c for c in chunks if c.type == "tool_call"]
            assert len(tool_chunks) >= 1

    async def test_chat_stream_error(self, provider, config):
        """Test streaming with error."""
        messages = [Message(id="1", role="user", content="Hello!")]

        error = Exception("Stream error")
        error.status_code = 503

        mock_stream = AsyncMock()
        mock_stream.__aenter__ = AsyncMock(side_effect=error)

        with patch.object(provider.client.messages, "stream", return_value=mock_stream):
            chunks = []
            async for chunk in provider.chat_stream(messages, None, config):
                chunks.append(chunk)

            # Should have error chunk
            error_chunks = [c for c in chunks if c.type == "error"]
            assert len(error_chunks) == 1
            assert "error" in error_chunks[0].payload


class TestStreamEventParsing:
    """Tests for stream event parsing."""

    def test_parse_text_delta(self, provider):
        """Test parsing text delta event."""

        @dataclass
        class TextDelta:
            type: str = "text_delta"
            text: str = "Hello"

        @dataclass
        class Event:
            type: str = "content_block_delta"
            delta: Any = None

        event = Event(delta=TextDelta())
        chunk = provider._parse_stream_event(event)

        assert chunk is not None
        assert chunk.type == "token"
        assert chunk.payload["content"] == "Hello"

    def test_parse_tool_use_start(self, provider):
        """Test parsing tool_use start event."""

        @dataclass
        class ToolUseBlock:
            type: str = "tool_use"
            id: str = "call_123"
            name: str = "get_weather"

        @dataclass
        class Event:
            type: str = "content_block_start"
            content_block: Any = None

        event = Event(content_block=ToolUseBlock())
        chunk = provider._parse_stream_event(event)

        assert chunk is not None
        assert chunk.type == "tool_call"
        assert chunk.payload["tool_call_id"] == "call_123"
        assert chunk.payload["tool_name"] == "get_weather"

    def test_parse_message_start(self, provider):
        """Test parsing message start event."""

        @dataclass
        class Message:
            usage: Any = None

        @dataclass
        class Event:
            type: str = "message_start"
            message: Any = None

        event = Event(message=Message(usage=MockUsage(input_tokens=10)))
        chunk = provider._parse_stream_event(event)

        assert chunk is not None
        assert chunk.type == "status"
        assert chunk.payload["input_tokens"] == 10

    def test_parse_unknown_event(self, provider):
        """Test parsing unknown event type."""

        @dataclass
        class Event:
            type: str = "unknown_event"

        event = Event()
        chunk = provider._parse_stream_event(event)

        assert chunk is None


class TestEdgeCases:
    """Tests for edge cases and error conditions."""

    def test_empty_response_content(self, provider):
        """Test parsing response with empty content."""
        response = MockMessage(
            content=[],
            usage=MockUsage(input_tokens=5, output_tokens=0),
        )

        messages = provider._parse_response(response)

        assert len(messages) == 1
        assert messages[0].content == ""
        assert messages[0].tool_calls is None

    def test_convert_empty_messages(self, provider):
        """Test converting empty message list."""
        system_msg, anthropic_msgs = provider._convert_messages([])

        assert system_msg is None
        assert len(anthropic_msgs) == 0

    def test_assistant_message_no_content(self, provider):
        """Test assistant message with only tool calls."""
        messages = [
            Message(
                id="1",
                role="assistant",
                content="",
                tool_calls=[{"id": "call_1", "name": "tool_a", "arguments": {"x": 1}}],
            )
        ]

        _, anthropic_msgs = provider._convert_messages(messages)

        assert len(anthropic_msgs) == 1
        content = anthropic_msgs[0]["content"]
        # Should not have empty text block
        assert all(block["type"] != "text" or block["text"] for block in content)
