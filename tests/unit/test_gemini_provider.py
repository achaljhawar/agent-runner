"""Tests for Gemini provider implementation."""

from unittest.mock import AsyncMock, Mock, patch

import pytest
from google.api_core import exceptions as google_exceptions
from google.genai import types

from agentrunner.core.config import AgentConfig
from agentrunner.core.exceptions import ModelResponseError, TokenLimitExceededError
from agentrunner.core.messages import Message
from agentrunner.core.tool_protocol import ToolDefinition
from agentrunner.providers.base import ProviderConfig
from agentrunner.providers.gemini_provider import GeminiProvider


@pytest.fixture
def gemini_provider():
    """Create GeminiProvider instance with mocked API."""
    with patch("google.genai.Client") as mock_client_class:
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        config = ProviderConfig(model="gemini-2.5-pro")
        provider = GeminiProvider(api_key="test-key", config=config)
        provider.client = mock_client
        return provider


@pytest.fixture
def agent_config():
    """Create test agent config."""
    return AgentConfig(
        max_rounds=10,
    )


@pytest.fixture
def sample_messages():
    """Create sample messages."""
    return [
        Message(id="1", role="system", content="You are a helpful assistant."),
        Message(id="2", role="user", content="Hello, how are you?"),
    ]


@pytest.fixture
def sample_tools():
    """Create sample tool definitions."""
    return [
        ToolDefinition(
            name="get_weather",
            description="Get weather for a location",
            parameters={
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name"},
                    "units": {"type": "string", "description": "Temperature units"},
                },
                "required": ["location"],
            },
        )
    ]


class TestGeminiProviderInit:
    """Tests for GeminiProvider initialization."""

    def test_init_with_default_model(self):
        """Test initialization with default model."""
        with patch("google.genai.Client") as mock_client_class:
            mock_client = Mock()
            mock_client_class.return_value = mock_client
            config = ProviderConfig(model="gemini-2.5-pro")
            provider = GeminiProvider(api_key="test-key", config=config)
            mock_client_class.assert_called_once_with(api_key="test-key")
            assert provider.config.model == "gemini-2.5-pro"

    def test_init_with_custom_model(self):
        """Test initialization with custom model."""
        with patch("google.genai.Client"):
            config = ProviderConfig(model="gemini-2.5-flash")
            provider = GeminiProvider(api_key="test-key", config=config)
            assert provider.config.model == "gemini-2.5-flash"


class TestGetModelInfo:
    """Tests for get_model_info method."""

    def test_get_model_info_pro(self, gemini_provider):
        """Test getting model info for Pro model."""
        info = gemini_provider.get_model_info()
        assert info.name == "gemini-2.5-pro"
        assert info.context_window == 2097152
        assert "input_per_1k" in info.pricing
        assert "output_per_1k" in info.pricing

    def test_get_model_info_flash(self):
        """Test getting model info for Flash model."""
        with patch("google.genai.Client"):
            config = ProviderConfig(model="gemini-2.5-flash")
            provider = GeminiProvider(api_key="test-key", config=config)
            info = provider.get_model_info()
            assert info.name == "gemini-2.5-flash"
            assert info.max_tokens == 1048576

    def test_get_model_info_unknown(self):
        """Test that unknown models raise ConfigurationError."""
        from agentrunner.core.exceptions import ConfigurationError

        with patch("google.genai.Client"):
            config = ProviderConfig(model="unknown-model")
            provider = GeminiProvider(api_key="test-key", config=config)

            with pytest.raises(ConfigurationError, match="Unknown model: unknown-model"):
                provider.get_model_info()


class TestCountTokens:
    """Tests for count_tokens method."""

    def test_count_tokens_success(self, gemini_provider):
        """Test successful token counting."""
        mock_result = Mock()
        mock_result.total_tokens = 42
        gemini_provider.client.models.count_tokens.return_value = mock_result

        count = gemini_provider.count_tokens("Hello world")
        assert count == 42
        gemini_provider.client.models.count_tokens.assert_called_once_with(
            model=gemini_provider._api_model_name, contents="Hello world"
        )

    def test_count_tokens_fallback(self, gemini_provider):
        """Test fallback when count_tokens fails."""
        gemini_provider.client.models.count_tokens.side_effect = Exception("API error")

        count = gemini_provider.count_tokens("Hello world test")
        # Fallback: len(text) // 4 = 16 // 4 = 4
        assert count == 4


class TestConvertMessages:
    """Tests for _convert_messages method."""

    def test_convert_simple_messages(self, gemini_provider):
        """Test converting simple user/assistant messages."""
        messages = [
            Message(id="1", role="user", content="Hello"),
            Message(id="2", role="assistant", content="Hi there"),
        ]

        gemini_messages, system_instruction = gemini_provider._convert_messages(messages)

        assert system_instruction is None
        assert len(gemini_messages) == 2
        assert gemini_messages[0].role == "user"
        assert gemini_messages[0].parts[0].text == "Hello"
        assert gemini_messages[1].role == "model"
        assert gemini_messages[1].parts[0].text == "Hi there"

    def test_convert_system_message(self, gemini_provider):
        """Test extracting system instruction."""
        messages = [
            Message(id="1", role="system", content="You are helpful"),
            Message(id="2", role="user", content="Hello"),
        ]

        gemini_messages, system_instruction = gemini_provider._convert_messages(messages)

        assert system_instruction == "You are helpful"
        # Should return as string when single user message with text only
        assert isinstance(gemini_messages, str)
        assert gemini_messages == "Hello"

    def test_convert_tool_calls(self, gemini_provider):
        """Test converting assistant message with tool calls."""
        messages = [
            Message(
                id="1",
                role="assistant",
                content="",
                tool_calls=[
                    {"id": "call_1", "name": "get_weather", "arguments": {"location": "NYC"}}
                ],
            )
        ]

        gemini_messages, _ = gemini_provider._convert_messages(messages)

        assert len(gemini_messages) == 1
        assert gemini_messages[0].role == "model"
        assert len(gemini_messages[0].parts) == 1
        assert gemini_messages[0].parts[0].function_call.name == "get_weather"
        assert gemini_messages[0].parts[0].function_call.args == {"location": "NYC"}

    def test_convert_tool_response(self, gemini_provider):
        """Test converting tool result message."""
        messages = [
            Message(
                id="1",
                role="tool",
                content="Weather is sunny",
                tool_call_id="get_weather",
                meta={"tool_name": "get_weather"},
            )
        ]

        gemini_messages, _ = gemini_provider._convert_messages(messages)

        assert len(gemini_messages) == 1
        assert gemini_messages[0].role == "user"
        # Tool messages have both text content and function_response
        assert len(gemini_messages[0].parts) == 2
        assert gemini_messages[0].parts[0].text == "Weather is sunny"
        assert gemini_messages[0].parts[1].function_response.name == "get_weather"
        assert gemini_messages[0].parts[1].function_response.response == {
            "result": "Weather is sunny"
        }


class TestConvertTools:
    """Tests for _convert_tools method."""

    def test_convert_simple_tool(self, gemini_provider):
        """Test converting simple tool definition."""
        tools = [
            ToolDefinition(
                name="test_tool",
                description="A test tool",
                parameters={
                    "type": "object",
                    "properties": {"param": {"type": "string", "description": "A parameter"}},
                    "required": ["param"],
                },
            )
        ]

        gemini_tools = gemini_provider._convert_tools(tools)

        # Returns list of Tool objects
        assert len(gemini_tools) == 1
        assert isinstance(gemini_tools[0], types.Tool)
        assert len(gemini_tools[0].function_declarations) == 1
        func_decl = gemini_tools[0].function_declarations[0]
        assert func_decl.name == "test_tool"
        assert func_decl.description == "A test tool"
        assert func_decl.parameters is not None

    def test_convert_multiple_tools(self, gemini_provider, sample_tools):
        """Test converting multiple tools."""
        tools = [
            *sample_tools,
            ToolDefinition(
                name="another_tool",
                description="Another tool",
                parameters={"type": "object", "properties": {}},
            ),
        ]

        gemini_tools = gemini_provider._convert_tools(tools)

        # All tools are wrapped in a single Tool object
        assert len(gemini_tools) == 1
        assert len(gemini_tools[0].function_declarations) == 2
        assert gemini_tools[0].function_declarations[0].name == "get_weather"
        assert gemini_tools[0].function_declarations[1].name == "another_tool"


class TestSanitizeSchema:
    """Tests for _sanitize_schema_for_gemini method."""

    def test_sanitize_simple_schema(self, gemini_provider):
        """Test sanitizing simple JSON schema."""
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string", "description": "Name"}},
            "required": ["name"],
            "additionalProperties": False,  # Should be removed
            "title": "Test Schema",  # Should be removed
        }

        sanitized_schema = gemini_provider._sanitize_schema_for_gemini(schema)

        assert sanitized_schema["type"] == "object"
        assert "name" in sanitized_schema["properties"]
        assert sanitized_schema["required"] == ["name"]
        assert "additionalProperties" not in sanitized_schema
        assert "title" not in sanitized_schema

    def test_sanitize_nested_schema(self, gemini_provider):
        """Test sanitizing nested schema with arrays."""
        schema = {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of items",
                    "default": [],  # Should be removed
                }
            },
            "$schema": "http://json-schema.org/draft-07/schema#",  # Should be removed
        }

        sanitized_schema = gemini_provider._sanitize_schema_for_gemini(schema)

        assert sanitized_schema["type"] == "object"
        assert "items" in sanitized_schema["properties"]
        assert sanitized_schema["properties"]["items"]["type"] == "array"
        assert "default" not in sanitized_schema["properties"]["items"]
        assert "$schema" not in sanitized_schema


@pytest.mark.asyncio
class TestChat:
    """Tests for chat method."""

    async def test_chat_success(self, gemini_provider, sample_messages, agent_config):
        """Test successful chat completion."""
        # Mock response
        mock_response = Mock()
        mock_candidate = Mock()
        mock_part = Mock()
        mock_part.text = "Hello! I'm doing well."
        mock_part.function_call = None
        mock_candidate.content.parts = [mock_part]
        mock_response.candidates = [mock_candidate]
        mock_response.usage_metadata = Mock(
            prompt_token_count=10, candidates_token_count=8, total_token_count=18
        )

        # Mock generate_content
        with patch("asyncio.to_thread", new_callable=AsyncMock) as mock_to_thread:
            mock_to_thread.return_value = mock_response

            response = await gemini_provider.chat(sample_messages, None, agent_config)

            assert len(response.messages) == 1
            assert response.messages[0].role == "assistant"
            assert response.messages[0].content == "Hello! I'm doing well."
            assert response.usage["prompt_tokens"] == 10
            assert response.usage["completion_tokens"] == 8
            assert response.usage["total_tokens"] == 18

    async def test_chat_with_tools(
        self, gemini_provider, sample_messages, sample_tools, agent_config
    ):
        """Test chat with tool definitions."""
        # Mock response with function call
        mock_response = Mock()
        mock_candidate = Mock()
        mock_part = Mock()
        mock_part.text = ""
        mock_function_call = Mock()
        mock_function_call.name = "get_weather"
        mock_function_call.args = {"location": "NYC"}
        mock_part.function_call = mock_function_call
        mock_candidate.content.parts = [mock_part]
        mock_response.candidates = [mock_candidate]
        mock_response.usage_metadata = Mock(
            prompt_token_count=20, candidates_token_count=5, total_token_count=25
        )

        with patch("asyncio.to_thread", new_callable=AsyncMock) as mock_to_thread:
            mock_to_thread.return_value = mock_response

            response = await gemini_provider.chat(sample_messages, sample_tools, agent_config)

            assert len(response.messages) == 1
            assert response.messages[0].tool_calls is not None
            assert len(response.messages[0].tool_calls) == 1
            assert response.messages[0].tool_calls[0]["name"] == "get_weather"

    async def test_chat_with_system_instruction(self, gemini_provider, agent_config):
        """Test chat with system instruction."""
        messages = [
            Message(id="1", role="system", content="Be concise"),
            Message(id="2", role="user", content="Hi"),
        ]

        mock_response = Mock()
        mock_candidate = Mock()
        mock_part = Mock()
        mock_part.text = "Hello"
        mock_part.function_call = None
        mock_candidate.content = Mock()
        mock_candidate.content.parts = [mock_part]
        mock_response.candidates = [mock_candidate]
        mock_response.usage_metadata = Mock(
            prompt_token_count=5, candidates_token_count=2, total_token_count=7
        )

        with patch("asyncio.to_thread", new_callable=AsyncMock) as mock_to_thread:
            mock_to_thread.return_value = mock_response

            response = await gemini_provider.chat(messages, None, agent_config)

            # Verify the call was made with system instruction in config
            mock_to_thread.assert_called_once()
            call_args = mock_to_thread.call_args
            # Check that system_instruction was passed in the config
            assert "config" in call_args.kwargs or len(call_args.args) >= 4
            assert response.messages[0].content == "Hello"

    async def test_chat_token_limit_error(self, gemini_provider, sample_messages, agent_config):
        """Test chat with token limit exceeded."""
        with patch("asyncio.to_thread", new_callable=AsyncMock) as mock_to_thread:
            mock_to_thread.side_effect = google_exceptions.InvalidArgument(
                "context_window exceeded"
            )

            with pytest.raises(TokenLimitExceededError):
                await gemini_provider.chat(sample_messages, None, agent_config)

    async def test_chat_api_error(self, gemini_provider, sample_messages, agent_config):
        """Test chat with API error."""
        with patch("asyncio.to_thread", new_callable=AsyncMock) as mock_to_thread:
            mock_to_thread.side_effect = google_exceptions.GoogleAPIError("API error")

            with pytest.raises(ModelResponseError):
                await gemini_provider.chat(sample_messages, None, agent_config)

    async def test_chat_unexpected_error(self, gemini_provider, sample_messages, agent_config):
        """Test chat with unexpected error."""
        with patch("asyncio.to_thread", new_callable=AsyncMock) as mock_to_thread:
            mock_to_thread.side_effect = ValueError("Unexpected")

            with pytest.raises(ModelResponseError):
                await gemini_provider.chat(sample_messages, None, agent_config)


@pytest.mark.asyncio
class TestChatStream:
    """Tests for chat_stream method."""

    async def test_chat_stream_text_only(self, gemini_provider, sample_messages, agent_config):
        """Test streaming text-only response."""
        # Mock streaming response
        mock_chunk1 = Mock()
        mock_chunk1.candidates = [Mock()]
        mock_chunk1.candidates[0].content = Mock()
        mock_chunk1.candidates[0].content.parts = [Mock()]
        mock_chunk1.candidates[0].content.parts[0].text = "Hello"
        mock_chunk1.candidates[0].content.parts[0].function_call = None

        mock_chunk2 = Mock()
        mock_chunk2.candidates = [Mock()]
        mock_chunk2.candidates[0].content = Mock()
        mock_chunk2.candidates[0].content.parts = [Mock()]
        mock_chunk2.candidates[0].content.parts[0].text = "Hello world"
        mock_chunk2.candidates[0].content.parts[0].function_call = None

        mock_stream = [mock_chunk1, mock_chunk2]

        gemini_provider.client.models.generate_content_stream.return_value = mock_stream

        chunks = []
        async for chunk in gemini_provider.chat_stream(sample_messages, None, agent_config):
            chunks.append(chunk)

        # Should have 2 token chunks + 1 status chunk
        assert len(chunks) >= 2
        assert chunks[0].type == "token"
        assert chunks[0].payload["content"] == "Hello"
        assert chunks[-1].type == "status"

    async def test_chat_stream_with_tool_call(
        self, gemini_provider, sample_messages, sample_tools, agent_config
    ):
        """Test streaming response with tool call."""
        mock_chunk = Mock()
        mock_chunk.candidates = [Mock()]
        mock_part = Mock()
        mock_function_call = Mock()
        mock_function_call.name = "get_weather"
        mock_function_call.args = {"location": "NYC"}
        mock_part.text = ""
        mock_part.function_call = mock_function_call
        mock_chunk.candidates[0].content = Mock()
        mock_chunk.candidates[0].content.parts = [mock_part]

        mock_stream = [mock_chunk]

        gemini_provider.client.models.generate_content_stream.return_value = mock_stream

        chunks = []
        async for chunk in gemini_provider.chat_stream(sample_messages, sample_tools, agent_config):
            chunks.append(chunk)

        # Find tool_call chunk
        tool_chunks = [c for c in chunks if c.type == "tool_call"]
        assert len(tool_chunks) == 1
        assert tool_chunks[0].payload["name"] == "get_weather"
        assert "id" in tool_chunks[0].payload

    async def test_chat_stream_api_error(self, gemini_provider, sample_messages, agent_config):
        """Test streaming with API error."""
        gemini_provider.client.models.generate_content_stream.side_effect = (
            google_exceptions.GoogleAPIError("API error")
        )

        chunks = []
        async for chunk in gemini_provider.chat_stream(sample_messages, None, agent_config):
            chunks.append(chunk)

        # Should have an error chunk
        assert len(chunks) == 1
        assert chunks[0].type == "error"

    async def test_chat_stream_empty_candidates(
        self, gemini_provider, sample_messages, agent_config
    ):
        """Test streaming with empty candidates."""
        mock_chunk = Mock()
        mock_chunk.candidates = []

        mock_stream = [mock_chunk]

        gemini_provider.client.models.generate_content_stream.return_value = mock_stream

        chunks = []
        async for chunk in gemini_provider.chat_stream(sample_messages, None, agent_config):
            chunks.append(chunk)

        # Should only have status chunk
        assert len(chunks) == 1
        assert chunks[0].type == "status"


class TestParseResponse:
    """Tests for _parse_response method."""

    def test_parse_text_only(self, gemini_provider):
        """Test parsing text-only response."""
        mock_response = Mock()
        mock_candidate = Mock()
        mock_part = Mock()
        mock_part.text = "Hello world"
        mock_part.function_call = None
        mock_candidate.content = Mock()
        mock_candidate.content.parts = [mock_part]
        mock_response.candidates = [mock_candidate]
        mock_response.usage_metadata = Mock(
            prompt_token_count=5, candidates_token_count=3, total_token_count=8
        )

        message, usage = gemini_provider._parse_response(mock_response)

        assert message.role == "assistant"
        assert message.content == "Hello world"
        assert message.tool_calls is None
        assert usage["prompt_tokens"] == 5
        assert usage["completion_tokens"] == 3

    def test_parse_with_function_call(self, gemini_provider):
        """Test parsing response with function call."""
        mock_response = Mock()
        mock_candidate = Mock()
        mock_part = Mock()
        mock_part.text = ""
        mock_function_call = Mock()
        mock_function_call.name = "test_func"
        mock_function_call.args = {"arg": "value"}
        mock_part.function_call = mock_function_call
        mock_candidate.content = Mock()
        mock_candidate.content.parts = [mock_part]
        mock_response.candidates = [mock_candidate]
        mock_response.usage_metadata = Mock(
            prompt_token_count=10, candidates_token_count=5, total_token_count=15
        )

        message, _usage = gemini_provider._parse_response(mock_response)

        assert message.tool_calls is not None
        assert len(message.tool_calls) == 1
        assert message.tool_calls[0]["name"] == "test_func"
        assert message.tool_calls[0]["arguments"] == {"arg": "value"}
        assert "id" in message.tool_calls[0]

    def test_parse_mixed_content(self, gemini_provider):
        """Test parsing response with text and function call."""
        mock_response = Mock()
        mock_candidate = Mock()

        mock_part1 = Mock()
        mock_part1.text = "I'll check that for you."
        mock_part1.function_call = None

        mock_part2 = Mock()
        mock_part2.text = ""
        mock_function_call = Mock()
        mock_function_call.name = "get_weather"
        mock_function_call.args = {"location": "NYC"}
        mock_part2.function_call = mock_function_call

        mock_candidate.content = Mock()
        mock_candidate.content.parts = [mock_part1, mock_part2]
        mock_response.candidates = [mock_candidate]
        mock_response.usage_metadata = Mock(
            prompt_token_count=15, candidates_token_count=10, total_token_count=25
        )

        message, _usage = gemini_provider._parse_response(mock_response)

        assert message.content == "I'll check that for you."
        assert message.tool_calls is not None
        assert len(message.tool_calls) == 1


class TestEdgeCases:
    """Tests for edge cases and error conditions."""

    def test_empty_message_list(self, gemini_provider):
        """Test converting empty message list."""
        gemini_messages, system_instruction = gemini_provider._convert_messages([])

        assert gemini_messages == []
        assert system_instruction is None

    def test_message_without_content(self, gemini_provider):
        """Test converting message without content."""
        messages = [Message(id="1", role="assistant", content="")]

        gemini_messages, _ = gemini_provider._convert_messages(messages)

        # Should still create a message but with empty parts
        assert len(gemini_messages) == 0  # No parts means no message

    def test_tool_without_properties(self, gemini_provider):
        """Test converting tool without properties."""
        tools = [
            ToolDefinition(name="simple_tool", description="Simple", parameters={"type": "object"})
        ]

        gemini_tools = gemini_provider._convert_tools(tools)

        assert len(gemini_tools) == 1
        assert len(gemini_tools[0].function_declarations) == 1
        assert gemini_tools[0].function_declarations[0].name == "simple_tool"
