"""Unit tests for ZAI provider."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentrunner.core.messages import Message
from agentrunner.core.tool_protocol import ToolDefinition
from agentrunner.providers.base import ProviderConfig
from agentrunner.providers.zai_provider import ZAIProvider


@pytest.fixture
def provider_config():
    """Create a test provider config."""
    return ProviderConfig(
        model="glm-4.6",
        temperature=0.7,
        max_tokens=2048,
    )


@pytest.fixture
def zai_provider(provider_config):
    """Create a ZAI provider instance."""
    with (
        patch("agentrunner.providers.zai_provider.OpenAI"),
        patch("agentrunner.providers.zai_provider.AsyncOpenAI"),
    ):
        provider = ZAIProvider(api_key="test-key", config=provider_config)
        return provider


class TestZAIProviderInit:
    """Test ZAI provider initialization."""

    def test_init_with_default_base_url(self, provider_config):
        """Test initialization with default base URL."""
        with (
            patch("agentrunner.providers.zai_provider.OpenAI") as mock_openai,
            patch("agentrunner.providers.zai_provider.AsyncOpenAI") as mock_async_openai,
        ):
            ZAIProvider(api_key="test-key", config=provider_config)

            # Check that clients were initialized with correct base URL
            mock_openai.assert_called_once_with(
                api_key="test-key",
                base_url="https://api.z.ai/api/paas/v4",
            )
            mock_async_openai.assert_called_once_with(
                api_key="test-key",
                base_url="https://api.z.ai/api/paas/v4",
            )

    def test_init_with_custom_base_url(self, provider_config):
        """Test initialization with custom base URL."""
        with (
            patch("agentrunner.providers.zai_provider.OpenAI") as mock_openai,
            patch("agentrunner.providers.zai_provider.AsyncOpenAI") as mock_async_openai,
        ):
            custom_url = "https://custom.api.endpoint/v1"
            ZAIProvider(
                api_key="test-key",
                config=provider_config,
                base_url=custom_url,
            )

            mock_openai.assert_called_once_with(api_key="test-key", base_url=custom_url)
            mock_async_openai.assert_called_once_with(api_key="test-key", base_url=custom_url)


class TestMessageConversion:
    """Test message conversion methods."""

    def test_convert_simple_messages(self, zai_provider):
        """Test converting simple user/assistant messages."""
        messages = [
            Message(id="1", role="system", content="You are helpful"),
            Message(id="2", role="user", content="Hello"),
            Message(id="3", role="assistant", content="Hi there!"),
        ]

        zai_messages = zai_provider._convert_messages_to_zai(messages)

        assert len(zai_messages) == 3
        assert zai_messages[0] == {"role": "system", "content": "You are helpful"}
        assert zai_messages[1] == {"role": "user", "content": "Hello"}
        assert zai_messages[2] == {"role": "assistant", "content": "Hi there!"}

    def test_convert_tool_message(self, zai_provider):
        """Test converting tool result messages."""
        messages = [
            Message(
                id="1",
                role="tool",
                content="result data",
                tool_call_id="call_123",
                meta={"tool_name": "test_tool"},
            )
        ]

        zai_messages = zai_provider._convert_messages_to_zai(messages)

        assert len(zai_messages) == 1
        assert zai_messages[0]["role"] == "tool"
        assert zai_messages[0]["content"] == "result data"
        assert zai_messages[0]["tool_call_id"] == "call_123"
        assert zai_messages[0]["name"] == "test_tool"

    def test_convert_assistant_with_tool_calls(self, zai_provider):
        """Test converting assistant message with tool calls."""
        messages = [
            Message(
                id="1",
                role="assistant",
                content="",
                tool_calls=[
                    {
                        "id": "call_123",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": '{"city": "NYC"}'},
                    }
                ],
            )
        ]

        zai_messages = zai_provider._convert_messages_to_zai(messages)

        assert len(zai_messages) == 1
        assert "tool_calls" in zai_messages[0]
        assert len(zai_messages[0]["tool_calls"]) == 1
        assert zai_messages[0]["tool_calls"][0]["id"] == "call_123"
        assert zai_messages[0]["tool_calls"][0]["function"]["name"] == "get_weather"


class TestToolConversion:
    """Test tool conversion methods."""

    def test_convert_tools(self, zai_provider):
        """Test converting tool definitions."""
        tools = [
            ToolDefinition(
                name="get_weather",
                description="Get weather for a city",
                parameters={
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            )
        ]

        zai_tools = zai_provider._convert_tools_to_zai(tools)

        assert len(zai_tools) == 1
        assert zai_tools[0]["type"] == "function"
        assert zai_tools[0]["function"]["name"] == "get_weather"
        assert zai_tools[0]["function"]["description"] == "Get weather for a city"
        assert "properties" in zai_tools[0]["function"]["parameters"]


class TestChat:
    """Test chat functionality."""

    @pytest.mark.asyncio
    async def test_chat_success(self, zai_provider):
        """Test successful chat completion."""
        # Mock response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Hello!"
        mock_response.choices[0].message.tool_calls = None
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5
        mock_response.usage.total_tokens = 15

        # Mock async client
        zai_provider.async_client.chat.completions.create = AsyncMock(return_value=mock_response)

        messages = [Message(id="1", role="user", content="Hi")]
        response = await zai_provider.chat(messages, tools=None)

        assert len(response.messages) == 1
        assert response.messages[0].content == "Hello!"
        assert response.usage["prompt_tokens"] == 10
        assert response.usage["completion_tokens"] == 5

    @pytest.mark.asyncio
    async def test_chat_with_tools(self, zai_provider):
        """Test chat with tool calling."""
        # Mock response with tool call
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = ""
        mock_tool_call = MagicMock()
        mock_tool_call.id = "call_123"
        mock_tool_call.function.name = "get_weather"
        mock_tool_call.function.arguments = '{"city": "NYC"}'
        mock_response.choices[0].message.tool_calls = [mock_tool_call]
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5
        mock_response.usage.total_tokens = 15

        zai_provider.async_client.chat.completions.create = AsyncMock(return_value=mock_response)

        messages = [Message(id="1", role="user", content="What's the weather?")]
        tools = [
            ToolDefinition(
                name="get_weather",
                description="Get weather",
                parameters={"type": "object", "properties": {"city": {"type": "string"}}},
            )
        ]

        response = await zai_provider.chat(messages, tools=tools)

        assert len(response.messages) == 1
        assert response.messages[0].tool_calls is not None
        assert len(response.messages[0].tool_calls) == 1
        assert response.messages[0].tool_calls[0]["function"]["name"] == "get_weather"


class TestModelInfo:
    """Test model info retrieval."""

    def test_get_model_info(self, zai_provider):
        """Test getting model information."""
        model_info = zai_provider.get_model_info()

        assert model_info.name == "glm-4.6"
        assert model_info.context_window == 128000
        assert "input_per_1k" in model_info.pricing
        assert "output_per_1k" in model_info.pricing


class TestTokenCounting:
    """Test token counting."""

    @pytest.mark.skip(reason="Skipping temporarily - CI environment issue")
    def test_count_tokens(self, zai_provider):
        """Test token counting."""
        text = "Hello, world!"
        token_count = zai_provider.count_tokens(text)

        # Should return a positive integer
        assert isinstance(token_count, int)
        assert token_count > 0


class TestToolCallParsing:
    """Test tool call parsing."""

    def test_parse_tool_calls(self, zai_provider):
        """Test parsing tool calls from message."""
        message = Message(
            id="1",
            role="assistant",
            content="",
            tool_calls=[
                {
                    "id": "call_123",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"city": "NYC"}'},
                }
            ],
        )

        tool_calls = zai_provider.parse_tool_calls(message)

        assert len(tool_calls) == 1
        assert tool_calls[0].id == "call_123"
        assert tool_calls[0].name == "get_weather"
        assert tool_calls[0].arguments == {"city": "NYC"}

    def test_parse_empty_tool_calls(self, zai_provider):
        """Test parsing message with no tool calls."""
        message = Message(id="1", role="assistant", content="Hello!")

        tool_calls = zai_provider.parse_tool_calls(message)

        assert len(tool_calls) == 0
