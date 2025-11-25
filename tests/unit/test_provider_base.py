"""Tests for provider base classes and data structures."""

import pytest

from agentrunner.core.messages import Message
from agentrunner.providers.base import (
    BaseLLMProvider,
    ModelInfo,
    ProviderConfig,
    ProviderResponse,
    StreamChunk,
)


class TestModelInfo:
    """Tests for ModelInfo dataclass."""

    def test_model_info_creation(self):
        """Test creating ModelInfo with valid data."""
        info = ModelInfo(
            name="gpt-5.1-2025-11-13",
            context_window=8192,
            pricing={"input_per_1k": 0.03, "output_per_1k": 0.06},
        )
        assert info.name == "gpt-5.1-2025-11-13"
        assert info.context_window == 8192
        assert info.pricing["input_per_1k"] == 0.03

    def test_model_info_default_pricing(self):
        """Test ModelInfo with default empty pricing dict."""
        info = ModelInfo(name="test-model", context_window=4096)
        assert info.pricing == {}

    def test_model_info_invalid_context_window(self):
        """Test ModelInfo rejects zero or negative context window."""
        with pytest.raises(ValueError, match="context_window must be positive"):
            ModelInfo(name="test", context_window=0)

        with pytest.raises(ValueError, match="context_window must be positive"):
            ModelInfo(name="test", context_window=-100)


class TestProviderConfig:
    """Tests for ProviderConfig dataclass validation."""

    def test_provider_config_valid(self):
        """Test creating ProviderConfig with valid data."""
        config = ProviderConfig(
            model="gpt-5.1-2025-11-13", temperature=0.7, max_tokens=1000, top_p=0.9
        )
        assert config.model == "gpt-5.1-2025-11-13"
        assert config.temperature == 0.7
        assert config.max_tokens == 1000
        assert config.top_p == 0.9

    def test_provider_config_empty_model(self):
        """Test ProviderConfig rejects empty model."""
        with pytest.raises(ValueError, match="model must not be empty"):
            ProviderConfig(model="")

    def test_provider_config_invalid_temperature(self):
        """Test ProviderConfig rejects invalid temperature."""
        with pytest.raises(ValueError, match="temperature must be 0.0-2.0"):
            ProviderConfig(model="gpt-5.1-2025-11-13", temperature=-0.1)

        with pytest.raises(ValueError, match="temperature must be 0.0-2.0"):
            ProviderConfig(model="gpt-5.1-2025-11-13", temperature=2.1)

    def test_provider_config_invalid_max_tokens(self):
        """Test ProviderConfig rejects invalid max_tokens."""
        with pytest.raises(ValueError, match="max_tokens must be positive"):
            ProviderConfig(model="gpt-5.1-2025-11-13", max_tokens=0)

        with pytest.raises(ValueError, match="max_tokens must be positive"):
            ProviderConfig(model="gpt-5.1-2025-11-13", max_tokens=-100)

    def test_provider_config_invalid_top_p(self):
        """Test ProviderConfig rejects invalid top_p."""
        with pytest.raises(ValueError, match="top_p must be 0.0-1.0"):
            ProviderConfig(model="gpt-5.1-2025-11-13", top_p=-0.1)

        with pytest.raises(ValueError, match="top_p must be 0.0-1.0"):
            ProviderConfig(model="gpt-5.1-2025-11-13", top_p=1.1)

    def test_provider_config_optional_none_values(self):
        """Test ProviderConfig allows None for optional values."""
        config = ProviderConfig(model="gpt-5.1-2025-11-13", max_tokens=None, top_p=None)
        assert config.max_tokens is None
        assert config.top_p is None


class TestProviderResponse:
    """Tests for ProviderResponse dataclass."""

    def test_provider_response_creation(self):
        """Test creating ProviderResponse with valid data."""
        message = Message(id="1", role="assistant", content="Hello!")
        response = ProviderResponse(
            messages=[message],
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        )
        assert len(response.messages) == 1
        assert response.messages[0].content == "Hello!"
        assert response.usage["total_tokens"] == 15

    def test_provider_response_default_usage(self):
        """Test ProviderResponse with default usage dict."""
        message = Message(id="1", role="assistant", content="Test")
        response = ProviderResponse(messages=[message])
        assert response.usage == {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

    def test_provider_response_empty_messages(self):
        """Test ProviderResponse rejects empty messages list."""
        with pytest.raises(ValueError, match="must contain at least one message"):
            ProviderResponse(messages=[])

    def test_provider_response_last_message_not_assistant(self):
        """Test ProviderResponse rejects if last message is not from assistant."""
        user_message = Message(id="1", role="user", content="Hello")
        with pytest.raises(ValueError, match="Last message.*must be from assistant"):
            ProviderResponse(messages=[user_message])

    def test_provider_response_multiple_messages_valid(self):
        """Test ProviderResponse with multiple messages where last is assistant."""
        user_msg = Message(id="1", role="user", content="Question")
        assistant_msg = Message(id="2", role="assistant", content="Answer")
        response = ProviderResponse(messages=[user_msg, assistant_msg])
        assert len(response.messages) == 2
        assert response.messages[-1].role == "assistant"


class TestStreamChunk:
    """Tests for StreamChunk dataclass."""

    def test_stream_chunk_token(self):
        """Test creating token StreamChunk."""
        chunk = StreamChunk(type="token", payload={"content": "Hello"})
        assert chunk.type == "token"
        assert chunk.payload["content"] == "Hello"

    def test_stream_chunk_tool_call(self):
        """Test creating tool_call StreamChunk."""
        chunk = StreamChunk(
            type="tool_call",
            payload={"id": "call_123", "name": "read_file", "arguments": {}},
        )
        assert chunk.type == "tool_call"
        assert chunk.payload["id"] == "call_123"

    def test_stream_chunk_status(self):
        """Test creating status StreamChunk."""
        chunk = StreamChunk(type="status", payload={"finish_reason": "stop"})
        assert chunk.type == "status"

    def test_stream_chunk_error(self):
        """Test creating error StreamChunk."""
        chunk = StreamChunk(
            type="error", payload={"error": "timeout", "message": "Request timeout"}
        )
        assert chunk.type == "error"

    def test_stream_chunk_default_payload(self):
        """Test StreamChunk with default empty payload."""
        chunk = StreamChunk(type="token")
        assert chunk.payload == {}

    def test_stream_chunk_invalid_type(self):
        """Test StreamChunk rejects invalid type."""
        with pytest.raises(ValueError, match="Invalid chunk type"):
            StreamChunk(type="invalid_type")


class TestBaseLLMProvider:
    """Tests for BaseLLMProvider abstract class."""

    def test_cannot_instantiate_base_provider(self):
        """Test that BaseLLMProvider cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseLLMProvider()  # type: ignore

    def test_subclass_must_implement_all_methods(self):
        """Test that subclass must implement all abstract methods."""

        class IncompleteProvider(BaseLLMProvider):
            """Provider that doesn't implement all methods."""

            pass

        with pytest.raises(TypeError):
            IncompleteProvider()  # type: ignore
