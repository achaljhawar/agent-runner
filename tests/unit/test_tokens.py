"""Unit tests for token counting and context management."""

from unittest.mock import Mock, patch

from agentrunner.core.messages import Message
from agentrunner.core.tokens import ContextManager, TokenCounter
from agentrunner.core.tool_protocol import ToolDefinition
from agentrunner.providers.base import ModelInfo


class TestTokenCounter:
    """Test cases for TokenCounter class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.counter = TokenCounter()

    @patch("agentrunner.core.tokens.tiktoken.get_encoding")
    @patch("agentrunner.core.tokens.tiktoken.encoding_for_model")
    def test_count_text_basic(self, mock_encoding_for_model, mock_get_encoding):
        """Test basic text token counting."""
        # Mock encoding
        mock_enc = Mock()
        mock_enc.encode.return_value = [1, 2, 3, 4, 5]  # 5 tokens
        mock_encoding_for_model.return_value = mock_enc

        result = self.counter.count_text("Hello world", "gpt-5.1-2025-11-13")

        assert result == 5
        mock_encoding_for_model.assert_called_once_with("gpt-5.1-2025-11-13")
        mock_enc.encode.assert_called_once_with("Hello world")

    @patch("agentrunner.core.tokens.tiktoken.get_encoding")
    @patch("agentrunner.core.tokens.tiktoken.encoding_for_model")
    def test_count_text_fallback_encoding(self, mock_encoding_for_model, mock_get_encoding):
        """Test fallback to cl100k_base for unknown models."""
        # Mock model not found
        mock_encoding_for_model.side_effect = KeyError("Model not found")

        mock_enc = Mock()
        mock_enc.encode.return_value = [1, 2, 3]  # 3 tokens
        mock_get_encoding.return_value = mock_enc

        result = self.counter.count_text("Hello", "unknown-model")

        assert result == 3
        mock_encoding_for_model.assert_called_once_with("unknown-model")
        mock_get_encoding.assert_called_once_with("cl100k_base")

    def test_count_text_empty(self):
        """Test counting empty text."""
        result = self.counter.count_text("", "gpt-5.1-2025-11-13")
        assert result == 0

    @patch("agentrunner.core.tokens.tiktoken.get_encoding")
    @patch("agentrunner.core.tokens.tiktoken.encoding_for_model")
    def test_count_text_caching(self, mock_encoding_for_model, mock_get_encoding):
        """Test that encodings are cached."""
        mock_enc = Mock()
        mock_enc.encode.return_value = [1, 2]
        mock_encoding_for_model.return_value = mock_enc

        # Call twice with same model
        self.counter.count_text("text1", "gpt-5.1-2025-11-13")
        self.counter.count_text("text2", "gpt-5.1-2025-11-13")

        # Should only create encoding once
        assert mock_encoding_for_model.call_count == 1

    @patch("agentrunner.core.tokens.tiktoken.get_encoding")
    @patch("agentrunner.core.tokens.tiktoken.encoding_for_model")
    def test_count_messages_basic(self, mock_encoding_for_model, mock_get_encoding):
        """Test basic message counting."""
        mock_enc = Mock()
        # Mock token counts: role=1, content=3, overhead=3
        mock_enc.encode.side_effect = lambda x: [1] if x == "user" else [1, 2, 3]
        mock_encoding_for_model.return_value = mock_enc

        messages = [Message(id="1", role="user", content="Hello")]

        result = self.counter.count_messages(messages, "gpt-5.1-2025-11-13")

        # 1 (role) + 3 (content) + 3 (message overhead) + 3 (conversation overhead) = 10
        assert result == 10

    def test_count_messages_empty(self):
        """Test counting empty message list."""
        result = self.counter.count_messages([], "gpt-5.1-2025-11-13")
        assert result == 0

    @patch("agentrunner.core.tokens.tiktoken.get_encoding")
    @patch("agentrunner.core.tokens.tiktoken.encoding_for_model")
    def test_count_messages_with_tool_calls(self, mock_encoding_for_model, mock_get_encoding):
        """Test counting messages with tool calls."""
        mock_enc = Mock()

        # Mock different token counts for different strings
        def mock_encode(text):
            if text == "assistant":
                return [1]
            elif text == "Hello":
                return [1, 2]
            elif text == "read_file":
                return [1, 2, 3]
            elif text == '{"path": "test.py"}':
                return [1, 2, 3, 4]
            return [1]

        mock_enc.encode.side_effect = mock_encode
        mock_encoding_for_model.return_value = mock_enc

        messages = [
            Message(
                id="1",
                role="assistant",
                content="Hello",
                tool_calls=[{"name": "read_file", "arguments": '{"path": "test.py"}'}],
            )
        ]

        result = self.counter.count_messages(messages, "gpt-5.1-2025-11-13")

        # 1 (role) + 2 (content) + 3 (message) + 3 (tool name) + 4 (args) + 5 (tool overhead) + 3 (conv) = 21
        assert result == 21

    @patch("agentrunner.core.tokens.tiktoken.get_encoding")
    @patch("agentrunner.core.tokens.tiktoken.encoding_for_model")
    def test_count_messages_with_tool_call_id(self, mock_encoding_for_model, mock_get_encoding):
        """Test counting messages with tool_call_id."""
        mock_enc = Mock()
        mock_enc.encode.side_effect = lambda x: [1] * len(x.split())  # 1 token per word
        mock_encoding_for_model.return_value = mock_enc

        messages = [Message(id="1", role="tool", content="File contents", tool_call_id="call_123")]

        result = self.counter.count_messages(messages, "gpt-5.1-2025-11-13")

        # 1 (role) + 2 (content) + 3 (message) + 1 (tool_call_id) + 2 (tool_call_id overhead) + 3 (conv) = 12
        assert result == 12

    @patch("agentrunner.core.tokens.tiktoken.get_encoding")
    def test_count_tools_basic(self, mock_get_encoding):
        """Test basic tool counting."""
        mock_enc = Mock()
        mock_enc.encode.side_effect = lambda x: [1] * len(x.split())  # 1 token per word
        mock_get_encoding.return_value = mock_enc

        tools = [
            ToolDefinition(
                name="read_file",
                description="Read a file",
                parameters={"type": "object", "properties": {"path": {"type": "string"}}},
            )
        ]

        result = self.counter.count_tools(tools)

        # Token count depends on parameter structure
        assert result > 0  # Basic sanity check

    def test_count_tools_empty(self):
        """Test counting empty tool list."""
        result = self.counter.count_tools([])
        assert result == 0

    @patch("agentrunner.core.tokens.tiktoken.get_encoding")
    def test_count_tools_no_parameters(self, mock_get_encoding):
        """Test counting tools without parameters."""
        mock_enc = Mock()
        mock_enc.encode.side_effect = lambda x: [1] * len(x.split())
        mock_get_encoding.return_value = mock_enc

        tools = [
            ToolDefinition(
                name="simple_tool", description="A simple tool", parameters={"type": "object"}
            )
        ]

        result = self.counter.count_tools(tools)

        # Token count depends on parameter structure
        assert result > 0  # Basic sanity check

    @patch("agentrunner.core.tokens.tiktoken.get_encoding")
    def test_count_tools_multiple(self, mock_get_encoding):
        """Test counting multiple tools."""
        mock_enc = Mock()
        mock_enc.encode.side_effect = lambda x: [1] * 2  # 2 tokens each
        mock_get_encoding.return_value = mock_enc

        tools = [
            ToolDefinition(name="tool1", description="First tool", parameters={"type": "object"}),
            ToolDefinition(name="tool2", description="Second tool", parameters={"type": "object"}),
        ]

        result = self.counter.count_tools(tools)

        # Should count tokens for both tools
        assert result > 0
        # Verify multiple tools are counted (should be more than single tool)
        single_result = self.counter.count_tools([tools[0]])
        assert result > single_result


class TestContextManager:
    """Test cases for ContextManager class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.counter = Mock(spec=TokenCounter)
        self.model_info = ModelInfo(name="gpt-5.1-2025-11-13", context_window=4096)
        self.context_manager = ContextManager(self.counter, self.model_info)

    def test_init(self):
        """Test context manager initialization."""
        assert self.context_manager.counter == self.counter
        assert self.context_manager.model_info == self.model_info
        assert self.context_manager.input_tokens_used == 0
        assert self.context_manager.output_tokens_used == 0

    def test_total_tokens(self):
        """Test total token calculation."""
        messages = [Message(id="1", role="user", content="Hello")]
        tools = [
            ToolDefinition(name="test", description="Test tool", parameters={"type": "object"})
        ]

        self.counter.count_messages.return_value = 100
        self.counter.count_tools.return_value = 50

        result = self.context_manager.total_tokens(messages, tools, response_buffer=200)

        assert result == 350  # 100 + 50 + 200
        self.counter.count_messages.assert_called_once_with(messages, "gpt-5.1-2025-11-13")
        self.counter.count_tools.assert_called_once_with(tools)

    def test_total_tokens_default_buffer(self):
        """Test total tokens with default response buffer."""
        messages = []
        tools = []

        self.counter.count_messages.return_value = 100
        self.counter.count_tools.return_value = 0

        result = self.context_manager.total_tokens(messages, tools)

        assert result == 1636  # 100 + 0 + 1536 (default buffer)

    def test_is_near_limit_default_threshold(self):
        """Test near limit check with default threshold."""
        self.context_manager.input_tokens_used = 3000
        self.context_manager.output_tokens_used = 300

        # 3300 / 4096 = 0.805 > 0.8
        assert self.context_manager.is_near_limit() is True

    def test_is_near_limit_custom_threshold(self):
        """Test near limit check with custom threshold."""
        self.context_manager.input_tokens_used = 2000
        self.context_manager.output_tokens_used = 500

        # 2500 / 4096 = 0.61 < 0.9
        assert self.context_manager.is_near_limit(threshold=0.9) is False

        # 2500 / 4096 = 0.61 > 0.5
        assert self.context_manager.is_near_limit(threshold=0.5) is True

    def test_is_near_limit_under_threshold(self):
        """Test near limit check under threshold."""
        self.context_manager.input_tokens_used = 1000
        self.context_manager.output_tokens_used = 500

        # 1500 / 4096 = 0.366 < 0.8
        assert self.context_manager.is_near_limit() is False

    def test_available_for_response(self):
        """Test available tokens calculation."""
        self.context_manager.input_tokens_used = 2000
        self.context_manager.output_tokens_used = 500

        result = self.context_manager.available_for_response()

        assert result == 1596  # 4096 - 2000 - 500

    def test_available_for_response_negative(self):
        """Test available tokens when over limit."""
        self.context_manager.input_tokens_used = 3000
        self.context_manager.output_tokens_used = 2000

        result = self.context_manager.available_for_response()

        assert result == 0  # max(0, 4096 - 5000)

    def test_record_usage(self):
        """Test recording token usage."""
        self.context_manager.record_usage(1500, 300)

        assert self.context_manager.input_tokens_used == 1500
        assert self.context_manager.output_tokens_used == 300

    def test_record_usage_cumulative(self):
        """Test cumulative token usage recording."""
        self.context_manager.record_usage(1000, 200)
        self.context_manager.record_usage(500, 100)

        assert self.context_manager.input_tokens_used == 1500
        assert self.context_manager.output_tokens_used == 300

    def test_reset_usage(self):
        """Test resetting token usage."""
        self.context_manager.input_tokens_used = 1000
        self.context_manager.output_tokens_used = 500

        self.context_manager.reset_usage()

        assert self.context_manager.input_tokens_used == 0
        assert self.context_manager.output_tokens_used == 0

    def test_get_usage_stats(self):
        """Test getting usage statistics."""
        self.context_manager.input_tokens_used = 2000
        self.context_manager.output_tokens_used = 500

        stats = self.context_manager.get_usage_stats()

        expected = {
            "input_tokens": 2000,
            "output_tokens": 500,
            "total_tokens": 2500,
            "max_tokens": 4096,
            "utilization": 2500 / 4096,
            "available": 1596,
        }

        assert stats == expected

    def test_get_usage_stats_zero_max_tokens(self):
        """Test usage stats with zero max tokens."""
        # Create a new model_info with zero context_window
        self.context_manager.model_info = ModelInfo(
            name="test", context_window=1
        )  # Can't be 0 due to validation
        self.context_manager.input_tokens_used = 100

        stats = self.context_manager.get_usage_stats()

        # With context_window=1 and usage=100, utilization should be very high
        assert stats["utilization"] > 1.0  # Over 100%


class TestModelInfo:
    """Test cases for ModelInfo dataclass."""

    def test_model_info_defaults(self):
        """Test ModelInfo with default values."""
        model = ModelInfo(name="gpt-5.1-2025-11-13", context_window=4096)

        assert model.name == "gpt-5.1-2025-11-13"
        assert model.max_tokens == 4096  # Should work via alias property
        assert model.context_window == 4096
        assert model.pricing == {}

    def test_model_info_with_pricing(self):
        """Test ModelInfo with pricing information."""
        model = ModelInfo(
            name="custom-model", context_window=2048, pricing={"input": 0.001, "output": 0.002}
        )

        assert model.name == "custom-model"
        assert model.max_tokens == 2048  # Should work via alias property
        assert model.context_window == 2048
        assert model.pricing == {"input": 0.001, "output": 0.002}
