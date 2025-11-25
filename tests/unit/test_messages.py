"""Comprehensive unit tests for messages module.

Tests the Message dataclass and MessageHistory class with >95% coverage.
"""

import uuid
from unittest.mock import patch

import pytest

from agentrunner.core.messages import Message, MessageHistory


class TestMessage:
    """Test cases for Message dataclass."""

    def test_message_creation_valid_system(self):
        """Test creating a valid system message."""
        msg = Message(
            id="test-id", role="system", content="You are a helpful assistant.", meta={"tokens": 10}
        )

        assert msg.id == "test-id"
        assert msg.role == "system"
        assert msg.content == "You are a helpful assistant."
        assert msg.tool_calls is None
        assert msg.tool_call_id is None
        assert msg.meta == {"tokens": 10}

    def test_message_creation_valid_user(self):
        """Test creating a valid user message."""
        msg = Message(
            id="test-id",
            role="user",
            content="Hello world",
        )

        assert msg.id == "test-id"
        assert msg.role == "user"
        assert msg.content == "Hello world"
        assert msg.tool_calls is None
        assert msg.tool_call_id is None
        assert msg.meta == {}

    def test_message_creation_valid_assistant_no_tools(self):
        """Test creating assistant message without tool calls."""
        msg = Message(
            id="test-id",
            role="assistant",
            content="I can help you with that.",
        )

        assert msg.id == "test-id"
        assert msg.role == "assistant"
        assert msg.content == "I can help you with that."
        assert msg.tool_calls is None
        assert msg.tool_call_id is None

    def test_message_creation_valid_assistant_with_tools(self):
        """Test creating assistant message with tool calls."""
        tool_calls = [{"id": "call-1", "name": "read_file", "arguments": {"file_path": "/test.py"}}]

        msg = Message(
            id="test-id", role="assistant", content="Let me read that file.", tool_calls=tool_calls
        )

        assert msg.id == "test-id"
        assert msg.role == "assistant"
        assert msg.content == "Let me read that file."
        assert msg.tool_calls == tool_calls
        assert msg.tool_call_id is None

    def test_message_creation_valid_tool(self):
        """Test creating a valid tool message."""
        msg = Message(
            id="test-id", role="tool", content="File contents here", tool_call_id="call-1"
        )

        assert msg.id == "test-id"
        assert msg.role == "tool"
        assert msg.content == "File contents here"
        assert msg.tool_calls is None
        assert msg.tool_call_id == "call-1"

    def test_message_creation_invalid_role(self):
        """Test that invalid role raises ValueError."""
        with pytest.raises(ValueError, match="Invalid role: invalid"):
            Message(id="test-id", role="invalid", content="test")

    def test_message_tool_missing_tool_call_id(self):
        """Test that tool message without tool_call_id raises ValueError."""
        with pytest.raises(ValueError, match="tool messages MUST have tool_call_id"):
            Message(id="test-id", role="tool", content="test")

    def test_message_tool_empty_tool_call_id(self):
        """Test that tool message with empty tool_call_id raises ValueError."""
        with pytest.raises(ValueError, match="tool messages MUST have tool_call_id"):
            Message(id="test-id", role="tool", content="test", tool_call_id="")

    def test_message_non_tool_with_tool_call_id(self):
        """Test that non-tool message with tool_call_id raises ValueError."""
        with pytest.raises(ValueError, match="Only tool messages may have tool_call_id"):
            Message(id="test-id", role="user", content="test", tool_call_id="call-1")

    def test_message_non_assistant_with_tool_calls(self):
        """Test that non-assistant message with tool_calls raises ValueError."""
        tool_calls = [{"id": "call-1", "name": "test", "arguments": {}}]

        with pytest.raises(ValueError, match="Only assistant messages may have tool_calls"):
            Message(id="test-id", role="user", content="test", tool_calls=tool_calls)


class TestMessageHistory:
    """Test cases for MessageHistory class."""

    def test_init_empty(self):
        """Test MessageHistory initialization."""
        history = MessageHistory()
        assert history.messages == []
        assert history.get() == []

    def test_add_message_valid(self):
        """Test adding a valid message."""
        history = MessageHistory()
        msg = Message(id="test", role="user", content="hello")

        history.add(msg)

        assert len(history.messages) == 1
        assert history.messages[0] == msg

    def test_add_message_invalid(self):
        """Test adding an invalid message raises ValueError."""
        history = MessageHistory()

        with pytest.raises(ValueError):
            invalid_msg = Message(id="test", role="invalid", content="test")
            history.add(invalid_msg)

    @patch("uuid.uuid4")
    @patch("agentrunner.core.messages.datetime")
    def test_add_system(self, mock_datetime, mock_uuid):
        """Test adding system message."""
        mock_uuid.return_value = uuid.UUID("12345678-1234-1234-1234-123456789abc")
        mock_datetime.now.return_value.isoformat.return_value = "2024-01-01T12:00:00+00:00"

        history = MessageHistory()
        history.add_system("You are helpful")

        assert len(history.messages) == 1
        msg = history.messages[0]
        assert msg.id == "12345678-1234-1234-1234-123456789abc"
        assert msg.role == "system"
        assert msg.content == "You are helpful"
        assert msg.meta == {"ts": "2024-01-01T12:00:00+00:00"}

    @patch("uuid.uuid4")
    @patch("agentrunner.core.messages.datetime")
    def test_add_user(self, mock_datetime, mock_uuid):
        """Test adding user message."""
        mock_uuid.return_value = uuid.UUID("12345678-1234-1234-1234-123456789abc")
        mock_datetime.now.return_value.isoformat.return_value = "2024-01-01T12:00:00+00:00"

        history = MessageHistory()
        history.add_user("Hello world")

        assert len(history.messages) == 1
        msg = history.messages[0]
        assert msg.id == "12345678-1234-1234-1234-123456789abc"
        assert msg.role == "user"
        assert msg.content == "Hello world"
        assert msg.meta == {"ts": "2024-01-01T12:00:00+00:00"}

    @patch("uuid.uuid4")
    @patch("agentrunner.core.messages.datetime")
    def test_add_assistant_no_tools(self, mock_datetime, mock_uuid):
        """Test adding assistant message without tools."""
        mock_uuid.return_value = uuid.UUID("12345678-1234-1234-1234-123456789abc")
        mock_datetime.now.return_value.isoformat.return_value = "2024-01-01T12:00:00+00:00"

        history = MessageHistory()
        history.add_assistant("I can help")

        assert len(history.messages) == 1
        msg = history.messages[0]
        assert msg.id == "12345678-1234-1234-1234-123456789abc"
        assert msg.role == "assistant"
        assert msg.content == "I can help"
        assert msg.tool_calls is None
        assert msg.meta == {"ts": "2024-01-01T12:00:00+00:00"}

    @patch("uuid.uuid4")
    @patch("agentrunner.core.messages.datetime")
    def test_add_assistant_with_tools(self, mock_datetime, mock_uuid):
        """Test adding assistant message with tool calls."""
        mock_uuid.return_value = uuid.UUID("12345678-1234-1234-1234-123456789abc")
        mock_datetime.now.return_value.isoformat.return_value = "2024-01-01T12:00:00+00:00"

        tool_calls = [{"id": "call-1", "name": "read_file", "arguments": {}}]

        history = MessageHistory()
        history.add_assistant("Let me read that", tool_calls)

        assert len(history.messages) == 1
        msg = history.messages[0]
        assert msg.id == "12345678-1234-1234-1234-123456789abc"
        assert msg.role == "assistant"
        assert msg.content == "Let me read that"
        assert msg.tool_calls == tool_calls
        assert msg.meta == {"ts": "2024-01-01T12:00:00+00:00"}

    @patch("uuid.uuid4")
    @patch("agentrunner.core.messages.datetime")
    def test_add_tool(self, mock_datetime, mock_uuid):
        """Test adding tool message."""
        mock_uuid.return_value = uuid.UUID("12345678-1234-1234-1234-123456789abc")
        mock_datetime.now.return_value.isoformat.return_value = "2024-01-01T12:00:00+00:00"

        history = MessageHistory()
        history.add_tool("File contents", "call-1")

        assert len(history.messages) == 1
        msg = history.messages[0]
        assert msg.id == "12345678-1234-1234-1234-123456789abc"
        assert msg.role == "tool"
        assert msg.content == "File contents"
        assert msg.tool_call_id == "call-1"
        assert msg.meta == {"ts": "2024-01-01T12:00:00+00:00"}

    def test_add_tool_empty_id(self):
        """Test adding tool message with empty tool_call_id raises ValueError."""
        history = MessageHistory()

        with pytest.raises(ValueError, match="tool_call_id cannot be empty"):
            history.add_tool("content", "")

    def test_get_returns_copy(self):
        """Test that get() returns a copy of messages."""
        history = MessageHistory()
        msg1 = Message(id="1", role="user", content="hello")
        msg2 = Message(id="2", role="assistant", content="hi")

        history.add(msg1)
        history.add(msg2)

        messages = history.get()
        assert len(messages) == 2
        assert messages[0] == msg1
        assert messages[1] == msg2

        # Verify it's a copy
        messages.append(Message(id="3", role="user", content="test"))
        assert len(history.get()) == 2

    def test_clear(self):
        """Test clearing message history."""
        history = MessageHistory()
        history.add_user("hello")
        history.add_assistant("hi")

        assert len(history.messages) == 2

        history.clear()

        assert len(history.messages) == 0
        assert history.get() == []

    def test_serialize_empty(self):
        """Test serializing empty history."""
        history = MessageHistory()

        data = history.serialize()

        assert data == {"messages": []}

    def test_serialize_with_messages(self):
        """Test serializing history with messages."""
        history = MessageHistory()

        # Add various message types
        msg1 = Message(id="1", role="system", content="You are helpful", meta={"tokens": 5})
        msg2 = Message(id="2", role="user", content="Hello")
        msg3 = Message(
            id="3",
            role="assistant",
            content="Hi there",
            tool_calls=[{"id": "call-1", "name": "test", "arguments": {}}],
            meta={"tokens": 8},
        )
        msg4 = Message(id="4", role="tool", content="result", tool_call_id="call-1")

        history.add(msg1)
        history.add(msg2)
        history.add(msg3)
        history.add(msg4)

        data = history.serialize()

        expected = {
            "messages": [
                {
                    "id": "1",
                    "role": "system",
                    "content": "You are helpful",
                    "tool_calls": None,
                    "tool_call_id": None,
                    "meta": {"tokens": 5},
                },
                {
                    "id": "2",
                    "role": "user",
                    "content": "Hello",
                    "tool_calls": None,
                    "tool_call_id": None,
                    "meta": {},
                },
                {
                    "id": "3",
                    "role": "assistant",
                    "content": "Hi there",
                    "tool_calls": [{"id": "call-1", "name": "test", "arguments": {}}],
                    "tool_call_id": None,
                    "meta": {"tokens": 8},
                },
                {
                    "id": "4",
                    "role": "tool",
                    "content": "result",
                    "tool_calls": None,
                    "tool_call_id": "call-1",
                    "meta": {},
                },
            ]
        }

        assert data == expected

    def test_deserialize_empty(self):
        """Test deserializing empty history."""
        data = {"messages": []}

        history = MessageHistory.deserialize(data)

        assert len(history.messages) == 0
        assert history.get() == []

    def test_deserialize_with_messages(self):
        """Test deserializing history with messages."""
        data = {
            "messages": [
                {
                    "id": "1",
                    "role": "system",
                    "content": "You are helpful",
                    "tool_calls": None,
                    "tool_call_id": None,
                    "meta": {"tokens": 5},
                },
                {
                    "id": "2",
                    "role": "user",
                    "content": "Hello",
                    "tool_calls": None,
                    "tool_call_id": None,
                    "meta": {},
                },
                {
                    "id": "3",
                    "role": "assistant",
                    "content": "Hi there",
                    "tool_calls": [{"id": "call-1", "name": "test", "arguments": {}}],
                    "tool_call_id": None,
                    "meta": {"tokens": 8},
                },
                {
                    "id": "4",
                    "role": "tool",
                    "content": "result",
                    "tool_calls": None,
                    "tool_call_id": "call-1",
                    "meta": {},
                },
            ]
        }

        history = MessageHistory.deserialize(data)

        assert len(history.messages) == 4

        msg1 = history.messages[0]
        assert msg1.id == "1"
        assert msg1.role == "system"
        assert msg1.content == "You are helpful"
        assert msg1.meta == {"tokens": 5}

        msg2 = history.messages[1]
        assert msg2.id == "2"
        assert msg2.role == "user"
        assert msg2.content == "Hello"

        msg3 = history.messages[2]
        assert msg3.id == "3"
        assert msg3.role == "assistant"
        assert msg3.content == "Hi there"
        assert msg3.tool_calls == [{"id": "call-1", "name": "test", "arguments": {}}]

        msg4 = history.messages[3]
        assert msg4.id == "4"
        assert msg4.role == "tool"
        assert msg4.content == "result"
        assert msg4.tool_call_id == "call-1"

    def test_deserialize_missing_messages_key(self):
        """Test deserializing data without 'messages' key raises ValueError."""
        data = {"other_key": []}

        with pytest.raises(ValueError, match="Missing 'messages' key in serialized data"):
            MessageHistory.deserialize(data)

    def test_deserialize_invalid_message_data(self):
        """Test deserializing with invalid message data raises ValueError."""
        data = {"messages": [{"id": "1"}]}  # Missing required fields

        with pytest.raises(ValueError, match="Invalid message data"):
            MessageHistory.deserialize(data)

    def test_serialize_deserialize_roundtrip(self):
        """Test that serialize/deserialize roundtrip preserves data."""
        original = MessageHistory()
        original.add_system("System message")
        original.add_user("User input")
        original.add_assistant(
            "Assistant response", [{"id": "call-1", "name": "test", "arguments": {}}]
        )
        original.add_tool("Tool result", "call-1")

        # Serialize
        data = original.serialize()

        # Deserialize
        restored = MessageHistory.deserialize(data)

        # Compare
        assert len(restored.messages) == len(original.messages)
        for orig_msg, rest_msg in zip(original.messages, restored.messages, strict=True):
            assert orig_msg.id == rest_msg.id
            assert orig_msg.role == rest_msg.role
            assert orig_msg.content == rest_msg.content
            assert orig_msg.tool_calls == rest_msg.tool_calls
            assert orig_msg.tool_call_id == rest_msg.tool_call_id
            assert orig_msg.meta == rest_msg.meta

    def test_normalize_user_input_trim_whitespace(self):
        """Test user input normalization trims trailing whitespace."""
        history = MessageHistory()

        normalized = history._normalize_user_input("hello world   \n   \t  ")

        assert normalized == "hello world"

    def test_normalize_user_input_collapse_newlines(self):
        """Test user input normalization collapses excessive newlines."""
        history = MessageHistory()

        content = "line1\n\n\n\n\nline2\n\n\nline3"
        normalized = history._normalize_user_input(content)

        assert normalized == "line1\n\nline2\n\nline3"

    def test_normalize_user_input_preserve_double_newlines(self):
        """Test user input normalization preserves double newlines."""
        history = MessageHistory()

        content = "paragraph1\n\nparagraph2"
        normalized = history._normalize_user_input(content)

        assert normalized == "paragraph1\n\nparagraph2"

    def test_normalize_user_input_combined(self):
        """Test user input normalization with combined issues."""
        history = MessageHistory()

        content = "   hello\n\n\n\nworld   \n\n\n   \t"
        normalized = history._normalize_user_input(content)

        assert normalized == "   hello\n\nworld"

    def test_add_user_calls_normalization(self):
        """Test that add_user calls input normalization."""
        history = MessageHistory()

        # Add user input with whitespace issues
        history.add_user("hello   \n\n\n\nworld   ")

        msg = history.messages[0]
        assert msg.content == "hello   \n\nworld"
        assert msg.role == "user"
