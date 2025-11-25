"""Message handling and conversation history.

Implements Message and MessageHistory per INTERFACES/MESSAGES.md and INTERFACES/MESSAGE_HISTORY.md.
"""

import re
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any


@dataclass
class Message:
    """A single message in a conversation.

    Represents messages between user, assistant, system, and tools
    following the schema defined in INTERFACES/MESSAGES.md.
    """

    id: str
    role: str
    content: str
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate message constraints after initialization."""
        if self.role not in ("system", "user", "assistant", "tool"):
            raise ValueError(f"Invalid role: {self.role}")

        if self.role == "tool" and not self.tool_call_id:
            raise ValueError("tool messages MUST have tool_call_id")

        if self.role != "tool" and self.tool_call_id:
            raise ValueError("Only tool messages may have tool_call_id")

        if self.role != "assistant" and self.tool_calls:
            raise ValueError("Only assistant messages may have tool_calls")


class MessageHistory:
    """Manages conversation history with ordered message storage.

    Enforces message shape constraints and provides serialization.
    See INTERFACES/MESSAGE_HISTORY.md for full specification.
    """

    def __init__(self) -> None:
        """Initialize empty message history."""
        self.messages: list[Message] = []

    def add(self, message: Message) -> None:
        """Add a message to the history.

        Args:
            message: The message to add

        Raises:
            ValueError: If message validation fails
        """
        # Message validation happens in Message.__post_init__
        self.messages.append(message)

    def add_system(self, content: str) -> None:
        """Add a system message.

        Args:
            content: System message content
        """
        message = Message(
            id=str(uuid.uuid4()),
            role="system",
            content=content,
            meta={"ts": datetime.now(UTC).isoformat()},
        )
        self.add(message)

    def add_user(self, content: str) -> None:
        """Add a user message with input normalization.

        Args:
            content: User message content (will be normalized)
        """
        normalized_content = self._normalize_user_input(content)
        message = Message(
            id=str(uuid.uuid4()),
            role="user",
            content=normalized_content,
            meta={"ts": datetime.now(UTC).isoformat()},
        )
        self.add(message)

    def add_assistant(self, content: str, tool_calls: list[dict[str, Any]] | None = None) -> None:
        """Add an assistant message.

        Args:
            content: Assistant message content
            tool_calls: Optional list of tool calls made by assistant
        """
        message = Message(
            id=str(uuid.uuid4()),
            role="assistant",
            content=content,
            tool_calls=tool_calls,
            meta={"ts": datetime.now(UTC).isoformat()},
        )
        self.add(message)

    def add_tool(self, content: str, tool_call_id: str, tool_name: str = "") -> None:
        """Add tool result message.

        Args:
            content: Tool execution result
            tool_call_id: ID linking to originating tool call
            tool_name: Name of the tool that was called (for OpenAI compatibility)

        Raises:
            ValueError: If tool_call_id is empty
        """
        if not tool_call_id:
            raise ValueError("tool_call_id cannot be empty")

        meta = {"ts": datetime.now(UTC).isoformat()}
        if tool_name:
            meta["tool_name"] = tool_name

        message = Message(
            id=str(uuid.uuid4()),
            role="tool",
            content=content,
            tool_call_id=tool_call_id,
            meta=meta,
        )
        self.add(message)

    def get(self) -> list[Message]:
        """Get all messages in insertion order.

        Returns:
            List of messages
        """
        return self.messages.copy()

    def clear(self) -> None:
        """Clear all messages from history."""
        self.messages = []

    def serialize(self) -> dict[str, Any]:
        """Serialize message history to dictionary for persistence.

        Returns:
            Dictionary representation suitable for JSON serialization
        """
        return {
            "messages": [
                {
                    "id": msg.id,
                    "role": msg.role,
                    "content": msg.content,
                    "tool_calls": msg.tool_calls,
                    "tool_call_id": msg.tool_call_id,
                    "meta": msg.meta,
                }
                for msg in self.messages
            ]
        }

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> "MessageHistory":
        """Deserialize message history from dictionary.

        Args:
            data: Dictionary representation from serialize()

        Returns:
            MessageHistory instance

        Raises:
            ValueError: If data format is invalid
        """
        if "messages" not in data:
            raise ValueError("Missing 'messages' key in serialized data")

        history = cls()

        for msg_data in data["messages"]:
            try:
                message = Message(
                    id=msg_data["id"],
                    role=msg_data["role"],
                    content=msg_data["content"],
                    tool_calls=msg_data.get("tool_calls"),
                    tool_call_id=msg_data.get("tool_call_id"),
                    meta=msg_data.get("meta", {}),
                )
                history.add(message)
            except (KeyError, TypeError) as e:
                raise ValueError(f"Invalid message data: {e}") from e

        return history

    def _normalize_user_input(self, content: str) -> str:
        """Normalize user input by trimming whitespace and collapsing newlines.

        Args:
            content: Raw user input

        Returns:
            Normalized content
        """
        # Trim trailing whitespace
        content = content.rstrip()

        # Collapse excessive newlines (more than 2 consecutive)
        # Optional: Detect and handle commands (placeholder for future)
        # Commands like /abort, /config profile=autonomous would be handled here

        return re.sub(r"\n{3,}", "\n\n", content)
