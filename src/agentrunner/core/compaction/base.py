"""Base classes and interfaces for context compaction strategies.

Implements the pluggable compaction architecture per INTERFACES/COMPACTION_SUMMARIZATION.md.
"""

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from agentrunner.core.messages import Message

if TYPE_CHECKING:
    from agentrunner.core.tokens import TokenCounter


@dataclass
class CompactionContext:
    """Context for compaction decisions.

    Used by both Agent (when to compact) and Provider (how to compact).
    """

    current_tokens: int
    target_tokens: int
    system_prompt: Message | None = None  # Never compact this
    recent_rounds: int = 3  # Keep last N rounds untouched
    preserve_errors: bool = True  # Keep tool errors
    metadata: dict[str, Any] = field(default_factory=dict)  # Additional context

    # Configuration fields (set at initialization)
    enabled: bool = False  # Compaction disabled (not working correctly yet)
    threshold: float = 0.8  # Trigger at this % of context window
    strategy: str = "claude_style"  # Strategy name: "claude_style", "aggressive", "noop"

    # Token counter for accurate token estimation (optional, falls back to len(text) // 4)
    token_counter: "TokenCounter | None" = None
    model: str | None = None  # Model name for token counting

    def __post_init__(self) -> None:
        """Validate compaction context."""
        if self.current_tokens < 0:
            raise ValueError("current_tokens must be non-negative")
        if self.target_tokens < 0:
            raise ValueError("target_tokens must be non-negative")
        if self.recent_rounds < 0:
            raise ValueError("recent_rounds must be non-negative")
        if not (0.0 <= self.threshold <= 1.0):
            raise ValueError(f"threshold must be 0.0-1.0, got {self.threshold}")
        valid_strategies = {"claude_style", "aggressive", "noop"}
        if self.strategy not in valid_strategies:
            raise ValueError(f"strategy must be one of {valid_strategies}, got {self.strategy}")

    def count_tokens(self, text: str) -> int:
        """Count tokens in text using real token counter or fallback estimation.

        Args:
            text: Text to count tokens for

        Returns:
            Number of tokens (accurate if token_counter provided, estimated otherwise)
        """
        if not text:
            return 0

        # Use real token counter if available
        if self.token_counter and self.model:
            return self.token_counter.count_text(text, self.model)

        # Fallback to rough estimation (4 chars per token)
        return max(1, len(text) // 4)


@dataclass
class CompactionResult:
    """Result of compaction operation."""

    messages: list[Message]
    tokens_saved: int
    strategy_used: str
    summary_created: bool = False
    audit_log: list[str] = field(default_factory=list)  # What was removed/summarized
    metadata: dict[str, Any] = field(default_factory=dict)  # Additional result data

    def __post_init__(self) -> None:
        """Validate compaction result."""
        if self.tokens_saved < 0:
            raise ValueError("tokens_saved must be non-negative")
        if not self.strategy_used:
            raise ValueError("strategy_used must be provided")


class CompactionStrategy(ABC):
    """Abstract base class for all compaction strategies.

    Compaction strategies reduce the token count of a message history
    while preserving the most important information for continued conversation.
    """

    @abstractmethod
    async def compact(
        self,
        messages: list[Message],
        target_tokens: int,
        context: CompactionContext,
    ) -> CompactionResult:
        """Compact messages to fit within target token budget.

        Args:
            messages: List of messages to compact
            target_tokens: Target number of tokens after compaction
            context: Compaction context with preferences and constraints

        Returns:
            CompactionResult with compacted messages and metadata

        Raises:
            ValueError: If inputs are invalid
            RuntimeError: If compaction fails or is impossible
        """
        raise NotImplementedError

    @abstractmethod
    def get_name(self) -> str:
        """Get the strategy name.

        Returns:
            Strategy name for registry and logging
        """
        raise NotImplementedError

    def get_description(self) -> str:
        """Get a human-readable description of the strategy.

        Returns:
            Strategy description
        """
        return f"Compaction strategy: {self.get_name()}"

    def validate_messages(self, messages: list[Message]) -> None:
        """Validate message list for compaction.

        Args:
            messages: Messages to validate

        Raises:
            ValueError: If messages are invalid for compaction
        """
        if not messages:
            raise ValueError("Cannot compact empty message list")

        # Check for valid message structure
        for i, msg in enumerate(messages):
            if not isinstance(msg, Message):
                raise ValueError(f"Message {i} is not a Message instance")
            if not msg.role:
                raise ValueError(f"Message {i} has no role")

    def find_system_prompt(self, messages: list[Message]) -> Message | None:
        """Find the system prompt in the message list.

        Args:
            messages: Messages to search

        Returns:
            System prompt message if found, None otherwise
        """
        for msg in messages:
            if msg.role == "system":
                return msg
        return None

    def identify_recent_rounds(self, messages: list[Message], recent_rounds: int) -> set[int]:
        """Identify indices of messages from recent conversation rounds.

        Args:
            messages: All messages
            recent_rounds: Number of recent rounds to preserve

        Returns:
            Set of message indices to preserve
        """
        if recent_rounds <= 0 or not messages:
            return set()

        # Find the last N complete rounds (user -> assistant -> tools -> user -> ...)
        preserve_indices = set()
        rounds_found = 0

        # Work backwards to find complete rounds
        i = len(messages) - 1
        while i >= 0 and rounds_found < recent_rounds:
            msg = messages[i]
            preserve_indices.add(i)

            # Count a round when we see a user message (start of round)
            if msg.role == "user":
                rounds_found += 1

            i -= 1

        return preserve_indices

    def create_summary_message(
        self, summarized_messages: list[Message], summary_text: str
    ) -> Message:
        """Create a summary message from a list of messages.

        Args:
            summarized_messages: Original messages being summarized
            summary_text: Human-readable summary text

        Returns:
            New summary message with assistant role
        """
        # Create metadata about what was summarized
        msg_count_by_role: dict[str, int] = {}
        for msg in summarized_messages:
            msg_count_by_role[msg.role] = msg_count_by_role.get(msg.role, 0) + 1

        summary_msg = Message(
            id=str(uuid.uuid4()),
            role="assistant",
            content=f"[Summary] {summary_text}",
            meta={
                "compaction_summary": True,
                "original_message_count": len(summarized_messages),
                "summarized_roles": msg_count_by_role,
            },
        )

        return summary_msg
