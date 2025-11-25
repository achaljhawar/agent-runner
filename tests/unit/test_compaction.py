"""Unit tests for context compaction strategies.

Tests CompactionStrategy implementations, registry functions, and dataclasses.
"""

from unittest.mock import Mock

import pytest

from agentrunner.core.compaction import (
    AggressiveCompactor,
    ClaudeStyleCompactor,
    CompactionContext,
    CompactionResult,
    CompactionStrategy,
    NoOpCompactor,
    clear_registry,
    get_compactor,
    get_default_compactor,
    list_compactors,
    register_compactor,
)
from agentrunner.core.messages import Message


class TestCompactionContext:
    """Test CompactionContext dataclass."""

    def test_basic_creation(self):
        """Test basic CompactionContext creation."""
        context = CompactionContext(
            current_tokens=1000,
            target_tokens=800,
            recent_rounds=3,
            preserve_errors=True,
            enabled=True,
            threshold=0.8,
            strategy="claude_style",
        )

        assert context.current_tokens == 1000
        assert context.target_tokens == 800
        assert context.system_prompt is None  # system_prompt is optional
        assert context.recent_rounds == 3
        assert context.preserve_errors is True
        assert context.metadata == {}

    def test_default_values(self):
        """Test CompactionContext with default values."""
        context = CompactionContext(current_tokens=100, target_tokens=50)

        assert context.system_prompt is None
        assert context.recent_rounds == 3
        assert context.preserve_errors is True
        assert context.metadata == {}

    def test_validation_negative_current_tokens(self):
        """Test validation fails for negative current_tokens."""
        with pytest.raises(ValueError, match="current_tokens must be non-negative"):
            CompactionContext(current_tokens=-1, target_tokens=100)

    def test_validation_negative_target_tokens(self):
        """Test validation fails for negative target_tokens."""
        with pytest.raises(ValueError, match="target_tokens must be non-negative"):
            CompactionContext(current_tokens=100, target_tokens=-1)

    def test_validation_negative_recent_rounds(self):
        """Test validation fails for negative recent_rounds."""
        with pytest.raises(ValueError, match="recent_rounds must be non-negative"):
            CompactionContext(current_tokens=100, target_tokens=50, recent_rounds=-1)


class TestCompactionResult:
    """Test CompactionResult dataclass."""

    def test_basic_creation(self):
        """Test basic CompactionResult creation."""
        messages = [Message(id="msg-1", role="user", content="test")]
        result = CompactionResult(
            messages=messages,
            tokens_saved=100,
            strategy_used="test_strategy",
            summary_created=True,
            audit_log=["test action"],
        )

        assert result.messages == messages
        assert result.tokens_saved == 100
        assert result.strategy_used == "test_strategy"
        assert result.summary_created is True
        assert result.audit_log == ["test action"]
        assert result.metadata == {}

    def test_default_values(self):
        """Test CompactionResult with default values."""
        messages = [Message(id="msg-1", role="user", content="test")]
        result = CompactionResult(
            messages=messages,
            tokens_saved=50,
            strategy_used="test",
        )

        assert result.summary_created is False
        assert result.audit_log == []
        assert result.metadata == {}

    def test_validation_negative_tokens_saved(self):
        """Test validation fails for negative tokens_saved."""
        messages = [Message(id="msg-1", role="user", content="test")]
        with pytest.raises(ValueError, match="tokens_saved must be non-negative"):
            CompactionResult(
                messages=messages,
                tokens_saved=-1,
                strategy_used="test",
            )

    def test_validation_empty_strategy_used(self):
        """Test validation fails for empty strategy_used."""
        messages = [Message(id="msg-1", role="user", content="test")]
        with pytest.raises(ValueError, match="strategy_used must be provided"):
            CompactionResult(
                messages=messages,
                tokens_saved=0,
                strategy_used="",
            )


class TestCompactionStrategyBase:
    """Test CompactionStrategy abstract base class."""

    def test_abstract_methods_not_implemented(self):
        """Test abstract methods must be implemented."""
        # Can't instantiate abstract class - this is expected behavior
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            CompactionStrategy()

    def test_validate_messages_empty_list(self):
        """Test validate_messages fails on empty list."""

        # Create a minimal concrete implementation for testing
        class TestStrategy(CompactionStrategy):
            def get_name(self):
                return "test"

            async def compact(self, messages, target_tokens, context):
                return CompactionResult(messages=messages, tokens_saved=0, strategy_used="test")

        strategy = TestStrategy()

        with pytest.raises(ValueError, match="Cannot compact empty message list"):
            strategy.validate_messages([])

    def test_validate_messages_invalid_type(self):
        """Test validate_messages fails on non-Message objects."""

        class TestStrategy(CompactionStrategy):
            def get_name(self):
                return "test"

            async def compact(self, messages, target_tokens, context):
                return CompactionResult(messages=messages, tokens_saved=0, strategy_used="test")

        strategy = TestStrategy()

        with pytest.raises(ValueError, match="Message 0 is not a Message instance"):
            strategy.validate_messages(["not a message"])

    def test_validate_messages_no_role(self):
        """Test validate_messages fails on messages with invalid data."""

        class TestStrategy(CompactionStrategy):
            def get_name(self):
                return "test"

            async def compact(self, messages, target_tokens, context):
                return CompactionResult(messages=messages, tokens_saved=0, strategy_used="test")

        strategy = TestStrategy()
        # Use a Mock that will pass isinstance but has empty role
        message = Mock(spec=Message)
        message.role = ""
        message.content = ""

        with pytest.raises(ValueError, match="Message 0"):
            strategy.validate_messages([message])

    def test_validate_messages_success(self):
        """Test validate_messages succeeds with valid messages."""

        class TestStrategy(CompactionStrategy):
            def get_name(self):
                return "test"

            async def compact(self, messages, target_tokens, context):
                return CompactionResult(messages=messages, tokens_saved=0, strategy_used="test")

        strategy = TestStrategy()
        messages = [
            Message(id="msg-1", role="user", content="Hello"),
            Message(id="msg-2", role="assistant", content="Hi there"),
        ]

        # Should not raise
        strategy.validate_messages(messages)

    def test_find_system_prompt_found(self):
        """Test find_system_prompt returns system message."""

        class TestStrategy(CompactionStrategy):
            def get_name(self):
                return "test"

            async def compact(self, messages, target_tokens, context):
                return CompactionResult(messages=messages, tokens_saved=0, strategy_used="test")

        strategy = TestStrategy()
        system_msg = Message(id="msg-1", role="system", content="System prompt")
        messages = [
            Message(id="msg-2", role="user", content="Hello"),
            system_msg,
            Message(id="msg-3", role="assistant", content="Hi"),
        ]

        found = strategy.find_system_prompt(messages)
        assert found == system_msg

    def test_find_system_prompt_not_found(self):
        """Test find_system_prompt returns None when no system message."""

        class TestStrategy(CompactionStrategy):
            def get_name(self):
                return "test"

            async def compact(self, messages, target_tokens, context):
                return CompactionResult(messages=messages, tokens_saved=0, strategy_used="test")

        strategy = TestStrategy()
        messages = [
            Message(id="msg-1", role="user", content="Hello"),
            Message(id="msg-2", role="assistant", content="Hi"),
        ]

        found = strategy.find_system_prompt(messages)
        assert found is None

    def test_identify_recent_rounds_basic(self):
        """Test identify_recent_rounds basic functionality."""

        class TestStrategy(CompactionStrategy):
            def get_name(self):
                return "test"

            async def compact(self, messages, target_tokens, context):
                return CompactionResult(messages=messages, tokens_saved=0, strategy_used="test")

        strategy = TestStrategy()
        messages = [
            Message(id="msg-1", role="user", content="First question"),
            Message(id="msg-2", role="assistant", content="First answer"),
            Message(id="msg-3", role="user", content="Second question"),
            Message(id="msg-4", role="assistant", content="Second answer"),
        ]

        indices = strategy.identify_recent_rounds(messages, 1)
        # Should preserve the last round (user + assistant)
        assert indices == {2, 3}

    def test_identify_recent_rounds_zero(self):
        """Test identify_recent_rounds with zero rounds."""

        class TestStrategy(CompactionStrategy):
            def get_name(self):
                return "test"

            async def compact(self, messages, target_tokens, context):
                return CompactionResult(messages=messages, tokens_saved=0, strategy_used="test")

        strategy = TestStrategy()
        messages = [Message(id="msg-1", role="user", content="Hello")]

        indices = strategy.identify_recent_rounds(messages, 0)
        assert indices == set()

    def test_identify_recent_rounds_more_than_available(self):
        """Test identify_recent_rounds with more rounds than available."""

        class TestStrategy(CompactionStrategy):
            def get_name(self):
                return "test"

            async def compact(self, messages, target_tokens, context):
                return CompactionResult(messages=messages, tokens_saved=0, strategy_used="test")

        strategy = TestStrategy()
        messages = [
            Message(id="msg-1", role="user", content="Question"),
            Message(id="msg-2", role="assistant", content="Answer"),
        ]

        indices = strategy.identify_recent_rounds(messages, 5)
        # Should preserve all messages
        assert indices == {0, 1}

    def test_create_summary_message(self):
        """Test create_summary_message."""

        class TestStrategy(CompactionStrategy):
            def get_name(self):
                return "test"

            async def compact(self, messages, target_tokens, context):
                return CompactionResult(messages=messages, tokens_saved=0, strategy_used="test")

        strategy = TestStrategy()
        original_messages = [
            Message(id="msg-1", role="user", content="Hello"),
            Message(id="msg-2", role="assistant", content="Hi there"),
        ]

        summary_msg = strategy.create_summary_message(original_messages, "Test summary")

        assert summary_msg.role == "assistant"
        assert summary_msg.content == "[Summary] Test summary"
        assert summary_msg.meta["compaction_summary"] is True
        assert summary_msg.meta["original_message_count"] == 2
        assert "user" in summary_msg.meta["summarized_roles"]
        assert "assistant" in summary_msg.meta["summarized_roles"]


class TestClaudeStyleCompactor:
    """Test ClaudeStyleCompactor strategy."""

    def test_get_name(self):
        """Test strategy name."""
        compactor = ClaudeStyleCompactor()
        assert compactor.get_name() == "claude_style"

    def test_get_description(self):
        """Test strategy description."""
        compactor = ClaudeStyleCompactor()
        description = compactor.get_description()
        assert "Preserve recent context" in description
        assert "drop old messages" in description

    @pytest.mark.asyncio
    async def test_compact_no_compaction_needed(self):
        """Test compaction when no compaction is needed."""
        compactor = ClaudeStyleCompactor()
        messages = [Message(id="msg-1", role="user", content="Hello")]
        context = CompactionContext(current_tokens=50, target_tokens=100)

        result = await compactor.compact(messages, 100, context)

        assert result.messages == messages
        assert result.tokens_saved == 0
        assert result.strategy_used == "claude_style"
        assert not result.summary_created
        assert "No compaction needed" in result.audit_log

    @pytest.mark.asyncio
    async def test_compact_drop_tool_messages(self):
        """Test dropping old tool messages."""
        compactor = ClaudeStyleCompactor()
        messages = [
            Message(id="msg-1", role="system", content="System prompt"),
            Message(id="msg-2", role="user", content="Do something"),
            Message(id="msg-3", role="assistant", content="OK", tool_calls=[{"name": "test"}]),
            Message(id="msg-4", role="tool", content="Success result", tool_call_id="1"),
            Message(id="msg-5", role="user", content="Recent question"),  # Recent round
            Message(id="msg-6", role="assistant", content="Recent answer"),  # Recent round
        ]

        context = CompactionContext(current_tokens=200, target_tokens=100, recent_rounds=1)

        result = await compactor.compact(messages, 100, context)

        # Should preserve system, recent user/assistant, but drop old tool message
        assert len(result.messages) < len(messages)
        assert result.tokens_saved > 0
        assert any("Dropped" in log for log in result.audit_log)

    @pytest.mark.asyncio
    async def test_compact_preserve_error_messages(self):
        """Test preserving tool messages with errors."""
        compactor = ClaudeStyleCompactor()
        messages = [
            Message(id="msg-1", role="user", content="Do something"),
            Message(id="msg-2", role="assistant", content="OK", tool_calls=[{"name": "test"}]),
            Message(id="msg-3", role="tool", content="ERROR: Failed to execute", tool_call_id="1"),
            Message(id="msg-4", role="user", content="Try again"),
        ]

        context = CompactionContext(
            current_tokens=200,
            target_tokens=100,
            recent_rounds=0,  # Don't preserve recent rounds
            preserve_errors=True,
        )

        result = await compactor.compact(messages, 100, context)

        # Error tool message should be preserved
        tool_messages = [msg for msg in result.messages if msg.role == "tool"]
        assert len(tool_messages) == 1
        assert "ERROR" in tool_messages[0].content


class TestAggressiveCompactor:
    """Test AggressiveCompactor strategy."""

    def test_get_name(self):
        """Test strategy name."""
        compactor = AggressiveCompactor()
        assert compactor.get_name() == "aggressive"

    def test_get_description(self):
        """Test strategy description."""
        compactor = AggressiveCompactor()
        description = compactor.get_description()
        assert "Aggressive" in description
        assert "maximum compression" in description

    @pytest.mark.asyncio
    async def test_compact_aggressive_tool_dropping(self):
        """Test aggressive dropping of successful tool messages."""
        compactor = AggressiveCompactor()
        messages = [
            Message(id="msg-1", role="system", content="System"),
            Message(id="msg-2", role="user", content="Question 1"),
            Message(id="msg-3", role="assistant", content="Answer 1"),
            Message(id="msg-4", role="tool", content="Success result", tool_call_id="1"),
            Message(id="msg-5", role="user", content="Question 2"),
            Message(id="msg-6", role="assistant", content="Answer 2"),
            Message(id="msg-7", role="tool", content="Another success", tool_call_id="2"),
            Message(id="msg-8", role="user", content="Recent question"),  # Preserve (recent)
        ]

        context = CompactionContext(current_tokens=300, target_tokens=100, recent_rounds=1)

        result = await compactor.compact(messages, 100, context)

        # Should aggressively drop successful tool messages
        tool_messages = [msg for msg in result.messages if msg.role == "tool"]
        assert len(tool_messages) == 0  # No tool messages should remain
        assert result.tokens_saved > 0

    @pytest.mark.asyncio
    async def test_compact_aggressive_summarization(self):
        """Test aggressive compaction (drops old messages aggressively)."""
        compactor = AggressiveCompactor()
        messages = [
            Message(id="msg-1", role="system", content="System prompt"),
            Message(id="msg-2", role="user", content="Old question 1"),
            Message(id="msg-3", role="assistant", content="Old answer 1"),
            Message(id="msg-4", role="user", content="Old question 2"),
            Message(id="msg-5", role="assistant", content="Old answer 2"),
            Message(id="msg-6", role="user", content="Recent question"),  # Preserve
            Message(id="msg-7", role="assistant", content="Recent answer"),  # Preserve
        ]

        context = CompactionContext(current_tokens=500, target_tokens=200, recent_rounds=1)

        result = await compactor.compact(messages, 200, context)

        # AggressiveCompactor drops messages but doesn't create summaries
        assert not result.summary_created
        # Should keep only system + recent round (forced to 1)
        assert len(result.messages) == 2  # system, recent assistant
        assert result.tokens_saved > 0


class TestNoOpCompactor:
    """Test NoOpCompactor strategy."""

    def test_get_name(self):
        """Test strategy name."""
        compactor = NoOpCompactor()
        assert compactor.get_name() == "noop"

    def test_get_description(self):
        """Test strategy description."""
        compactor = NoOpCompactor()
        description = compactor.get_description()
        assert "No-op" in description
        assert "makes no changes" in description

    @pytest.mark.asyncio
    async def test_compact_no_changes(self):
        """Test NoOp compaction makes no changes."""
        compactor = NoOpCompactor()
        messages = [
            Message(id="msg-1", role="user", content="Hello"),
            Message(id="msg-2", role="assistant", content="Hi there"),
        ]

        context = CompactionContext(current_tokens=200, target_tokens=50)

        result = await compactor.compact(messages, 50, context)

        assert result.messages == messages  # Unchanged
        assert result.tokens_saved == 0
        assert result.strategy_used == "noop"
        assert not result.summary_created
        assert len(result.audit_log) >= 1

    @pytest.mark.asyncio
    async def test_compact_warning_over_target(self):
        """Test NoOp compaction warns when over target."""
        compactor = NoOpCompactor()
        messages = [Message(id="msg-1", role="user", content="Hello")]

        context = CompactionContext(current_tokens=200, target_tokens=50)

        result = await compactor.compact(messages, 50, context)

        # Should warn about exceeding target
        warning_logs = [log for log in result.audit_log if "WARNING" in log]
        assert len(warning_logs) >= 1
        # NoOp returns original messages unchanged
        assert len(result.messages) == len(messages)
        assert result.tokens_saved == 0


class TestCompactionRegistry:
    """Test compaction strategy registry functions."""

    def setup_method(self):
        """Clear registry before each test."""
        clear_registry()

    def test_register_compactor_success(self):
        """Test successful compactor registration."""

        class TestCompactor(CompactionStrategy):
            def get_name(self):
                return "test"

            async def compact(self, messages, target_tokens, context):
                return CompactionResult(
                    messages=messages,
                    tokens_saved=0,
                    strategy_used="test",
                )

        register_compactor("test_strategy", TestCompactor)

        # Should be able to get it
        compactor = get_compactor("test_strategy")
        assert isinstance(compactor, TestCompactor)

    def test_register_compactor_duplicate_name(self):
        """Test registering duplicate name fails."""

        class TestCompactor(CompactionStrategy):
            def get_name(self):
                return "test"

            async def compact(self, messages, target_tokens, context):
                pass

        register_compactor("duplicate", TestCompactor)

        with pytest.raises(ValueError, match="Strategy 'duplicate' is already registered"):
            register_compactor("duplicate", TestCompactor)

    def test_register_compactor_invalid_class(self):
        """Test registering invalid class fails."""

        class NotACompactor:
            pass

        with pytest.raises(ValueError, match="Strategy class must inherit from CompactionStrategy"):
            register_compactor("invalid", NotACompactor)

    def test_register_compactor_empty_name(self):
        """Test registering with empty name fails."""

        class TestCompactor(CompactionStrategy):
            def get_name(self):
                return "test"

            async def compact(self, messages, target_tokens, context):
                pass

        with pytest.raises(ValueError, match="Strategy name cannot be empty"):
            register_compactor("", TestCompactor)

    def test_get_compactor_built_in(self):
        """Test getting built-in compactors."""
        compactor = get_compactor("claude_style")
        assert isinstance(compactor, ClaudeStyleCompactor)

        compactor = get_compactor("aggressive")
        assert isinstance(compactor, AggressiveCompactor)

        compactor = get_compactor("noop")
        assert isinstance(compactor, NoOpCompactor)

    def test_get_compactor_not_found(self):
        """Test getting non-existent compactor fails."""
        with pytest.raises(ValueError, match="Unknown compaction strategy: nonexistent"):
            get_compactor("nonexistent")

    def test_get_compactor_empty_name(self):
        """Test getting compactor with empty name fails."""
        with pytest.raises(ValueError, match="Strategy name cannot be empty"):
            get_compactor("")

    def test_get_compactor_with_kwargs(self):
        """Test getting compactor with constructor arguments."""
        # This would work if strategies accepted kwargs
        compactor = get_compactor("noop")
        assert isinstance(compactor, NoOpCompactor)

    def test_list_compactors(self):
        """Test listing available compactors."""
        compactors = list_compactors()

        assert "claude_style" in compactors
        assert "aggressive" in compactors
        assert "noop" in compactors

    def test_list_compactors_with_custom(self):
        """Test listing includes custom registered compactors."""

        class CustomCompactor(CompactionStrategy):
            def get_name(self):
                return "custom"

            async def compact(self, messages, target_tokens, context):
                pass

        register_compactor("custom", CustomCompactor)
        compactors = list_compactors()

        assert "custom" in compactors

    def test_clear_registry(self):
        """Test clearing registry removes custom strategies."""

        class CustomCompactor(CompactionStrategy):
            def get_name(self):
                return "custom"

            async def compact(self, messages, target_tokens, context):
                pass

        register_compactor("custom", CustomCompactor)
        assert "custom" in list_compactors()

        clear_registry()
        assert "custom" not in list_compactors()
        # Built-ins should still be available
        assert "claude_style" in list_compactors()

    def test_get_default_compactor(self):
        """Test getting default compactor."""
        compactor = get_default_compactor()
        assert isinstance(compactor, ClaudeStyleCompactor)


class TestIntegrationScenarios:
    """Test realistic integration scenarios."""

    @pytest.mark.asyncio
    async def test_mixed_conversation_compaction(self):
        """Test compaction of a mixed conversation with tools and errors."""
        compactor = ClaudeStyleCompactor()
        messages = [
            Message(id="msg-1", role="system", content="You are a helpful assistant"),
            Message(id="msg-2", role="user", content="Create a file"),
            Message(
                id="msg-3",
                role="assistant",
                content="I'll create the file",
                tool_calls=[{"name": "write_file"}],
            ),
            Message(id="msg-4", role="tool", content="File created successfully", tool_call_id="1"),
            Message(id="msg-5", role="user", content="Read the file"),
            Message(
                id="msg-6",
                role="assistant",
                content="I'll read it",
                tool_calls=[{"name": "read_file"}],
            ),
            Message(id="msg-7", role="tool", content="ERROR: File not found", tool_call_id="2"),
            Message(id="msg-8", role="user", content="What happened?"),  # Recent
            Message(id="msg-9", role="assistant", content="There was an error..."),  # Recent
        ]

        context = CompactionContext(current_tokens=500, target_tokens=200, recent_rounds=1)

        result = await compactor.compact(messages, 200, context)

        # Should preserve system, error tool message, and recent conversation
        assert len(result.messages) < len(messages)
        assert result.tokens_saved > 0

        # Check error message is preserved
        error_messages = [
            msg for msg in result.messages if msg.role == "tool" and "ERROR" in (msg.content or "")
        ]
        assert len(error_messages) == 1

        # Check recent messages are preserved
        recent_user_messages = [msg for msg in result.messages[-3:] if msg.role == "user"]
        assert len(recent_user_messages) >= 1
        assert "What happened?" in recent_user_messages[-1].content

    @pytest.mark.asyncio
    async def test_strategy_comparison(self):
        """Test different strategies on same input."""
        messages = [
            Message(id="msg-1", role="user", content="Question 1"),
            Message(id="msg-2", role="assistant", content="Answer 1"),
            Message(id="msg-3", role="tool", content="Tool result 1", tool_call_id="1"),
            Message(id="msg-4", role="user", content="Question 2"),
            Message(id="msg-5", role="assistant", content="Answer 2"),
            Message(id="msg-6", role="tool", content="Tool result 2", tool_call_id="2"),
        ]

        context = CompactionContext(current_tokens=300, target_tokens=150)

        # Test Claude style
        claude_compactor = ClaudeStyleCompactor()
        claude_result = await claude_compactor.compact(messages, 150, context)

        # Test Aggressive
        aggressive_compactor = AggressiveCompactor()
        aggressive_result = await aggressive_compactor.compact(messages, 150, context)

        # Test NoOp
        noop_compactor = NoOpCompactor()
        noop_result = await noop_compactor.compact(messages, 150, context)

        # Aggressive should save more tokens than Claude style
        assert aggressive_result.tokens_saved >= claude_result.tokens_saved

        # NoOp should save no tokens
        assert noop_result.tokens_saved == 0
        assert len(noop_result.messages) == len(messages)

        # All should return valid results
        for result in [claude_result, aggressive_result, noop_result]:
            assert isinstance(result, CompactionResult)
            assert result.strategy_used in ["claude_style", "aggressive", "noop"]
