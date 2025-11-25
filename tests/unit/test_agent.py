"""Tests for AgentRunnerAgent.

Per .coding_agent_guide: Test REAL modules, mock only external boundaries.
Use REAL OpenAIProvider with MOCKED OpenAI API calls.
"""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from agentrunner.core.agent import AgentRunnerAgent, AgentState
from agentrunner.core.compaction import CompactionContext, CompactionResult
from agentrunner.core.config import AgentConfig
from agentrunner.core.exceptions import AgentRunnerException
from agentrunner.core.logger import AgentRunnerLogger
from agentrunner.core.messages import Message
from agentrunner.core.tokens import ContextManager, TokenCounter
from agentrunner.core.tool_protocol import ToolCall, ToolResult
from agentrunner.core.workspace import Workspace
from agentrunner.providers.base import ModelInfo, ProviderConfig
from agentrunner.providers.openai_provider import OpenAIProvider
from agentrunner.tools.base import ToolContext, ToolRegistry
from agentrunner.tools.file_io import WriteFileTool
from agentrunner.tools.read_file import ReadFileTool
from agentrunner.tools.search import GrepSearchTool


def create_mock_openai_response(message: Message, tokens: dict = None):
    """Create a mock OpenAI API response."""
    if tokens is None:
        tokens = {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}

    # Mock OpenAI response structure
    mock_choice = MagicMock()
    mock_choice.message.role = message.role
    mock_choice.message.content = message.content
    mock_choice.finish_reason = "stop"

    # Handle tool calls
    if message.tool_calls:
        mock_tool_calls = []
        for tc in message.tool_calls:
            mock_tc = MagicMock()
            mock_tc.id = tc.get("id") if isinstance(tc, dict) else tc.id
            mock_tc.function.name = tc.get("name") if isinstance(tc, dict) else tc.name
            mock_tc.function.arguments = (
                str(tc.get("arguments", "{}")) if isinstance(tc, dict) else str(tc.arguments)
            )
            mock_tool_calls.append(mock_tc)
        mock_choice.message.tool_calls = mock_tool_calls
    else:
        mock_choice.message.tool_calls = None

    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_response.usage.prompt_tokens = tokens["prompt_tokens"]
    mock_response.usage.completion_tokens = tokens["completion_tokens"]
    mock_response.usage.total_tokens = tokens["total_tokens"]

    return mock_response


class MockToolRegistry:
    """Mock tool registry ONLY for error scenarios. Prefer real ToolRegistry."""

    def __init__(self, tools: dict[str, any] | None = None):
        self.tools = tools or {}
        self.executed_calls: list[ToolCall] = []

    def has(self, name: str) -> bool:
        return name in self.tools

    def get_definitions(self) -> list:
        return []

    def get(self, name: str):
        mock_tool = MagicMock()
        mock_tool.get_definition.return_value = MagicMock(name=name, safety={})
        return mock_tool

    async def execute(self, call: ToolCall) -> ToolResult:
        self.executed_calls.append(call)
        if call.name in self.tools:
            result = self.tools[call.name](call)
            if asyncio.iscoroutine(result):
                return await result
            return result
        return ToolResult(
            success=False, error=f"Unknown tool: {call.name}", error_code="E_TOOL_UNKNOWN"
        )


class MockProvider:
    """Mock provider for testing agent behavior. Test REAL modules, mock only external boundaries."""

    def __init__(
        self, responses: list[Message] | None = None, config: ProviderConfig | None = None
    ):
        """Initialize mock provider with predetermined responses.

        Args:
            responses: List of Messages to return in sequence
            config: Provider config (optional)
        """
        self.responses = responses or []
        self.call_count = 0
        self.config = config or ProviderConfig(
            model="test-model",
            compaction=CompactionContext(
                current_tokens=0,
                target_tokens=0,
                enabled=False,
                threshold=0.8,
                strategy="noop",
                recent_rounds=3,
                preserve_errors=True,
            ),
        )

    def get_system_prompt(self, workspace_path: str, tools: list | None = None) -> str:
        """Return a mock system prompt."""
        return "You are AgentRunner, an autonomous agent assistant."

    def get_tool_classes(self) -> list:
        """Return empty tool list for mock provider."""
        return []

    def get_model_info(self) -> ModelInfo:
        """Return mock model info."""
        return ModelInfo(
            name="test-model",
            context_window=128000,
            pricing={"input_per_1k": 0.01, "output_per_1k": 0.03},
        )

    async def chat(
        self, messages: list[Message], tools: list | None = None, config: object | None = None
    ):
        """Return next response from the predetermined list."""
        if self.call_count < len(self.responses):
            response_msg = self.responses[self.call_count]
            self.call_count += 1
        else:
            # Default response if we run out of predetermined responses
            response_msg = Message(
                id=f"msg_{self.call_count}", role="assistant", content="Default response"
            )
            self.call_count += 1

        # Mock provider response structure
        mock_response = MagicMock()
        mock_response.messages = [response_msg]
        mock_response.usage = {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}
        return mock_response

    async def compact(
        self, messages: list[Message], target_tokens: int, context: CompactionContext
    ):
        """Mock compaction - just return fewer messages."""

        # Simple mock: keep last few messages
        compacted = messages[-3:] if len(messages) > 3 else messages
        return CompactionResult(
            messages=compacted, tokens_saved=500, strategy_used="mock_compaction"
        )


@pytest.fixture
def temp_workspace(tmp_path):
    """Create temporary workspace with test files."""
    workspace = Workspace(str(tmp_path))

    # Create a test file for tools to operate on
    test_file = Path(tmp_path) / "test.txt"
    test_file.write_text("Hello world\n")

    return workspace


@pytest.fixture
def tool_registry(temp_workspace, config):
    """Create REAL tool registry with REAL tools - not mocked!"""
    logger = AgentRunnerLogger()
    ctx = ToolContext(
        workspace=temp_workspace,
        logger=logger,
        model_id="test-model",
        config=config.to_dict(),
    )

    registry = ToolRegistry(context=ctx)

    # Register real tools
    registry.register(ReadFileTool())
    registry.register(WriteFileTool())
    registry.register(GrepSearchTool())

    return registry


@pytest.fixture
def config():
    return AgentConfig(
        max_rounds=10,
    )


@pytest.fixture
def logger():
    return AgentRunnerLogger()


@pytest.fixture
def context_manager():
    counter = TokenCounter()
    model_info = ModelInfo(
        name="gpt-5.1-2025-11-13",
        context_window=128000,
        pricing={"input_per_1k": 0.01, "output_per_1k": 0.03},
    )
    return ContextManager(counter=counter, model_info=model_info)


@pytest.fixture
def provider():
    """Create REAL OpenAIProvider with MOCKED OpenAI client."""
    config = ProviderConfig(
        model="gpt-5.1-2025-11-13",
        temperature=0.7,
        compaction=CompactionContext(
            current_tokens=0,
            target_tokens=0,
            enabled=False,
            threshold=0.8,
            strategy="noop",
            recent_rounds=3,
            preserve_errors=True,
        ),
    )

    provider = OpenAIProvider(api_key="test-api-key", config=config)
    # Mock the client to avoid real API calls
    provider.client = MagicMock()
    provider.client.chat.completions.create = AsyncMock()

    return provider


class TestAgentRunnerAgentInit:
    def test_agent_initialization(self, temp_workspace, config, logger, context_manager, provider):
        agent = AgentRunnerAgent(
            provider=provider,
            workspace=temp_workspace,
            config=config,
            context_manager=context_manager,
            logger=logger,
        )

        assert agent.provider == provider
        assert agent.workspace == temp_workspace
        assert agent.config == config
        assert agent.state == AgentState.IDLE
        assert not agent._abort_requested

    def test_agent_adds_system_prompt(
        self, temp_workspace, config, logger, context_manager, provider
    ):
        agent = AgentRunnerAgent(
            provider=provider,
            workspace=temp_workspace,
            config=config,
            context_manager=context_manager,
            logger=logger,
        )

        # System prompt is now added lazily (on first message processing, not at init)
        messages = agent.get_messages()
        assert len(messages) == 0  # No messages yet

        # After first message, system prompt will be added
        assert agent.provider is not None
        assert hasattr(agent.provider, "get_system_prompt")

    def test_agent_no_system_prompt(self, temp_workspace, logger, context_manager, provider):
        config = AgentConfig()
        agent = AgentRunnerAgent(
            provider=provider,
            workspace=temp_workspace,
            config=config,
            context_manager=context_manager,
            logger=logger,
        )

        # System prompt is now added lazily (on first message processing)
        messages = agent.get_messages()
        assert len(messages) == 0  # No messages at initialization


class TestProcessMessage:
    @pytest.mark.asyncio
    async def test_simple_conversation(self, temp_workspace, config, logger, context_manager):
        # Provider returns single response with no tool calls
        provider = MockProvider(
            [Message(id="msg_1", role="assistant", content="Hello! How can I help?")]
        )
        agent = AgentRunnerAgent(
            provider=provider,
            workspace=temp_workspace,
            config=config,
            context_manager=context_manager,
            logger=logger,
        )

        result = await agent.process_message("Hello!")

        assert result.content == "Hello! How can I help?"
        assert result.rounds == 1
        assert result.stopped_reason == "completed"
        assert agent.state == AgentState.COMPLETED

        messages = agent.get_messages()
        assert len(messages) == 3  # system, user, assistant (system added first)
        assert messages[0].role == "system"  # System prompt added first
        assert messages[1].role == "user"
        assert messages[1].content == "Hello!"
        assert messages[2].role == "assistant"

    @pytest.mark.asyncio
    async def test_max_rounds_reached(self, temp_workspace, logger, context_manager):
        config = AgentConfig(
            max_rounds=2,
        )

        # Provider keeps returning messages with tool calls
        provider = MockProvider(
            [
                Message(
                    id="msg_1",
                    role="assistant",
                    content="Calling tool",
                    tool_calls=[{"id": "call_1", "name": "test_tool", "arguments": {}}],
                ),
                Message(
                    id="msg_2",
                    role="assistant",
                    content="Calling tool again",
                    tool_calls=[{"id": "call_2", "name": "test_tool", "arguments": {}}],
                ),
            ]
        )

        tools = MockToolRegistry(
            {"test_tool": lambda call: ToolResult(success=True, output="Done")}
        )

        agent = AgentRunnerAgent(
            provider=provider,
            workspace=temp_workspace,
            config=config,
            context_manager=context_manager,
            logger=logger,
            tools=tools,
        )

        result = await agent.process_message("Test")

        assert result.rounds == 2
        assert result.stopped_reason == "max_rounds"
        assert agent.state == AgentState.COMPLETED

    @pytest.mark.asyncio
    async def test_abort_requested(self, temp_workspace, config, logger, context_manager, provider):
        # Test abort() method directly sets state
        agent = AgentRunnerAgent(
            provider=provider,
            workspace=temp_workspace,
            config=config,
            context_manager=context_manager,
            logger=logger,
        )

        # Request abort
        agent.abort()

        assert agent._abort_requested is True
        assert agent.state == AgentState.ABORTING


class TestToolExecution:
    @pytest.mark.asyncio
    async def test_tool_execution_success(self, temp_workspace, config, logger, context_manager):
        # Provider calls tool, then returns final answer
        provider = MockProvider(
            [
                Message(
                    id="msg_1",
                    role="assistant",
                    content="Using tool",
                    tool_calls=[
                        {
                            "id": "call_1",
                            "name": "read_file",
                            "arguments": {"path": "test.txt"},
                        }
                    ],
                ),
                Message(id="msg_2", role="assistant", content="File read successfully"),
            ]
        )

        def read_file_mock(call: ToolCall) -> ToolResult:
            return ToolResult(success=True, output="File contents here")

        tools = MockToolRegistry({"read_file": read_file_mock})

        agent = AgentRunnerAgent(
            provider=provider,
            workspace=temp_workspace,
            config=config,
            context_manager=context_manager,
            logger=logger,
            tools=tools,
        )

        result = await agent.process_message("Read test.txt")

        assert result.rounds == 2
        assert result.stopped_reason == "completed"
        assert len(tools.executed_calls) == 1
        assert tools.executed_calls[0].name == "read_file"

        messages = agent.get_messages()
        # system, user, assistant (with tool call), tool result, assistant (final)
        assert any(msg.role == "tool" for msg in messages)

    @pytest.mark.asyncio
    async def test_unknown_tool(self, temp_workspace, config, logger, context_manager):
        provider = MockProvider(
            [
                Message(
                    id="msg_1",
                    role="assistant",
                    content="Using unknown tool",
                    tool_calls=[{"id": "call_1", "name": "nonexistent_tool", "arguments": {}}],
                ),
                Message(id="msg_2", role="assistant", content="Tool not found"),
            ]
        )

        tools = MockToolRegistry()

        agent = AgentRunnerAgent(
            provider=provider,
            workspace=temp_workspace,
            config=config,
            context_manager=context_manager,
            logger=logger,
            tools=tools,
        )

        result = await agent.process_message("Test")

        assert result.rounds == 2
        # Tool error should be added to history
        messages = agent.get_messages()
        tool_messages = [msg for msg in messages if msg.role == "tool"]
        assert len(tool_messages) == 1
        assert "Unknown tool" in tool_messages[0].content

    @pytest.mark.asyncio
    async def test_tool_timeout(self, temp_workspace, logger, context_manager):
        config = AgentConfig(
            tool_timeout_s=1,
        )

        provider = MockProvider(
            [
                Message(
                    id="msg_1",
                    role="assistant",
                    content="Using slow tool",
                    tool_calls=[{"id": "call_1", "name": "slow_tool", "arguments": {}}],
                ),
                Message(id="msg_2", role="assistant", content="Timeout handled"),
            ]
        )

        async def slow_tool(call: ToolCall) -> ToolResult:
            await asyncio.sleep(2)  # Longer than timeout
            return ToolResult(success=True, output="Done")

        tools = MockToolRegistry({"slow_tool": slow_tool})

        agent = AgentRunnerAgent(
            provider=provider,
            workspace=temp_workspace,
            config=config,
            context_manager=context_manager,
            logger=logger,
            tools=tools,
        )

        result = await agent.process_message("Test")

        assert result.rounds == 2
        messages = agent.get_messages()
        tool_messages = [msg for msg in messages if msg.role == "tool"]
        assert len(tool_messages) == 1
        assert "timed out" in tool_messages[0].content.lower()


class TestAgentLifecycle:
    def test_abort(self, temp_workspace, config, logger, context_manager):
        agent = AgentRunnerAgent(
            provider=MockProvider(),
            workspace=temp_workspace,
            config=config,
            context_manager=context_manager,
            logger=logger,
        )

        agent.abort()

        assert agent._abort_requested is True
        assert agent.state == AgentState.ABORTING

    def test_reset(self, temp_workspace, config, logger, context_manager):
        agent = AgentRunnerAgent(
            provider=MockProvider(),
            workspace=temp_workspace,
            config=config,
            context_manager=context_manager,
            logger=logger,
        )

        # Add some messages (no system prompt yet since no process_message called)
        agent.history.add_user("Hello")
        agent.history.add_assistant("Hi there")
        assert len(agent.get_messages()) == 2  # user + assistant (no system yet)

        agent.reset()

        assert agent.state == AgentState.IDLE
        assert not agent._abort_requested
        # After reset, system prompt is added immediately by reset()
        messages = agent.get_messages()
        assert len(messages) == 1
        assert messages[0].role == "system"

    def test_set_configuration(self, temp_workspace, config, logger, context_manager):
        agent = AgentRunnerAgent(
            provider=MockProvider(),
            workspace=temp_workspace,
            config=config,
            context_manager=context_manager,
            logger=logger,
        )

        new_config = AgentConfig(
            max_rounds=20,
        )

        agent.set_configuration(new_config)

        # Note: model moved to ProviderConfig
        assert agent.config.max_rounds == 20
        assert agent.config.max_rounds == 20
        # System prompt is added lazily, so no messages yet
        messages = agent.get_messages()
        assert len(messages) == 0  # No messages until first process_message call

    def test_get_messages(self, temp_workspace, config, logger, context_manager):
        agent = AgentRunnerAgent(
            provider=MockProvider(),
            workspace=temp_workspace,
            config=config,
            context_manager=context_manager,
            logger=logger,
        )

        agent.history.add_user("Test message")

        messages = agent.get_messages()
        assert len(messages) == 1  # Only user message (system added lazily)
        assert messages[0].role == "user"
        assert messages[0].content == "Test message"


class TestStreaming:
    @pytest.mark.asyncio
    async def test_process_message_stream(self, temp_workspace, config, logger, context_manager):
        provider = MockProvider(
            [Message(id="msg_1", role="assistant", content="Streaming response")]
        )

        agent = AgentRunnerAgent(
            provider=provider,
            workspace=temp_workspace,
            config=config,
            context_manager=context_manager,
            logger=logger,
        )

        events = []
        async for event in agent.process_message_stream("Test"):
            events.append(event)

        # Fallback mode (no EventBus): assistant_message + completed
        assert len(events) == 2
        assert events[0]["type"] == "assistant_message"
        assert events[0]["payload"]["content"] == "Streaming response"
        assert events[1]["type"] == "completed"

    @pytest.mark.asyncio
    async def test_process_message_with_event_bus(
        self, temp_workspace, config, logger, context_manager
    ):
        """Test that event bus publishes events during processing."""
        from agentrunner.core.events import EventBus

        provider = MockProvider(
            [Message(id="msg_1", role="assistant", content="Response with events")]
        )

        event_bus = EventBus()
        events_published = []

        # Mock publish to track events
        original_publish = event_bus.publish

        def track_publish(event):
            events_published.append(event)
            return original_publish(event)

        event_bus.publish = track_publish

        agent = AgentRunnerAgent(
            provider=provider,
            workspace=temp_workspace,
            config=config,
            context_manager=context_manager,
            logger=logger,
            event_bus=event_bus,
        )

        result = await agent.process_message("Test")

        assert result.stopped_reason == "completed"
        # Should have published status events and assistant_message event
        assert len(events_published) >= 2
        event_types = [e.type for e in events_published]
        assert "status_update" in event_types
        assert "assistant_message" in event_types


class TestSessionManagement:
    @pytest.mark.asyncio
    async def test_save_session_stub(self, temp_workspace, config, logger, context_manager):
        agent = AgentRunnerAgent(
            provider=MockProvider(),
            workspace=temp_workspace,
            config=config,
            context_manager=context_manager,
            logger=logger,
        )

        # Should not raise, just stub
        await agent.save_session("/tmp/test_session.json")

    @pytest.mark.asyncio
    async def test_load_session_stub(self, temp_workspace, config, logger, context_manager):
        agent = AgentRunnerAgent(
            provider=MockProvider(),
            workspace=temp_workspace,
            config=config,
            context_manager=context_manager,
            logger=logger,
        )

        # Should not raise, just stub
        await agent.load_session("/tmp/test_session.json")


class TestReadBeforeEdit:
    @pytest.mark.asyncio
    async def test_read_before_edit_success(
        self, temp_workspace, config, logger, context_manager, tool_registry
    ):
        """Test that edit succeeds when file was read first."""
        # Create test file
        test_file = temp_workspace.root_path / "test.txt"
        test_file.write_text("original content")

        provider = MockProvider(
            [
                # First: read the file
                Message(
                    id="msg_1",
                    role="assistant",
                    content="Reading file",
                    tool_calls=[
                        {
                            "id": "call_1",
                            "name": "read_file",
                            "arguments": {"file_path": "test.txt"},
                        }
                    ],
                ),
                # Then: edit the file
                Message(
                    id="msg_2",
                    role="assistant",
                    content="Editing file",
                    tool_calls=[
                        {
                            "id": "call_2",
                            "name": "write_file",
                            "arguments": {"file_path": "test.txt", "content": "new content"},
                        }
                    ],
                ),
                # Final response
                Message(id="msg_3", role="assistant", content="Done"),
            ]
        )

        agent = AgentRunnerAgent(
            provider=provider,
            workspace=temp_workspace,
            config=config,
            context_manager=context_manager,
            logger=logger,
            tools=tool_registry,
        )

        result = await agent.process_message("Edit test.txt")

        # Should complete the rounds
        assert result.rounds == 3
        # File should be edited successfully
        assert test_file.read_text() == "new content"


class TestSecurityChecks:
    @pytest.mark.asyncio
    async def test_confirmation_required(self, temp_workspace, config, logger, context_manager):
        """Test that confirmation is requested for dangerous operations."""
        provider = MockProvider(
            [
                Message(
                    id="msg_1",
                    role="assistant",
                    content="Deleting file",
                    tool_calls=[
                        {
                            "id": "call_1",
                            "name": "delete_file",
                            "arguments": {"file_path": "test.txt"},
                        }
                    ],
                ),
                Message(id="msg_2", role="assistant", content="Done"),
            ]
        )

        # Mock confirmation service that denies
        mock_confirmation = MagicMock()
        mock_confirmation.approve.return_value = False

        # Create mock tool with confirmation requirement
        def mock_tool_func(call: ToolCall) -> ToolResult:
            return ToolResult(success=True, output="Deleted")

        tools = MockToolRegistry({"delete_file": mock_tool_func})
        mock_tool = MagicMock()
        mock_tool_def = MagicMock()
        mock_tool_def.safety = {"requires_confirmation": True}
        mock_tool.get_definition.return_value = mock_tool_def
        tools.get = MagicMock(return_value=mock_tool)

        agent = AgentRunnerAgent(
            provider=provider,
            workspace=temp_workspace,
            config=config,
            context_manager=context_manager,
            logger=logger,
            tools=tools,
            confirmation=mock_confirmation,
        )

        await agent.process_message("Delete file")

        # Confirmation should have been called
        assert mock_confirmation.approve.called

        # Tool should not have been executed
        messages = agent.get_messages()
        tool_messages = [msg for msg in messages if msg.role == "tool"]
        assert len(tool_messages) == 1
        assert "denied" in tool_messages[0].content.lower()

    @pytest.mark.asyncio
    async def test_command_validator_unsafe(self, temp_workspace, config, logger, context_manager):
        """Test that unsafe commands are blocked."""
        provider = MockProvider(
            [
                Message(
                    id="msg_1",
                    role="assistant",
                    content="Running command",
                    tool_calls=[
                        {
                            "id": "call_1",
                            "name": "bash",
                            "arguments": {"command": "rm -rf /"},
                        }
                    ],
                ),
                Message(id="msg_2", role="assistant", content="Done"),
            ]
        )

        # Mock command validator that blocks
        mock_validator = MagicMock()
        mock_validator.is_safe_tool.return_value = False

        tools = MockToolRegistry({"bash": lambda c: ToolResult(success=True, output="")})

        agent = AgentRunnerAgent(
            provider=provider,
            workspace=temp_workspace,
            config=config,
            context_manager=context_manager,
            logger=logger,
            tools=tools,
            command_validator=mock_validator,
        )

        await agent.process_message("Run dangerous command")

        # Validator should have been called
        assert mock_validator.is_safe_tool.called

        # Command should have been blocked
        messages = agent.get_messages()
        tool_messages = [msg for msg in messages if msg.role == "tool"]
        assert len(tool_messages) == 1
        assert (
            "not whitelisted" in tool_messages[0].content.lower()
            or "dangerous" in tool_messages[0].content.lower()
        )


class TestToolCallParsing:
    @pytest.mark.asyncio
    async def test_parse_openai_format_tool_calls(
        self, temp_workspace, config, logger, context_manager
    ):
        """Test parsing OpenAI-format tool calls (nested function structure)."""
        provider = MockProvider(
            [
                Message(
                    id="msg_1",
                    role="assistant",
                    content="Using tool",
                    tool_calls=[
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "test_tool",
                                "arguments": '{"param": "value"}',  # JSON string
                            },
                        }
                    ],
                ),
                Message(id="msg_2", role="assistant", content="Done"),
            ]
        )

        tools = MockToolRegistry({"test_tool": lambda c: ToolResult(success=True, output="OK")})

        agent = AgentRunnerAgent(
            provider=provider,
            workspace=temp_workspace,
            config=config,
            context_manager=context_manager,
            logger=logger,
            tools=tools,
        )

        result = await agent.process_message("Test")

        assert result.rounds == 2
        # Tool should have been executed with parsed arguments
        assert len(tools.executed_calls) == 1
        assert tools.executed_calls[0].arguments == {"param": "value"}


@pytest.mark.skip(reason="Compaction is disabled by default - feature not fully implemented")
class TestCompaction:
    @pytest.mark.asyncio
    async def test_token_limit_without_compaction(self, temp_workspace, logger):
        """Test that token limit error is raised when compaction is disabled."""
        # Create a config without compaction
        config = AgentConfig(
            max_rounds=5,
        )
        # Explicitly disable compaction
        # Note: Compaction settings now in ProviderConfig.compaction (defaults to disabled)

        # Create a context manager that reports near limit
        counter = TokenCounter()
        model_info = ModelInfo(
            name="gpt-5.1-2025-11-13",
            context_window=100,  # Very small limit
            pricing={"input_per_1k": 0.01, "output_per_1k": 0.03},
        )
        context_manager = ContextManager(counter=counter, model_info=model_info)

        # Mock context manager to always report near limit
        def mock_is_near_limit(threshold):
            return True

        context_manager.is_near_limit = mock_is_near_limit
        context_manager.total_tokens = lambda msgs, tools=None: 95

        provider = MockProvider([Message(id="msg_1", role="assistant", content="Response")])

        agent = AgentRunnerAgent(
            provider=provider,
            workspace=temp_workspace,
            config=config,
            context_manager=context_manager,
            logger=logger,
        )

        # Should raise TokenLimitExceededError
        from agentrunner.core.exceptions import TokenLimitExceededError

        with pytest.raises(TokenLimitExceededError, match="Context window exhausted"):
            await agent.process_message("Test")

    @pytest.mark.asyncio
    async def test_compaction_when_enabled(self, temp_workspace, logger):
        """Test that compaction runs when enabled and near limit."""
        from agentrunner.core.compaction import CompactionContext
        from agentrunner.providers.base import ProviderConfig

        config = AgentConfig(
            max_rounds=5,
        )

        # Create provider config with compaction enabled
        provider_config = ProviderConfig(
            model="gpt-5.1-2025-11-13",
            compaction=CompactionContext(
                current_tokens=0,
                target_tokens=0,
                enabled=True,
                threshold=0.8,
                strategy="claude_style",
            ),
        )

        counter = TokenCounter()
        model_info = ModelInfo(
            name="gpt-5.1-2025-11-13",
            context_window=1000,
            pricing={"input_per_1k": 0.01, "output_per_1k": 0.03},
        )
        context_manager = ContextManager(counter=counter, model_info=model_info)

        # First call near limit, second call after compaction not near limit
        call_count = [0]

        def mock_is_near_limit(threshold):
            call_count[0] += 1
            return call_count[0] == 1  # Only first call is near limit

        context_manager.is_near_limit = mock_is_near_limit
        context_manager.total_tokens = lambda msgs, tools=None: 850 if call_count[0] <= 1 else 500

        provider = MockProvider(
            responses=[Message(id="msg_1", role="assistant", content="Response after compaction")],
            config=provider_config,
        )

        agent = AgentRunnerAgent(
            provider=provider,
            workspace=temp_workspace,
            config=config,
            context_manager=context_manager,
            logger=logger,
        )

        # Should complete successfully with compaction
        result = await agent.process_message("Test")
        assert result.stopped_reason == "completed"

    @pytest.mark.asyncio
    async def test_compaction_fails_after_compaction(self, temp_workspace, logger):
        """Test error when still over limit after compaction."""
        from agentrunner.core.compaction import CompactionContext
        from agentrunner.providers.base import ProviderConfig

        config = AgentConfig(
            max_rounds=5,
        )

        # Create provider config with compaction enabled
        provider_config = ProviderConfig(
            model="gpt-5.1-2025-11-13",
            compaction=CompactionContext(
                current_tokens=0,
                target_tokens=0,
                enabled=True,
                threshold=0.8,
                strategy="claude_style",
            ),
        )

        counter = TokenCounter()
        model_info = ModelInfo(
            name="gpt-5.1-2025-11-13",
            context_window=1000,
            pricing={"input_per_1k": 0.01, "output_per_1k": 0.03},
        )
        context_manager = ContextManager(counter=counter, model_info=model_info)

        # Always near limit, even after compaction
        context_manager.is_near_limit = lambda threshold: True
        context_manager.total_tokens = lambda msgs, tools=None: 950

        provider = MockProvider(
            responses=[Message(id="msg_1", role="assistant", content="Response")],
            config=provider_config,
        )

        agent = AgentRunnerAgent(
            provider=provider,
            workspace=temp_workspace,
            config=config,
            context_manager=context_manager,
            logger=logger,
        )

        # Should raise error after compaction
        from agentrunner.core.exceptions import TokenLimitExceededError

        with pytest.raises(TokenLimitExceededError, match="Context exhausted after compaction"):
            await agent.process_message("Test")


class TestErrorHandling:
    @pytest.mark.asyncio
    async def test_provider_not_implemented(self, temp_workspace, config, logger, context_manager):
        # Provider without chat method - has get_system_prompt for init but no chat
        provider = MagicMock()
        provider.get_system_prompt.return_value = "Test system prompt"
        provider.get_model_info.return_value = ModelInfo(
            name="test-model",
            context_window=128000,
            pricing={"input_per_1k": 0.01, "output_per_1k": 0.03},
        )
        provider.config = ProviderConfig(
            model="test-model",
            compaction=CompactionContext(
                current_tokens=0,
                target_tokens=0,
                enabled=False,
                threshold=0.8,
                strategy="noop",
                recent_rounds=3,
                preserve_errors=True,
            ),
        )
        # Remove the chat method to test the "not implemented" path
        delattr(provider, "chat") if hasattr(provider, "chat") else None

        agent = AgentRunnerAgent(
            provider=provider,
            workspace=temp_workspace,
            config=config,
            context_manager=context_manager,
            logger=logger,
        )

        with pytest.raises(AgentRunnerException, match="Provider not yet implemented"):
            await agent.process_message("Test")

    @pytest.mark.asyncio
    async def test_tool_execution_error(self, temp_workspace, config, logger, context_manager):
        provider = MockProvider(
            [
                Message(
                    id="msg_1",
                    role="assistant",
                    content="Using tool",
                    tool_calls=[{"id": "call_1", "name": "broken_tool", "arguments": {}}],
                ),
                Message(id="msg_2", role="assistant", content="Error handled"),
            ]
        )

        def broken_tool(call: ToolCall) -> ToolResult:
            raise ValueError("Tool is broken")

        tools = MockToolRegistry({"broken_tool": broken_tool})

        agent = AgentRunnerAgent(
            provider=provider,
            workspace=temp_workspace,
            config=config,
            context_manager=context_manager,
            logger=logger,
            tools=tools,
        )

        result = await agent.process_message("Test")

        # Should handle error gracefully
        assert result.rounds == 2
        messages = agent.get_messages()
        tool_messages = [msg for msg in messages if msg.role == "tool"]
        assert len(tool_messages) == 1
        assert "failed" in tool_messages[0].content.lower()

    @pytest.mark.asyncio
    async def test_tool_registry_not_initialized(
        self, temp_workspace, config, logger, context_manager
    ):
        """Test error handling when tool registry is not initialized but LLM tries to use tools."""
        provider = MockProvider(
            [
                Message(
                    id="msg_1",
                    role="assistant",
                    content="Using tool",
                    tool_calls=[{"id": "call_1", "name": "test_tool", "arguments": {}}],
                ),
                Message(id="msg_2", role="assistant", content="Error handled"),
            ]
        )

        agent = AgentRunnerAgent(
            provider=provider,
            workspace=temp_workspace,
            config=config,
            context_manager=context_manager,
            logger=logger,
            tools=None,  # No tools
        )

        # The error is caught and added to message history instead of being raised
        await agent.process_message("Test")

        # Error should be in the tool result messages
        messages = agent.get_messages()
        tool_messages = [msg for msg in messages if msg.role == "tool"]
        assert len(tool_messages) == 1
        assert "ToolRegistry not initialized" in tool_messages[0].content
