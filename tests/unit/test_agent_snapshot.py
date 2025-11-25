"""Unit tests for AgentRunnerAgent snapshot/restore functionality."""

from unittest.mock import MagicMock

import pytest

from agentrunner.core.agent import AgentRunnerAgent, AgentState
from agentrunner.core.config import AgentConfig
from agentrunner.core.logger import AgentRunnerLogger
from agentrunner.core.messages import Message
from agentrunner.core.tokens import ContextManager
from agentrunner.core.workspace import Workspace
from agentrunner.providers.base import ProviderConfig
from agentrunner.providers.openai_provider import OpenAIProvider


class TestAgentSnapshot:
    """Tests for agent snapshot and restore methods."""

    @pytest.fixture
    def workspace(self, tmp_path):
        """Create a test workspace."""
        return Workspace(str(tmp_path))

    @pytest.fixture
    def provider(self):
        """Create a mock provider."""
        provider_config = ProviderConfig(model="gpt-5.1-2025-11-13")
        provider = OpenAIProvider(config=provider_config, api_key="test-key")
        # Mock the OpenAI client
        provider.client = MagicMock()
        return provider

    @pytest.fixture
    def agent(self, provider, workspace):
        """Create a test agent."""
        from agentrunner.core.tokens import TokenCounter

        config = AgentConfig(max_rounds=10)
        counter = TokenCounter()
        context_manager = ContextManager(counter, provider.get_model_info())
        logger = AgentRunnerLogger()

        agent = AgentRunnerAgent(
            provider=provider,
            workspace=workspace,
            config=config,
            context_manager=context_manager,
            logger=logger,
        )

        return agent

    def test_snapshot_initial_state(self, agent):
        """Test snapshot of agent in initial state."""
        snapshot = agent.snapshot()

        assert "model" in snapshot
        assert snapshot["model"] == "gpt-5.1-2025-11-13"
        assert "messages" in snapshot
        assert isinstance(snapshot["messages"], list)
        assert "input_tokens" in snapshot
        assert snapshot["input_tokens"] >= 0
        assert "output_tokens" in snapshot
        assert snapshot["output_tokens"] >= 0
        assert "rounds" in snapshot
        assert snapshot["rounds"] == 0
        assert "workspace_root" in snapshot
        assert "state" in snapshot
        assert snapshot["state"] == "idle"

    def test_snapshot_with_messages(self, agent):
        """Test snapshot captures message history."""
        agent.history.add(Message(id="msg1", role="user", content="Hello"))
        agent.history.add(Message(id="msg2", role="assistant", content="Hi there!"))

        snapshot = agent.snapshot()

        assert len(snapshot["messages"]) == 2  # 2 messages (system prompt is lazy-loaded)
        assert snapshot["messages"][0]["role"] == "user"
        assert snapshot["messages"][0]["content"] == "Hello"
        assert snapshot["messages"][1]["role"] == "assistant"
        assert snapshot["messages"][1]["content"] == "Hi there!"

    def test_snapshot_with_meta(self, agent):
        """Test snapshot captures message metadata."""
        agent.history.add(Message(id="msg1", role="user", content="Test", meta={"custom": "data"}))

        snapshot = agent.snapshot()

        assert snapshot["messages"][0]["meta"] == {"custom": "data"}

    def test_snapshot_captures_tokens(self, agent):
        """Test snapshot captures token counts."""
        agent.context_manager.input_tokens_used = 1000
        agent.context_manager.output_tokens_used = 500

        snapshot = agent.snapshot()

        assert snapshot["input_tokens"] == 1000
        assert snapshot["output_tokens"] == 500

    def test_snapshot_counts_rounds(self, agent):
        """Test snapshot counts assistant rounds correctly."""
        agent.history.add(Message(id="msg1", role="user", content="Q1"))
        agent.history.add(Message(id="msg2", role="assistant", content="A1"))
        agent.history.add(Message(id="msg3", role="user", content="Q2"))
        agent.history.add(Message(id="msg4", role="assistant", content="A2"))

        snapshot = agent.snapshot()

        assert snapshot["rounds"] == 2

    def test_snapshot_captures_state(self, agent):
        """Test snapshot captures agent state."""
        agent.state = AgentState.THINKING

        snapshot = agent.snapshot()

        assert snapshot["state"] == "thinking"

    def test_from_snapshot_creates_agent(self, provider, workspace):
        """Test from_snapshot class method creates agent."""
        from agentrunner.core.events import EventBus
        from agentrunner.tools.base import ToolRegistry

        tools = ToolRegistry(workspace)
        event_bus = EventBus()

        snapshot = {
            "model": "gpt-5.1-2025-11-13",
            "messages": [
                {"id": "msg1", "role": "user", "content": "Hello", "meta": {}},
                {"id": "msg2", "role": "assistant", "content": "Hi", "meta": {}},
            ],
            "input_tokens": 50,
            "output_tokens": 50,
            "rounds": 1,
            "workspace_root": str(workspace.root_path),
            "state": "idle",
        }

        agent = AgentRunnerAgent.from_snapshot(
            snapshot=snapshot,
            provider=provider,
            workspace=workspace,
            tools=tools,
            event_bus=event_bus,
        )

        assert agent is not None
        assert isinstance(agent, AgentRunnerAgent)
        assert len(agent.history.messages) == 2
        assert agent.context_manager.input_tokens_used == 50
        assert agent.context_manager.output_tokens_used == 50
        assert agent.state == AgentState.IDLE

    def test_from_snapshot_restores_messages(self, provider, workspace):
        """Test from_snapshot restores message history."""
        from agentrunner.tools.base import ToolRegistry

        tools = ToolRegistry(workspace)

        snapshot = {
            "model": "gpt-5.1-2025-11-13",
            "messages": [
                {"id": "msg1", "role": "user", "content": "Hello", "meta": {"id": "1"}},
                {"id": "msg2", "role": "assistant", "content": "Hi", "meta": {"id": "2"}},
            ],
            "input_tokens": 50,
            "output_tokens": 50,
            "rounds": 1,
            "workspace_root": str(workspace.root_path),
            "state": "idle",
        }

        agent = AgentRunnerAgent.from_snapshot(
            snapshot=snapshot,
            provider=provider,
            workspace=workspace,
            tools=tools,
        )

        messages = agent.history.messages
        assert len(messages) == 2
        assert messages[0].role == "user"
        assert messages[0].content == "Hello"
        assert messages[0].meta["id"] == "1"
        assert messages[1].role == "assistant"
        assert messages[1].content == "Hi"

    def test_from_snapshot_restores_state(self, provider, workspace):
        """Test from_snapshot restores agent state."""
        from agentrunner.tools.base import ToolRegistry

        tools = ToolRegistry(workspace)

        snapshot = {
            "model": "gpt-5.1-2025-11-13",
            "messages": [],
            "input_tokens": 0,
            "output_tokens": 0,
            "rounds": 0,
            "workspace_root": str(workspace.root_path),
            "state": "completed",
        }

        agent = AgentRunnerAgent.from_snapshot(
            snapshot=snapshot,
            provider=provider,
            workspace=workspace,
            tools=tools,
        )

        assert agent.state == AgentState.COMPLETED

    def test_from_snapshot_handles_invalid_state(self, provider, workspace):
        """Test from_snapshot handles invalid state gracefully."""
        from agentrunner.tools.base import ToolRegistry

        tools = ToolRegistry(workspace)

        snapshot = {
            "model": "gpt-5.1-2025-11-13",
            "messages": [],
            "input_tokens": 0,
            "output_tokens": 0,
            "rounds": 0,
            "workspace_root": str(workspace.root_path),
            "state": "invalid_state",
        }

        agent = AgentRunnerAgent.from_snapshot(
            snapshot=snapshot,
            provider=provider,
            workspace=workspace,
            tools=tools,
        )

        # Should default to IDLE on invalid state
        assert agent.state == AgentState.IDLE

    def test_snapshot_roundtrip(self, agent):
        """Test that snapshot -> restore produces equivalent agent."""
        from agentrunner.core.messages import Message

        # Add some state
        agent.history.add(Message(id="msg1", role="user", content="Test"))
        agent.history.add(Message(id="msg2", role="assistant", content="Response"))
        agent.context_manager.input_tokens_used = 300
        agent.context_manager.output_tokens_used = 200
        agent.state = AgentState.COMPLETED

        # Take snapshot
        snapshot = agent.snapshot()

        # Create new agent from snapshot
        restored = AgentRunnerAgent.from_snapshot(
            snapshot=snapshot,
            provider=agent.provider,
            workspace=agent.workspace,
            tools=agent.tools,
            event_bus=agent.event_bus,
        )

        # Verify equivalence
        assert len(restored.history.messages) == len(agent.history.messages)
        assert restored.context_manager.input_tokens_used == agent.context_manager.input_tokens_used
        assert (
            restored.context_manager.output_tokens_used == agent.context_manager.output_tokens_used
        )
        assert restored.state == agent.state

    def test_snapshot_with_created_at(self, agent):
        """Test snapshot includes created_at timestamp."""
        agent._created_at = "2025-10-28T12:00:00"

        snapshot = agent.snapshot()

        assert snapshot["created_at"] == "2025-10-28T12:00:00"

    def test_from_snapshot_restores_created_at(self, provider, workspace):
        """Test from_snapshot restores created_at timestamp."""
        from agentrunner.tools.base import ToolRegistry

        tools = ToolRegistry(workspace)

        snapshot = {
            "model": "gpt-5.1-2025-11-13",
            "messages": [],
            "input_tokens": 0,
            "output_tokens": 0,
            "rounds": 0,
            "workspace_root": str(workspace.root_path),
            "state": "idle",
            "created_at": "2025-10-28T12:00:00",
        }

        agent = AgentRunnerAgent.from_snapshot(
            snapshot=snapshot,
            provider=provider,
            workspace=workspace,
            tools=tools,
        )

        assert agent._created_at == "2025-10-28T12:00:00"
