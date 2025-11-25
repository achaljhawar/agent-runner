"""Tests for base tool framework and registry."""

import pytest

from agentrunner.core.exceptions import ToolExecutionError
from agentrunner.core.logger import AgentRunnerLogger
from agentrunner.core.tool_protocol import ToolCall, ToolDefinition, ToolResult
from agentrunner.core.workspace import Workspace
from agentrunner.tools.base import BaseTool, ToolContext, ToolRegistry


class MockTool(BaseTool):
    """Mock tool for testing."""

    def __init__(self, name: str = "mock_tool", should_fail: bool = False):
        self.name = name
        self.should_fail = should_fail
        self.execution_count = 0

    async def execute(self, call: ToolCall, context: ToolContext) -> ToolResult:
        self.execution_count += 1
        if self.should_fail:
            raise ValueError("Mock tool failure")
        return ToolResult(
            success=True,
            output=f"Executed {call.name} with args: {call.arguments}",
        )

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description="A mock tool for testing",
            parameters={
                "type": "object",
                "properties": {"arg": {"type": "string"}},
                "required": [],
            },
        )


@pytest.fixture
def temp_workspace(tmp_path):
    return Workspace(str(tmp_path))


@pytest.fixture
def logger():
    return AgentRunnerLogger()


@pytest.fixture
def tool_context(temp_workspace, logger):
    return ToolContext(workspace=temp_workspace, logger=logger, model_id="test-model")


class TestToolContext:
    def test_tool_context_creation(self, temp_workspace, logger):
        context = ToolContext(workspace=temp_workspace, logger=logger, model_id="test-model")
        assert context.workspace == temp_workspace
        assert context.logger == logger
        assert context.model_id == "test-model"
        assert context.config == {}

    def test_tool_context_with_config(self, temp_workspace, logger):
        context = ToolContext(
            workspace=temp_workspace,
            logger=logger,
            model_id="test-model",
            config={"timeout": 30},
        )
        assert context.config["timeout"] == 30


class TestBaseTool:
    def test_get_name(self):
        tool = MockTool(name="test_tool")
        assert tool.get_name() == "test_tool"

    def test_get_description(self):
        tool = MockTool()
        assert tool.get_description() == "A mock tool for testing"

    @pytest.mark.asyncio
    async def test_execute(self, tool_context):
        tool = MockTool()
        call = ToolCall(id="call_1", name="mock_tool", arguments={"arg": "value"})

        result = await tool.execute(call, tool_context)

        assert result.success is True
        assert "Executed mock_tool" in result.output
        assert tool.execution_count == 1

    def test_abstract_methods_cannot_be_called(self):
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            BaseTool()


class TestToolRegistry:
    def test_registry_initialization(self, tool_context):
        registry = ToolRegistry(tool_context)
        assert registry.context == tool_context
        assert len(registry.list_tools()) == 0

    def test_register_tool(self, tool_context):
        registry = ToolRegistry(tool_context)
        tool = MockTool(name="test_tool")

        registry.register(tool)

        assert registry.has("test_tool")
        assert registry.get("test_tool") == tool
        assert "test_tool" in registry.list_tools()

    def test_register_duplicate_tool_fails(self, tool_context):
        registry = ToolRegistry(tool_context)
        tool1 = MockTool(name="same_tool")
        tool2 = MockTool(name="same_tool")

        registry.register(tool1)

        with pytest.raises(ValueError, match="Tool already registered"):
            registry.register(tool2)

    def test_has_tool(self, tool_context):
        registry = ToolRegistry(tool_context)
        tool = MockTool(name="exists")

        registry.register(tool)

        assert registry.has("exists") is True
        assert registry.has("nonexistent") is False

    def test_get_tool(self, tool_context):
        registry = ToolRegistry(tool_context)
        tool = MockTool(name="my_tool")

        registry.register(tool)

        retrieved = registry.get("my_tool")
        assert retrieved == tool
        assert registry.get("nonexistent") is None

    @pytest.mark.asyncio
    async def test_execute_tool(self, tool_context):
        registry = ToolRegistry(tool_context)
        tool = MockTool(name="exec_tool")
        registry.register(tool)

        call = ToolCall(id="call_1", name="exec_tool", arguments={"arg": "test"})
        result = await registry.execute(call)

        assert result.success is True
        assert "Executed exec_tool" in result.output
        assert tool.execution_count == 1

    @pytest.mark.asyncio
    async def test_execute_unknown_tool(self, tool_context):
        registry = ToolRegistry(tool_context)

        call = ToolCall(id="call_1", name="unknown_tool", arguments={})
        result = await registry.execute(call)

        assert result.success is False
        assert "Unknown tool" in result.error
        assert result.error_code == "E_TOOL_UNKNOWN"

    @pytest.mark.asyncio
    async def test_execute_tool_failure(self, tool_context):
        registry = ToolRegistry(tool_context)
        tool = MockTool(name="failing_tool", should_fail=True)
        registry.register(tool)

        call = ToolCall(id="call_1", name="failing_tool", arguments={})

        with pytest.raises(ToolExecutionError, match="failed"):
            await registry.execute(call)

    def test_get_definitions(self, tool_context):
        registry = ToolRegistry(tool_context)
        tool1 = MockTool(name="tool1")
        tool2 = MockTool(name="tool2")

        registry.register(tool1)
        registry.register(tool2)

        definitions = registry.get_definitions()

        assert len(definitions) == 2
        assert all(isinstance(d, ToolDefinition) for d in definitions)
        names = [d.name for d in definitions]
        assert "tool1" in names
        assert "tool2" in names

    def test_list_tools(self, tool_context):
        registry = ToolRegistry(tool_context)
        registry.register(MockTool(name="tool_a"))
        registry.register(MockTool(name="tool_b"))
        registry.register(MockTool(name="tool_c"))

        tools = registry.list_tools()

        assert len(tools) == 3
        assert "tool_a" in tools
        assert "tool_b" in tools
        assert "tool_c" in tools

    def test_clear(self, tool_context):
        registry = ToolRegistry(tool_context)
        registry.register(MockTool(name="tool1"))
        registry.register(MockTool(name="tool2"))

        assert len(registry.list_tools()) == 2

        registry.clear()

        assert len(registry.list_tools()) == 0
        assert not registry.has("tool1")
        assert not registry.has("tool2")

    @pytest.mark.asyncio
    async def test_multiple_executions(self, tool_context):
        registry = ToolRegistry(tool_context)
        tool = MockTool(name="multi_exec")
        registry.register(tool)

        for i in range(3):
            call = ToolCall(id=f"call_{i}", name="multi_exec", arguments={"n": i})
            result = await registry.execute(call)
            assert result.success is True

        assert tool.execution_count == 3
