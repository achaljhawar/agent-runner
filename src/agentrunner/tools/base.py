"""Base tool framework and registry.

Defines the abstract interface for all tools and the registry for managing them.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

# Import TYPE_CHECKING to avoid circular imports
from typing import TYPE_CHECKING, Any

from agentrunner.core.exceptions import ToolExecutionError
from agentrunner.core.logger import AgentRunnerLogger
from agentrunner.core.tool_protocol import ToolCall, ToolDefinition, ToolResult
from agentrunner.core.workspace import Workspace

if TYPE_CHECKING:
    from agentrunner.core.events import EventBus


@dataclass
class ToolContext:
    """Context provided to tools during execution.

    CRITICAL: Keep in sync with multi-agent architecture.
    """

    workspace: Workspace
    logger: AgentRunnerLogger
    model_id: str
    event_bus: "EventBus | None" = None
    config: dict[str, Any] = field(default_factory=dict)

    # Unix UID/GID for process isolation (optional, defaults to current user)
    session_uid: int | None = None
    session_gid: int | None = None

    # Additional context for deployment state, database connections, etc.
    deployment_context: dict[str, Any] = field(default_factory=dict)


class BaseTool(ABC):
    """Abstract base class for all tools.

    To create a new tool:
    1. Subclass BaseTool
    2. Implement execute() method
    3. Implement get_definition() to return ToolDefinition
    4. Register with ToolRegistry
    """

    @abstractmethod
    async def execute(self, call: ToolCall, context: ToolContext) -> ToolResult:
        """Execute the tool with given arguments.

        Args:
            call: Tool call with name and arguments
            context: Execution context (workspace, logger, config)

        Returns:
            ToolResult with success status and output/error

        Raises:
            ToolExecutionError: On execution failures
        """
        raise NotImplementedError

    @abstractmethod
    def get_definition(self) -> ToolDefinition:
        """Get the tool's definition for LLM.

        Returns:
            ToolDefinition with name, description, and JSON Schema
        """
        raise NotImplementedError

    def get_name(self) -> str:
        """Get tool name.

        Returns:
            Tool name
        """
        return self.get_definition().name

    def get_description(self) -> str:
        """Get tool description.

        Returns:
            Tool description
        """
        return self.get_definition().description


class ToolRegistry:
    """Registry for managing and executing tools.

    Manages both native AgentRunner tools and external tools.
    Native tools are the core framework tools, while external tools
    can be added dynamically from other sources.
    """

    def __init__(self, context: ToolContext) -> None:
        """Initialize tool registry.

        Args:
            context: Tool execution context
        """
        self.context = context
        self._tools: dict[str, BaseTool] = {}
        self._native_tools: set[str] = set()  # Track native tool names

    def register(self, tool: BaseTool, is_native: bool = True) -> None:
        """Register a tool.

        Args:
            tool: Tool instance to register
            is_native: True if this is a native AgentRunner tool, False for external tools

        Raises:
            ValueError: If tool with same name already registered
        """
        name = tool.get_name()
        if name in self._tools:
            raise ValueError(f"Tool already registered: {name}")

        self._tools[name] = tool

        if is_native:
            self._native_tools.add(name)
            self.context.logger.debug("Native tool registered", tool_name=name)
        else:
            self.context.logger.debug("External tool registered", tool_name=name)

    def has(self, name: str) -> bool:
        """Check if tool is registered.

        Args:
            name: Tool name

        Returns:
            True if tool exists
        """
        return name in self._tools

    def get(self, name: str) -> BaseTool | None:
        """Get tool by name.

        Args:
            name: Tool name

        Returns:
            Tool instance or None if not found
        """
        return self._tools.get(name)

    async def execute(self, call: ToolCall) -> ToolResult:
        """Execute a tool call.

        Args:
            call: Tool call to execute

        Returns:
            ToolResult from tool execution

        Raises:
            ToolExecutionError: If tool execution fails
        """
        if not self.has(call.name):
            self.context.logger.warn("Unknown tool requested", tool_name=call.name)
            return ToolResult(
                success=False,
                error=f"Unknown tool: {call.name}",
                error_code="E_TOOL_UNKNOWN",
            )

        tool = self._tools[call.name]

        try:
            self.context.logger.debug("Executing tool", tool_name=call.name)
            result = await tool.execute(call, self.context)
            self.context.logger.info(
                "Tool executed",
                tool_name=call.name,
                success=result.success,
            )
            return result
        except Exception as e:
            self.context.logger.error(
                "Tool execution failed",
                tool_name=call.name,
                error=str(e),
            )
            raise ToolExecutionError(
                error_code="E_VALIDATION",
                message=f"Tool {call.name} failed: {e}",
            ) from e

    def get_native_tools(self) -> list[ToolDefinition]:
        """Get native AgentRunner tool definitions.

        Returns:
            List of native ToolDefinitions
        """
        return [
            self._tools[name].get_definition() for name in self._native_tools if name in self._tools
        ]

    def get_external_tools(self) -> list[ToolDefinition]:
        """Get external tool definitions.

        Returns:
            List of external ToolDefinitions
        """
        external_names = set(self._tools.keys()) - self._native_tools
        return [self._tools[name].get_definition() for name in external_names]

    def get_definitions(self) -> list[ToolDefinition]:
        """Get all tool definitions for LLM.

        Returns:
            List of all ToolDefinitions (native + external)
        """
        return [tool.get_definition() for tool in self._tools.values()]

    def list_tools(self) -> list[str]:
        """List all registered tool names.

        Returns:
            List of tool names
        """
        return list(self._tools.keys())

    def is_native(self, tool_name: str) -> bool:
        """Check if a tool is a native tool.

        Args:
            tool_name: Name of the tool

        Returns:
            True if tool is native, False if external or not found
        """
        return tool_name in self._native_tools

    def clear(self) -> None:
        """Clear all registered tools."""
        self._tools.clear()
        self._native_tools.clear()
        self.context.logger.debug("Tool registry cleared")
