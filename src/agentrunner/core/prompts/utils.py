"""Utility functions for working with prompts."""

from typing import TYPE_CHECKING, Any

from agentrunner.core.prompts.base import PromptSection, SystemPromptBuilder
from agentrunner.core.prompts.sections import get_default_sections

if TYPE_CHECKING:
    from agentrunner.core.tool_protocol import ToolDefinition


def generate_tool_section_from_definitions(tools: list["ToolDefinition"]) -> PromptSection:
    """Generate tool documentation section from actual tool definitions.

    Dynamically creates tool documentation based on registered tools,
    ensuring system prompt accurately reflects available tools.

    Args:
        tools: List of tool definitions

    Returns:
        PromptSection with tool documentation
    """
    if not tools:
        return PromptSection(
            name="available_tools",
            content="# Available Tools\n\nNo tools currently registered.",
            priority=200,
        )

    content_parts = ["# Available Tools\n"]

    # Group tools by category (inferred from name patterns)
    file_tools = []
    search_tools = []
    execution_tools = []
    image_tools = []
    other_tools = []

    for tool in tools:
        name_lower = tool.name.lower()
        if any(x in name_lower for x in ["read", "write", "edit", "create", "delete", "file"]):
            file_tools.append(tool)
        elif any(x in name_lower for x in ["grep", "glob", "search", "find", "list"]):
            search_tools.append(tool)
        elif any(
            x in name_lower for x in ["bash", "execute", "run", "deploy", "scaffold", "clean"]
        ):
            execution_tools.append(tool)
        elif any(x in name_lower for x in ["image", "video", "generation", "fetch"]):
            image_tools.append(tool)
        else:
            other_tools.append(tool)

    # Format each category
    if file_tools:
        content_parts.append("\n## File Operations\n")
        for tool in file_tools:
            content_parts.append(f"- **{tool.name}**: {tool.description}\n")

    if search_tools:
        content_parts.append("\n## Search & Discovery\n")
        for tool in search_tools:
            content_parts.append(f"- **{tool.name}**: {tool.description}\n")

    if execution_tools:
        content_parts.append("\n## Execution & Deployment\n")
        for tool in execution_tools:
            content_parts.append(f"- **{tool.name}**: {tool.description}\n")

    if image_tools:
        content_parts.append("\n## Media & Generation\n")
        for tool in image_tools:
            content_parts.append(f"- **{tool.name}**: {tool.description}\n")

    if other_tools:
        content_parts.append("\n## Other Tools\n")
        for tool in other_tools:
            content_parts.append(f"- **{tool.name}**: {tool.description}\n")

    return PromptSection(
        name="available_tools",
        content="".join(content_parts),
        priority=200,  # After identity, before guidelines
    )


def build_prompt(
    workspace_root: str,
    model_name: str | None = None,
    available_tools: list[str] | None = None,
    tool_definitions: list["ToolDefinition"] | None = None,
    max_rounds: int | None = None,
    **extra_variables: Any,
) -> str:
    """Build system prompt with default sections.

    Args:
        workspace_root: Absolute path to workspace root
        model_name: Current model name (optional)
        available_tools: List of available tool names (optional, deprecated)
        tool_definitions: Tool definitions for dynamic tool section (optional, preferred)
        max_rounds: Maximum tool execution rounds (optional)
        **extra_variables: Additional variables for template substitution

    Returns:
        Complete system prompt string
    """
    builder = SystemPromptBuilder()

    # Add all default sections except tools if we have dynamic definitions
    for section in get_default_sections():
        # Skip static tool section if we have dynamic definitions
        if section.name == "available_tools" and tool_definitions is not None:
            continue
        builder.add_section(section)

    # Add dynamic tool section if provided
    if tool_definitions is not None:
        tool_section = generate_tool_section_from_definitions(tool_definitions)
        builder.add_section(tool_section)

    # Load custom sections if they exist
    builder.load_custom_sections(workspace_root)

    # Build variables
    variables: dict[str, Any] = {
        "workspace_root": workspace_root,
    }

    if model_name:
        variables["model_name"] = model_name

    if available_tools:
        variables["available_tools"] = ", ".join(available_tools)

    if max_rounds:
        variables["max_rounds"] = max_rounds

    # Add any extra variables
    variables.update(extra_variables)

    return builder.build(variables=variables)
