"""Workspace cleaning tool for preparing fresh environments.

Implements CleanWorkspaceTool for resetting the current model's workspace.
CRITICAL: Use with extreme caution - this deletes files permanently.
"""

import shutil
from pathlib import Path

from agentrunner.core.exceptions import E_NOT_FOUND, E_VALIDATION
from agentrunner.core.tool_protocol import ToolCall, ToolDefinition, ToolResult
from agentrunner.tools.base import BaseTool, ToolContext


class CleanWorkspaceTool(BaseTool):
    """Clean the current model's workspace, removing all project files.

    CRITICAL WARNINGS:
    - This tool PERMANENTLY DELETES files and subdirectories
    - Use ONLY before scaffolding to avoid conflicts
    - NEVER use on workspace root or parent directories
    - NEVER use on directories with important existing code
    """

    async def execute(self, call: ToolCall, context: ToolContext) -> ToolResult:
        """Reset the current workspace by deleting all contents."""
        workspace_root = Path(context.workspace.root_path)

        if not workspace_root.exists() or not workspace_root.is_dir():
            return ToolResult(
                success=False,
                error="Workspace directory not found",
                error_code=E_NOT_FOUND,
            )

        # Count items before cleaning
        items_to_clean = list(workspace_root.iterdir())
        item_count = len(items_to_clean)

        if item_count == 0:
            context.logger.info("Workspace already empty", path=str(workspace_root))
            return ToolResult(
                success=True,
                output="Workspace already clean",
                data={"items_removed": 0, "path": str(workspace_root)},
            )

        # Clean directory contents
        try:
            for item in items_to_clean:
                if item.is_dir():
                    shutil.rmtree(item)
                else:
                    item.unlink()

            context.logger.info(
                "Workspace cleaned",
                path=str(workspace_root),
                items_removed=item_count,
            )

            return ToolResult(
                success=True,
                output=f"Cleaned {item_count} items from workspace",
                data={"items_removed": item_count, "path": str(workspace_root)},
            )

        except Exception as e:
            context.logger.error(
                "Failed to clean workspace",
                path=str(workspace_root),
                error=str(e),
            )
            return ToolResult(
                success=False,
                error=f"Failed to clean workspace: {e}",
                error_code=E_VALIDATION,
            )

    def get_definition(self) -> ToolDefinition:
        """Get tool definition.

        Returns:
            ToolDefinition for clean_workspace
        """
        return ToolDefinition(
            name="clean_workspace",
            description=(
                "DANGEROUS: Reset the current workspace (PERMANENT DELETION!).\n\n"
                "This tool removes every file and subdirectory inside the current model's workspace\n"
                "while preserving the workspace folder itself. Use it to recover from failed scaffolding\n"
                "attempts or to start fresh before re-running project generators.\n\n"
                "CRITICAL:\n"
                "- Files are deleted PERMANENTLY (not recoverable)\n"
                "- Hidden files (.git, .env, etc.) are also deleted\n"
                "- Use only when you are certain you want a clean slate\n\n"
                "Example workflow:\n"
                "1. Failed create-next-app left .npm/.tmp/Library in workspace\n"
                "2. clean_directory()\n"
                "3. bash('npx --yes create-next-app@latest . --typescript --tailwind --eslint')\n\n"
                "Returns: Number of items removed and confirmation message"
            ),
            parameters={
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        )
