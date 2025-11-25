"""File I/O tools: write, delete, create.

Implements CreateFileTool, WriteFileTool, and DeleteFileTool.
Note: ReadFileTool is in read_file.py for enhanced line number support.
"""

import difflib
from pathlib import Path

from agentrunner.core.tool_protocol import ToolCall, ToolDefinition, ToolResult
from agentrunner.tools.base import BaseTool, ToolContext


class CreateFileTool(BaseTool):
    """Create new file (no read-before-write requirement for new files)."""

    async def execute(self, call: ToolCall, context: ToolContext) -> ToolResult:
        """Execute create file operation.

        Args:
            call: Tool call with file_path and content
            context: Tool execution context

        Returns:
            ToolResult with diff and files_changed or error
        """
        file_path = call.arguments.get("file_path")
        content = call.arguments.get("content")

        if not file_path:
            return ToolResult(
                success=False,
                error="file_path is required",
                error_code="E_VALIDATION",
            )

        if content is None:
            return ToolResult(
                success=False,
                error="content is required",
                error_code="E_VALIDATION",
            )

        # Validate workspace boundaries
        try:
            abs_path = context.workspace.resolve_path(file_path)
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Invalid path: {e}",
                error_code="E_PERMISSIONS",
            )

        if not context.workspace.is_path_safe(abs_path):
            return ToolResult(
                success=False,
                error=f"Path outside workspace: {file_path}",
                error_code="E_PERMISSIONS",
            )

        path_obj = Path(abs_path)

        # Create parent directories if needed
        try:
            path_obj.parent.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Failed to create parent directory: {e}",
                error_code="E_PERMISSIONS",
            )

        # Write file
        try:
            path_obj.write_text(content)
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Failed to write file: {e}",
                error_code="E_PERMISSIONS",
            )

        # Generate diff (from nothing to new content)
        diff_content = self._generate_diff("", content, file_path)

        context.logger.info("File created", path=file_path, size=len(content))

        # Publish file_created event
        context.logger.info(
            "CreateFileTool: About to publish file_created event",
            path=file_path,
            has_event_bus=context.event_bus is not None,
        )

        if context.event_bus:
            from datetime import UTC, datetime

            from agentrunner.core.events import StreamEvent

            event = StreamEvent(
                type="file_created",
                data={
                    "path": file_path,
                    "size": path_obj.stat().st_size,
                    "line_count": len(content.splitlines()),
                },
                model_id=context.model_id,  # Tag event with model ID
                ts=datetime.now(UTC).isoformat(),
            )

            context.logger.info(
                "CreateFileTool: Publishing file_created event",
                path=file_path,
                event_id=event.id,
                event_type=event.type,
                subscriber_count=context.event_bus.subscriber_count,
            )

            context.event_bus.publish(event)

        return ToolResult(
            success=True,
            output=f"Created {file_path}",
            diffs=[{"file": file_path, "format": "unified", "content": diff_content}],
            files_changed=[abs_path],
        )

    def _generate_diff(self, old_content: str, new_content: str, file_path: str) -> str:
        """Generate unified diff for file creation."""

        new_lines = new_content.split("\n")

        diff_lines = [
            "--- /dev/null",
            f"+++ {file_path}",
            f"@@ -0,0 +1,{len(new_lines)} @@",
        ]
        diff_lines.extend(f"+{line}" for line in new_lines)

        return "\n".join(diff_lines)

    def get_definition(self) -> ToolDefinition:
        """Get tool definition."""
        return ToolDefinition(
            name="create_file",
            description="Create a new file with content. Use for files that don't exist yet.",
            parameters={
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Path to new file"},
                    "content": {"type": "string", "description": "File content"},
                },
                "required": ["file_path", "content"],
            },
            safety={},  # No read-before-write for new files
        )


class WriteFileTool(BaseTool):
    """Write or overwrite existing file with diff generation."""

    async def execute(self, call: ToolCall, context: ToolContext) -> ToolResult:
        """Execute write file operation.

        Args:
            call: Tool call with file_path, content, optional overwrite
            context: Tool execution context

        Returns:
            ToolResult with diff and files_changed or error
        """
        file_path = call.arguments.get("file_path")
        content = call.arguments.get("content")
        overwrite = call.arguments.get("overwrite", True)

        if not file_path:
            return ToolResult(
                success=False,
                error="file_path is required",
                error_code="E_VALIDATION",
            )

        if content is None:
            return ToolResult(
                success=False,
                error="content is required",
                error_code="E_VALIDATION",
            )

        # Validate workspace boundaries
        try:
            abs_path = context.workspace.resolve_path(file_path)
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Invalid path: {e}",
                error_code="E_PERMISSIONS",
            )

        if not context.workspace.is_path_safe(abs_path):
            return ToolResult(
                success=False,
                error=f"Path outside workspace: {file_path}",
                error_code="E_PERMISSIONS",
            )

        path_obj = Path(abs_path)
        file_exists = path_obj.exists()

        # Check overwrite flag
        if file_exists and not overwrite:
            return ToolResult(
                success=False,
                error=f"File exists and overwrite=false: {file_path}",
                error_code="E_VALIDATION",
            )

        # Read old content for diff generation
        old_content = ""
        if file_exists:
            try:
                old_content = path_obj.read_text()
            except Exception as e:
                context.logger.warn(
                    "Failed to read existing file for diff", path=file_path, error=str(e)
                )

        # Create parent directories if needed
        try:
            path_obj.parent.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Failed to create parent directories: {e}",
                error_code="E_PERMISSIONS",
            )

        # Write the file
        try:
            path_obj.write_text(content)
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Failed to write file: {e}",
                error_code="E_PERMISSIONS",
            )

        # Publish file event
        if context.event_bus:
            from datetime import UTC, datetime

            from agentrunner.core.events import EventType, StreamEvent

            event_type: EventType = "file_created" if not file_exists else "file_modified"
            event_data = {
                "path": file_path,
                "size": path_obj.stat().st_size,
                "line_count": len(content.splitlines()),
            }

            if event_type == "file_modified":
                event_data["old_size"] = len(old_content)
                event_data["new_size"] = path_obj.stat().st_size

            context.event_bus.publish(
                StreamEvent(
                    type=event_type,
                    data=event_data,
                    model_id=context.model_id,  # Tag event with model ID
                    ts=datetime.now(UTC).isoformat(),
                )
            )

        # Generate diff if overwriting
        diffs = None
        if file_exists:
            old_lines = old_content.splitlines(keepends=True)
            new_lines = content.splitlines(keepends=True)
            diff_lines = list(
                difflib.unified_diff(
                    old_lines,
                    new_lines,
                    fromfile=f"a/{file_path}",
                    tofile=f"b/{file_path}",
                    lineterm="",
                )
            )
            if diff_lines:
                diff_text = "\n".join(diff_lines)
                diffs = [{"path": file_path, "diff": diff_text}]

        action = "updated" if file_exists else "created"
        context.logger.info(f"File {action}", path=file_path, size=len(content))

        return ToolResult(
            success=True,
            output=f"File {action}: {file_path}",
            diffs=diffs,
            files_changed=[file_path],
        )

    def get_definition(self) -> ToolDefinition:
        """Get tool definition.

        Returns:
            ToolDefinition for write_file
        """
        return ToolDefinition(
            name="write_file",
            description=("Write or create file. Generates diff when overwriting existing file."),
            parameters={
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Path to file"},
                    "content": {"type": "string", "description": "File content"},
                    "overwrite": {
                        "type": "boolean",
                        "description": "Whether to overwrite existing file",
                        "default": True,
                    },
                },
                "required": ["file_path", "content"],
            },
            safety={},
        )


class DeleteFileTool(BaseTool):
    """Delete file with confirmation requirement."""

    async def execute(self, call: ToolCall, context: ToolContext) -> ToolResult:
        """Execute delete file operation.

        Args:
            call: Tool call with file_path
            context: Tool execution context

        Returns:
            ToolResult with success or error
        """
        file_path = call.arguments.get("file_path")

        if not file_path:
            return ToolResult(
                success=False,
                error="file_path is required",
                error_code="E_VALIDATION",
            )

        # Validate workspace boundaries
        try:
            abs_path = context.workspace.resolve_path(file_path)
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Invalid path: {e}",
                error_code="E_PERMISSIONS",
            )

        if not context.workspace.is_path_safe(abs_path):
            return ToolResult(
                success=False,
                error=f"Path outside workspace: {file_path}",
                error_code="E_PERMISSIONS",
            )

        # Check if file exists
        path_obj = Path(abs_path)
        if not path_obj.exists():
            return ToolResult(
                success=False,
                error=f"File not found: {file_path}",
                error_code="E_NOT_FOUND",
            )

        if not path_obj.is_file():
            return ToolResult(
                success=False,
                error=f"Not a file: {file_path}",
                error_code="E_VALIDATION",
            )

        # Delete the file
        try:
            path_obj.unlink()
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Failed to delete file: {e}",
                error_code="E_PERMISSIONS",
            )

        context.logger.info("File deleted", path=file_path)

        return ToolResult(
            success=True,
            output=f"File deleted: {file_path}",
            files_changed=[file_path],
        )

    def get_definition(self) -> ToolDefinition:
        """Get tool definition.

        Returns:
            ToolDefinition for delete_file
        """
        return ToolDefinition(
            name="delete_file",
            description="Delete a file. Requires confirmation.",
            parameters={
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Path to file"},
                },
                "required": ["file_path"],
            },
            safety={"requires_confirmation": True},
        )
