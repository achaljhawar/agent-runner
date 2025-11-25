"""Read file tool with line numbering and offset/limit support."""

from pathlib import Path

from agentrunner.core.tool_protocol import (
    E_NOT_FOUND,
    E_PERMISSIONS,
    E_VALIDATION,
    ToolCall,
    ToolDefinition,
    ToolResult,
)
from agentrunner.tools.base import BaseTool, ToolContext

MAX_NUM_TOKENS = 25_000
MAX_LINES_READ = 1_001


def _add_line_numbers(lines: list[str], start_line: int) -> list[str]:
    """Add line numbers to lines.

    Args:
        lines: Lines to number
        start_line: Starting line number (1-based)

    Returns:
        List of formatted lines with line numbers
    """
    return [f"{i:>6}|{line}" for i, line in enumerate(lines, start=start_line)]


def _create_lines_not_shown_str(num_lines: int) -> str:
    """Create message for lines not shown.

    Args:
        num_lines: Number of lines not shown

    Returns:
        Message string or empty if no lines hidden
    """
    if num_lines <= 0:
        return ""
    if num_lines == 1:
        return "... [1 line not shown] ..."
    return f"... [{num_lines} lines not shown] ..."


class ReadFileTool(BaseTool):
    """Read file contents with line numbering and optional offset/limit."""

    async def execute(self, call: ToolCall, context: ToolContext) -> ToolResult:
        """Execute read file operation.

        Args:
            call: Tool call with target_file, optional offset, optional limit
            context: Tool execution context

        Returns:
            ToolResult with line-numbered content or error
        """
        target_file = call.arguments.get("target_file")
        offset = call.arguments.get("offset", 1)
        limit = call.arguments.get("limit", 1_000_000)

        if not target_file:
            return ToolResult(
                success=False,
                error="target_file is required",
                error_code=E_VALIDATION,
            )

        if not isinstance(offset, int) or offset < 1:
            return ToolResult(
                success=False,
                error="offset must be >= 1",
                error_code=E_VALIDATION,
            )

        if not isinstance(limit, int) or limit < 1:
            return ToolResult(
                success=False,
                error="limit must be >= 1",
                error_code=E_VALIDATION,
            )

        # Validate workspace boundaries
        try:
            abs_path = context.workspace.resolve_path(target_file)
        except (ValueError, OSError) as e:
            return ToolResult(
                success=False,
                error=f"Invalid path: {e}",
                error_code=E_PERMISSIONS,
            )
        except Exception as e:
            # Catch WorkspaceSecurityError and other workspace-related errors
            return ToolResult(
                success=False,
                error=f"Invalid path: {e}",
                error_code=E_PERMISSIONS,
            )

        if not context.workspace.is_path_safe(abs_path):
            return ToolResult(
                success=False,
                error=f"Path outside workspace: {target_file}",
                error_code=E_PERMISSIONS,
            )

        # Check if file exists
        path_obj = Path(abs_path)
        if not path_obj.exists():
            relative_path = context.workspace.get_relative(abs_path)
            return ToolResult(
                success=False,
                error=f"Could not find file '{relative_path}' in the workspace.",
                error_code=E_NOT_FOUND,
            )

        if not path_obj.is_file():
            relative_path = context.workspace.get_relative(abs_path)
            return ToolResult(
                success=False,
                error=f"Could not find file '{relative_path}' in the workspace.",
                error_code=E_NOT_FOUND,
            )

        # Read file content
        try:
            content_bytes = path_obj.read_bytes()
            # Check for binary file (null bytes)
            if b"\x00" in content_bytes:
                file_size = len(content_bytes)
                relative_path = context.workspace.get_relative(abs_path)
                context.logger.info("Binary file detected", path=relative_path, size=file_size)
                return ToolResult(
                    success=True,
                    output=f"Binary file: {relative_path} ({file_size} bytes)",
                    data={"binary": True, "size": file_size},
                )

            text = content_bytes.decode("utf-8")
        except PermissionError:
            relative_path = context.workspace.get_relative(abs_path)
            return ToolResult(
                success=False,
                error=f"Permission denied: {relative_path}",
                error_code=E_PERMISSIONS,
            )
        except (OSError, UnicodeDecodeError) as e:
            return ToolResult(
                success=False,
                error=f"Failed to read file: {e}",
                error_code=E_VALIDATION,
            )

        # Handle empty file
        if not text.strip():
            return ToolResult(
                success=True,
                output="File is empty.",
                data={"total_lines": 0, "offset": offset, "limit": limit, "lines_shown": 0},
            )

        # Split into lines preserving trailing newlines
        lines = text.splitlines(keepends=True)
        if not lines:
            lines = [""]

        total_lines = len(lines)

        # Validate offset
        if offset > total_lines:
            return ToolResult(
                success=False,
                error=f"Warning: the file exists but is shorter than the provided offset ({offset}). The file has {total_lines} lines.",
                error_code=E_VALIDATION,
            )

        # Calculate actual range
        start_line = offset
        actual_limit = min(limit, MAX_LINES_READ)
        end_line = min(start_line + actual_limit - 1, total_lines)

        # Get selected lines
        selected_lines = lines[start_line - 1 : end_line]

        # Validate token limit
        selected_text = "".join(selected_lines)
        num_tokens = len(selected_text) // 4
        if num_tokens > MAX_NUM_TOKENS:
            return ToolResult(
                success=False,
                error=(
                    f"File content ({num_tokens} tokens) exceeds maximum allowed tokens "
                    f"({MAX_NUM_TOKENS} tokens).\nPlease use offset and limit parameters "
                    "to read shorter range, or use the 'grep_search' to search for specific content."
                ),
                error_code=E_VALIDATION,
            )

        # Format output with line numbers
        formatted_lines = _add_line_numbers(selected_lines, start_line)
        content = "\n".join(formatted_lines)

        # Add headers/footers for hidden lines
        if start_line > 1:
            header = _create_lines_not_shown_str(start_line - 1)
            content = f"{header}\n{content}"

        if end_line < total_lines:
            footer = _create_lines_not_shown_str(total_lines - end_line)
            content = f"{content}\n{footer}"

        context.logger.info(
            "File read",
            path=target_file,
            total_lines=total_lines,
            lines_shown=len(selected_lines),
        )

        return ToolResult(
            success=True,
            output=content,
            data={
                "total_lines": total_lines,
                "offset": offset,
                "limit": limit,
                "lines_shown": len(selected_lines),
            },
        )

    def get_definition(self) -> ToolDefinition:
        """Get tool definition.

        Returns:
            ToolDefinition for read_file
        """
        return ToolDefinition(
            name="read_file",
            description=(
                "Reads a file from the local filesystem. You can access any file directly by using this tool.\n"
                "If the User provides a path to a file assume that path is valid. It is okay to read a file that does not exist; an error will be returned.\n\n"
                "Usage:\n"
                "- You can optionally specify a line offset and limit (especially handy for long files), but it's recommended to read the whole file by not providing these parameters.\n"
                "- Lines in the output are numbered starting at 1, using following format: LINE_NUMBER|LINE_CONTENT.\n"
                "- You have the capability to call multiple tools in a single response. It is always better to speculatively read multiple files as a batch that are potentially useful.\n"
                "- If you read a file that exists but has empty contents you will receive 'File is empty.'."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "target_file": {
                        "type": "string",
                        "description": "The path of the file to read. You can use either a relative path in the workspace or an absolute path. If an absolute path is provided, it will be preserved as is.",
                    },
                    "offset": {
                        "type": "integer",
                        "description": "The line number to start reading from. Only provide if the file is too large to read at once.",
                        "default": 1,
                        "minimum": 1,
                    },
                    "limit": {
                        "type": "integer",
                        "description": "The number of lines to read. Only provide if the file is too large to read at once.",
                        "default": 1000000,
                        "minimum": 1,
                    },
                },
                "required": ["target_file"],
            },
        )
