"""Grep search tool using ripgrep."""

import enum
import os
import re
import shlex

from agentrunner.core.exceptions import WorkspaceSecurityError
from agentrunner.core.executors import SubprocessExecutor
from agentrunner.core.tool_protocol import (
    E_PERMISSIONS,
    E_VALIDATION,
    ToolCall,
    ToolDefinition,
    ToolResult,
)
from agentrunner.tools.base import BaseTool, ToolContext

FILE_COUNT_LIMIT = 10_000
CONTENT_LINE_LIMIT = 2_000
MAX_CHARS = 10_000
MAX_CHARS_PER_LINE = 1_000


class OutputMode(str, enum.Enum):
    """Output mode for grep results."""

    CONTENT = "content"
    FILES_WITH_MATCHES = "files_with_matches"
    COUNT = "count"


def _count_matches(output_lines: list[str]) -> int:
    """Count matching lines in ripgrep output.

    Args:
        output_lines: Lines from ripgrep output

    Returns:
        Number of matching lines
    """
    match_pattern = re.compile(r"^\d+:")
    return sum(bool(match_pattern.match(line)) for line in output_lines)


def _first_idx_exceed_cum_limit(lines: list[str], limit: int) -> int:
    """Find first index where cumulative length exceeds limit.

    Args:
        lines: Lines to check
        limit: Character limit

    Returns:
        First index exceeding limit, or len(lines) if none
    """
    cum_len = 0
    for i, line in enumerate(lines):
        cum_len += len(line)
        if cum_len > limit:
            return i
    return len(lines)


def _trim_line(line: str) -> str:
    """Trim line to max length.

    Args:
        line: Line to trim

    Returns:
        Trimmed line with indicator if truncated
    """
    if len(line) > MAX_CHARS_PER_LINE:
        return line[:MAX_CHARS_PER_LINE] + " [... omitted end of long line]"
    return line


class GrepSearchTool(BaseTool):
    """Search for patterns using ripgrep."""

    def __init__(self) -> None:
        """Initialize grep search tool."""
        self._executor: SubprocessExecutor | None = None

    def _get_executor(self, context: ToolContext) -> SubprocessExecutor:
        """Get or create subprocess executor with UID/GID isolation.

        Args:
            context: Tool execution context

        Returns:
            Configured SubprocessExecutor
        """
        if self._executor is None:
            uid = getattr(context, "session_uid", None)
            gid = getattr(context, "session_gid", None)
            self._executor = SubprocessExecutor(
                uid=uid,
                gid=gid,
                max_output_lines=10000,  # High limit for grep output
                max_output_bytes=10 * 1024 * 1024,  # 10MB for large search results
            )
        return self._executor

    async def execute(self, call: ToolCall, context: ToolContext) -> ToolResult:
        """Execute grep search.

        Args:
            call: Tool call with pattern and search options
            context: Tool execution context

        Returns:
            ToolResult with search results or error
        """
        pattern = call.arguments.get("pattern")
        path = call.arguments.get("path", "")
        glob_pattern = call.arguments.get("glob", "")
        output_mode_str = call.arguments.get("output_mode", "content")
        before_context = call.arguments.get("-B", 0)
        after_context = call.arguments.get("-A", 0)
        context_lines = call.arguments.get("-C", 0)
        case_insensitive = call.arguments.get("-i", False)
        file_type = call.arguments.get("type", "")
        head_limit = call.arguments.get("head_limit", 10_000)
        multiline = call.arguments.get("multiline", False)

        if not pattern:
            return ToolResult(
                success=False,
                error="pattern is required",
                error_code=E_VALIDATION,
            )

        try:
            output_mode = OutputMode(output_mode_str)
        except ValueError:
            return ToolResult(
                success=False,
                error=f"Invalid output_mode: {output_mode_str}",
                error_code=E_VALIDATION,
            )

        # Resolve search path
        search_path = path if path else "."
        try:
            abs_path = context.workspace.resolve_path(search_path)
            if not context.workspace.is_path_safe(abs_path):
                return ToolResult(
                    success=False,
                    error=f"Path outside workspace: {search_path}",
                    error_code=E_PERMISSIONS,
                )
        except WorkspaceSecurityError as e:
            return ToolResult(
                success=False,
                error=str(e),
                error_code=E_PERMISSIONS,
            )
        except (ValueError, OSError) as e:
            return ToolResult(
                success=False,
                error=f"Invalid path: {e}",
                error_code=E_PERMISSIONS,
            )

        # Build ripgrep command - use full path for reliability
        import shutil

        rg_path = shutil.which("rg") or "rg"
        command_parts = [rg_path, "--heading", "--with-filename", "--line-number", "--color=never"]

        if case_insensitive:
            command_parts.append("--ignore-case")

        if glob_pattern:
            command_parts.extend(["--glob", glob_pattern])

        if multiline:
            command_parts.extend(["-U", "--multiline-dotall"])

        if context_lines:
            command_parts.extend(["-C", str(context_lines)])
        if before_context:
            command_parts.extend(["-B", str(before_context)])
        if after_context:
            command_parts.extend(["-A", str(after_context)])

        # Adjust head_limit based on output mode
        if output_mode == OutputMode.CONTENT:
            head_limit = min(head_limit, CONTENT_LINE_LIMIT)
        elif output_mode == OutputMode.FILES_WITH_MATCHES:
            head_limit = min(head_limit, FILE_COUNT_LIMIT)
            command_parts.append("-l")
        elif output_mode == OutputMode.COUNT:
            head_limit = min(head_limit, FILE_COUNT_LIMIT)
            command_parts.append("-c")

        if file_type:
            types = file_type.split(",")
            for t in types:
                command_parts.extend(["--type", t.strip()])

        command_parts.extend(["-e", pattern, "--", str(abs_path)])

        # Execute ripgrep using secure executor
        executor = self._get_executor(context)
        command = " ".join(shlex.quote(arg) for arg in command_parts)

        # Preserve PATH so ripgrep can be found
        env = {"PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin")}

        try:
            exec_result = await executor.execute_command(
                command=command,
                cwd=str(context.workspace.root_path),
                env=env,
                timeout=30,
            )

            output = exec_result.stdout
            exit_code = exec_result.exit_code

            # Handle ripgrep exit codes
            if exit_code == 1 and not output:
                # No matches found
                workspace_path = str(context.workspace.root_path)
                return ToolResult(
                    success=True,
                    output=f'<workspace_result workspace_path="{workspace_path}">\nNo matches found\n</workspace_result>',
                )

            # Exit code 2 indicates error in regex or pattern
            rg_error_exit_code = 2
            if exit_code == rg_error_exit_code:
                # Error in regex or pattern
                error_msg = exec_result.stderr or exec_result.stdout
                return ToolResult(
                    success=False,
                    error=f"Error calling tool: {error_msg} (exit 2, root: {context.workspace.root_path})",
                    error_code=E_VALIDATION,
                )

            if exit_code != 0:
                return ToolResult(
                    success=False,
                    error=f"Unknown error (exit {exit_code}, root: {context.workspace.root_path})",
                    error_code=E_VALIDATION,
                )

            # Format output and collect metrics
            formatted_output, metrics = self._format_output_with_metrics(
                output, output_mode, head_limit, search_path, context
            )

            context.logger.info(
                "Grep search completed",
                pattern=pattern[:50],
                path=search_path,
                mode=output_mode.value,
                matches=metrics.get("match_count", 0),
                files=metrics.get("file_count", 0),
            )

            return ToolResult(
                success=True,
                output=formatted_output,
                data=metrics,
            )

        except TimeoutError:
            context.logger.error("Grep search timed out", pattern=pattern[:50], path=search_path)
            return ToolResult(
                success=False,
                error="Command timed out",
                error_code=E_VALIDATION,
            )
        except OSError as e:
            context.logger.error("Ripgrep execution failed", pattern=pattern[:50], error=str(e))
            return ToolResult(
                success=False,
                error=f"ripgrep (rg) execution failed: {e}. Please ensure ripgrep is installed.",
                error_code=E_VALIDATION,
            )

    def _format_output_with_metrics(
        self,
        output: str,
        output_mode: OutputMode,
        head_limit: int,
        search_path: str,
        context: ToolContext,
    ) -> tuple[str, dict[str, int]]:
        """Format ripgrep output and collect metrics.

        Args:
            output: Raw ripgrep output
            output_mode: Output mode
            head_limit: Maximum lines to show
            search_path: Search path used
            context: Tool context

        Returns:
            Tuple of (formatted output string, metrics dictionary)
        """
        if not output.strip():
            workspace_path = str(context.workspace.root_path)
            return (
                f'<workspace_result workspace_path="{workspace_path}">\nNo matches found\n</workspace_result>',
                {"match_count": 0, "file_count": 0},
            )

        output_lines = output.splitlines()

        # Remove ./ prefix if path was not specified
        if not search_path:
            output_lines = [line[2:] if line.startswith("./") else line for line in output_lines]

        is_truncated = len(output_lines) > head_limit
        output_lines = output_lines[:head_limit]

        workspace_path = str(context.workspace.root_path)

        # Calculate metrics based on output mode
        if output_mode == OutputMode.CONTENT:
            formatted = self._format_content_output(output_lines, is_truncated)
            match_count = _count_matches(output_lines)
            file_count = len(
                {
                    line.split(":")[0]
                    for line in output_lines
                    if ":" in line and not line.startswith("-")
                }
            )
        elif output_mode == OutputMode.FILES_WITH_MATCHES:
            formatted = self._format_files_output(output_lines, is_truncated)
            match_count = len(output_lines)
            file_count = len(output_lines)
        elif output_mode == OutputMode.COUNT:
            formatted = self._format_count_output(output_lines, is_truncated)
            file_count = len(output_lines)
            match_count = sum(int(line.split(":")[-1]) for line in output_lines if ":" in line)
        else:
            formatted = "\n".join(output_lines)
            match_count = 0
            file_count = 0

        metrics = {"match_count": match_count, "file_count": file_count}
        formatted_output = f'<workspace_result workspace_path="{workspace_path}">\n{formatted}\n</workspace_result>'

        return formatted_output, metrics

    def _format_content_output(self, lines: list[str], is_truncated: bool) -> str:
        """Format content output mode.

        Args:
            lines: Output lines
            is_truncated: Whether output was truncated

        Returns:
            Formatted string
        """
        is_truncated_str = "at least " if is_truncated else ""
        num_matches = _count_matches(lines)

        trimmed_lines = [_trim_line(line) for line in lines]
        cut_idx = _first_idx_exceed_cum_limit(trimmed_lines, MAX_CHARS)

        result_lines = [f"Found {is_truncated_str}{num_matches} matching lines"]
        result_lines.extend(trimmed_lines[:cut_idx])

        if cut_idx < len(trimmed_lines):
            remaining = _count_matches(trimmed_lines[cut_idx:])
            result_lines.append(f"... [{is_truncated_str}{remaining} lines truncated] ...")

        return "\n".join(result_lines)

    def _format_files_output(self, lines: list[str], is_truncated: bool) -> str:
        """Format files_with_matches output mode.

        Args:
            lines: Output lines
            is_truncated: Whether output was truncated

        Returns:
            Formatted string
        """
        is_truncated_str = "at least " if is_truncated else ""
        trimmed_lines = [_trim_line(line) for line in lines]
        cut_idx = _first_idx_exceed_cum_limit(trimmed_lines, MAX_CHARS)

        result_lines = [f"Found {is_truncated_str}{len(lines)} files"]
        result_lines.extend(trimmed_lines[:cut_idx])

        if len(trimmed_lines) > cut_idx:
            result_lines.append(
                f"... [{is_truncated_str}{len(trimmed_lines) - cut_idx} lines truncated] ..."
            )

        return "\n".join(result_lines)

    def _format_count_output(self, lines: list[str], is_truncated: bool) -> str:
        """Format count output mode.

        Args:
            lines: Output lines
            is_truncated: Whether output was truncated

        Returns:
            Formatted string
        """
        is_truncated_str = "at least " if is_truncated else ""
        sum_matches = 0

        for line in lines:
            try:
                count_str = line.split(":")[-1]
                sum_matches += int(count_str)
            except (ValueError, IndexError):
                pass

        trimmed_lines = [_trim_line(line) for line in lines]
        cut_idx = _first_idx_exceed_cum_limit(trimmed_lines, MAX_CHARS)

        result_lines = [f"Found {sum_matches} across {is_truncated_str}{len(lines)} files"]
        result_lines.extend(trimmed_lines[:cut_idx])

        if len(trimmed_lines) > cut_idx:
            result_lines.append(
                f"... [{is_truncated_str}{len(trimmed_lines) - cut_idx} lines truncated] ..."
            )

        return "\n".join(result_lines)

    def get_definition(self) -> ToolDefinition:
        """Get tool definition.

        Returns:
            ToolDefinition for grep_search
        """
        return ToolDefinition(
            name="grep",
            description=(
                "A powerful search tool built on ripgrep\n\n"
                "Usage:\n"
                "- Prefer grep_search for exact symbol/string searches. Whenever possible, use this instead of terminal grep/rg. This tool is faster and respects .gitignore.\n"
                '- Supports full regex syntax, e.g. "log.*Error", "function\\s+\\w+". Ensure you escape special chars to get exact matches, e.g. "functionCall\\("\n'
                "- Avoid overly broad glob patterns (e.g., '--glob *') as they bypass .gitignore rules and may be slow\n"
                "- Only use 'type' (or 'glob' for file types) when certain of the file type needed. Note: import paths may not match source file types (.js vs .ts)\n"
                '- Output modes: "content" shows matching lines (default), "files_with_matches" shows only file paths, "count" shows match counts per file\n'
                "- Pattern syntax: Uses ripgrep (not grep) - literal braces need escaping (e.g. use interface\\{\\} to find interface{} in Go code)\n"
                "- Multiline matching: By default patterns match within single lines only. For cross-line patterns like struct \\{[\\s\\S]*?field, use multiline: true.\n"
                '- Results are capped for responsiveness; truncated results show "at least" counts.\n'
                "- Content output follows ripgrep format: '-' for context lines, ':' for match lines, and all lines grouped by file.\n"
                '- Unsaved or out of workspace active editors are also searched and show "(unsaved)" or "(out of workspace)". Use absolute paths to read/edit these files.'
            ),
            parameters={
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "The regular expression pattern to search for in file contents (rg --regexp)",
                    },
                    "path": {
                        "type": "string",
                        "description": "File or directory to search in (rg pattern -- PATH). Defaults to workspace root.",
                        "default": "",
                    },
                    "glob": {
                        "type": "string",
                        "description": 'Glob pattern (rg --glob GLOB -- PATH) to filter files (e.g. "*.js", "*.{ts,tsx}").',
                        "default": "",
                    },
                    "output_mode": {
                        "type": "string",
                        "enum": ["content", "files_with_matches", "count"],
                        "description": 'Output mode: "content" shows matching lines (supports -A/-B/-C context, -n line numbers, head_limit), "files_with_matches" shows only file paths (supports head_limit), "count" shows match counts (supports head_limit). Defaults to "content".',
                        "default": "content",
                    },
                    "-B": {
                        "type": "integer",
                        "description": 'Number of lines to show before each match (rg -B). Requires output_mode: "content", ignored otherwise.',
                        "default": 0,
                    },
                    "-A": {
                        "type": "integer",
                        "description": 'Number of lines to show after each match (rg -A). Requires output_mode: "content", ignored otherwise.',
                        "default": 0,
                    },
                    "-C": {
                        "type": "integer",
                        "description": 'Number of lines to show before and after each match (rg -C). Requires output_mode: "content", ignored otherwise.',
                        "default": 0,
                    },
                    "-i": {
                        "type": "boolean",
                        "description": "Case insensitive search (rg -i) Defaults to false",
                        "default": False,
                    },
                    "type": {
                        "type": "string",
                        "description": "File type to search (rg --type). Common types: js, py, rust, go, java, etc. More efficient than glob for standard file types.",
                        "default": "",
                    },
                    "head_limit": {
                        "type": "integer",
                        "description": 'Limit output to first N lines/entries, equivalent to "| head -N". Works across all output modes: content (limits output lines), files_with_matches (limits file paths), count (limits count entries). When unspecified, shows all ripgrep results.',
                        "default": 10000,
                    },
                    "multiline": {
                        "type": "boolean",
                        "description": "Enable multiline mode where . matches newlines and patterns can span lines (rg -U --multiline-dotall). Default: false.",
                        "default": False,
                    },
                },
                "required": ["pattern"],
            },
        )
