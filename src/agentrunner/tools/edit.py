"""File editing tools: string replace, multi-edit, and line insertion.

Implements EditFileTool, MultiEditTool, and InsertLinesTool.
"""

import difflib
from dataclasses import dataclass
from pathlib import Path

from agentrunner.core.tool_protocol import (
    E_NOT_FOUND,
    E_NOT_UNIQUE,
    E_PERMISSIONS,
    E_VALIDATION,
    ToolCall,
    ToolDefinition,
    ToolResult,
)
from agentrunner.tools.base import BaseTool, ToolContext


@dataclass
class EditOperation:
    """Single edit operation for MultiEdit."""

    old_string: str
    new_string: str
    replace_all: bool = False


def _generate_unified_diff(old_content: str, new_content: str, filepath: str) -> list[str]:
    """Generate unified diff between old and new content.

    Args:
        old_content: Original content
        new_content: Modified content
        filepath: Path for diff headers

    Returns:
        List of diff lines
    """
    old_lines = old_content.splitlines(keepends=True)
    new_lines = new_content.splitlines(keepends=True)

    return list(
        difflib.unified_diff(
            old_lines,
            new_lines,
            fromfile=f"a/{filepath}",
            tofile=f"b/{filepath}",
            lineterm="",
        )
    )


def _fuzzy_match(content: str, target: str) -> list[tuple[int, int]]:
    """Find fuzzy matches for target string in content.

    Uses fuzzy matching to handle whitespace variations.

    Args:
        content: Content to search in
        target: Target string to find

    Returns:
        List of (start, end) positions of matches
    """

    # Normalize whitespace for fuzzy matching
    def normalize_ws(s: str) -> str:
        """Normalize whitespace in string."""
        lines = s.splitlines()
        return "\n".join(line.rstrip() for line in lines)

    normalized_content = normalize_ws(content)
    normalized_target = normalize_ws(target)

    matches = []
    start = 0

    while True:
        pos = normalized_content.find(normalized_target, start)
        if pos == -1:
            break

        # Map back to original content
        original_start = len(content[: pos + normalized_content[:pos].count(" ")])
        original_end = original_start + len(target)

        matches.append((original_start, original_end))
        start = pos + len(normalized_target)

    return matches


def _find_sophisticated_fuzzy_match(content: str, search_str: str) -> str | None:
    """Find fuzzy match using structural understanding (function/class boundaries).

    Uses Python-specific patterns to detect function/class boundaries and match
    code blocks based on structure rather than exact text.

    Args:
        content: File content to search in
        search_str: Search string (should contain function/class definition)

    Returns:
        Matched function/class block or None if not found
    """
    import re

    # Try to find function or class definition in search string
    func_match = re.search(r"(def|class)\s+(\w+)", search_str)
    if not func_match:
        return None

    keyword, name = func_match.groups()
    content_lines = content.splitlines()

    # Find the function/class start in content
    start_idx = -1
    for i, line in enumerate(content_lines):
        # Look for "def name" or "class name" with colon
        if f"{keyword} {name}" in line and ":" in line:
            start_idx = i
            break

    if start_idx == -1:
        return None

    # Determine base indentation level
    base_indent = len(content_lines[start_idx]) - len(content_lines[start_idx].lstrip())

    # Find the end of the function/class block by tracking indentation
    end_idx = start_idx + 1
    for i in range(start_idx + 1, len(content_lines)):
        line = content_lines[i]

        # Skip empty lines and comments
        if not line.strip() or line.strip().startswith("#"):
            continue

        # Calculate current line's indentation
        line_indent = len(line) - len(line.lstrip())

        # If we hit a line at same or lower indentation, we've reached the end
        if line_indent <= base_indent:
            end_idx = i
            break
    else:
        # Reached end of file
        end_idx = len(content_lines)

    # Extract the matched block
    actual_block = "\n".join(content_lines[start_idx:end_idx])

    # Normalize both strings for comparison
    def normalize_for_comparison(s: str) -> str:
        """Normalize string for structural comparison."""
        # Replace quotes, normalize whitespace
        return re.sub(r"\s+", " ", s.replace('"', "'").replace("`", "'"))

    search_normalized = normalize_for_comparison(search_str)
    actual_normalized = normalize_for_comparison(actual_block)

    # Check if structures are similar (allow some flexibility)
    # If the search string appears to match the structure, return the actual block
    if _is_similar_structure(search_normalized, actual_normalized):
        return actual_block

    return None


def _is_similar_structure(search: str, actual: str) -> bool:
    """Check if two normalized strings have similar structure.

    Args:
        search: Normalized search string
        actual: Normalized actual string

    Returns:
        True if structures are similar enough
    """
    # Extract key tokens (function names, keywords, etc.)
    import re

    def extract_tokens(s: str) -> set[str]:
        """Extract significant tokens from string."""
        # Get identifiers and keywords
        tokens = set(re.findall(r"\b[a-zA-Z_]\w*\b", s))
        return tokens

    search_tokens = extract_tokens(search)
    actual_tokens = extract_tokens(actual)

    # Calculate similarity (Jaccard index)
    if not search_tokens:
        return False

    intersection = search_tokens & actual_tokens
    union = search_tokens | actual_tokens

    similarity = len(intersection) / len(union) if union else 0

    # Consider similar if >70% token overlap
    return similarity > 0.7


def _apply_exact_replace(
    content: str, old_string: str, new_string: str, replace_all: bool
) -> tuple[str | None, str | None]:
    """Apply exact string replacement.

    Args:
        content: File content
        old_string: String to replace
        new_string: Replacement string
        replace_all: Replace all occurrences

    Returns:
        Tuple of (result_content, error_message)
    """
    count = content.count(old_string)

    if count == 0:
        return None, E_NOT_FOUND

    if count > 1 and not replace_all:
        return None, E_NOT_UNIQUE

    if replace_all:
        result = content.replace(old_string, new_string)
    else:
        result = content.replace(old_string, new_string, 1)

    return result, None


def _apply_fuzzy_replace(
    content: str, old_string: str, new_string: str, replace_all: bool
) -> tuple[str | None, str | None]:
    """Apply fuzzy string replacement.

    First tries sophisticated fuzzy matching (for function/class blocks),
    then falls back to simple whitespace normalization.

    Args:
        content: File content
        old_string: String to replace
        new_string: Replacement string
        replace_all: Replace all occurrences

    Returns:
        Tuple of (result_content, error_message)
    """
    # First, try sophisticated fuzzy match (function/class boundary detection)
    if "\n" in old_string:  # Only for multi-line strings
        sophisticated_match = _find_sophisticated_fuzzy_match(content, old_string)
        if sophisticated_match:
            # Replace the matched block
            result = content.replace(sophisticated_match, new_string, 1 if not replace_all else -1)
            return result, None

    # Fall back to basic fuzzy matching
    matches = _fuzzy_match(content, old_string)

    if not matches:
        return None, E_NOT_FOUND

    if len(matches) > 1 and not replace_all:
        return None, E_NOT_UNIQUE

    # Replace in reverse order to preserve positions
    result = content
    for start, end in reversed(matches if replace_all else [matches[0]]):
        result = result[:start] + new_string + result[end:]

    return result, None


class EditFileTool(BaseTool):
    """String replacement tool with exact and fuzzy matching."""

    async def execute(self, call: ToolCall, context: ToolContext) -> ToolResult:
        """Execute string replacement in file.

        Args:
            call: Tool call with file_path, old_string, new_string, replace_all
            context: Tool execution context

        Returns:
            ToolResult with diff on success
        """
        try:
            file_path = call.arguments["file_path"]
            old_string = call.arguments["old_string"]
            new_string = call.arguments["new_string"]
            replace_all = call.arguments.get("replace_all", False)
        except KeyError as e:
            return ToolResult(
                success=False,
                error=f"Missing required argument: {e}",
                error_code=E_VALIDATION,
            )

        # Validate workspace path
        try:
            abs_path = context.workspace.resolve_path(file_path)
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Invalid path: {e}",
                error_code=E_PERMISSIONS,
            )

        if not context.workspace.is_path_safe(abs_path):
            return ToolResult(
                success=False,
                error=f"Path outside workspace: {file_path}",
                error_code=E_PERMISSIONS,
            )

        # Read file
        path_obj = Path(abs_path)
        if not path_obj.exists():
            return ToolResult(
                success=False,
                error=f"File not found: {file_path}",
                error_code=E_NOT_FOUND,
            )

        try:
            old_content = path_obj.read_text()
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Failed to read file: {e}",
                error_code=E_VALIDATION,
            )

        # Try exact match first
        new_content, error = _apply_exact_replace(old_content, old_string, new_string, replace_all)

        # If exact match fails and contains newlines, try fuzzy match
        if error == E_NOT_FOUND and "\n" in old_string:
            context.logger.info("Exact match failed, trying fuzzy match")
            new_content, error = _apply_fuzzy_replace(
                old_content, old_string, new_string, replace_all
            )

        if error or new_content is None:
            error_messages = {
                E_NOT_FOUND: f"String not found in file: {file_path}",
                E_NOT_UNIQUE: f"Multiple matches found, use replace_all=true: {file_path}",
            }
            return ToolResult(
                success=False,
                error=error_messages.get(error or E_VALIDATION, f"Replace failed: {error}"),
                error_code=error or E_VALIDATION,
            )

        # Write file (new_content is guaranteed to be str here)
        try:
            path_obj.write_text(new_content)
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Failed to write file: {e}",
                error_code=E_VALIDATION,
            )

        # Publish file_modified event
        if context.event_bus:
            from datetime import UTC, datetime

            from agentrunner.core.events import StreamEvent

            context.event_bus.publish(
                StreamEvent(
                    type="file_modified",
                    data={
                        "path": file_path,
                        "old_size": len(old_content),
                        "new_size": path_obj.stat().st_size,
                        "line_count": len(new_content.splitlines()),
                    },
                    model_id=context.model_id,  # Tag event with model ID
                    ts=datetime.now(UTC).isoformat(),
                )
            )

        # Generate diff
        diff_lines = _generate_unified_diff(old_content, new_content, file_path)

        context.logger.info("File edited", file_path=file_path, replace_all=replace_all)

        return ToolResult(
            success=True,
            output=f"File edited: {file_path}",
            diffs=[{"path": file_path, "diff": "\n".join(diff_lines)}],
            files_changed=[file_path],
        )

    def get_definition(self) -> ToolDefinition:
        """Get tool definition.

        Returns:
            ToolDefinition for EditFileTool
        """
        return ToolDefinition(
            name="edit_file",
            description="Replace exact string in file with exact/fuzzy matching",
            parameters={
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Path to file"},
                    "old_string": {
                        "type": "string",
                        "description": "String to replace",
                    },
                    "new_string": {
                        "type": "string",
                        "description": "Replacement string",
                    },
                    "replace_all": {
                        "type": "boolean",
                        "default": False,
                        "description": "Replace all occurrences",
                    },
                },
                "required": ["file_path", "old_string", "new_string"],
            },
            safety={"requires_confirmation": False},
        )


class MultiEditTool(BaseTool):
    """Atomic batch edit tool."""

    async def execute(self, call: ToolCall, context: ToolContext) -> ToolResult:
        """Execute multiple edits atomically.

        Args:
            call: Tool call with file_path and edits list
            context: Tool execution context

        Returns:
            ToolResult with combined diff on success
        """
        try:
            file_path = call.arguments["file_path"]
            edits_raw = call.arguments["edits"]
        except KeyError as e:
            return ToolResult(
                success=False,
                error=f"Missing required argument: {e}",
                error_code=E_VALIDATION,
            )

        if not isinstance(edits_raw, list) or len(edits_raw) == 0:
            return ToolResult(
                success=False,
                error="Edits must be a non-empty list",
                error_code=E_VALIDATION,
            )

        # Parse edits
        try:
            edits = [
                EditOperation(
                    old_string=e["old_string"],
                    new_string=e["new_string"],
                    replace_all=e.get("replace_all", False),
                )
                for e in edits_raw
            ]
        except (KeyError, TypeError) as e:
            return ToolResult(
                success=False,
                error=f"Invalid edit format: {e}",
                error_code=E_VALIDATION,
            )

        # Validate workspace path
        try:
            abs_path = context.workspace.resolve_path(file_path)
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Invalid path: {e}",
                error_code=E_PERMISSIONS,
            )

        if not context.workspace.is_path_safe(abs_path):
            return ToolResult(
                success=False,
                error=f"Path outside workspace: {file_path}",
                error_code=E_PERMISSIONS,
            )

        # Read file
        path_obj = Path(abs_path)
        if not path_obj.exists():
            return ToolResult(
                success=False,
                error=f"File not found: {file_path}",
                error_code=E_NOT_FOUND,
            )

        try:
            original_content = path_obj.read_text()
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Failed to read file: {e}",
                error_code=E_VALIDATION,
            )

        # Apply edits sequentially
        current_content = original_content
        for i, edit in enumerate(edits):
            # Try exact match first
            new_content, error = _apply_exact_replace(
                current_content, edit.old_string, edit.new_string, edit.replace_all
            )

            # Try fuzzy match if exact fails and contains newlines
            if error == E_NOT_FOUND and "\n" in edit.old_string:
                new_content, error = _apply_fuzzy_replace(
                    current_content, edit.old_string, edit.new_string, edit.replace_all
                )

            if error or new_content is None:
                error_messages = {
                    E_NOT_FOUND: f"Edit {i+1}: String not found",
                    E_NOT_UNIQUE: f"Edit {i+1}: Multiple matches, use replace_all=true",
                }
                return ToolResult(
                    success=False,
                    error=error_messages.get(error or E_VALIDATION, f"Edit {i+1} failed: {error}"),
                    error_code=error or E_VALIDATION,
                )

            current_content = new_content  # new_content is guaranteed to be str here

        # Write file
        try:
            path_obj.write_text(current_content)
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Failed to write file: {e}",
                error_code=E_VALIDATION,
            )

        # Publish file_modified event
        if context.event_bus:
            from datetime import UTC, datetime

            from agentrunner.core.events import StreamEvent

            context.event_bus.publish(
                StreamEvent(
                    type="file_modified",
                    data={
                        "path": file_path,
                        "old_size": len(original_content),
                        "new_size": path_obj.stat().st_size,
                        "line_count": len(current_content.splitlines()),
                    },
                    model_id=context.model_id,
                    ts=datetime.now(UTC).isoformat(),
                )
            )

        # Generate diff
        diff_lines = _generate_unified_diff(original_content, current_content, file_path)

        context.logger.info("Multi-edit completed", file_path=file_path, edit_count=len(edits))

        return ToolResult(
            success=True,
            output=f"Applied {len(edits)} edits to {file_path}",
            diffs=[{"path": file_path, "diff": "\n".join(diff_lines)}],
            files_changed=[file_path],
        )

    def get_definition(self) -> ToolDefinition:
        """Get tool definition.

        Returns:
            ToolDefinition for MultiEditTool
        """
        return ToolDefinition(
            name="multi_edit",
            description="Apply multiple string replacements atomically",
            parameters={
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Path to file"},
                    "edits": {
                        "type": "array",
                        "description": "List of edit operations",
                        "items": {
                            "type": "object",
                            "properties": {
                                "old_string": {
                                    "type": "string",
                                    "description": "String to replace",
                                },
                                "new_string": {
                                    "type": "string",
                                    "description": "Replacement string",
                                },
                                "replace_all": {
                                    "type": "boolean",
                                    "default": False,
                                    "description": "Replace all occurrences",
                                },
                            },
                            "required": ["old_string", "new_string"],
                        },
                        "minItems": 1,
                    },
                },
                "required": ["file_path", "edits"],
            },
            safety={"requires_confirmation": False},
        )


class InsertLinesTool(BaseTool):
    """Insert content at specific line number."""

    async def execute(self, call: ToolCall, context: ToolContext) -> ToolResult:
        """Insert content at line number.

        Args:
            call: Tool call with file_path, line, content
            context: Tool execution context

        Returns:
            ToolResult with diff on success
        """
        try:
            file_path = call.arguments["file_path"]
            line = call.arguments["line"]
            content = call.arguments["content"]
        except KeyError as e:
            return ToolResult(
                success=False,
                error=f"Missing required argument: {e}",
                error_code=E_VALIDATION,
            )

        if not isinstance(line, int) or line < 1:
            return ToolResult(
                success=False,
                error="Line must be a positive integer",
                error_code=E_VALIDATION,
            )

        # Validate workspace path
        try:
            abs_path = context.workspace.resolve_path(file_path)
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Invalid path: {e}",
                error_code=E_PERMISSIONS,
            )

        if not context.workspace.is_path_safe(abs_path):
            return ToolResult(
                success=False,
                error=f"Path outside workspace: {file_path}",
                error_code=E_PERMISSIONS,
            )

        # Read file
        path_obj = Path(abs_path)
        if not path_obj.exists():
            return ToolResult(
                success=False,
                error=f"File not found: {file_path}",
                error_code=E_NOT_FOUND,
            )

        try:
            old_content = path_obj.read_text()
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Failed to read file: {e}",
                error_code=E_VALIDATION,
            )

        # Split into lines
        lines = old_content.splitlines(keepends=True)

        # Validate line number
        if line > len(lines) + 1:
            return ToolResult(
                success=False,
                error=f"Line {line} exceeds file length {len(lines)}",
                error_code=E_VALIDATION,
            )

        # Ensure content ends with newline if not already
        insert_content = content
        if insert_content and not insert_content.endswith("\n"):
            insert_content += "\n"

        # Insert at line (1-based)
        insert_index = line - 1
        if insert_index >= len(lines):
            # Append to end
            new_lines = [*lines, insert_content]
        else:
            # Insert before line
            new_lines = [*lines[:insert_index], insert_content, *lines[insert_index:]]

        new_content = "".join(new_lines)

        # Write file
        try:
            path_obj.write_text(new_content)
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Failed to write file: {e}",
                error_code=E_VALIDATION,
            )

        # Publish file_modified event
        if context.event_bus:
            from datetime import UTC, datetime

            from agentrunner.core.events import StreamEvent

            context.event_bus.publish(
                StreamEvent(
                    type="file_modified",
                    data={
                        "path": file_path,
                        "old_size": len(old_content),
                        "new_size": path_obj.stat().st_size,
                        "line_count": len(new_content.splitlines()),
                    },
                    model_id=context.model_id,  # Tag event with model ID
                    ts=datetime.now(UTC).isoformat(),
                )
            )

        # Generate diff
        diff_lines = _generate_unified_diff(old_content, new_content, file_path)

        context.logger.info("Lines inserted", file_path=file_path, line=line)

        return ToolResult(
            success=True,
            output=f"Inserted content at line {line} in {file_path}",
            diffs=[{"path": file_path, "diff": "\n".join(diff_lines)}],
            files_changed=[file_path],
        )

    def get_definition(self) -> ToolDefinition:
        """Get tool definition.

        Returns:
            ToolDefinition for InsertLinesTool
        """
        return ToolDefinition(
            name="insert_lines",
            description="Insert content at specific line number",
            parameters={
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Path to file"},
                    "line": {
                        "type": "integer",
                        "minimum": 1,
                        "description": "Line number (1-based) to insert at",
                    },
                    "content": {
                        "type": "string",
                        "description": "Content to insert",
                    },
                },
                "required": ["file_path", "line", "content"],
            },
            safety={"requires_confirmation": False},
        )
