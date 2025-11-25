"""Batch file creation tools.

Implements batch file operations for efficient project scaffolding.
"""

import asyncio
import contextlib
from dataclasses import dataclass
from difflib import unified_diff
from pathlib import Path

from agentrunner.core.exceptions import WorkspaceSecurityError
from agentrunner.core.tool_protocol import (
    E_PERMISSIONS,
    E_VALIDATION,
    ToolCall,
    ToolDefinition,
    ToolResult,
)
from agentrunner.tools.base import BaseTool, ToolContext


@dataclass
class FileSpec:
    """Specification for a file to create."""

    path: str
    content: str


class BatchCreateFilesTool(BaseTool):
    """Create multiple files atomically in a single operation.

    Creates all files or none (atomic with rollback). Useful for project
    scaffolding and large-scale code generation.
    """

    async def execute(self, call: ToolCall, context: ToolContext) -> ToolResult:
        """Execute batch file creation.

        Args:
            call: Tool call with files parameter
            context: Execution context with workspace and logger

        Returns:
            ToolResult with success status and created files
        """
        try:
            # Validate input
            files_data = call.arguments.get("files", [])
            if not files_data:
                return ToolResult(
                    success=False,
                    error="No files specified",
                    error_code=E_VALIDATION,
                )

            if not isinstance(files_data, list):
                return ToolResult(
                    success=False,
                    error="Files must be a list",
                    error_code=E_VALIDATION,
                )

            # Parse file specifications
            file_specs = []
            for idx, file_data in enumerate(files_data):
                if not isinstance(file_data, dict):
                    return ToolResult(
                        success=False,
                        error=f"File {idx} is not a dictionary",
                        error_code=E_VALIDATION,
                    )

                if "path" not in file_data or "content" not in file_data:
                    return ToolResult(
                        success=False,
                        error=f"File {idx} missing 'path' or 'content'",
                        error_code=E_VALIDATION,
                    )

                file_specs.append(FileSpec(path=file_data["path"], content=file_data["content"]))

            # Validate all paths first
            resolved_paths = []
            for spec in file_specs:
                try:
                    abs_path = context.workspace.resolve_path(spec.path)
                    if not context.workspace.is_path_safe(abs_path):
                        return ToolResult(
                            success=False,
                            error=f"Path outside workspace: {spec.path}",
                            error_code=E_PERMISSIONS,
                        )
                    resolved_paths.append(abs_path)
                except WorkspaceSecurityError:
                    return ToolResult(
                        success=False,
                        error=f"Path outside workspace: {spec.path}",
                        error_code=E_PERMISSIONS,
                    )
                except Exception as e:
                    return ToolResult(
                        success=False,
                        error=f"Invalid path {spec.path}: {e}",
                        error_code=E_VALIDATION,
                    )

            # Check for existing files
            existing_files = []
            for abs_path in resolved_paths:
                if Path(abs_path).exists():
                    existing_files.append(abs_path)

            if existing_files:
                relative_paths = []
                for path in existing_files:
                    try:
                        relative_paths.append(context.workspace.get_relative(path))
                    except Exception:
                        relative_paths.append(path)

                return ToolResult(
                    success=False,
                    error=f"Files already exist: {', '.join(relative_paths)}",
                    error_code=E_VALIDATION,
                )

            # Create files atomically
            context.logger.info(
                "Creating batch files",
                file_count=len(file_specs),
            )

            created_files = []
            try:
                # Create parent directories and files in parallel
                await asyncio.gather(
                    *[
                        self._create_file_async(abs_path, spec.content)
                        for abs_path, spec in zip(resolved_paths, file_specs, strict=True)
                    ]
                )

                created_files = resolved_paths

                # Publish file_created events for each file
                context.logger.info(
                    "About to publish file_created events",
                    file_count=len(created_files),
                    has_event_bus=context.event_bus is not None,
                )

                if context.event_bus:
                    from datetime import UTC, datetime

                    from agentrunner.core.events import StreamEvent

                    for abs_path, spec in zip(resolved_paths, file_specs, strict=True):
                        relative_path = context.workspace.get_relative(abs_path)
                        event = StreamEvent(
                            type="file_created",
                            data={
                                "path": relative_path,
                                "size": len(spec.content),
                                "line_count": len(spec.content.splitlines()),
                            },
                            model_id=context.model_id,  # Tag event with model ID
                            ts=datetime.now(UTC).isoformat(),
                        )

                        context.logger.info(
                            "BatchCreateFilesTool: Publishing file_created event",
                            path=relative_path,
                            event_id=event.id,
                            event_type=event.type,
                            subscriber_count=context.event_bus.subscriber_count,
                        )

                        context.event_bus.publish(event)

                # Generate combined diff
                diffs = []
                for abs_path, spec in zip(resolved_paths, file_specs, strict=True):
                    relative_path = context.workspace.get_relative(abs_path)
                    diff_lines = list(
                        unified_diff(
                            [],
                            spec.content.splitlines(keepends=True),
                            fromfile=f"a/{relative_path}",
                            tofile=f"b/{relative_path}",
                            lineterm="",
                        )
                    )
                    if diff_lines:
                        diffs.append(
                            {
                                "file": relative_path,
                                "diff": "".join(diff_lines),
                            }
                        )

                # Get relative paths for result
                relative_files = []
                for abs_path in created_files:
                    try:
                        relative_files.append(context.workspace.get_relative(abs_path))
                    except Exception:
                        relative_files.append(abs_path)

                context.logger.info(
                    "Batch files created",
                    file_count=len(created_files),
                )

                return ToolResult(
                    success=True,
                    output=f"Created {len(created_files)} files successfully",
                    files_changed=relative_files,
                    diffs=diffs,
                )

            except Exception as e:
                # Rollback: delete any files that were created
                context.logger.warn(
                    "Batch creation failed, rolling back",
                    created_count=len(created_files),
                    error=str(e),
                )

                await asyncio.gather(
                    *[
                        self._delete_file_async(path)
                        for path in created_files
                        if Path(path).exists()
                    ]
                )

                return ToolResult(
                    success=False,
                    error=f"Failed to create files (rolled back): {e}",
                    error_code=E_VALIDATION,
                )

        except Exception as e:
            context.logger.error(
                "Batch creation error",
                error=str(e),
            )
            return ToolResult(
                success=False,
                error=f"Batch creation failed: {e}",
                error_code=E_VALIDATION,
            )

    async def _create_file_async(self, path: str, content: str) -> None:
        """Create a single file asynchronously.

        Args:
            path: Absolute path to file
            content: File content

        Raises:
            OSError: If file creation fails
        """
        path_obj = Path(path)
        # Create parent directories
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        # Write file
        await asyncio.to_thread(path_obj.write_text, content, encoding="utf-8")

    async def _delete_file_async(self, path: str) -> None:
        """Delete a single file asynchronously.

        Args:
            path: Absolute path to file
        """
        with contextlib.suppress(Exception):
            await asyncio.to_thread(Path(path).unlink)

    def get_definition(self) -> ToolDefinition:
        """Get tool definition for LLM.

        Returns:
            ToolDefinition with schema and safety flags
        """
        return ToolDefinition(
            name="batch_create_files",
            description=(
                "Create multiple files atomically. "
                "All files are created or none (with rollback on failure). "
                "Useful for project scaffolding."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "files": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "path": {
                                    "type": "string",
                                    "description": "File path (relative to workspace)",
                                },
                                "content": {
                                    "type": "string",
                                    "description": "File content",
                                },
                            },
                            "required": ["path", "content"],
                        },
                        "minItems": 1,
                        "description": "List of files to create",
                    },
                },
                "required": ["files"],
            },
            safety={
                "requires_confirmation": True,
            },
        )
