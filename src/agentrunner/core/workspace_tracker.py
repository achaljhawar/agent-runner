"""WorkspaceTracker for tracking files in memory."""

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass
class FileInfo:
    """Information about a file in the workspace."""

    path: str
    size: int
    modified_at: datetime
    operation: str  # "created", "modified", "deleted"


class WorkspaceTracker:
    """Tracks files in workspace for LLM context"""

    def __init__(self, workspace_root: str):
        """Initialize tracker with workspace root directory."""
        self.workspace_root = Path(workspace_root)
        self.files: dict[str, FileInfo] = {}

    def track_file(self, file_path: str, operation: str, size: int | None = None) -> None:
        """Track a file operation"""
        abs_path = (self.workspace_root / file_path).resolve()

        try:
            rel_path = abs_path.relative_to(self.workspace_root.resolve())
        except ValueError:
            # Path is outside workspace, use as-is but validate
            rel_path = Path(file_path)

        rel_path_str = str(rel_path)

        if operation == "deleted":
            self.files.pop(rel_path_str, None)
        else:
            # Get file size if not provided
            if size is None and abs_path.exists():
                size = abs_path.stat().st_size

            self.files[rel_path_str] = FileInfo(
                path=rel_path_str,
                size=size or 0,
                modified_at=datetime.now(UTC),
                operation=operation,
            )

    def get_file_tree(self) -> str:
        """Generate compact file tree for LLM"""
        if not self.files:
            return "No files in workspace."

        # Build tree structure
        tree: dict[str, Any] = {}
        for file_path in sorted(self.files.keys()):
            parts = Path(file_path).parts
            current = tree
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            # Leaf node (file)
            if parts:
                current[parts[-1]] = None

        # Format as tree
        lines: list[str] = []
        self._format_tree(tree, lines, prefix="")
        return "\n".join(lines)

    def _format_tree(
        self, tree: dict[str, Any], lines: list[str], prefix: str = "", is_last: bool = True
    ) -> None:
        """Recursively format tree structure"""
        items = sorted(tree.items())
        for i, (name, subtree) in enumerate(items):
            is_last_item = i == len(items) - 1
            connector = "└── " if is_last_item else "├── "
            lines.append(f"{prefix}{connector}{name}")

            if subtree is not None:  # Directory
                extension = "    " if is_last_item else "│   "
                self._format_tree(subtree, lines, prefix + extension, is_last_item)

    def get_files_list(self) -> list[str]:
        """Get flat list of file paths"""
        return sorted(self.files.keys())

    def get_modified_files(self, since: datetime) -> list[str]:
        """Get files modified since timestamp"""
        return [path for path, info in self.files.items() if info.modified_at > since]

    def get_stats(self) -> dict[str, int]:
        """Get workspace statistics"""
        # Count directories by checking if any file path starts with this directory
        directories = set()
        for file_path in self.files.keys():
            path = Path(file_path)
            # Add all parent directories
            for parent in path.parents:
                if parent != Path("."):
                    directories.add(str(parent))

        return {
            "total_files": len(self.files),
            "total_size": sum(info.size for info in self.files.values()),
            "created_count": sum(1 for info in self.files.values() if info.operation == "created"),
            "modified_count": sum(
                1 for info in self.files.values() if info.operation == "modified"
            ),
            "total_directories": len(directories),
        }

    def clear(self) -> None:
        """Clear all tracked files"""
        self.files.clear()
