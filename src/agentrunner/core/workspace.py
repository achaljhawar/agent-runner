"""Workspace sandboxing and path validation.

Implements Workspace per INTERFACES/WORKSPACE_SECURITY.md.
Provides secure path resolution and prevents path traversal attacks.
"""

import contextlib
from pathlib import Path
from typing import Any

from agentrunner.core.exceptions import E_PERMISSIONS, E_VALIDATION, WorkspaceSecurityError


class Workspace:
    """Manages workspace sandboxing and path validation.

    Ensures all file operations stay within a designated workspace root
    and prevents path traversal attacks including symlink escapes.
    """

    def __init__(self, root_path: str) -> None:
        """Initialize workspace with root path.

        Args:
            root_path: Absolute path to workspace root directory

        Raises:
            WorkspaceSecurityError: If root_path is not absolute or doesn't exist
        """
        # Convert to absolute path first, then check if original was relative
        path_obj = Path(root_path)
        if not path_obj.is_absolute():
            raise WorkspaceSecurityError(
                f"Workspace root must be absolute: {root_path}",
                path=root_path,
                reason="not_absolute",
                error_code=E_VALIDATION,
            )

        self.root_path = path_obj.resolve()

        # Create root directory if it doesn't exist
        self.root_path.mkdir(parents=True, exist_ok=True)

        # Store the real resolved path to handle symlinks in root path itself
        self._real_root = self.root_path.resolve()

    def resolve_path(self, path: str) -> str:
        """Convert relative/absolute path to absolute within workspace root.

        Args:
            path: Path to resolve (relative or absolute)

        Returns:
            Absolute path within workspace root

        Raises:
            WorkspaceSecurityError: If resolved path is outside workspace
        """
        if not path:
            raise WorkspaceSecurityError(
                "Empty path not allowed", path=path, error_code=E_VALIDATION
            )

        # Handle home directory expansion
        if path.startswith("~"):
            # Don't allow home directory expansion outside workspace
            raise WorkspaceSecurityError(
                "Home directory paths not allowed",
                path=path,
                reason="home_expansion",
                error_code=E_PERMISSIONS,
            )

        # Convert to Path object and handle relative/absolute paths
        path_obj = Path(path)

        if path_obj.is_absolute():
            # Absolute path - must be within workspace
            resolved = path_obj.resolve()
        else:
            # Relative path - resolve against workspace root
            resolved = (self.root_path / path_obj).resolve()

        # Security check: ensure resolved path is within workspace
        if not self.is_path_safe(str(resolved)):
            raise WorkspaceSecurityError(
                f"Path outside workspace: {path} -> {resolved}",
                path=path,
                reason="outside_workspace",
                error_code=E_PERMISSIONS,
            )

        return str(resolved)

    def is_path_safe(self, abs_path: str) -> bool:
        """Check if absolute path is safe (within workspace root).

        Args:
            abs_path: Absolute path to check

        Returns:
            True if path is within workspace root, False otherwise
        """
        if not abs_path:
            return False

        try:
            path_obj = Path(abs_path).resolve()

            # Check if path is under workspace root
            # Use str comparison of resolved paths to handle symlinks properly
            try:
                path_obj.relative_to(self._real_root)
                return True
            except ValueError:
                # Path is not under workspace root
                return False

        except (OSError, ValueError):
            # Invalid path or permission error
            return False

    def get_relative(self, abs_path: str) -> str:
        """Get workspace-relative path for display.

        Args:
            abs_path: Absolute path within workspace

        Returns:
            Relative path from workspace root

        Raises:
            WorkspaceSecurityError: If path is outside workspace
        """
        if not self.is_path_safe(abs_path):
            raise WorkspaceSecurityError(
                f"Path outside workspace: {abs_path}",
                path=abs_path,
                reason="outside_workspace",
                error_code=E_PERMISSIONS,
            )

        try:
            path_obj = Path(abs_path).resolve()
            return str(path_obj.relative_to(self._real_root))
        except ValueError as e:
            raise WorkspaceSecurityError(
                f"Cannot get relative path: {abs_path}",
                path=abs_path,
                reason="relative_error",
                error_code=E_VALIDATION,
            ) from e

    def list_files(self) -> list[str]:
        """List all files within workspace (recursive).

        Returns:
            List of absolute paths to all files in workspace
        """
        files = []

        try:
            for item in self.root_path.rglob("*"):
                if item.is_file():
                    files.append(str(item))
        except (OSError, PermissionError):
            # Handle permission errors gracefully
            pass

        return sorted(files)

    def info(self) -> dict[str, Any]:
        """Get workspace information.

        Returns:
            Dictionary with root, size, and file_count
        """
        file_count = 0
        total_size = 0

        try:
            for item in self.root_path.rglob("*"):
                if item.is_file():
                    file_count += 1
                    with contextlib.suppress(OSError, PermissionError):
                        total_size += item.stat().st_size
        except (OSError, PermissionError):
            # Handle permission errors gracefully
            pass

        return {"root": str(self.root_path), "size": total_size, "file_count": file_count}
