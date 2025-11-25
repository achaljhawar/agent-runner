"""Unit tests for workspace security module.

Tests workspace sandboxing, path validation, and security features.
Uses real filesystem with temporary directories.
"""

from pathlib import Path
from unittest import mock

import pytest

from agentrunner.core.workspace import Workspace, WorkspaceSecurityError


class TestWorkspaceInit:
    """Test workspace initialization."""

    def test_init_with_absolute_path(self, tmp_path):
        """Test initialization with absolute path."""
        workspace = Workspace(str(tmp_path))
        assert workspace.root_path == tmp_path.resolve()
        assert workspace._real_root == tmp_path.resolve()

    def test_init_creates_directory(self, tmp_path):
        """Test initialization creates directory if it doesn't exist."""
        new_dir = tmp_path / "new_workspace"
        workspace = Workspace(str(new_dir))
        assert new_dir.exists()
        assert workspace.root_path == new_dir.resolve()

    def test_init_with_relative_path_raises_error(self):
        """Test initialization with relative path raises error."""
        with pytest.raises(WorkspaceSecurityError) as exc_info:
            Workspace("relative/path")
        assert "must be absolute" in str(exc_info.value)
        assert exc_info.value.reason == "not_absolute"
        assert exc_info.value.error_code == "E_VALIDATION"

    def test_init_handles_symlink_in_root(self, tmp_path):
        """Test initialization handles symlinks in root path."""
        real_dir = tmp_path / "real"
        real_dir.mkdir()
        symlink_dir = tmp_path / "symlink"
        symlink_dir.symlink_to(real_dir)

        workspace = Workspace(str(symlink_dir))
        # Should resolve to the real directory
        assert workspace._real_root == real_dir.resolve()


class TestResolvePath:
    """Test path resolution functionality."""

    def test_resolve_absolute_path_within_workspace(self, tmp_path):
        """Test resolving absolute path within workspace."""
        workspace = Workspace(str(tmp_path))
        file_path = tmp_path / "file.txt"

        resolved = workspace.resolve_path(str(file_path))
        assert resolved == str(file_path)

    def test_resolve_relative_path(self, tmp_path):
        """Test resolving relative path."""
        workspace = Workspace(str(tmp_path))

        resolved = workspace.resolve_path("file.txt")
        assert resolved == str(tmp_path / "file.txt")

    def test_resolve_nested_relative_path(self, tmp_path):
        """Test resolving nested relative path."""
        workspace = Workspace(str(tmp_path))

        resolved = workspace.resolve_path("subdir/file.txt")
        assert resolved == str(tmp_path / "subdir" / "file.txt")

    def test_resolve_path_with_dots(self, tmp_path):
        """Test resolving path with . and .. components."""
        workspace = Workspace(str(tmp_path))

        resolved = workspace.resolve_path("./subdir/../file.txt")
        assert resolved == str(tmp_path / "file.txt")

    def test_resolve_empty_path_raises_error(self, tmp_path):
        """Test resolving empty path raises error."""
        workspace = Workspace(str(tmp_path))

        with pytest.raises(WorkspaceSecurityError):
            workspace.resolve_path("")

    def test_resolve_home_path_raises_error(self, tmp_path):
        """Test resolving home directory path raises error."""
        workspace = Workspace(str(tmp_path))

        with pytest.raises(WorkspaceSecurityError) as exc_info:
            workspace.resolve_path("~/file.txt")
        assert exc_info.value.reason == "home_expansion"
        assert exc_info.value.error_code == "E_PERMISSIONS"

    def test_resolve_path_outside_workspace_raises_error(self, tmp_path):
        """Test resolving path outside workspace raises error."""
        workspace = Workspace(str(tmp_path))
        outside_path = tmp_path.parent / "outside.txt"

        with pytest.raises(WorkspaceSecurityError) as exc_info:
            workspace.resolve_path(str(outside_path))
        assert exc_info.value.reason == "outside_workspace"
        assert exc_info.value.error_code == "E_PERMISSIONS"

    def test_resolve_path_traversal_attack(self, tmp_path):
        """Test path traversal attack prevention."""
        workspace = Workspace(str(tmp_path))

        # Various path traversal attempts
        attacks = [
            "../../../etc/passwd",
            "../../outside.txt",
            "subdir/../../outside.txt",
            str(tmp_path.parent / "outside.txt"),
            "/etc/passwd",
            "/tmp/evil.txt",
        ]

        for attack in attacks:
            with pytest.raises(WorkspaceSecurityError):
                workspace.resolve_path(attack)


class TestIsPathSafe:
    """Test path safety validation."""

    def test_safe_path_within_workspace(self, tmp_path):
        """Test safe path within workspace."""
        workspace = Workspace(str(tmp_path))
        file_path = str(tmp_path / "file.txt")

        assert workspace.is_path_safe(file_path) is True

    def test_safe_nested_path(self, tmp_path):
        """Test safe nested path."""
        workspace = Workspace(str(tmp_path))
        file_path = str(tmp_path / "deep" / "nested" / "file.txt")

        assert workspace.is_path_safe(file_path) is True

    def test_unsafe_path_outside_workspace(self, tmp_path):
        """Test unsafe path outside workspace."""
        workspace = Workspace(str(tmp_path))
        outside_path = str(tmp_path.parent / "outside.txt")

        assert workspace.is_path_safe(outside_path) is False

    def test_unsafe_empty_path(self, tmp_path):
        """Test unsafe empty path."""
        workspace = Workspace(str(tmp_path))

        assert workspace.is_path_safe("") is False

    def test_unsafe_root_path(self, tmp_path):
        """Test unsafe root path."""
        workspace = Workspace(str(tmp_path))

        assert workspace.is_path_safe("/") is False

    def test_unsafe_system_paths(self, tmp_path):
        """Test unsafe system paths."""
        workspace = Workspace(str(tmp_path))

        system_paths = ["/etc/passwd", "/tmp/evil.txt", "/usr/bin/malware", "/home/user/.bashrc"]

        for path in system_paths:
            assert workspace.is_path_safe(path) is False

    def test_symlink_escape_blocked(self, tmp_path):
        """Test symlink escape attempts are blocked."""
        workspace = Workspace(str(tmp_path))

        # Create a symlink pointing outside workspace
        outside_dir = tmp_path.parent / "outside"
        outside_dir.mkdir(exist_ok=True)
        outside_file = outside_dir / "secret.txt"
        outside_file.write_text("secret data")

        symlink_path = tmp_path / "escape_link"
        symlink_path.symlink_to(outside_file)

        # The symlink itself should not be considered safe
        # because it resolves outside the workspace
        resolved_link = symlink_path.resolve()
        assert workspace.is_path_safe(str(resolved_link)) is False


class TestGetRelative:
    """Test relative path conversion."""

    def test_get_relative_for_workspace_file(self, tmp_path):
        """Test getting relative path for file in workspace."""
        workspace = Workspace(str(tmp_path))
        file_path = str(tmp_path / "file.txt")

        relative = workspace.get_relative(file_path)
        assert relative == "file.txt"

    def test_get_relative_for_nested_file(self, tmp_path):
        """Test getting relative path for nested file."""
        workspace = Workspace(str(tmp_path))
        file_path = str(tmp_path / "subdir" / "file.txt")

        relative = workspace.get_relative(file_path)
        assert relative == "subdir/file.txt"

    def test_get_relative_for_outside_path_raises_error(self, tmp_path):
        """Test getting relative path for outside file raises error."""
        workspace = Workspace(str(tmp_path))
        outside_path = str(tmp_path.parent / "outside.txt")

        with pytest.raises(WorkspaceSecurityError) as exc_info:
            workspace.get_relative(outside_path)
        assert exc_info.value.reason == "outside_workspace"
        assert exc_info.value.error_code == "E_PERMISSIONS"

    def test_get_relative_for_workspace_root(self, tmp_path):
        """Test getting relative path for workspace root."""
        workspace = Workspace(str(tmp_path))

        relative = workspace.get_relative(str(tmp_path))
        assert relative == "."


class TestListFiles:
    """Test file listing functionality."""

    def test_list_files_empty_workspace(self, tmp_path):
        """Test listing files in empty workspace."""
        workspace = Workspace(str(tmp_path))

        files = workspace.list_files()
        assert files == []

    def test_list_files_with_files(self, tmp_path):
        """Test listing files with actual files."""
        workspace = Workspace(str(tmp_path))

        # Create test files
        (tmp_path / "file1.txt").write_text("content1")
        (tmp_path / "file2.py").write_text("content2")
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        (subdir / "file3.md").write_text("content3")

        files = workspace.list_files()
        expected = [
            str(tmp_path / "file1.txt"),
            str(tmp_path / "file2.py"),
            str(tmp_path / "subdir" / "file3.md"),
        ]

        assert sorted(files) == sorted(expected)

    def test_list_files_ignores_directories(self, tmp_path):
        """Test listing files ignores directories."""
        workspace = Workspace(str(tmp_path))

        # Create directories and files
        (tmp_path / "file.txt").write_text("content")
        (tmp_path / "dir1").mkdir()
        (tmp_path / "dir2").mkdir()

        files = workspace.list_files()
        assert files == [str(tmp_path / "file.txt")]

    def test_list_files_handles_permission_errors(self, tmp_path):
        """Test listing files handles permission errors gracefully."""
        workspace = Workspace(str(tmp_path))

        # Create a file first
        (tmp_path / "accessible.txt").write_text("content")

        # Mock permission error for one part of the traversal
        with mock.patch.object(Path, "rglob") as mock_rglob:
            mock_rglob.side_effect = PermissionError("Access denied")

            files = workspace.list_files()
            assert files == []  # Should handle gracefully


class TestInfo:
    """Test workspace info functionality."""

    def test_info_empty_workspace(self, tmp_path):
        """Test info for empty workspace."""
        workspace = Workspace(str(tmp_path))

        info = workspace.info()
        assert info["root"] == str(tmp_path)
        assert info["size"] == 0
        assert info["file_count"] == 0

    def test_info_with_files(self, tmp_path):
        """Test info with files."""
        workspace = Workspace(str(tmp_path))

        # Create test files with known content
        content1 = "Hello, World!"
        content2 = "Python is great!"
        (tmp_path / "file1.txt").write_text(content1)
        (tmp_path / "file2.txt").write_text(content2)

        info = workspace.info()
        assert info["root"] == str(tmp_path)
        assert info["file_count"] == 2
        assert info["size"] == len(content1) + len(content2)

    def test_info_with_nested_files(self, tmp_path):
        """Test info with nested files."""
        workspace = Workspace(str(tmp_path))

        # Create nested structure
        (tmp_path / "file1.txt").write_text("content1")
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        (subdir / "file2.txt").write_text("content2")

        info = workspace.info()
        assert info["file_count"] == 2
        assert info["size"] == len("content1") + len("content2")

    def test_info_handles_permission_errors(self, tmp_path):
        """Test info handles permission errors gracefully."""
        workspace = Workspace(str(tmp_path))

        # Mock permission error
        with mock.patch.object(Path, "rglob") as mock_rglob:
            mock_rglob.side_effect = PermissionError("Access denied")

            info = workspace.info()
            assert info["root"] == str(tmp_path)
            assert info["file_count"] == 0
            assert info["size"] == 0


class TestSecurityScenarios:
    """Test various security scenarios and edge cases."""

    def test_symlink_within_workspace_allowed(self, tmp_path):
        """Test symlinks within workspace are allowed."""
        workspace = Workspace(str(tmp_path))

        # Create target file and symlink within workspace
        target_file = tmp_path / "target.txt"
        target_file.write_text("target content")

        symlink = tmp_path / "link.txt"
        symlink.symlink_to(target_file)

        # Symlink within workspace should be safe
        assert workspace.is_path_safe(str(symlink.resolve())) is True

    def test_complex_path_traversal_attempts(self, tmp_path):
        """Test complex path traversal attempts."""
        workspace = Workspace(str(tmp_path))

        # Create some structure
        subdir = tmp_path / "subdir"
        subdir.mkdir()

        complex_attacks = [
            "subdir/../../../etc/passwd",
            "subdir/./../../outside.txt",
            "subdir/../subdir/../../etc/passwd",
            "./subdir/../../../etc/passwd",
            "subdir/../subdir/../subdir/../../etc/passwd",
        ]

        for attack in complex_attacks:
            with pytest.raises(WorkspaceSecurityError):
                workspace.resolve_path(attack)

    def test_absolute_path_within_workspace_allowed(self, tmp_path):
        """Test absolute paths within workspace are allowed."""
        workspace = Workspace(str(tmp_path))

        # Create file
        file_path = tmp_path / "file.txt"
        file_path.write_text("content")

        # Absolute path within workspace should work
        resolved = workspace.resolve_path(str(file_path))
        assert resolved == str(file_path)

    def test_case_sensitivity_handled(self, tmp_path):
        """Test case sensitivity is handled properly."""
        workspace = Workspace(str(tmp_path))

        # This test may behave differently on case-insensitive filesystems
        # but the important thing is it doesn't crash
        resolved = workspace.resolve_path("File.TXT")
        assert resolved == str(tmp_path / "File.TXT")

    def test_unicode_paths_handled(self, tmp_path):
        """Test Unicode paths are handled properly."""
        workspace = Workspace(str(tmp_path))

        unicode_filename = "файл.txt"  # Russian characters
        resolved = workspace.resolve_path(unicode_filename)
        assert resolved == str(tmp_path / unicode_filename)

    def test_very_long_paths_handled(self, tmp_path):
        """Test very long paths are handled."""
        workspace = Workspace(str(tmp_path))

        # Create a reasonably long path
        long_path = "/".join(["subdir"] * 10) + "/file.txt"
        resolved = workspace.resolve_path(long_path)
        expected = tmp_path
        for _ in range(10):
            expected = expected / "subdir"
        expected = expected / "file.txt"
        assert resolved == str(expected)


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_workspace_security_error_attributes(self, tmp_path):
        """Test WorkspaceSecurityError has proper attributes."""
        workspace = Workspace(str(tmp_path))

        try:
            workspace.resolve_path("~/file.txt")
        except WorkspaceSecurityError as e:
            assert e.path == "~/file.txt"
            assert e.reason == "home_expansion"
            assert e.error_code == "E_PERMISSIONS"
            assert "Home directory paths not allowed" in str(e)

    def test_nonexistent_workspace_directory_created(self, tmp_path):
        """Test nonexistent workspace directory is created."""
        nonexistent = tmp_path / "new" / "workspace"
        workspace = Workspace(str(nonexistent))

        assert nonexistent.exists()
        assert workspace.root_path == nonexistent

    def test_resolve_path_with_special_characters(self, tmp_path):
        """Test resolving paths with special characters."""
        workspace = Workspace(str(tmp_path))

        special_chars = [
            "file with spaces.txt",
            "file-with-dashes.txt",
            "file_with_underscores.txt",
        ]

        for filename in special_chars:
            resolved = workspace.resolve_path(filename)
            assert resolved == str(tmp_path / filename)
