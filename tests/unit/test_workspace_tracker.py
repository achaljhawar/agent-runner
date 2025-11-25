"""Tests for WorkspaceTracker."""

import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from agentrunner.core.workspace_tracker import FileInfo, WorkspaceTracker


@pytest.fixture
def temp_workspace():
    """Create temporary workspace directory."""
    with tempfile.TemporaryDirectory() as temp_dir:
        tracker = WorkspaceTracker(temp_dir)
        yield tracker


def test_track_file_created(temp_workspace):
    """Test tracking a created file."""
    tracker = temp_workspace
    tracker.track_file("main.py", "created", 100)

    assert "main.py" in tracker.files
    file_info = tracker.files["main.py"]
    assert file_info.path == "main.py"
    assert file_info.size == 100
    assert file_info.operation == "created"
    assert isinstance(file_info.modified_at, datetime)


def test_track_file_modified(temp_workspace):
    """Test tracking a modified file."""
    tracker = temp_workspace
    tracker.track_file("main.py", "created", 100)
    tracker.track_file("main.py", "modified", 150)

    file_info = tracker.files["main.py"]
    assert file_info.size == 150
    assert file_info.operation == "modified"


def test_track_file_deleted(temp_workspace):
    """Test tracking a deleted file."""
    tracker = temp_workspace
    tracker.track_file("main.py", "created", 100)
    assert "main.py" in tracker.files

    tracker.track_file("main.py", "deleted")
    assert "main.py" not in tracker.files


def test_get_file_tree_empty(temp_workspace):
    """Test file tree for empty workspace."""
    tracker = temp_workspace
    tree = tracker.get_file_tree()
    assert tree == "No files in workspace."


def test_get_file_tree_single_file(temp_workspace):
    """Test file tree with single file."""
    tracker = temp_workspace
    tracker.track_file("main.py", "created", 100)

    tree = tracker.get_file_tree()
    assert "main.py" in tree
    assert "└──" in tree


def test_get_file_tree_nested_structure(temp_workspace):
    """Test file tree with nested directory structure."""
    tracker = temp_workspace
    tracker.track_file("src/main.py", "created", 100)
    tracker.track_file("src/utils.py", "created", 50)
    tracker.track_file("src/components/Button.tsx", "created", 75)
    tracker.track_file("README.md", "created", 200)

    tree = tracker.get_file_tree()

    # Should contain all files and directories
    assert "src" in tree
    assert "main.py" in tree
    assert "utils.py" in tree
    assert "components" in tree
    assert "Button.tsx" in tree
    assert "README.md" in tree

    # Should have tree structure characters
    assert "├──" in tree or "└──" in tree
    assert "│" in tree or "    " in tree


def test_get_files_list(temp_workspace):
    """Test getting flat list of files."""
    tracker = temp_workspace
    tracker.track_file("src/main.py", "created", 100)
    tracker.track_file("README.md", "created", 50)
    tracker.track_file("src/utils.py", "created", 75)

    files = tracker.get_files_list()

    # Should be sorted
    assert files == ["README.md", "src/main.py", "src/utils.py"]


def test_get_modified_files(temp_workspace):
    """Test getting files modified since timestamp."""
    tracker = temp_workspace

    # Track files at different times
    now = datetime.now(UTC)
    past = now - timedelta(hours=1)

    tracker.track_file("old_file.py", "created", 100)
    # Manually set older timestamp
    tracker.files["old_file.py"].modified_at = past

    tracker.track_file("new_file.py", "created", 100)

    # Get files modified after the past timestamp
    recent_files = tracker.get_modified_files(past + timedelta(minutes=30))

    assert "new_file.py" in recent_files
    assert "old_file.py" not in recent_files


def test_get_stats(temp_workspace):
    """Test workspace statistics."""
    tracker = temp_workspace
    tracker.track_file("src/main.py", "created", 100)
    tracker.track_file("src/utils.py", "modified", 50)
    tracker.track_file("tests/test_main.py", "created", 75)
    tracker.track_file("README.md", "created", 200)

    stats = tracker.get_stats()

    assert stats["total_files"] == 4
    assert stats["total_size"] == 425
    assert stats["created_count"] == 3
    assert stats["modified_count"] == 1
    assert stats["total_directories"] >= 2  # src, tests


def test_clear(temp_workspace):
    """Test clearing all tracked files."""
    tracker = temp_workspace
    tracker.track_file("main.py", "created", 100)
    tracker.track_file("utils.py", "created", 50)

    assert len(tracker.files) == 2

    tracker.clear()

    assert len(tracker.files) == 0
    assert tracker.get_file_tree() == "No files in workspace."


def test_track_file_without_size(temp_workspace):
    """Test tracking file without explicit size."""
    tracker = temp_workspace

    # Create an actual file to test auto-size detection
    test_file = tracker.workspace_root / "test.txt"
    test_file.write_text("Hello world!")

    tracker.track_file("test.txt", "created")

    file_info = tracker.files["test.txt"]
    assert file_info.size == len("Hello world!")


def test_track_file_nonexistent_without_size(temp_workspace):
    """Test tracking nonexistent file without size."""
    tracker = temp_workspace
    tracker.track_file("nonexistent.txt", "created")

    file_info = tracker.files["nonexistent.txt"]
    assert file_info.size == 0


def test_complex_directory_structure(temp_workspace):
    """Test complex nested directory structure."""
    tracker = temp_workspace

    files = [
        "src/main.py",
        "src/components/Button.tsx",
        "src/components/Input.tsx",
        "src/utils/helpers.py",
        "src/utils/constants.py",
        "tests/unit/test_main.py",
        "tests/integration/test_api.py",
        "docs/README.md",
        "package.json",
    ]

    for file_path in files:
        tracker.track_file(file_path, "created", 100)

    tree = tracker.get_file_tree()

    # Verify all directories and files appear
    for file_path in files:
        parts = Path(file_path).parts
        for part in parts:
            assert part in tree

    # Verify tree structure
    assert "├──" in tree or "└──" in tree

    # Verify stats
    stats = tracker.get_stats()
    assert stats["total_files"] == len(files)


def test_file_info_dataclass():
    """Test FileInfo dataclass."""
    now = datetime.utcnow()
    file_info = FileInfo(path="src/main.py", size=1200, modified_at=now, operation="created")

    assert file_info.path == "src/main.py"
    assert file_info.size == 1200
    assert file_info.modified_at == now
    assert file_info.operation == "created"
