"""Tests for context strategies."""

import tempfile

import pytest

from agentrunner.core.context_strategy import (
    CursorStyleContext,
    FullTreeContext,
    MinimalContext,
    create_context_strategy,
)
from agentrunner.core.workspace_tracker import WorkspaceTracker


@pytest.fixture
def temp_workspace():
    """Create temporary workspace with tracker."""
    with tempfile.TemporaryDirectory() as temp_dir:
        tracker = WorkspaceTracker(temp_dir)
        yield tracker


def test_cursor_style_context_empty(temp_workspace):
    """Test Cursor style context with empty workspace."""
    tracker = temp_workspace
    strategy = CursorStyleContext()

    context = strategy.build_workspace_context(tracker)

    assert "WORKSPACE STATE:" in context
    assert "Total files: 0" in context
    assert "FILE TREE:" in context
    assert "No files in workspace." in context
    assert "To read a file, use the read_file tool." in context


def test_cursor_style_context_with_files(temp_workspace):
    """Test Cursor style context with files."""
    tracker = temp_workspace
    tracker.track_file("src/main.py", "created", 100)
    tracker.track_file("README.md", "created", 50)

    strategy = CursorStyleContext()
    context = strategy.build_workspace_context(tracker)

    assert "WORKSPACE STATE:" in context
    assert "Total files: 2" in context
    assert "Files created: 2" in context
    assert "Files modified: 0" in context
    assert "FILE TREE:" in context
    assert "main.py" in context
    assert "README.md" in context
    assert "src" in context


def test_cursor_style_format_size():
    """Test file size formatting in Cursor style."""
    strategy = CursorStyleContext()

    assert strategy._format_size(500) == "500.0 B"
    assert strategy._format_size(1536) == "1.5 KB"
    assert strategy._format_size(2048 * 1024) == "2.0 MB"
    assert strategy._format_size(3 * 1024 * 1024 * 1024) == "3.0 GB"
    assert strategy._format_size(2 * 1024 * 1024 * 1024 * 1024) == "2.0 TB"


def test_full_tree_context_empty(temp_workspace):
    """Test Full tree context with empty workspace."""
    tracker = temp_workspace
    strategy = FullTreeContext()

    context = strategy.build_workspace_context(tracker)

    assert context == "WORKSPACE: Empty (no files)"


def test_full_tree_context_with_files(temp_workspace):
    """Test Full tree context with files."""
    tracker = temp_workspace
    tracker.track_file("src/main.py", "created", 100)
    tracker.track_file("src/utils.py", "modified", 50)
    tracker.track_file("README.md", "created", 200)

    strategy = FullTreeContext()
    context = strategy.build_workspace_context(tracker)

    assert "WORKSPACE FILES:" in context
    assert "[CREATED] README.md" in context
    assert "[CREATED] src/main.py" in context
    assert "[MODIFIED] src/utils.py" in context
    assert "100.0 B" in context
    assert "50.0 B" in context
    assert "200.0 B" in context


def test_minimal_context(temp_workspace):
    """Test Minimal context strategy."""
    tracker = temp_workspace
    tracker.track_file("src/main.py", "created", 100)
    tracker.track_file("README.md", "created", 50)

    strategy = MinimalContext()
    context = strategy.build_workspace_context(tracker)

    assert "WORKSPACE: 2 files." in context
    assert "Use list_directory or read_file to explore." in context


def test_minimal_context_empty(temp_workspace):
    """Test Minimal context with empty workspace."""
    tracker = temp_workspace
    strategy = MinimalContext()

    context = strategy.build_workspace_context(tracker)

    assert "WORKSPACE: 0 files." in context


def test_strategy_names():
    """Test strategy name methods."""
    cursor = CursorStyleContext()
    full = FullTreeContext()
    minimal = MinimalContext()

    assert cursor.get_name() == "compact"
    assert full.get_name() == "full_tree"
    assert minimal.get_name() == "minimal"


def test_create_context_strategy_cursor():
    """Test factory function for cursor strategy (backwards compat alias)."""
    strategy = create_context_strategy("cursor")

    assert isinstance(strategy, CursorStyleContext)
    assert strategy.get_name() == "compact"


def test_create_context_strategy_full():
    """Test factory function for full tree strategy."""
    strategy = create_context_strategy("full")

    assert isinstance(strategy, FullTreeContext)
    assert strategy.get_name() == "full_tree"


def test_create_context_strategy_minimal():
    """Test factory function for minimal strategy."""
    strategy = create_context_strategy("minimal")

    assert isinstance(strategy, MinimalContext)
    assert strategy.get_name() == "minimal"


def test_create_context_strategy_unknown():
    """Test factory function with unknown strategy."""
    with pytest.raises(ValueError, match="Unknown context strategy: unknown"):
        create_context_strategy("unknown")


def test_create_context_strategy_error_message():
    """Test error message includes available strategies."""
    with pytest.raises(ValueError, match="Available: compact, cursor, full, minimal"):
        create_context_strategy("invalid")


def test_context_with_large_workspace(temp_workspace):
    """Test context strategies with larger workspace."""
    tracker = temp_workspace

    # Create a more complex workspace structure
    files = [
        "src/main.py",
        "src/components/Button.tsx",
        "src/components/Input.tsx",
        "src/utils/helpers.py",
        "src/utils/constants.py",
        "tests/test_main.py",
        "tests/test_utils.py",
        "package.json",
        "README.md",
        "tsconfig.json",
    ]

    for i, file_path in enumerate(files):
        operation = "modified" if i % 3 == 0 else "created"
        tracker.track_file(file_path, operation, 100 + i * 50)

    # Test each strategy
    cursor = CursorStyleContext()
    full = FullTreeContext()
    minimal = MinimalContext()

    cursor_context = cursor.build_workspace_context(tracker)
    full_context = full.build_workspace_context(tracker)
    minimal_context = minimal.build_workspace_context(tracker)

    # Cursor should show tree structure
    assert "src" in cursor_context
    assert "components" in cursor_context
    assert "Total files: 10" in cursor_context

    # Full should show all files with metadata
    assert "WORKSPACE FILES:" in full_context
    assert "[CREATED]" in full_context
    assert "[MODIFIED]" in full_context
    assert all(file_path in full_context for file_path in files)

    # Minimal should be very brief
    assert "WORKSPACE: 10 files." in minimal_context
    assert len(minimal_context.split("\n")) <= 3  # Very concise


def test_context_strategy_max_tokens_parameter():
    """Test that all strategies accept max_tokens parameter (even if unused)."""
    tracker = WorkspaceTracker("/tmp")
    tracker.track_file("test.py", "created", 100)

    cursor = CursorStyleContext()
    full = FullTreeContext()
    minimal = MinimalContext()

    # Should not raise errors with max_tokens parameter
    cursor.build_workspace_context(tracker, max_tokens=1000)
    full.build_workspace_context(tracker, max_tokens=1000)
    minimal.build_workspace_context(tracker, max_tokens=1000)
