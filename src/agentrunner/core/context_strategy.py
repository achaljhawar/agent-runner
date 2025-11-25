"""Context strategy interfaces for pluggable context management."""

from abc import ABC, abstractmethod

from .workspace_tracker import WorkspaceTracker


class ContextStrategy(ABC):
    """Interface for different context management approaches"""

    @abstractmethod
    def build_workspace_context(
        self, tracker: WorkspaceTracker, max_tokens: int | None = None
    ) -> str:
        """Build workspace context for LLM"""
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Get strategy name for logging/debugging"""
        pass


class CompactContext(ContextStrategy):
    """Compact context: Show file tree + let LLM explore on-demand"""

    def build_workspace_context(
        self, tracker: WorkspaceTracker, max_tokens: int | None = None
    ) -> str:
        """Build compact file tree summary"""
        stats = tracker.get_stats()
        tree = tracker.get_file_tree()

        context = f"""WORKSPACE STATE:
- Total files: {stats['total_files']}
- Total size: {self._format_size(stats['total_size'])}
- Files created: {stats['created_count']}
- Files modified: {stats['modified_count']}

FILE TREE:
{tree}

To read a file, use the read_file tool.
To search for files, use the search_files tool.
To list directory contents, use the list_directory tool."""

        return context

    def _format_size(self, size_bytes: int) -> str:
        """Format file size for humans"""
        size_float = float(size_bytes)
        for unit in ["B", "KB", "MB", "GB"]:
            if size_float < 1024.0:
                return f"{size_float:.1f} {unit}"
            size_float /= 1024.0
        return f"{size_float:.1f} TB"

    def get_name(self) -> str:
        return "compact"


# Backwards compatibility alias
CursorStyleContext = CompactContext


class FullTreeContext(ContextStrategy):
    """Full tree: Show complete file tree with all metadata"""

    def build_workspace_context(
        self, tracker: WorkspaceTracker, max_tokens: int | None = None
    ) -> str:
        """Build detailed file tree with metadata"""
        files = tracker.get_files_list()

        if not files:
            return "WORKSPACE: Empty (no files)"

        lines = ["WORKSPACE FILES:"]
        for file_path in files:
            info = tracker.files[file_path]
            size = self._format_size(info.size)
            op = info.operation.upper()
            lines.append(f"  [{op}] {file_path} ({size})")

        return "\n".join(lines)

    def _format_size(self, size_bytes: int) -> str:
        """Format file size for humans"""
        size_float = float(size_bytes)
        for unit in ["B", "KB", "MB", "GB"]:
            if size_float < 1024.0:
                return f"{size_float:.1f} {unit}"
            size_float /= 1024.0
        return f"{size_float:.1f} TB"

    def get_name(self) -> str:
        return "full_tree"


class MinimalContext(ContextStrategy):
    """Minimal: Only show file count, let LLM explore everything"""

    def build_workspace_context(
        self, tracker: WorkspaceTracker, max_tokens: int | None = None
    ) -> str:
        """Build minimal context"""
        stats = tracker.get_stats()
        return (
            f"WORKSPACE: {stats['total_files']} files. Use list_directory or read_file to explore."
        )

    def get_name(self) -> str:
        return "minimal"


def create_context_strategy(strategy_name: str) -> ContextStrategy:
    """Factory function for creating context strategies"""
    strategies: dict[str, type[ContextStrategy]] = {
        "compact": CompactContext,
        "cursor": CompactContext,  # Alias for backwards compatibility
        "full": FullTreeContext,
        "minimal": MinimalContext,
    }

    strategy_class = strategies.get(strategy_name)
    if strategy_class is None:
        raise ValueError(
            f"Unknown context strategy: {strategy_name}. "
            f"Available: {', '.join(strategies.keys())}"
        )

    return strategy_class()
