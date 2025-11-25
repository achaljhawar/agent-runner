"""Command execution abstraction for tools.

Provides a pluggable interface for executing shell commands in different
environments (subprocess, Docker, sandboxed containers, etc.) without changing tool code.

This design pattern allows tools to remain agnostic of their execution environment.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class CommandResult:
    """Result from command execution.

    Attributes:
        stdout: Standard output from command
        stderr: Standard error from command
        exit_code: Command exit code (0 = success)
        duration_ms: Execution time in milliseconds
        metadata: Optional executor-specific metadata
    """

    stdout: str
    stderr: str
    exit_code: int
    duration_ms: int
    metadata: dict[str, Any] | None = None


class CommandExecutor(ABC):
    """Abstract interface for executing shell commands.

    This abstraction allows tools (like BashTool) to execute commands
    without knowing the underlying execution mechanism. Implementations
    can use subprocess, Docker, sandboxed containers, or any other backend.

    This enables flexible deployment across different security contexts.
    """

    @abstractmethod
    async def execute_command(
        self,
        command: str,
        cwd: str,
        env: dict[str, str],
        timeout: int,
    ) -> CommandResult:
        """Execute a shell command.

        Args:
            command: Shell command to execute
            cwd: Working directory for command
            env: Environment variables
            timeout: Timeout in seconds

        Returns:
            CommandResult with stdout, stderr, exit code, and timing

        Raises:
            TimeoutError: If command exceeds timeout
            OSError: If command execution fails
        """
        ...

    @abstractmethod
    def get_name(self) -> str:
        """Get executor name for logging/debugging.

        Returns:
            Human-readable executor name (e.g., "subprocess", "docker", "container")
        """
        ...
