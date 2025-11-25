"""Command executor implementations.

Provides different backends for executing shell commands:
- SubprocessExecutor: Direct subprocess with optional UID/GID isolation
- DockerExecutor: Containerized execution (future)
"""

from agentrunner.core.executors.subprocess_executor import SubprocessExecutor

__all__ = ["SubprocessExecutor"]
