"""Subprocess-based command executor with optional UID/GID isolation.

This is the default executor, extracted from BashTool to allow
command execution logic to be reused and tested independently.
"""

import asyncio
import os
import sys

from agentrunner.core.command_executor import CommandExecutor, CommandResult


class SubprocessExecutor(CommandExecutor):
    """Execute commands using asyncio.subprocess with optional UID/GID isolation.

    Provides Unix privilege dropping for multi-tenant security.
    """

    def __init__(
        self,
        uid: int | None = None,
        gid: int | None = None,
        max_output_lines: int = 1000,
        max_output_bytes: int = 1024 * 1024,  # 1MB
    ) -> None:
        """Initialize subprocess executor.

        Args:
            uid: Unix UID to drop privileges to (None = current user)
            gid: Unix GID to drop privileges to (None = current group)
            max_output_lines: Maximum output lines before truncation
            max_output_bytes: Maximum output bytes before truncation (1MB default)
        """
        self.uid = uid
        self.gid = gid
        self.max_output_lines = max_output_lines
        self.max_output_bytes = max_output_bytes

    def get_name(self) -> str:
        """Get executor name."""
        return "subprocess"

    async def execute_command(
        self,
        command: str,
        cwd: str,
        env: dict[str, str],
        timeout: int,
    ) -> CommandResult:
        """Execute command using asyncio.subprocess.

        Supports Unix UID/GID privilege dropping for multi-tenant isolation.

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

        def setup_subprocess() -> None:
            """Setup subprocess: create new process group and optionally drop privileges."""
            # Create new process group (so we can kill entire tree on timeout)
            os.setpgrp()

            # Drop privileges if requested
            if self.gid is not None and self.uid is not None:
                try:
                    print(
                        f"[SubprocessExecutor] Before drop: UID={os.getuid()}, GID={os.getgid()}",
                        file=sys.stderr,
                    )
                    os.setgid(self.gid)
                    os.setuid(self.uid)
                    print(
                        f"[SubprocessExecutor] After drop: UID={os.getuid()}, GID={os.getgid()}",
                        file=sys.stderr,
                    )
                except (PermissionError, OSError) as e:
                    print(
                        f"[SubprocessExecutor] Warning: Could not drop privileges: {e}",
                        file=sys.stderr,
                    )

        start_time = asyncio.get_event_loop().time()
        preexec = setup_subprocess if sys.platform != "win32" else None

        try:
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
                env=env,
                preexec_fn=preexec,
            )

            try:
                stdout_data, stderr_data = await asyncio.wait_for(
                    process.communicate(), timeout=timeout
                )
                exit_code = process.returncode or 0

            except TimeoutError:
                try:
                    if sys.platform != "win32":
                        os.killpg(os.getpgid(process.pid), 9)
                    else:
                        process.kill()
                    await process.wait()
                except ProcessLookupError:
                    pass

                raise TimeoutError(f"Command timed out after {timeout} seconds") from None

        except TimeoutError:
            raise TimeoutError(f"Command timed out after {timeout} seconds") from None
        except (OSError, ValueError) as e:
            raise OSError(f"Failed to execute command: {e}") from e

        end_time = asyncio.get_event_loop().time()
        duration_ms = int((end_time - start_time) * 1000)

        try:
            stdout = stdout_data.decode("utf-8", errors="replace")
            stderr = stderr_data.decode("utf-8", errors="replace")
        except Exception as e:
            raise OSError(f"Failed to decode command output: {e}") from e

        # Check both byte size and line count limits (whichever hits first)
        total_bytes = len(stdout_data) + len(stderr_data)
        stdout_lines = stdout.splitlines()
        stderr_lines = stderr.splitlines()
        total_lines = len(stdout_lines) + len(stderr_lines)

        # Truncate if either limit is exceeded
        if total_bytes > self.max_output_bytes or total_lines > self.max_output_lines:
            if total_bytes > self.max_output_bytes:
                truncated_msg = f"... [Output truncated - exceeded {self.max_output_bytes} bytes]"
                half_limit = self.max_output_bytes // 2
                if len(stdout_data) > half_limit:
                    stdout = (
                        stdout_data[:half_limit].decode("utf-8", errors="replace")
                        + "\n"
                        + truncated_msg
                    )
                if len(stderr_data) > half_limit:
                    stderr = (
                        stderr_data[:half_limit].decode("utf-8", errors="replace")
                        + "\n"
                        + truncated_msg
                    )
            elif total_lines > self.max_output_lines:
                truncated_msg = f"... [Output truncated - exceeded {self.max_output_lines} lines]"
                half_limit = self.max_output_lines // 2
                if len(stdout_lines) > half_limit:
                    stdout = "\n".join(stdout_lines[:half_limit]) + "\n" + truncated_msg
                if len(stderr_lines) > half_limit:
                    stderr = "\n".join(stderr_lines[:half_limit]) + "\n" + truncated_msg

        return CommandResult(
            stdout=stdout,
            stderr=stderr,
            exit_code=exit_code,
            duration_ms=duration_ms,
            metadata={
                "uid": self.uid,
                "gid": self.gid,
                "platform": sys.platform,
                "privileges_dropped": self.uid is not None,
            },
        )
