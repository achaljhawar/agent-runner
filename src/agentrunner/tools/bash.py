"""Bash execution tool with Unix UID/GID isolation.

Implements secure bash command execution using OS-level user isolation.
Each session runs commands as a unique UID, preventing cross-session access.
"""

import os
from datetime import UTC, datetime
from pathlib import Path

from agentrunner.core.events import StreamEvent
from agentrunner.core.exceptions import E_TIMEOUT, E_VALIDATION
from agentrunner.core.executors.subprocess_executor import SubprocessExecutor
from agentrunner.core.tool_protocol import ToolCall, ToolDefinition, ToolResult
from agentrunner.security.command_validator import CommandValidator
from agentrunner.tools.base import BaseTool, ToolContext


class BashTool(BaseTool):
    """Execute bash commands with Unix UID/GID isolation for security."""

    def __init__(
        self,
        command_validator: CommandValidator | None = None,
        default_timeout: int = 300,  # 5 minutes for npm install, builds, etc.
        max_output_lines: int = 1000,
        max_output_bytes: int = 1024 * 1024,  # 1MB
    ) -> None:
        """Initialize bash tool.

        Args:
            command_validator: Command validator for security checks (defaults to permissive validator)
            default_timeout: Default command timeout in seconds (300s for npm install, builds)
            max_output_lines: Maximum output lines before truncation
            max_output_bytes: Maximum output bytes before truncation (1MB default)
        """
        self.command_validator = command_validator or CommandValidator(allow_unlisted=True)
        self.default_timeout = default_timeout
        self.max_output_lines = max_output_lines
        self.max_output_bytes = max_output_bytes
        self._executor: SubprocessExecutor | None = None

    def _get_executor(self, context: ToolContext) -> SubprocessExecutor:
        """Get or create command executor with UID/GID from context."""
        if self._executor is None:
            uid = getattr(context, "session_uid", None)
            gid = getattr(context, "session_gid", None)
            self._executor = SubprocessExecutor(
                uid=uid,
                gid=gid,
                max_output_lines=self.max_output_lines,
                max_output_bytes=self.max_output_bytes,
            )
        return self._executor

    async def execute(self, call: ToolCall, context: ToolContext) -> ToolResult:
        """Execute bash command with Unix UID/GID isolation.

        The command runs as a unique UID per session, with workspace owned by that UID.
        This provides OS-level isolation - the kernel prevents cross-session access.

        Args:
            call: Tool call with command and optional timeout
            context: Tool execution context

        Returns:
            ToolResult with command output, exit code, and execution details

        Note:
            Follows Robustness Principle: Be liberal in what you accept (ignore unknown
            parameters from LLM), be strict in what you require (validate used parameters).
        """
        command = call.arguments.get("command")
        timeout = call.arguments.get("timeout", self.default_timeout)

        if not command:
            return ToolResult(
                success=False,
                error="command is required",
                error_code=E_VALIDATION,
            )

        if not isinstance(command, str):
            return ToolResult(
                success=False,
                error="command must be a string",
                error_code=E_VALIDATION,
            )

        if timeout <= 0 or timeout > 300:  # Max 5 minutes
            return ToolResult(
                success=False,
                error="timeout must be between 1 and 300 seconds",
                error_code=E_VALIDATION,
            )

        # Security validation (optional basic check for obviously dangerous commands)
        # if not self.command_validator.is_safe(command):
        #     cmd_info = self.command_validator.parse(command)
        #     context.logger.warn(
        #         "Unsafe command blocked",
        #         command=command,
        #         risk_level=cmd_info.risk_level,
        #         warnings=cmd_info.warnings,
        #     )
        #     return ToolResult(
        #         success=False,
        #         error=f"Command not allowed: {'; '.join(cmd_info.warnings)}",
        #         error_code=E_UNSAFE,
        #         data={
        #             "risk_level": cmd_info.risk_level,
        #             "warnings": cmd_info.warnings,
        #         },
        #     )

        # Publish bash_started event for real-time UI updates
        if context.event_bus:
            start_event = StreamEvent(
                type="bash_started",
                data={
                    "command": command,
                    "cwd": str(context.workspace.root_path),
                },
                model_id=context.model_id,
                ts=datetime.now(UTC).isoformat(),
            )
            context.logger.info(
                "[BashTool] Publishing bash_started event",
                event_id=start_event.id,
                event_type=start_event.type,
                command=command,
            )
            context.event_bus.publish(start_event)

        # Execute command with Unix UID/GID isolation
        try:
            result = await self._execute_command(command, timeout, context)

            context.logger.info(
                "Bash command executed",
                command=command,
                exit_code=result.data.get("exit_code") if result.data else None,
                duration_ms=result.data.get("duration_ms") if result.data else None,
            )

            # Publish bash_executed event for real-time UI updates
            if context.event_bus:
                event = StreamEvent(
                    type="bash_executed",
                    data={
                        "command": command,
                        "exit_code": result.data.get("exit_code", -1) if result.data else -1,
                        "stdout": result.data.get("stdout", "") if result.data else "",
                        "stderr": result.data.get("stderr", "") if result.data else "",
                        "duration_ms": result.data.get("duration_ms", 0) if result.data else 0,
                        "success": result.success,
                        "cwd": str(context.workspace.root_path),
                    },
                    model_id=context.model_id,
                    ts=datetime.now(UTC).isoformat(),
                )
                context.logger.info(
                    "[BashTool] Publishing bash_executed event",
                    event_id=event.id,
                    event_type=event.type,
                    command=command,
                )
                context.event_bus.publish(event)

            return result

        except (OSError, ValueError) as e:
            context.logger.error("Bash command execution failed", command=command, error=str(e))
            return ToolResult(
                success=False,
                error=f"Command execution failed: {e}",
                error_code=E_VALIDATION,
                data={},
            )

    async def _execute_command(
        self, command: str, timeout: int, context: ToolContext
    ) -> ToolResult:
        """Execute command with Unix UID/GID isolation and capture output.

        Args:
            command: Shell command to execute
            timeout: Timeout in seconds
            context: Tool execution context

        Returns:
            ToolResult with execution results
        """
        import time

        overall_start = time.time()

        workspace_root = str(context.workspace.root_path)
        executor = self._get_executor(context)

        # Log execution info
        session_uid = getattr(context, "session_uid", None)
        session_gid = getattr(context, "session_gid", None)
        current_uid = os.getuid()
        current_gid = os.getgid()

        setup_time = time.time() - overall_start
        context.logger.info(
            "[BASH] Executing command with UID/GID isolation",
            command=command[:100],
            cwd=workspace_root,
            current_uid=current_uid,
            current_gid=current_gid,
            target_uid=session_uid,
            target_gid=session_gid,
            will_drop_privileges=session_uid is not None,
            setup_ms=int(setup_time * 1000),
        )

        # Set up environment
        env_start = time.time()
        env = self._create_secure_env(context)
        env_time = time.time() - env_start
        context.logger.info("[BASH] Environment created", env_ms=int(env_time * 1000))

        # Execute command using executor
        exec_start = time.time()
        context.logger.info("[BASH] Calling executor.execute_command()")
        try:
            result = await executor.execute_command(
                command=command,
                cwd=workspace_root,
                env=env,
                timeout=timeout,
            )
            exec_time = time.time() - exec_start
            context.logger.info(
                "[BASH] executor.execute_command() returned", exec_ms=int(exec_time * 1000)
            )
        except TimeoutError:
            # Agent loop will emit tool_call_completed with success=False
            return ToolResult(
                success=False,
                error=f"Command timed out after {timeout} seconds",
                error_code=E_TIMEOUT,
                data={"timeout": timeout},
            )
        except OSError as e:
            context.logger.error(
                "Command execution failed",
                command=command,
                error=str(e),
            )
            return ToolResult(
                success=False,
                error=f"Failed to execute command: {e}",
                error_code=E_VALIDATION,
                data={},
            )

        # Extract results from executor
        stdout = result.stdout
        stderr = result.stderr
        exit_code = result.exit_code
        duration_ms = result.duration_ms

        # Prepare output
        output_parts = []
        if stdout.strip():
            output_parts.append(f"STDOUT:\n{stdout}")
        if stderr.strip():
            output_parts.append(f"STDERR:\n{stderr}")

        output = "\n\n".join(output_parts) if output_parts else "(no output)"
        success = exit_code == 0

        # Log command result
        context.logger.info(
            "[BASH] Command completed",
            exit_code=exit_code,
            success=success,
            duration_ms=duration_ms,
            stdout_length=len(stdout),
            stderr_length=len(stderr),
            stdout_preview=stdout[:200] if stdout else "(empty)",
            stderr_preview=stderr[:200] if stderr else "(empty)",
        )

        return ToolResult(
            success=success,
            output=output,
            error=f"Command exited with code {exit_code}" if not success else None,
            data={
                "exit_code": exit_code,
                "stdout": stdout,
                "stderr": stderr,
                "duration_ms": duration_ms,
                "cwd": workspace_root,
            },
        )

    def _create_secure_env(self, context: ToolContext) -> dict[str, str]:
        """Create secure environment for command execution.

        Args:
            context: Tool execution context

        Returns:
            Environment dictionary with minimal safe variables
        """
        workspace_root_path = Path(context.workspace.root_path)
        session_root_path = workspace_root_path.parent

        # Allocate cache directories outside the project root so scaffolding directories stay empty
        env_root = session_root_path / ".agentrunner-env" / context.model_id
        home_path = env_root / "home"
        tmp_path = env_root / "tmp"

        try:
            home_path.mkdir(parents=True, exist_ok=True)
            tmp_path.mkdir(parents=True, exist_ok=True)

            # Chown to session UID so subprocess can write (if UIDs are configured)
            if context.session_uid is not None and context.session_gid is not None:
                try:
                    # Chown workspace root first (critical for EFS)
                    os.chown(str(workspace_root_path), context.session_uid, context.session_gid)
                    # Then chown env directories
                    os.chown(str(env_root), context.session_uid, context.session_gid)
                    os.chown(str(home_path), context.session_uid, context.session_gid)
                    os.chown(str(tmp_path), context.session_uid, context.session_gid)
                    context.logger.debug(
                        f"[BASH] Chowned directories to UID={context.session_uid}, GID={context.session_gid}"
                    )
                except (OSError, PermissionError) as e:
                    context.logger.warn(
                        f"[BASH] Failed to chown directories (not running as root?): {str(e)}"
                    )

        except (OSError, PermissionError) as e:
            context.logger.warn(
                f"[BASH] Failed to create isolated env directories; falling back to workspace root: {str(e)}",
            )
            home_path = workspace_root_path
            tmp_path = workspace_root_path / ".tmp"
            tmp_path.mkdir(parents=True, exist_ok=True)

        workspace_root_str = str(workspace_root_path)

        # Start with minimal environment
        secure_env = {
            "HOME": str(home_path),
            "PWD": workspace_root_str,
            "TMPDIR": str(tmp_path),
            "USER": f"session_{context.model_id}",
            "SHELL": "/bin/bash",
            "TERM": os.environ.get("TERM", "xterm-256color"),
            "LANG": os.environ.get("LANG", "en_US.UTF-8"),
            "LC_ALL": os.environ.get("LC_ALL", "en_US.UTF-8"),
        }

        # Add essential PATH (prefer workspace bins first, then global npm bins)
        path_dirs = [
            str(
                workspace_root_path / "node_modules" / ".bin"
            ),  # Local project bins (highest priority)
            str(workspace_root_path / ".venv" / "bin"),
            str(workspace_root_path / "venv" / "bin"),
            "/usr/local/lib/node_modules/.bin",  # Global npm binaries (eslint, typescript, etc.)
            "/opt/homebrew/lib/node_modules/.bin",  # macOS Homebrew global npm bins
            "/opt/homebrew/bin",  # macOS Homebrew (Apple Silicon + Intel)
            "/usr/local/bin",
            "/usr/bin",
            "/bin",
        ]
        secure_env["PATH"] = ":".join(path_dirs)

        # Preserve specific safe environment variables if they exist
        safe_vars = [
            "PYTHON",
            "PYTHONPATH",
            "PYTHONIOENCODING",
            "NODE_OPTIONS",
            "NPM_CONFIG_PREFIX",
        ]
        for var in safe_vars:
            if var in os.environ:
                secure_env[var] = os.environ[var]

        # Don't preserve NODE_ENV - let Node.js tools set it correctly
        # Preserving it causes issues with Next.js when it has non-standard values

        return secure_env

    def get_definition(self) -> ToolDefinition:
        """Get tool definition for LLM.

        Returns:
            ToolDefinition with schema for bash execution
        """
        return ToolDefinition(
            name="bash",
            description=(
                "Execute bash commands in the workspace. "
                "Commands run with filesystem isolation - cannot access other sessions. "
                "Use for running builds, tests, installing dependencies, etc."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The bash command to execute",
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Timeout in seconds (default: 300, max: 300). Suitable for npm install, builds.",
                        "default": 300,
                    },
                },
                "required": ["command"],
            },
        )
