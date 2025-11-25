"""Command validation for bash tool security.

Validates commands against whitelist/blacklist and detects dangerous patterns.
See INTERFACES/WORKSPACE_SECURITY.md for specification.
"""

import re
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

from agentrunner.core.tool_protocol import ToolCall


@dataclass
class CommandInfo:
    """Parsed command information."""

    command: str
    binary: str
    arguments: list[str]
    is_safe: bool
    risk_level: str  # safe|low|medium|high|critical
    warnings: list[str] = field(default_factory=list)
    paths_referenced: list[str] = field(default_factory=list)


class CommandValidator:
    """Validates bash commands for security."""

    DEFAULT_WHITELIST: ClassVar[set[str]] = {
        # Read-only commands
        "ls",
        "cat",
        "head",
        "tail",
        "grep",
        "find",
        "echo",
        "pwd",
        "which",
        "whoami",
        "date",
        "wc",
        "sort",
        "uniq",
        "diff",
        "tree",
        "file",
        "stat",
        "rg",
        "fd",
        # Python execution (read code, run tests)
        "python",
        "python3",
        "pytest",
        # Build tools (compilation only)
        "cargo",
        "rustc",
        "go",
        "make",
        # Safe utilities
        "touch",
        "cp",
        "mv",
    }

    DEFAULT_BLACKLIST: ClassVar[set[str]] = {
        "dd",
        "mkfs",
        "fdisk",
        "parted",
        "format",
        "sudo",
        "su",
        "reboot",
        "shutdown",
        "poweroff",
        "init",
        "systemctl",
        "service",
        ":(){:|:&};:",  # Fork bomb
        "eval",
        "exec",
    }

    DANGEROUS_PATTERNS: ClassVar[list[str]] = [
        r"rm\s+-rf\s+/",  # rm -rf /
        r"rm\s+-rf\s+\*",  # rm -rf *
        r">\s*/dev/sd",  # Write to disk
        r"dd\s+.*of=/dev",  # dd to device
        r"mkfs",  # Format filesystem
        r":\(\)\{.*\|\:.*\}",  # Fork bomb variants
        r"curl.*\|.*sh",  # Pipe to shell
        r"wget.*\|.*sh",  # Pipe to shell
        r"chmod\s+777",  # Overly permissive
        r"/etc/passwd",  # System files
        r"/etc/shadow",
        r"sudo\s+",  # Privilege escalation
    ]

    def __init__(
        self,
        whitelist: set[str] | None = None,
        blacklist: set[str] | None = None,
        allow_unlisted: bool = False,
    ) -> None:
        """Initialize command validator.

        Args:
            whitelist: Allowed commands (None = use defaults)
            blacklist: Forbidden commands (None = use defaults)
            allow_unlisted: If True, allow commands not in whitelist
        """
        self.whitelist = whitelist if whitelist is not None else self.DEFAULT_WHITELIST
        self.blacklist = blacklist if blacklist is not None else self.DEFAULT_BLACKLIST
        self.allow_unlisted = allow_unlisted

    def parse(self, cmd: str) -> CommandInfo:
        """Parse command and extract information.

        Args:
            cmd: Command string to parse

        Returns:
            CommandInfo with parsed details
        """
        cmd = cmd.strip()

        if not cmd:
            return CommandInfo(
                command=cmd,
                binary="",
                arguments=[],
                is_safe=False,
                risk_level="critical",
                warnings=["Empty command"],
            )

        try:
            tokens = shlex.split(cmd)
        except ValueError as e:
            return CommandInfo(
                command=cmd,
                binary="",
                arguments=[],
                is_safe=False,
                risk_level="critical",
                warnings=[f"Failed to parse command: {e}"],
            )

        binary = tokens[0] if tokens else ""
        arguments = tokens[1:] if len(tokens) > 1 else []

        # Extract referenced paths
        paths = self._extract_paths(arguments)

        # Determine safety
        is_safe, risk_level, warnings = self._assess_safety(cmd, binary, arguments)

        return CommandInfo(
            command=cmd,
            binary=binary,
            arguments=arguments,
            is_safe=is_safe,
            risk_level=risk_level,
            warnings=warnings,
            paths_referenced=paths,
        )

    def is_safe(self, cmd: str) -> bool:
        """Check if command is safe to execute.

        Args:
            cmd: Command string

        Returns:
            True if safe, False otherwise
        """
        info = self.parse(cmd)
        return info.is_safe

    def is_safe_tool(self, call: ToolCall) -> bool:
        """Check if tool call is safe (for bash tool).

        Args:
            call: ToolCall to validate

        Returns:
            True if safe
        """
        if call.name != "bash":
            return True

        cmd = call.arguments.get("command", "")
        return self.is_safe(cmd)

    def _assess_safety(
        self, cmd: str, binary: str, arguments: list[str]
    ) -> tuple[bool, str, list[str]]:
        """Assess command safety.

        Returns:
            (is_safe, risk_level, warnings)
        """
        warnings = []

        # Check dangerous patterns
        for pattern in self.DANGEROUS_PATTERNS:
            if re.search(pattern, cmd, re.IGNORECASE):
                return (
                    False,
                    "critical",
                    [f"Dangerous pattern detected: {pattern}"],
                )

        # Check blacklist
        binary_base = Path(binary).name if "/" in binary else binary
        if binary_base in self.blacklist:
            return (False, "critical", [f"Blacklisted command: {binary_base}"])

        # Check whitelist
        if binary_base not in self.whitelist:
            if not self.allow_unlisted:
                return (
                    False,
                    "high",
                    [f"Command not in whitelist: {binary_base}"],
                )
            warnings.append(f"Unlisted command (allowed): {binary_base}")

        # Check for dangerous flags
        dangerous_flags = self._check_dangerous_flags(binary_base, arguments)
        if dangerous_flags:
            return (False, "high", dangerous_flags)

        # Passed all checks
        risk_level = "low" if binary_base in self.whitelist else "medium"
        return (True, risk_level, warnings)

    def _check_dangerous_flags(self, binary: str, arguments: list[str]) -> list[str]:
        """Check for dangerous command flags.

        Returns:
            List of warnings (empty if safe)
        """
        warnings = []
        args_str = " ".join(arguments)

        # Check rm flags
        if (
            binary == "rm"
            and ("-rf" in args_str or "-fr" in args_str)
            and ("/" in args_str or "*" in args_str)
        ):
            warnings.append("Dangerous rm flags with wildcard")

        # Check chmod
        if binary == "chmod" and ("777" in args_str or "666" in args_str):
            warnings.append("Overly permissive chmod")

        # Check curl/wget
        if binary in ("curl", "wget") and ("|" in args_str or "sh" in args_str):
            warnings.append("Piping download to shell")

        return warnings

    def _extract_paths(self, arguments: list[str]) -> list[str]:
        """Extract file paths from arguments.

        Args:
            arguments: Command arguments

        Returns:
            List of potential paths
        """
        paths = []

        for arg in arguments:
            if arg.startswith("-"):
                continue

            if ("/" in arg or "." in arg) and not arg.startswith("http"):
                paths.append(arg)

        return paths
