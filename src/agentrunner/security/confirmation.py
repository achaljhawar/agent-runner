"""User confirmation service for dangerous operations.

Handles user confirmation for file operations, bash commands, and destructive actions.
Supports "remember my choice" per session and batch confirmation.
See INTERFACES/WORKSPACE_SECURITY.md for specification.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, ClassVar


class ConfirmationLevel(Enum):
    """Confirmation requirement levels."""

    FILE_OPS = "file_ops"  # File operations (overwrite/delete large files)
    BASH = "bash"  # Bash commands that could be dangerous
    DESTRUCTIVE = "destructive"  # Any irreversible operation


@dataclass
class ActionDescriptor:
    """Describes an action requiring confirmation."""

    level: ConfirmationLevel
    operation: str  # e.g., "delete_file", "execute_bash", "overwrite_file"
    description: str  # Human-readable description
    target: str = ""  # Target file/command/etc.
    details: dict[str, Any] = field(default_factory=dict)  # Additional context
    batch_key: str = ""  # Key for grouping similar operations

    def __post_init__(self) -> None:
        """Generate batch key if not provided."""
        if not self.batch_key:
            # Default batch key is operation + target type
            target_type = "file" if "/" in self.target else "command"
            self.batch_key = f"{self.operation}_{target_type}"


@dataclass
class ConfirmationChoice:
    """User's confirmation choice with session memory."""

    approved: bool
    remember_for_session: bool = False
    apply_to_batch: bool = False


class ConfirmationService:
    """Handles user confirmation for dangerous operations."""

    # Default operations that require confirmation at each level
    DEFAULT_CONFIRMATIONS: ClassVar[dict[ConfirmationLevel, set[str]]] = {
        ConfirmationLevel.FILE_OPS: {
            "delete_file",
            "overwrite_file",
            "delete_directory",
            "move_file",
            "copy_file_overwrite",
        },
        ConfirmationLevel.BASH: {
            "execute_bash",
            "run_command",
        },
        ConfirmationLevel.DESTRUCTIVE: {
            "format_disk",
            "remove_system_file",
            "modify_permissions",
            "install_package",
            "uninstall_package",
        },
    }

    def __init__(
        self,
        auto_approve: bool = False,
        confirmation_map: dict[ConfirmationLevel, set[str]] | None = None,
    ) -> None:
        """Initialize confirmation service.

        Args:
            auto_approve: If True, automatically approve all confirmations (testing)
            confirmation_map: Custom mapping of levels to operations requiring confirmation
        """
        self.auto_approve = auto_approve
        self.confirmation_map = (
            confirmation_map if confirmation_map is not None else self.DEFAULT_CONFIRMATIONS
        )

        # Session memory for "remember my choice"
        self._session_choices: dict[str, bool] = {}

        # Batch choices for similar operations
        self._batch_choices: dict[str, bool] = {}

        # Per-operation session flags for remembering user choices
        self._session_flags = {
            "file_operations": False,
            "bash_commands": False,
            "all_operations": False,
        }

    def approve(self, action: ActionDescriptor) -> bool:
        """Request user approval for an action.

        Args:
            action: ActionDescriptor describing the action

        Returns:
            True if approved, False if denied

        Raises:
            WorkspaceSecurityError: If confirmation is denied
        """
        # Auto-approve if configured (for testing/automation)
        if self.auto_approve:
            return True

        # Check session flags first (global "remember for all" choice)
        if self._session_flags["all_operations"]:
            return True

        # Check operation-specific session flags
        if action.level == ConfirmationLevel.FILE_OPS and self._session_flags["file_operations"]:
            return True
        if action.level == ConfirmationLevel.BASH and self._session_flags["bash_commands"]:
            return True

        # Check if we need confirmation for this operation
        if not self._requires_confirmation(action):
            return True

        # Check session memory
        session_key = self._get_session_key(action)
        if session_key in self._session_choices:
            return self._session_choices[session_key]

        # Check batch choices
        if action.batch_key in self._batch_choices:
            return self._batch_choices[action.batch_key]

        # Request confirmation from user
        choice = self._request_confirmation(action)

        # Store choices if requested
        if choice.remember_for_session:
            self._session_choices[session_key] = choice.approved
            # Also update session flags if applicable
            if action.level == ConfirmationLevel.FILE_OPS:
                self._session_flags["file_operations"] = choice.approved
            elif action.level == ConfirmationLevel.BASH:
                self._session_flags["bash_commands"] = choice.approved

        if choice.apply_to_batch:
            self._batch_choices[action.batch_key] = choice.approved

        return choice.approved

    def get_session_flags(self) -> dict[str, bool]:
        """Get current session flags for operation types.

        Returns:
            Dictionary with file_operations, bash_commands, all_operations flags
        """
        return self._session_flags.copy()

    def clear_session_choices(self) -> None:
        """Clear all session-remembered choices."""
        self._session_choices.clear()
        # Also clear session flags
        self._session_flags = {
            "file_operations": False,
            "bash_commands": False,
            "all_operations": False,
        }

    def clear_batch_choices(self) -> None:
        """Clear all batch choices."""
        self._batch_choices.clear()

    def clear_all_choices(self) -> None:
        """Clear all remembered choices."""
        self.clear_session_choices()
        self.clear_batch_choices()

    def get_session_choices(self) -> dict[str, bool]:
        """Get current session choices (for testing/debugging)."""
        return self._session_choices.copy()

    def get_batch_choices(self) -> dict[str, bool]:
        """Get current batch choices (for testing/debugging)."""
        return self._batch_choices.copy()

    def _requires_confirmation(self, action: ActionDescriptor) -> bool:
        """Check if action requires confirmation.

        Args:
            action: ActionDescriptor to check

        Returns:
            True if confirmation is required
        """
        required_operations = self.confirmation_map.get(action.level, set())
        return action.operation in required_operations

    def _get_session_key(self, action: ActionDescriptor) -> str:
        """Generate session key for remembering choices.

        Args:
            action: ActionDescriptor

        Returns:
            Unique key for this type of action
        """
        return f"{action.level.value}_{action.operation}"

    def _request_confirmation(self, action: ActionDescriptor) -> ConfirmationChoice:
        """Request confirmation from user.

        This is a stub implementation that always denies by default.
        In a real implementation, this would show a UI dialog or prompt.

        Args:
            action: ActionDescriptor describing the action

        Returns:
            ConfirmationChoice with user's decision
        """
        # Stub implementation - in real usage, this would show UI/prompt
        # For now, we'll deny by default for safety
        print("\n🚨 CONFIRMATION REQUIRED 🚨")
        print(f"Level: {action.level.value}")
        print(f"Operation: {action.operation}")
        print(f"Description: {action.description}")
        if action.target:
            print(f"Target: {action.target}")
        if action.details:
            print(f"Details: {action.details}")

        # In a real implementation, this would be replaced with actual user input
        # For testing purposes, we'll return a denial
        return ConfirmationChoice(approved=False)
