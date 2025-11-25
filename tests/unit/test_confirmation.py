"""Unit tests for confirmation service.

Tests ConfirmationService, ActionDescriptor, and related functionality.
"""

from unittest.mock import patch

from agentrunner.security.confirmation import (
    ActionDescriptor,
    ConfirmationChoice,
    ConfirmationLevel,
    ConfirmationService,
)


class TestActionDescriptor:
    """Test ActionDescriptor dataclass."""

    def test_basic_creation(self):
        """Test basic ActionDescriptor creation."""
        action = ActionDescriptor(
            level=ConfirmationLevel.FILE_OPS,
            operation="delete_file",
            description="Delete important file",
            target="/workspace/important.txt",
        )

        assert action.level == ConfirmationLevel.FILE_OPS
        assert action.operation == "delete_file"
        assert action.description == "Delete important file"
        assert action.target == "/workspace/important.txt"
        assert action.details == {}
        assert action.batch_key == "delete_file_file"

    def test_auto_batch_key_generation(self):
        """Test automatic batch key generation."""
        # File target
        action1 = ActionDescriptor(
            level=ConfirmationLevel.FILE_OPS,
            operation="delete_file",
            description="Delete file",
            target="/path/to/file.txt",
        )
        assert action1.batch_key == "delete_file_file"

        # Command target
        action2 = ActionDescriptor(
            level=ConfirmationLevel.BASH,
            operation="execute_bash",
            description="Run command",
            target="rm -rf temp",
        )
        assert action2.batch_key == "execute_bash_command"

    def test_custom_batch_key(self):
        """Test custom batch key overrides auto-generation."""
        action = ActionDescriptor(
            level=ConfirmationLevel.DESTRUCTIVE,
            operation="format_disk",
            description="Format disk",
            target="/dev/sda1",
            batch_key="custom_batch",
        )
        assert action.batch_key == "custom_batch"

    def test_with_details(self):
        """Test ActionDescriptor with details."""
        details = {"size": "10GB", "files_affected": 150}
        action = ActionDescriptor(
            level=ConfirmationLevel.FILE_OPS,
            operation="delete_directory",
            description="Delete large directory",
            target="/workspace/large_dir",
            details=details,
        )

        assert action.details == details
        assert action.details["size"] == "10GB"
        assert action.details["files_affected"] == 150


class TestConfirmationChoice:
    """Test ConfirmationChoice dataclass."""

    def test_basic_choice(self):
        """Test basic confirmation choice."""
        choice = ConfirmationChoice(approved=True)
        assert choice.approved is True
        assert choice.remember_for_session is False
        assert choice.apply_to_batch is False

    def test_choice_with_memory(self):
        """Test choice with session and batch memory."""
        choice = ConfirmationChoice(
            approved=False,
            remember_for_session=True,
            apply_to_batch=True,
        )
        assert choice.approved is False
        assert choice.remember_for_session is True
        assert choice.apply_to_batch is True


class TestConfirmationService:
    """Test ConfirmationService class."""

    def test_basic_initialization(self):
        """Test basic service initialization."""
        service = ConfirmationService()
        assert service.auto_approve is False
        assert service.confirmation_map == ConfirmationService.DEFAULT_CONFIRMATIONS
        assert service._session_choices == {}
        assert service._batch_choices == {}

    def test_auto_approve_initialization(self):
        """Test initialization with auto-approve."""
        service = ConfirmationService(auto_approve=True)
        assert service.auto_approve is True

    def test_custom_confirmation_map(self):
        """Test initialization with custom confirmation map."""
        custom_map = {
            ConfirmationLevel.FILE_OPS: {"custom_operation"},
        }
        service = ConfirmationService(confirmation_map=custom_map)
        assert service.confirmation_map == custom_map

    def test_auto_approve_bypasses_confirmation(self):
        """Test auto-approve bypasses all confirmation checks."""
        service = ConfirmationService(auto_approve=True)
        action = ActionDescriptor(
            level=ConfirmationLevel.DESTRUCTIVE,
            operation="format_disk",
            description="Format entire disk",
        )

        result = service.approve(action)
        assert result is True

    def test_approve_operation_not_requiring_confirmation(self):
        """Test approving operation that doesn't require confirmation."""
        service = ConfirmationService()
        action = ActionDescriptor(
            level=ConfirmationLevel.FILE_OPS,
            operation="read_file",  # Not in default confirmations
            description="Read a file",
        )

        result = service.approve(action)
        assert result is True

    def test_approve_operation_requiring_confirmation_denied(self):
        """Test operation requiring confirmation that gets denied."""
        service = ConfirmationService()
        action = ActionDescriptor(
            level=ConfirmationLevel.FILE_OPS,
            operation="delete_file",
            description="Delete important file",
        )

        # Mock the _request_confirmation to return denial
        with patch.object(service, "_request_confirmation") as mock_request:
            mock_request.return_value = ConfirmationChoice(approved=False)
            result = service.approve(action)

        assert result is False
        mock_request.assert_called_once_with(action)

    def test_approve_operation_requiring_confirmation_approved(self):
        """Test operation requiring confirmation that gets approved."""
        service = ConfirmationService()
        action = ActionDescriptor(
            level=ConfirmationLevel.BASH,
            operation="execute_bash",
            description="Run bash command",
        )

        with patch.object(service, "_request_confirmation") as mock_request:
            mock_request.return_value = ConfirmationChoice(approved=True)
            result = service.approve(action)

        assert result is True
        mock_request.assert_called_once_with(action)

    def test_session_memory_approval(self):
        """Test session memory remembers approval."""
        service = ConfirmationService()
        action = ActionDescriptor(
            level=ConfirmationLevel.FILE_OPS,
            operation="delete_file",
            description="Delete file",
        )

        # First call - mock approval with session memory
        with patch.object(service, "_request_confirmation") as mock_request:
            mock_request.return_value = ConfirmationChoice(
                approved=True,
                remember_for_session=True,
            )
            result1 = service.approve(action)

        assert result1 is True
        mock_request.assert_called_once()

        # Second call - should use session memory, no prompt
        mock_request.reset_mock()
        result2 = service.approve(action)

        assert result2 is True
        mock_request.assert_not_called()  # Should not prompt again

    def test_session_memory_denial(self):
        """Test session memory remembers denial."""
        service = ConfirmationService()
        action = ActionDescriptor(
            level=ConfirmationLevel.DESTRUCTIVE,
            operation="format_disk",
            description="Format disk",
        )

        # First call - mock denial with session memory
        with patch.object(service, "_request_confirmation") as mock_request:
            mock_request.return_value = ConfirmationChoice(
                approved=False,
                remember_for_session=True,
            )
            result1 = service.approve(action)

        assert result1 is False
        mock_request.assert_called_once()

        # Second call - should use session memory
        mock_request.reset_mock()
        result2 = service.approve(action)

        assert result2 is False
        mock_request.assert_not_called()

    def test_batch_confirmation_approval(self):
        """Test batch confirmation applies to similar operations."""
        service = ConfirmationService()

        action1 = ActionDescriptor(
            level=ConfirmationLevel.FILE_OPS,
            operation="delete_file",
            description="Delete file 1",
            target="/file1.txt",
        )
        action2 = ActionDescriptor(
            level=ConfirmationLevel.FILE_OPS,
            operation="delete_file",
            description="Delete file 2",
            target="/file2.txt",
        )

        # Both should have same batch key
        assert action1.batch_key == action2.batch_key

        # First call - approve with batch confirmation
        with patch.object(service, "_request_confirmation") as mock_request:
            mock_request.return_value = ConfirmationChoice(
                approved=True,
                apply_to_batch=True,
            )
            result1 = service.approve(action1)

        assert result1 is True
        mock_request.assert_called_once()

        # Second call - should use batch approval
        mock_request.reset_mock()
        result2 = service.approve(action2)

        assert result2 is True
        mock_request.assert_not_called()

    def test_batch_confirmation_denial(self):
        """Test batch confirmation denies similar operations."""
        service = ConfirmationService()

        action1 = ActionDescriptor(
            level=ConfirmationLevel.BASH,
            operation="execute_bash",
            description="Run command 1",
            target="dangerous_command_1",
        )
        action2 = ActionDescriptor(
            level=ConfirmationLevel.BASH,
            operation="execute_bash",
            description="Run command 2",
            target="dangerous_command_2",
        )

        # First call - deny with batch confirmation
        with patch.object(service, "_request_confirmation") as mock_request:
            mock_request.return_value = ConfirmationChoice(
                approved=False,
                apply_to_batch=True,
            )
            result1 = service.approve(action1)

        assert result1 is False

        # Second call - should use batch denial
        mock_request.reset_mock()
        result2 = service.approve(action2)

        assert result2 is False
        mock_request.assert_not_called()

    def test_session_takes_precedence_over_batch(self):
        """Test session memory takes precedence over batch choices."""
        service = ConfirmationService()

        # Set up batch denial
        service._batch_choices["delete_file_file"] = False

        action = ActionDescriptor(
            level=ConfirmationLevel.FILE_OPS,
            operation="delete_file",
            description="Delete file",
            target="/test.txt",
        )

        # Set session approval (should override batch)
        session_key = service._get_session_key(action)
        service._session_choices[session_key] = True

        result = service.approve(action)
        assert result is True  # Session choice overrides batch

    def test_clear_session_choices(self):
        """Test clearing session choices."""
        service = ConfirmationService()
        service._session_choices["test_key"] = True

        assert len(service._session_choices) == 1
        service.clear_session_choices()
        assert len(service._session_choices) == 0

    def test_clear_batch_choices(self):
        """Test clearing batch choices."""
        service = ConfirmationService()
        service._batch_choices["test_batch"] = True

        assert len(service._batch_choices) == 1
        service.clear_batch_choices()
        assert len(service._batch_choices) == 0

    def test_clear_all_choices(self):
        """Test clearing all choices."""
        service = ConfirmationService()
        service._session_choices["session_key"] = True
        service._batch_choices["batch_key"] = False

        assert len(service._session_choices) == 1
        assert len(service._batch_choices) == 1

        service.clear_all_choices()

        assert len(service._session_choices) == 0
        assert len(service._batch_choices) == 0

    def test_get_session_choices(self):
        """Test getting session choices returns copy."""
        service = ConfirmationService()
        service._session_choices["test"] = True

        choices = service.get_session_choices()
        assert choices == {"test": True}

        # Modify returned dict shouldn't affect internal state
        choices["new"] = False
        assert "new" not in service._session_choices

    def test_get_batch_choices(self):
        """Test getting batch choices returns copy."""
        service = ConfirmationService()
        service._batch_choices["test"] = False

        choices = service.get_batch_choices()
        assert choices == {"test": False}

        # Modify returned dict shouldn't affect internal state
        choices["new"] = True
        assert "new" not in service._batch_choices

    def test_requires_confirmation_true(self):
        """Test _requires_confirmation returns True for configured operations."""
        service = ConfirmationService()
        action = ActionDescriptor(
            level=ConfirmationLevel.FILE_OPS,
            operation="delete_file",
            description="Delete file",
        )

        result = service._requires_confirmation(action)
        assert result is True

    def test_requires_confirmation_false(self):
        """Test _requires_confirmation returns False for unconfigured operations."""
        service = ConfirmationService()
        action = ActionDescriptor(
            level=ConfirmationLevel.FILE_OPS,
            operation="read_file",  # Not in default confirmations
            description="Read file",
        )

        result = service._requires_confirmation(action)
        assert result is False

    def test_requires_confirmation_custom_map(self):
        """Test _requires_confirmation with custom confirmation map."""
        custom_map = {
            ConfirmationLevel.FILE_OPS: {"custom_operation"},
        }
        service = ConfirmationService(confirmation_map=custom_map)

        # Default operation should not require confirmation with custom map
        action1 = ActionDescriptor(
            level=ConfirmationLevel.FILE_OPS,
            operation="delete_file",
            description="Delete file",
        )
        assert service._requires_confirmation(action1) is False

        # Custom operation should require confirmation
        action2 = ActionDescriptor(
            level=ConfirmationLevel.FILE_OPS,
            operation="custom_operation",
            description="Custom operation",
        )
        assert service._requires_confirmation(action2) is True

    def test_get_session_key(self):
        """Test session key generation."""
        service = ConfirmationService()
        action = ActionDescriptor(
            level=ConfirmationLevel.BASH,
            operation="execute_bash",
            description="Run command",
        )

        key = service._get_session_key(action)
        assert key == "bash_execute_bash"

    def test_request_confirmation_stub(self):
        """Test _request_confirmation stub implementation."""
        service = ConfirmationService()
        action = ActionDescriptor(
            level=ConfirmationLevel.DESTRUCTIVE,
            operation="format_disk",
            description="Format disk",
            target="/dev/sda1",
            details={"size": "1TB"},
        )

        # Capture stdout to verify the stub prints confirmation details
        with patch("builtins.print") as mock_print:
            choice = service._request_confirmation(action)

        # Should deny by default for safety
        assert choice.approved is False
        assert choice.remember_for_session is False
        assert choice.apply_to_batch is False

        # Verify it printed the confirmation details
        assert mock_print.call_count >= 5  # Multiple print calls for details

    def test_confirmation_levels_enum(self):
        """Test ConfirmationLevel enum values."""
        assert ConfirmationLevel.FILE_OPS.value == "file_ops"
        assert ConfirmationLevel.BASH.value == "bash"
        assert ConfirmationLevel.DESTRUCTIVE.value == "destructive"

    def test_default_confirmations_structure(self):
        """Test DEFAULT_CONFIRMATIONS has expected structure."""
        defaults = ConfirmationService.DEFAULT_CONFIRMATIONS

        # Should have all three levels
        assert ConfirmationLevel.FILE_OPS in defaults
        assert ConfirmationLevel.BASH in defaults
        assert ConfirmationLevel.DESTRUCTIVE in defaults

        # Each level should have operations
        assert len(defaults[ConfirmationLevel.FILE_OPS]) > 0
        assert len(defaults[ConfirmationLevel.BASH]) > 0
        assert len(defaults[ConfirmationLevel.DESTRUCTIVE]) > 0

        # Check some expected operations
        assert "delete_file" in defaults[ConfirmationLevel.FILE_OPS]
        assert "execute_bash" in defaults[ConfirmationLevel.BASH]
        assert "format_disk" in defaults[ConfirmationLevel.DESTRUCTIVE]
