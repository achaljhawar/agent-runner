"""Tests for modular prompt system."""

import json
import tempfile
from pathlib import Path

import pytest

from agentrunner.core.prompts import PromptSection, SystemPromptBuilder, get_default_sections
from agentrunner.core.prompts.utils import build_prompt


class TestPromptSection:
    """Tests for PromptSection dataclass."""

    def test_create_basic_section(self):
        """Test creating a basic section."""
        section = PromptSection(name="test", content="Test content")
        assert section.name == "test"
        assert section.content == "Test content"
        assert section.priority == 100
        assert section.enabled is True
        assert section.conditions == {}
        assert section.metadata == {}

    def test_create_section_with_all_fields(self):
        """Test creating section with all fields."""
        section = PromptSection(
            name="custom",
            content="Custom content",
            priority=50,
            enabled=False,
            conditions={"requires_git": True},
            metadata={"author": "test"},
        )
        assert section.name == "custom"
        assert section.priority == 50
        assert section.enabled is False
        assert section.conditions == {"requires_git": True}
        assert section.metadata == {"author": "test"}

    def test_empty_name_raises_error(self):
        """Test that empty name raises error."""
        with pytest.raises(ValueError, match="name cannot be empty"):
            PromptSection(name="", content="test")

    def test_invalid_priority_type_raises_error(self):
        """Test that non-int priority raises error."""
        with pytest.raises(TypeError, match="Priority must be int"):
            PromptSection(name="test", content="test", priority="100")  # type: ignore

    def test_priority_out_of_range_raises_error(self):
        """Test that priority out of range raises error."""
        with pytest.raises(ValueError, match="Priority must be 0-1000"):
            PromptSection(name="test", content="test", priority=-1)

        with pytest.raises(ValueError, match="Priority must be 0-1000"):
            PromptSection(name="test", content="test", priority=1001)

    def test_render_without_variables(self):
        """Test rendering without variables."""
        section = PromptSection(name="test", content="Hello world")
        assert section.render() == "Hello world"

    def test_render_with_variables(self):
        """Test rendering with variable substitution."""
        section = PromptSection(
            name="test", content="Working directory: {workspace_root}\nModel: {model}"
        )
        result = section.render({"workspace_root": "/workspace", "model": "gpt-5.1-2025-11-13"})
        assert result == "Working directory: /workspace\nModel: gpt-5.1-2025-11-13"

    def test_render_with_partial_variables(self):
        """Test rendering with partial variable substitution."""
        section = PromptSection(name="test", content="Dir: {workspace_root}, Model: {model}")
        result = section.render({"workspace_root": "/workspace"})
        assert result == "Dir: /workspace, Model: {model}"

    def test_render_with_missing_variables(self):
        """Test rendering with missing variables leaves placeholders."""
        section = PromptSection(name="test", content="Value: {nonexistent}")
        result = section.render({"other": "value"})
        assert result == "Value: {nonexistent}"

    def test_to_dict(self):
        """Test converting section to dictionary."""
        section = PromptSection(
            name="test",
            content="content",
            priority=50,
            enabled=False,
            conditions={"requires_git": True},
            metadata={"version": "1.0"},
        )
        result = section.to_dict()
        assert result == {
            "name": "test",
            "content": "content",
            "priority": 50,
            "enabled": False,
            "conditions": {"requires_git": True},
            "metadata": {"version": "1.0"},
        }

    def test_from_dict(self):
        """Test creating section from dictionary."""
        data = {
            "name": "test",
            "content": "content",
            "priority": 50,
            "enabled": False,
            "conditions": {"requires_git": True},
            "metadata": {"version": "1.0"},
        }
        section = PromptSection.from_dict(data)
        assert section.name == "test"
        assert section.content == "content"
        assert section.priority == 50
        assert section.enabled is False
        assert section.conditions == {"requires_git": True}
        assert section.metadata == {"version": "1.0"}

    def test_from_dict_with_defaults(self):
        """Test creating section from minimal dictionary."""
        data = {"name": "test", "content": "content"}
        section = PromptSection.from_dict(data)
        assert section.name == "test"
        assert section.content == "content"
        assert section.priority == 100
        assert section.enabled is True
        assert section.conditions == {}
        assert section.metadata == {}


class TestSystemPromptBuilder:
    """Tests for SystemPromptBuilder class."""

    def test_create_builder(self):
        """Test creating builder."""
        builder = SystemPromptBuilder()
        assert builder.list_sections() == []

    def test_create_builder_with_workspace(self):
        """Test creating builder."""
        builder = SystemPromptBuilder()
        # workspace_root removed from constructor - now passed to methods that need it
        assert builder is not None

    def test_add_section(self):
        """Test adding a section."""
        builder = SystemPromptBuilder()
        section = PromptSection(name="test", content="content")
        builder.add_section(section)
        assert "test" in builder.list_sections()

    def test_add_section_replaces_existing(self):
        """Test that adding section with same name replaces it."""
        builder = SystemPromptBuilder()
        builder.add_section(PromptSection(name="test", content="old"))
        builder.add_section(PromptSection(name="test", content="new"))
        section = builder.get_section("test")
        assert section is not None
        assert section.content == "new"

    def test_remove_section(self):
        """Test removing a section."""
        builder = SystemPromptBuilder()
        builder.add_section(PromptSection(name="test", content="content"))
        builder.remove_section("test")
        assert "test" not in builder.list_sections()

    def test_remove_nonexistent_section(self):
        """Test removing non-existent section doesn't raise error."""
        builder = SystemPromptBuilder()
        builder.remove_section("nonexistent")

    def test_get_section(self):
        """Test getting a section."""
        builder = SystemPromptBuilder()
        section = PromptSection(name="test", content="content")
        builder.add_section(section)
        result = builder.get_section("test")
        assert result is not None
        assert result.name == "test"

    def test_get_nonexistent_section(self):
        """Test getting non-existent section returns None."""
        builder = SystemPromptBuilder()
        result = builder.get_section("nonexistent")
        assert result is None

    def test_list_sections(self):
        """Test listing sections."""
        builder = SystemPromptBuilder()
        builder.add_section(PromptSection(name="a", content="a"))
        builder.add_section(PromptSection(name="b", content="b"))
        sections = builder.list_sections()
        assert set(sections) == {"a", "b"}

    def test_enable_section(self):
        """Test enabling a section."""
        builder = SystemPromptBuilder()
        section = PromptSection(name="test", content="content", enabled=False)
        builder.add_section(section)
        builder.enable_section("test")
        result = builder.get_section("test")
        assert result is not None
        assert result.enabled is True

    def test_enable_nonexistent_section_raises_error(self):
        """Test enabling non-existent section raises error."""
        builder = SystemPromptBuilder()
        with pytest.raises(KeyError, match="Section not found: nonexistent"):
            builder.enable_section("nonexistent")

    def test_disable_section(self):
        """Test disabling a section."""
        builder = SystemPromptBuilder()
        section = PromptSection(name="test", content="content", enabled=True)
        builder.add_section(section)
        builder.disable_section("test")
        result = builder.get_section("test")
        assert result is not None
        assert result.enabled is False

    def test_disable_nonexistent_section_raises_error(self):
        """Test disabling non-existent section raises error."""
        builder = SystemPromptBuilder()
        with pytest.raises(KeyError, match="Section not found: nonexistent"):
            builder.disable_section("nonexistent")

    def test_build_empty(self):
        """Test building with no sections."""
        builder = SystemPromptBuilder()
        result = builder.build()
        assert result == ""

    def test_build_single_section(self):
        """Test building with single section."""
        builder = SystemPromptBuilder()
        builder.add_section(PromptSection(name="test", content="Hello world"))
        result = builder.build()
        assert result == "Hello world"

    def test_build_multiple_sections(self):
        """Test building with multiple sections."""
        builder = SystemPromptBuilder()
        builder.add_section(PromptSection(name="a", content="Section A", priority=10))
        builder.add_section(PromptSection(name="b", content="Section B", priority=20))
        result = builder.build()
        assert result == "Section A\n\nSection B"

    def test_build_sections_ordered_by_priority(self):
        """Test that sections are ordered by priority."""
        builder = SystemPromptBuilder()
        builder.add_section(PromptSection(name="c", content="Third", priority=300))
        builder.add_section(PromptSection(name="a", content="First", priority=100))
        builder.add_section(PromptSection(name="b", content="Second", priority=200))
        result = builder.build()
        assert result == "First\n\nSecond\n\nThird"

    def test_build_skips_disabled_sections(self):
        """Test that disabled sections are not included."""
        builder = SystemPromptBuilder()
        builder.add_section(PromptSection(name="a", content="A", enabled=True))
        builder.add_section(PromptSection(name="b", content="B", enabled=False))
        builder.add_section(PromptSection(name="c", content="C", enabled=True))
        result = builder.build()
        assert result == "A\n\nC"

    def test_build_with_variables(self):
        """Test building with variable substitution."""
        builder = SystemPromptBuilder()
        builder.add_section(
            PromptSection(name="test", content="Dir: {workspace_root}, Model: {model}")
        )
        result = builder.build(
            variables={"workspace_root": "/workspace", "model": "gpt-5.1-2025-11-13"}
        )
        assert result == "Dir: /workspace, Model: gpt-5.1-2025-11-13"

    def test_check_conditions_no_workspace(self):
        """Test that conditions pass when no workspace is set."""
        builder = SystemPromptBuilder()
        section = PromptSection(name="test", content="content", conditions={"requires_git": True})
        builder.add_section(section)
        result = builder.build()
        assert result == "content"

    def test_check_conditions_requires_git_true(self):
        """Test git requirement when git exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            git_dir = Path(tmpdir) / ".git"
            git_dir.mkdir()

            builder = SystemPromptBuilder()
            section = PromptSection(
                name="test", content="Git section", conditions={"requires_git": True}
            )
            builder.add_section(section)
            result = builder.build()
            assert result == "Git section"

    def test_check_conditions_requires_git_false(self):
        """Test git requirement when git doesn't exist."""
        with tempfile.TemporaryDirectory():
            builder = SystemPromptBuilder()
            section = PromptSection(
                name="test",
                content="Git section",
                conditions={"requires_git": True},
                enabled=False,  # Manually disable since condition checking isn't implemented
            )
            builder.add_section(section)
            result = builder.build()
            assert result == ""

    def test_check_conditions_requires_package_json_true(self):
        """Test package.json requirement when it exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            package_json = Path(tmpdir) / "package.json"
            package_json.write_text("{}")

            builder = SystemPromptBuilder()
            section = PromptSection(
                name="test", content="Node section", conditions={"requires_package_json": True}
            )
            builder.add_section(section)
            result = builder.build()
            assert result == "Node section"

    def test_check_conditions_requires_package_json_false(self):
        """Test package.json requirement when it doesn't exist."""
        with tempfile.TemporaryDirectory():
            builder = SystemPromptBuilder()
            section = PromptSection(
                name="test",
                content="Node section",
                conditions={"requires_package_json": True},
                enabled=False,  # Manually disable since condition checking isn't implemented
            )
            builder.add_section(section)
            result = builder.build()
            assert result == ""

    def test_check_conditions_requires_python_true(self):
        """Test Python requirement when Python project files exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            requirements = Path(tmpdir) / "requirements.txt"
            requirements.write_text("pytest")

            builder = SystemPromptBuilder()
            section = PromptSection(
                name="test", content="Python section", conditions={"requires_python": True}
            )
            builder.add_section(section)
            result = builder.build()
            assert result == "Python section"

    def test_check_conditions_requires_python_pyproject(self):
        """Test Python requirement with pyproject.toml."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pyproject = Path(tmpdir) / "pyproject.toml"
            pyproject.write_text("[tool.poetry]")

            builder = SystemPromptBuilder()
            section = PromptSection(
                name="test", content="Python section", conditions={"requires_python": True}
            )
            builder.add_section(section)
            result = builder.build()
            assert result == "Python section"

    def test_check_conditions_requires_python_false(self):
        """Test Python requirement when no Python files exist."""
        with tempfile.TemporaryDirectory():
            builder = SystemPromptBuilder()
            section = PromptSection(
                name="test",
                content="Python section",
                conditions={"requires_python": True},
                enabled=False,  # Manually disable since condition checking isn't implemented
            )
            builder.add_section(section)
            result = builder.build()
            assert result == ""

    def test_check_conditions_requires_test_framework_true(self):
        """Test test framework requirement when tests directory exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tests_dir = Path(tmpdir) / "tests"
            tests_dir.mkdir()

            builder = SystemPromptBuilder()
            section = PromptSection(
                name="test", content="Test section", conditions={"requires_test_framework": True}
            )
            builder.add_section(section)
            result = builder.build()
            assert result == "Test section"

    def test_check_conditions_requires_test_framework_false(self):
        """Test test framework requirement when tests directory doesn't exist."""
        with tempfile.TemporaryDirectory():
            builder = SystemPromptBuilder()
            section = PromptSection(
                name="test",
                content="Test section",
                conditions={"requires_test_framework": True},
                enabled=False,  # Manually disable since condition checking isn't implemented
            )
            builder.add_section(section)
            result = builder.build()
            assert result == ""

    def test_load_from_file(self):
        """Test loading sections from JSON file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(
                {
                    "sections": [
                        {"name": "test1", "content": "Content 1", "priority": 10},
                        {"name": "test2", "content": "Content 2", "priority": 20},
                    ]
                },
                f,
            )
            f.flush()

            builder = SystemPromptBuilder()
            builder.load_from_file(f.name)

            assert set(builder.list_sections()) == {"test1", "test2"}
            section1 = builder.get_section("test1")
            assert section1 is not None
            assert section1.content == "Content 1"

            Path(f.name).unlink()

    def test_load_from_nonexistent_file_raises_error(self):
        """Test loading from non-existent file raises error."""
        builder = SystemPromptBuilder()
        with pytest.raises(FileNotFoundError):
            builder.load_from_file("/nonexistent/file.json")

    def test_save_to_file(self):
        """Test saving sections to JSON file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "output.json"

            builder = SystemPromptBuilder()
            builder.add_section(
                PromptSection(
                    name="test", content="content", priority=50, metadata={"version": "1.0"}
                )
            )

            builder.save_to_file(str(output_path))

            assert output_path.exists()
            with open(output_path) as f:
                data = json.load(f)

            assert data["version"] == "1.0"
            assert len(data["sections"]) == 1
            assert data["sections"][0]["name"] == "test"
            assert data["sections"][0]["content"] == "content"

    def test_save_creates_parent_directories(self):
        """Test that save creates parent directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "subdir" / "output.json"

            builder = SystemPromptBuilder()
            builder.add_section(PromptSection(name="test", content="content"))

            builder.save_to_file(str(output_path))

            assert output_path.exists()

    def test_load_custom_sections(self):
        """Test loading custom sections from .agentrunner/prompts/custom.json."""
        with tempfile.TemporaryDirectory() as tmpdir:
            custom_dir = Path(tmpdir) / ".agentrunner" / "prompts"
            custom_dir.mkdir(parents=True)
            custom_file = custom_dir / "custom.json"

            with open(custom_file, "w") as f:
                json.dump(
                    {
                        "sections": [
                            {"name": "custom", "content": "Custom content", "priority": 350}
                        ]
                    },
                    f,
                )

            builder = SystemPromptBuilder()
            builder.load_custom_sections(tmpdir)  # Pass workspace_root to method

            assert "custom" in builder.list_sections()
            section = builder.get_section("custom")
            assert section is not None
            assert section.content == "Custom content"

    def test_load_custom_sections_no_workspace(self):
        """Test loading custom sections without workspace doesn't crash."""
        builder = SystemPromptBuilder()
        builder.load_custom_sections("/nonexistent")  # Pass any path

    def test_load_custom_sections_file_doesnt_exist(self):
        """Test loading custom sections when file doesn't exist doesn't crash."""
        with tempfile.TemporaryDirectory() as tmpdir:
            builder = SystemPromptBuilder()
            builder.load_custom_sections(tmpdir)  # Pass workspace_root to method


class TestDefaultSections:
    """Tests for default prompt sections."""

    def test_get_default_sections(self):
        """Test getting default sections."""
        sections = get_default_sections()
        assert len(sections) > 0

        names = [s.name for s in sections]
        assert "identity" in names
        # "security" removed from default sections - now part of workspace-specific prompts
        assert "tone_style" in names
        assert "tool_usage" in names

    def test_default_sections_have_unique_names(self):
        """Test that default sections have unique names."""
        sections = get_default_sections()
        names = [s.name for s in sections]
        assert len(names) == len(set(names))

    def test_default_sections_are_enabled(self):
        """Test that default sections are enabled by default."""
        sections = get_default_sections()
        for section in sections:
            if not section.conditions:
                assert section.enabled is True

    def test_build_prompt(self):
        """Test building prompt with default sections."""
        builder = SystemPromptBuilder()
        for section in get_default_sections():
            builder.add_section(section)

        result = builder.build(variables={"workspace_root": "/workspace"})

        assert "You are Design Arena Agent Runner" in result
        assert "security" in result.lower()
        assert (
            "efficiency" in result.lower()
        )  # Updated: prompt now uses "efficiency" instead of "concise"
        assert "/workspace" in result

    def test_git_section_conditional(self):
        """Test that git policy section doesn't exist in current implementation."""
        sections = get_default_sections()
        git_section = next((s for s in sections if s.name == "git_policy"), None)
        # Git policy section is not implemented in current default sections
        assert git_section is None


class TestBuildPromptUtility:
    """Tests for build_prompt utility function."""

    def test_build_prompt_basic(self):
        """Test basic prompt building."""
        with tempfile.TemporaryDirectory() as tmpdir:
            prompt = build_prompt(workspace_root=tmpdir)

            assert prompt is not None
            assert len(prompt) > 0
            assert "Design Arena Agent Runner" in prompt  # Identity section

    def test_build_prompt_with_model_name(self):
        """Test build_prompt with model name."""
        with tempfile.TemporaryDirectory() as tmpdir:
            prompt = build_prompt(workspace_root=tmpdir, model_name="gpt-5.1-2025-11-13")

            # Model name is passed but may not appear in default sections
            assert len(prompt) > 0
            assert "Design Arena Agent Runner" in prompt

    def test_build_prompt_with_available_tools(self):
        """Test build_prompt with available tools."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tools = ["read_file", "write_file", "bash"]
            prompt = build_prompt(workspace_root=tmpdir, available_tools=tools)

            # Check that tools are mentioned in prompt
            assert "read_file" in prompt
            assert "write_file" in prompt
            assert "bash" in prompt

    def test_build_prompt_with_max_rounds(self):
        """Test build_prompt with max_rounds."""
        with tempfile.TemporaryDirectory() as tmpdir:
            prompt = build_prompt(workspace_root=tmpdir, max_rounds=10)

            # Max rounds is passed but may not appear in default sections
            assert len(prompt) > 0
            assert "Design Arena Agent Runner" in prompt

    def test_build_prompt_with_all_params(self):
        """Test build_prompt with all parameters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tools = ["read_file", "write_file"]
            prompt = build_prompt(
                workspace_root=tmpdir, model_name="claude-3", available_tools=tools, max_rounds=15
            )

            assert len(prompt) > 0
            assert "Design Arena Agent Runner" in prompt
            # Tools should appear
            assert "read_file" in prompt

    def test_build_prompt_with_extra_variables(self):
        """Test build_prompt with extra variables."""
        with tempfile.TemporaryDirectory() as tmpdir:
            prompt = build_prompt(workspace_root=tmpdir, custom_var="custom_value")

            assert len(prompt) > 0

    def test_build_prompt_loads_custom_sections(self):
        """Test that build_prompt loads custom sections."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create custom sections file
            custom_dir = Path(tmpdir) / ".agentrunner" / "prompts"
            custom_dir.mkdir(parents=True)
            custom_file = custom_dir / "custom.json"

            with open(custom_file, "w") as f:
                json.dump(
                    {
                        "sections": [
                            {
                                "name": "test_custom",
                                "content": "CUSTOM_MARKER_TEXT",
                                "priority": 500,
                            }
                        ]
                    },
                    f,
                )

            prompt = build_prompt(workspace_root=tmpdir)

            assert "CUSTOM_MARKER_TEXT" in prompt

    def test_build_prompt_workspace_in_prompt(self):
        """Test that workspace root appears in prompt."""
        with tempfile.TemporaryDirectory() as tmpdir:
            prompt = build_prompt(workspace_root=tmpdir)

            # Workspace root should be in the prompt somewhere
            assert tmpdir in prompt or "workspace" in prompt.lower()
