"""Core prompt system classes.

Implements modular, composable system prompt architecture per INTERFACES/PROMPTS.md.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class PromptSection:
    """A single section of the system prompt.

    Sections are composable building blocks that can be enabled, disabled,
    reordered, or customized per-project.
    """

    name: str
    content: str
    priority: int = 100
    enabled: bool = True
    conditions: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate section."""
        if not self.name:
            raise ValueError("Section name cannot be empty")
        if not isinstance(self.priority, int):
            raise TypeError(f"Priority must be int, got {type(self.priority)}")
        if self.priority < 0 or self.priority > 1000:
            raise ValueError(f"Priority must be 0-1000, got {self.priority}")

    def render(self, variables: dict[str, Any] | None = None) -> str:
        """Render section content with variable substitution.

        Args:
            variables: Variables for {key} substitution

        Returns:
            Rendered content with variables substituted
        """
        if not variables:
            return self.content

        content = self.content
        for key, value in variables.items():
            placeholder = f"{{{key}}}"
            if placeholder in content:
                content = content.replace(placeholder, str(value))

        return content

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "content": self.content,
            "priority": self.priority,
            "enabled": self.enabled,
            "conditions": self.conditions,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PromptSection":
        """Create section from dictionary."""
        return cls(
            name=data["name"],
            content=data["content"],
            priority=data.get("priority", 100),
            enabled=data.get("enabled", True),
            conditions=data.get("conditions", {}),
            metadata=data.get("metadata", {}),
        )


class SystemPromptBuilder:
    """Builds system prompts from modular sections.

    Manages section ordering and variable substitution.
    """

    def __init__(self) -> None:
        """Initialize builder."""
        self._sections: dict[str, PromptSection] = {}

    def add_section(self, section: PromptSection) -> None:
        """Add or replace a section.

        Args:
            section: Section to add
        """
        self._sections[section.name] = section

    def remove_section(self, name: str) -> None:
        """Remove a section by name.

        Args:
            name: Section name to remove
        """
        self._sections.pop(name, None)

    def get_section(self, name: str) -> PromptSection | None:
        """Get a section by name.

        Args:
            name: Section name

        Returns:
            Section or None if not found
        """
        return self._sections.get(name)

    def list_sections(self) -> list[str]:
        """List all section names.

        Returns:
            List of section names
        """
        return list(self._sections.keys())

    def enable_section(self, name: str) -> None:
        """Enable a section.

        Args:
            name: Section name

        Raises:
            KeyError: If section not found
        """
        section = self._sections.get(name)
        if section is None:
            raise KeyError(f"Section not found: {name}")
        section.enabled = True

    def disable_section(self, name: str) -> None:
        """Disable a section.

        Args:
            name: Section name

        Raises:
            KeyError: If section not found
        """
        section = self._sections.get(name)
        if section is None:
            raise KeyError(f"Section not found: {name}")
        section.enabled = False

    def build(self, variables: dict[str, Any] | None = None) -> str:
        """Build final prompt from enabled sections.

        Args:
            variables: Variables for template substitution

        Returns:
            Complete system prompt string
        """
        enabled_sections = [s for s in self._sections.values() if s.enabled]

        enabled_sections.sort(key=lambda s: s.priority)

        rendered_parts = [s.render(variables) for s in enabled_sections]

        return "\n\n".join(rendered_parts)

    def load_from_file(self, path: str) -> None:
        """Load sections from JSON file.

        Args:
            path: Path to JSON file

        Raises:
            FileNotFoundError: If file doesn't exist
            json.JSONDecodeError: If file isn't valid JSON
        """
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        sections = data.get("sections", [])
        for section_data in sections:
            section = PromptSection.from_dict(section_data)
            self.add_section(section)

    def save_to_file(self, path: str) -> None:
        """Save sections to JSON file.

        Args:
            path: Path to save JSON file
        """
        data = {
            "version": "1.0",
            "sections": [s.to_dict() for s in self._sections.values()],
        }

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def load_custom_sections(self, workspace_root: str) -> None:
        """Load custom sections from .agentrunner/prompts/custom.json if it exists.

        Args:
            workspace_root: Workspace path to look for custom sections
        """
        custom_path = Path(workspace_root) / ".agentrunner" / "prompts" / "custom.json"
        if custom_path.exists():
            self.load_from_file(str(custom_path))
