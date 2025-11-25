# Contributing to Agent Runner

Thank you for your interest in contributing! 🎉

## Getting Started

### Prerequisites

**Python Environment:**
- Python 3.11+ (project requires >=3.11)

**System Dependencies:**
- `ripgrep` - Fast search tool used by GrepSearchTool (not available via pip)
  - **macOS**: `brew install ripgrep`
  - **Ubuntu/Debian**: `sudo apt-get install ripgrep`

### Development Setup

```bash
# Clone the repository
git clone https://github.com/Design-Arena/agent-runner.git
cd agent-runner

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install in development mode with all dependencies
pip install -e ".[dev,providers]"

# Run tests
pytest

# Run tests with coverage
pytest --cov=agentrunner --cov-report=html

# Format and lint code
black src/ tests/
ruff check src/ tests/ --fix

# Type check
mypy src/
```

## Architecture Overview

```
agentrunner/
├── core/           # Agent logic, session management, config
│   ├── agent.py              # Main agent orchestration
│   ├── messages.py           # Message protocol
│   ├── tool_protocol.py      # Tool definitions
│   └── prompts/              # System prompt management
├── providers/      # LLM provider integrations
│   ├── base.py               # BaseLLMProvider interface
│   ├── anthropic_provider.py # Anthropic/Claude
│   ├── openai_provider.py    # OpenAI/GPT
│   └── ...                   # Other providers
├── tools/          # Tool implementations
│   ├── base.py               # BaseTool interface
│   ├── file_io.py            # File operations
│   ├── bash.py               # Command execution
│   └── ...                   # Other tools
├── security/       # Command validation, sandboxing
└── cli/            # Command-line interface
```

## Contributing Guidelines

### Adding a New Tool

1. Create a new file in `src/agentrunner/tools/`
2. Subclass `BaseTool` and implement required methods:
   - `execute()` - Tool execution logic
   - `get_definition()` - Return `ToolDefinition`
3. Add the tool to the default set in `BaseLLMProvider.get_tool_classes()`
4. Write comprehensive tests in `tests/unit/`
5. Update documentation

**Example:**
```python
from agentrunner.tools.base import BaseTool, ToolContext
from agentrunner.core.tool_protocol import ToolCall, ToolDefinition, ToolResult

class MyTool(BaseTool):
    async def execute(self, call: ToolCall, context: ToolContext) -> ToolResult:
        # Implementation
        return ToolResult(success=True, output="Done")
    
    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="my_tool",
            description="What the tool does",
            parameters={...}  # JSON Schema
        )
```

### Adding a New Provider

1. Create a new file in `src/agentrunner/providers/`
2. Subclass `BaseLLMProvider` and implement required methods:
   - `chat()` - Execute chat completion
   - `chat_stream()` - Streaming chat completion
   - `get_model_info()` - Return model capabilities
   - `count_tokens()` - Token counting
   - `get_system_prompt()` - Build system prompt
3. Register models in `src/agentrunner/providers/registry.py`
4. Write comprehensive tests in `tests/unit/`
5. Update documentation

### Testing

- Write unit tests for all new functionality
- Ensure existing tests pass: `pytest`
- Aim for >80% code coverage
- Test edge cases and error conditions

### Pull Request Process

1. **Fork** the repository
2. **Create a feature branch**: `git checkout -b feature/my-feature`
3. **Make your changes** with clear, atomic commits
4. **Write/update tests** to cover your changes
5. **Run tests and linting**: `pytest && black src/ tests/ && ruff check src/ tests/`
6. **Update documentation** if needed
7. **Submit PR** with clear description of changes

### Commit Message Guidelines

- Use clear, descriptive commit messages
- Start with a verb: "Add", "Fix", "Update", "Remove"
- Keep first line under 72 characters
- Add detailed description if needed

**Examples:**
```
Add batch file creation tool
Fix token counting for Gemini provider
Update README with new tool documentation
```

## Important: Tool Consistency

**All providers must use the same default tool set.** Do not create provider-specific tool overrides unless absolutely necessary for API compatibility. The default tools are defined in `BaseLLMProvider.get_tool_classes()`.

## Questions or Issues?

- **Bug reports**: Open an issue with reproduction steps
- **Feature requests**: Open an issue describing the feature
- **Questions**: Open a discussion or issue

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

