"""CLI entry points for Agent Runner.

Implements click-based CLI
"""

import asyncio
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import click
import pyfiglet
from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from agentrunner.core.agent import AgentRunnerAgent
from agentrunner.core.cli_session import CLISession
from agentrunner.core.config import load_config
from agentrunner.core.events import StreamEvent
from agentrunner.core.exceptions import AgentRunnerException, ConfigurationError
from agentrunner.core.factory import create_agent
from agentrunner.core.messages import MessageHistory
from agentrunner.core.session import SessionManager
from agentrunner.core.workspace import Workspace
from agentrunner.providers.base import ProviderConfig

# Load .env file from current directory or parent directories
load_dotenv()

console = Console()


def print_logo() -> None:
    """Print Agent Runner logo with ASCII art icon and text."""
    # Agent Runner logo icon
    logo_icon = """
 ...   ..            .    ... 
    .:*@+ .       . .%%=..    
 :=. %@%- .        ..*@@= -=. 
.%@@-=#+=. .      . :+**:*@@= 
 =%@@@@@*..       . :@@@@@@#. 
:---=+*+. .        . -**+=---.
=@@#-#@@- ..      .. %@%==@@@.
.+%@@@%*-   .   ..  .=#@@@%#- 
 .:--.=%@-..      ..%@#::--:  
.-@@@#@@@=%%+...:#@+*@@%%@@*..
. -*#%*=::@@@:  *@@*.-+#%#+...
 .    -#@@@@@-..+@@@@%+.      
  ....:+##*=+@@@@++##*=.....  
          .##+--+##.          
      .......    ......       
"""

    # Print icon with cyan color
    console.print(logo_icon, style="bold cyan")

    # Generate AGENT RUNNER text with pyfiglet
    logo_text = pyfiglet.figlet_format("AGENT RUNNER", font="small")

    # Print text with cyan color
    console.print(logo_text, style="bold cyan")


def get_default_model() -> str:
    """Get default model from environment or hardcoded default."""
    return os.getenv("AGENTRUNNER_MODEL", "gpt-5-codex")


def get_default_temperature() -> float:
    """Get default temperature from environment or hardcoded default."""
    temp_str = os.getenv("AGENTRUNNER_TEMPERATURE", "0.7")
    try:
        return float(temp_str)
    except ValueError:
        console.print(
            f"[yellow]Warning: Invalid AGENTRUNNER_TEMPERATURE '{temp_str}', using 0.7[/yellow]"
        )
        return 0.7


def get_default_max_tokens() -> int | None:
    """Get default max_tokens from environment or None."""
    max_tokens_str = os.getenv("AGENTRUNNER_MAX_TOKENS")
    if max_tokens_str:
        try:
            return int(max_tokens_str)
        except ValueError:
            console.print(
                f"[yellow]Warning: Invalid AGENTRUNNER_MAX_TOKENS '{max_tokens_str}', ignoring[/yellow]"
            )
    return None


@dataclass
class SessionMetrics:
    """Tracks metrics for a CLI session."""

    total_messages: int = 0
    total_rounds: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_tokens: int = 0
    total_cost: float = 0.0
    response_times: list[float] = field(default_factory=list)


def subscribe_cli_session_to_events(
    agent: AgentRunnerAgent, cli_session: CLISession, verbose: bool = False
) -> None:
    """Subscribe CLI session to agent's EventBus for event persistence.

    This enables full execution trace in CLI session history,
    including tool calls, outputs, file changes, etc.

    Args:
        agent: AgentRunnerAgent instance with EventBus
        cli_session: CLISession to persist events to
        verbose: If True, display events in real-time to console
    """

    def event_handler(event: StreamEvent) -> None:
        """Save event to CLI session and optionally display."""
        try:
            cli_session.save_event(event)

            # Display events in verbose mode
            if verbose:
                display_event(event)
        except Exception:
            pass  # Don't break execution if event logging fails

    if agent.event_bus:
        agent.event_bus.subscribe(event_handler)


def display_event(event: StreamEvent) -> None:
    """Display event in real-time to console (verbose mode).
    Args:
        event: StreamEvent to display
    """
    event_type = event.type
    data = event.data

    if event_type == "user_message":
        content = data.get("content", "")
        console.print()  # Blank line for spacing
        console.print(f"[bold white]→[/bold white] {content}")

    elif event_type == "status_update":
        status = data.get("status", "")
        detail = data.get("detail", "")

        if status == "thinking":
            console.print(f"[dim cyan]  ○ {detail}...[/dim cyan]")
        elif status == "executing_tools":
            num_tools = detail.split()[1] if "tool" in detail else "?"
            console.print(f"[cyan]  ◆ Executing {num_tools} tool(s)...[/cyan]")
        elif status == "idle":
            # Skip "Completed" status in verbose mode - the summary box shows it
            pass

    elif event_type == "tool_call_started":
        tool_name = data.get("tool_name", "unknown")
        console.print(f"[cyan]    ▸ {tool_name}[/cyan]", end="")

        # Show key arguments inline
        args = data.get("arguments", {})
        if args:
            # Show most relevant arg (usually path or command)
            key_args = []
            for key in ["file_path", "path", "command", "query", "name"]:
                if key in args:
                    val = args[key]
                    if isinstance(val, str) and len(val) < 50:
                        key_args.append(f"[dim]{val}[/dim]")
                        break

            if key_args:
                console.print(f" [dim]({key_args[0]})[/dim]")
            else:
                console.print()
        else:
            console.print()

    elif event_type == "tool_call_completed":
        tool_name = data.get("tool_name", "unknown")
        success = data.get("success", False)
        duration_ms = data.get("duration_ms", 0)

        if success:
            # Format timing
            if duration_ms > 0:
                if duration_ms < 1000:
                    time_str = f"[dim]{duration_ms}ms[/dim]"
                else:
                    time_str = f"[dim]{duration_ms/1000:.1f}s[/dim]"
            else:
                time_str = ""

            console.print(f"[green]    ✓ {tool_name}[/green] {time_str}")

            # Show compact output preview
            output = data.get("output", "")
            if output and isinstance(output, str):
                # Clean and truncate output
                output_clean = output.strip()
                if len(output_clean) > 0 and len(output_clean) < 100:
                    # For short outputs, show first line
                    first_line = output_clean.split("\n")[0]
                    if len(first_line) < 80:
                        console.print(f"[dim]      → {first_line}[/dim]")
        else:
            error = data.get("output", "Unknown error")
            console.print(f"[red]    ✗ {tool_name}[/red]")
            # Show error on next line, truncated
            error_msg = str(error)[:200]
            console.print(f"[dim red]      {error_msg}[/dim red]")

    elif event_type == "assistant_message":
        content = data.get("content", "")

        # Style matches tool execution for consistency
        console.print()
        console.print("[cyan]  ◆ Assistant Response[/cyan]")

        # Content indented like tool output (with → prefix for first line)
        lines = content.split("\n")
        if lines and lines[0].strip():
            console.print(f"[dim]    → {lines[0]}[/dim]")
            for line in lines[1:]:
                if line.strip():
                    console.print(f"[dim]      {line}[/dim]")
                else:
                    console.print()
        else:
            for line in lines:
                if line.strip():
                    console.print(f"[dim]    → {line}[/dim]")
                else:
                    console.print()

    elif event_type == "files_changed":
        files = data.get("files", [])
        if files:
            console.print(f"[yellow]  ◆ Modified {len(files)} file(s)[/yellow]")
            for file_path in files[:5]:  # Show first 5
                # Show relative path if it's long
                display_path = file_path
                if len(file_path) > 60:
                    display_path = "..." + file_path[-57:]
                console.print(f"[dim]    • {display_path}[/dim]")
            if len(files) > 5:
                console.print(f"[dim]    • ... {len(files) - 5} more[/dim]")


@click.group(invoke_without_command=True)
@click.version_option(version="0.0.1-alpha", prog_name="agentrunner")
@click.pass_context
def cli(ctx: click.Context) -> None:
    """Design Arena Agent Runner.

    Make any model into an autonomous coding agent. Configure agent behavior, tools, and constraints however you need.
    """
    # Show logo if no subcommand provided
    if ctx.invoked_subcommand is None:
        print_logo()
        console.print("[bold]Available Commands:[/bold]\n")
        console.print("  [cyan]agentrunner chat[/cyan]     - Start interactive chat session")
        console.print(
            "  [cyan]agentrunner run[/cyan]      - Run agent with specification (autonomous)"
        )
        console.print("  [cyan]agentrunner review[/cyan]   - Review code (read-only mode)")
        console.print("  [cyan]agentrunner sessions[/cyan] - Manage CLI session history")
        console.print("  [cyan]agentrunner config[/cyan]   - Manage configuration profiles")
        console.print("  [cyan]agentrunner session[/cyan]  - Manage agent sessions\n")
        console.print("[dim]Run 'agentrunner --help' for more information[/dim]\n")


@cli.command()
@click.argument("path", required=False, default=".")
@click.option("--workspace", "-w", default=".", help="Workspace directory")
@click.option("--profile", "-p", default="default", help="Configuration profile")
@click.option(
    "--model",
    "-m",
    default=None,
    help="Model ID from registry (default: from AGENTRUNNER_MODEL or gpt-5-codex)",
)
@click.option(
    "--temperature",
    "-t",
    type=float,
    default=None,
    help="Sampling temperature (default: from AGENTRUNNER_TEMPERATURE or 0.7)",
)
def review(
    path: str, workspace: str, profile: str, model: str | None, temperature: float | None
) -> None:
    """Review code with read-only tools (no modifications).

    Examples:
        agentrunner review
        agentrunner review src/api.py
        agentrunner review --workspace ./my-project
        agentrunner review --model claude-3-5-sonnet-20241022
    """
    from pathlib import Path

    from agentrunner.core.config import load_config

    # Use environment defaults if not specified
    if model is None:
        model = get_default_model()
    if temperature is None:
        temperature = get_default_temperature()

    try:
        # Load agent config (orchestration settings)
        agent_config = load_config(profile, Path(workspace))

        # Create provider config
        provider_config = ProviderConfig(
            model=model,
            temperature=temperature,
        )

        agent = create_agent(
            workspace_path=workspace,
            provider_config=provider_config,
            agent_config=agent_config,
            profile=profile,
            strict_commands=True,  # Review mode is always strict
            require_confirmation=True,  # Review mode always requires confirmation
        )

        # Remove write tools for read-only mode
        if agent.tools:
            write_tools = [
                "create_file",
                "write_file",
                "edit_file",
                "multi_edit",
                "insert_lines",
                "delete_file",
                "batch_create_files",
                "bash",
            ]
            for tool_name in write_tools:
                if agent.tools.has(tool_name):
                    agent.tools._tools.pop(tool_name, None)

        spec = f"Review the code at {path}. Analyze for issues, suggest improvements, check for bugs and security issues."

        console.print(Panel(f"[bold cyan]Reviewing: {path}[/bold cyan]", expand=False))
        console.print(f"[dim]Workspace: {workspace}[/dim]")
        console.print(f"[dim]Model: {model}[/dim]")
        console.print("[dim]Mode: Read-only (no modifications)[/dim]\n")

        result = asyncio.run(agent.process_message(spec))

        console.print(Panel("[bold green]✓ Review Complete[/bold green]", expand=False))

        if result.content:
            console.print(Markdown(result.content))

        console.print(f"\n[dim]Rounds: {result.rounds} | Tokens: {result.total_tokens:,}[/dim]")

    except ConfigurationError as e:
        console.print(f"[bold red]Configuration error:[/bold red] {e}")
        sys.exit(1)
    except AgentRunnerException as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        sys.exit(1)


@cli.command()
@click.argument("spec", required=False)
@click.option("--workspace", "-w", default=".", help="Workspace directory")
@click.option("--profile", "-p", default="default", help="Configuration profile")
@click.option(
    "--model",
    "-m",
    default=None,
    help="Model ID from registry (default: from AGENTRUNNER_MODEL or gpt-5-codex)",
)
@click.option(
    "--temperature",
    "-t",
    type=float,
    default=None,
    help="Sampling temperature (default: from AGENTRUNNER_TEMPERATURE or 0.7)",
)
@click.option("--max-rounds", type=int, help="Override max rounds from config")
@click.option("--verbose", "-v", is_flag=True, help="Show tool calls and events in real-time")
@click.option(
    "--allow-any-command",
    is_flag=True,
    help="Disable strict command validation (security: allow any bash command)",
)
@click.option(
    "--no-confirm",
    is_flag=True,
    help="Skip confirmation prompts (security: auto-approve all operations)",
)
def run(
    spec: str | None,
    workspace: str,
    profile: str,
    model: str | None,
    temperature: float | None,
    max_rounds: int | None,
    verbose: bool,
    allow_any_command: bool,
    no_confirm: bool,
) -> None:
    """Run agent with specification (autonomous mode).

    Examples:
        agentrunner run "Build a REST API for todos"
        agentrunner run "Add authentication" --workspace ./my-project
        agentrunner run --model claude-3-5-sonnet-20241022
    """
    from pathlib import Path

    from agentrunner.core.cli_session import CLISession
    from agentrunner.core.config import load_config
    from agentrunner.core.messages import Message

    # Use environment defaults if not specified
    if model is None:
        model = get_default_model()
    if temperature is None:
        temperature = get_default_temperature()

    try:
        # Load agent config (orchestration settings)
        agent_config = load_config(profile, Path(workspace))
        if max_rounds:
            agent_config.max_rounds = max_rounds

        # Create provider config
        provider_config = ProviderConfig(
            model=model,
            temperature=temperature,
        )

        agent = create_agent(
            workspace_path=workspace,
            provider_config=provider_config,
            agent_config=agent_config,
            profile=profile,
            strict_commands=not allow_any_command,
            require_confirmation=not no_confirm,
        )

        # Initialize CLI session for history persistence
        cli_session = CLISession(workspace_root=workspace)

        # Subscribe to events for full execution trace
        subscribe_cli_session_to_events(agent, cli_session, verbose=verbose)

        if not spec:
            console.print(
                "[yellow]No specification provided. Use 'agentrunner chat' for interactive mode.[/yellow]"
            )
            return

        # Display logo
        print_logo()

        console.print(Panel(f"[bold cyan]Running agent: {spec}[/bold cyan]", expand=False))
        console.print(f"[dim]Workspace: {workspace}[/dim]")
        console.print(f"[dim]Model: {model}[/dim]")
        console.print(f"[dim]Max rounds: {agent.config.max_rounds}[/dim]")
        console.print(f"[dim]Session ID: {cli_session.session_id}[/dim]\n")

        result = asyncio.run(agent.process_message(spec))

        # Persist user message and agent response to session history
        import uuid

        user_msg = Message(id=str(uuid.uuid4()), role="user", content=spec)
        assistant_msg = Message(
            id=str(uuid.uuid4()), role="assistant", content=result.content or ""
        )
        cli_session.save_message(user_msg)
        cli_session.save_message(assistant_msg)

        # Clean completion summary
        console.print()
        console.print("[bold green]✓ Task Complete[/bold green]")
        console.print(f"[dim]├─ Rounds: {result.rounds}[/dim]")
        console.print(f"[dim]├─ Tokens: {result.total_tokens:,}[/dim]")
        console.print(f"[dim]└─ Session: {cli_session.session_file}[/dim]")
        console.print()

    except ConfigurationError as e:
        console.print(f"[bold red]Configuration error:[/bold red] {e}")
        sys.exit(1)
    except AgentRunnerException as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        sys.exit(1)


@cli.command()
@click.option("--workspace", "-w", default=".", help="Workspace directory")
@click.option("--profile", "-p", default="default", help="Configuration profile")
@click.option(
    "--model",
    "-m",
    default=None,
    help="Model ID from registry (default: from AGENTRUNNER_MODEL or gpt-5-codex)",
)
@click.option(
    "--temperature",
    "-t",
    type=float,
    default=None,
    help="Sampling temperature (default: from AGENTRUNNER_TEMPERATURE or 0.7)",
)
@click.option("--verbose", "-v", is_flag=True, help="Show tool calls and events in real-time")
@click.option(
    "--allow-any-command",
    is_flag=True,
    help="Disable strict command validation (security: allow any bash command)",
)
@click.option(
    "--no-confirm",
    is_flag=True,
    help="Skip confirmation prompts (security: auto-approve all operations)",
)
def chat(
    workspace: str,
    profile: str,
    model: str | None,
    temperature: float | None,
    verbose: bool,
    allow_any_command: bool,
    no_confirm: bool,
) -> None:
    """Start interactive chat session.

    Examples:
        agentrunner chat
        agentrunner chat --workspace ./my-project
        agentrunner chat --model claude-3-5-sonnet-20241022
    """
    import time
    from pathlib import Path

    from agentrunner.core.cli_session import CLISession
    from agentrunner.core.config import load_config

    # Use environment defaults if not specified
    if model is None:
        model = get_default_model()
    if temperature is None:
        temperature = get_default_temperature()

    try:
        # Load agent config (orchestration settings)
        agent_config = load_config(profile, Path(workspace))

        # Create provider config
        provider_config = ProviderConfig(
            model=model,
            temperature=temperature,
        )

        agent = create_agent(
            workspace_path=workspace,
            provider_config=provider_config,
            agent_config=agent_config,
            profile=profile,
            strict_commands=not allow_any_command,
            require_confirmation=not no_confirm,
        )

        # Initialize CLI session for history persistence
        cli_session = CLISession(workspace_root=workspace)

        # Subscribe to events for full execution trace
        subscribe_cli_session_to_events(agent, cli_session, verbose=verbose)

        # Session metrics
        session_metrics = SessionMetrics()

        # Get pricing from provider
        model_info = agent.provider.get_model_info()
        input_cost_per_1k = model_info.pricing.get("input_per_1k", 0)
        output_cost_per_1k = model_info.pricing.get("output_per_1k", 0)

        # Display logo
        print_logo()

        console.print(
            Panel(
                "[bold cyan]Interactive Session[/bold cyan]\n\n"
                f"Model: {model}\n"
                f"Workspace: {workspace}\n"
                f"Profile: {profile}\n"
                f"Session ID: {cli_session.session_id}\n"
                f"History: {cli_session.session_file}\n\n"
                "[dim]Type 'exit' or 'quit' to end session[/dim]",
                expand=False,
            )
        )

        while True:
            try:
                user_input = console.input("\n[bold green]You:[/bold green] ")

                if not user_input.strip():
                    continue

                if user_input.strip().lower() in ("exit", "quit", "q"):
                    # Print session summary
                    if session_metrics.total_messages > 0:
                        avg_time = sum(session_metrics.response_times) / len(
                            session_metrics.response_times
                        )
                        console.print("\n[bold]Session Summary:[/bold]")
                        console.print(f"Messages: {session_metrics.total_messages}")
                        console.print(f"Total Rounds: {session_metrics.total_rounds}")
                        console.print(f"Input Tokens: {session_metrics.total_input_tokens:,}")
                        console.print(f"Output Tokens: {session_metrics.total_output_tokens:,}")
                        console.print(f"Total Tokens: {session_metrics.total_tokens:,}")
                        console.print(f"Total Cost: ${session_metrics.total_cost:.4f}")
                        console.print(f"Avg Response Time: {avg_time:.2f}s")
                    console.print("[dim]Ending session...[/dim]")
                    break

                # Time the response
                start_time = time.time()
                result = asyncio.run(agent.process_message(user_input))
                elapsed_time = time.time() - start_time

                # Persist user message and agent response to session history
                import uuid

                from agentrunner.core.messages import Message

                user_msg = Message(id=str(uuid.uuid4()), role="user", content=user_input)
                assistant_msg = Message(
                    id=str(uuid.uuid4()), role="assistant", content=result.content or ""
                )
                cli_session.save_message(user_msg)
                cli_session.save_message(assistant_msg)

                # Use ACTUAL token values from provider response
                input_tokens_msg = result.input_tokens
                output_tokens_msg = result.output_tokens

                # Calculate cost for this message using actual provider pricing
                msg_cost = (input_tokens_msg / 1000) * input_cost_per_1k + (
                    output_tokens_msg / 1000
                ) * output_cost_per_1k

                # Update session metrics
                session_metrics.total_messages += 1
                session_metrics.total_rounds += result.rounds
                session_metrics.total_input_tokens += input_tokens_msg
                session_metrics.total_output_tokens += output_tokens_msg
                session_metrics.total_tokens += result.total_tokens
                session_metrics.total_cost += msg_cost
                session_metrics.response_times.append(elapsed_time)

                console.print("\n[bold cyan]Agent:[/bold cyan]")
                if result.content:
                    console.print(Markdown(result.content))

                # Show detailed metrics
                console.print(
                    f"\n[dim]This: {result.rounds} rounds | "
                    f"In: {input_tokens_msg:,} | Out: {output_tokens_msg:,} | "
                    f"${msg_cost:.4f} | {elapsed_time:.2f}s[/dim]"
                )
                console.print(
                    f"[dim]Session: {session_metrics.total_messages} msgs | "
                    f"{session_metrics.total_rounds} rounds | "
                    f"{session_metrics.total_tokens:,} tokens | "
                    f"${session_metrics.total_cost:.4f} | "
                    f"avg {sum(session_metrics.response_times)/len(session_metrics.response_times):.1f}s[/dim]"
                )

            except KeyboardInterrupt:
                console.print("\n[dim]Interrupted. Type 'exit' to quit.[/dim]")
                continue

    except ConfigurationError as e:
        console.print(f"[bold red]Configuration error:[/bold red] {e}")
        sys.exit(1)
    except AgentRunnerException as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        sys.exit(1)


@cli.group()
def sessions() -> None:
    """Manage CLI session history."""
    pass


@sessions.command("list")
def sessions_list() -> None:
    """List all saved CLI sessions."""
    from agentrunner.core.cli_session import CLISession

    sessions = CLISession.list_sessions()

    if not sessions:
        console.print("[yellow]No saved sessions found[/yellow]")
        return

    console.print("[bold]Saved Sessions:[/bold]\n")
    for session in sessions:
        console.print(f"  • [cyan]{session['session_id']}[/cyan]")
        console.print(f"    Created: {session.get('created_at', 'Unknown')}")
        console.print(f"    Workspace: {session.get('workspace_root', 'Unknown')}")
        console.print(
            f"    Messages: {session.get('message_count', 0)} | Events: {session.get('event_count', 0)}"
        )
        console.print()


@sessions.command("show")
@click.argument("session_id")
@click.option("--events", "-e", is_flag=True, help="Show detailed event trace")
@click.option(
    "--event-type", "-t", multiple=True, help="Filter by event type (can specify multiple)"
)
def sessions_show(session_id: str, events: bool, event_type: tuple[str, ...]) -> None:
    """Show session history with optional event details.

    Args:
        session_id: Session ID to display
        events: Show detailed event trace (tool calls, outputs, etc.)
        event_type: Filter events by type (e.g., tool_call_started, bash_executed)

    Examples:
        agentrunner sessions show cli_abc123
        agentrunner sessions show cli_abc123 --events
        agentrunner sessions show cli_abc123 --events --event-type bash_executed
        agentrunner sessions show cli_abc123 -e -t tool_call_started -t tool_call_completed
    """
    from agentrunner.core.cli_session import CLISession

    session = CLISession(session_id=session_id)
    metadata = session.get_metadata()

    if not metadata:
        console.print(f"[bold red]Session not found:[/bold red] {session_id}")
        return

    messages = session.load_messages()
    event_filter = list(event_type) if event_type else None
    session_events = session.load_events(event_types=event_filter) if events else []

    console.print(f"[bold]Session: {session_id}[/bold]\n")
    console.print(f"Created: {metadata.get('created_at', 'Unknown')}")
    console.print(f"Workspace: {metadata.get('workspace_root', 'Unknown')}")
    console.print(f"Messages: {len(messages)}")
    if events:
        console.print(f"Events: {len(session_events)}")
    console.print()

    if events and session_events:
        # Show detailed event trace
        console.print("[bold]Event Trace:[/bold]\n")

        for event in session_events:
            event_type_str = event.get("type", "unknown")
            timestamp = event.get("timestamp", "")
            data = event.get("data", {})

            # Color code by event type
            if event_type_str.startswith("tool_call"):
                color = "yellow"
            elif event_type_str.startswith("bash"):
                color = "magenta"
            elif event_type_str.startswith("file"):
                color = "blue"
            elif "error" in event_type_str:
                color = "red"
            else:
                color = "dim"

            console.print(f"[{color}]▶ {event_type_str}[/{color}] [dim]{timestamp}[/dim]")

            # Show relevant data based on event type
            if event_type_str == "tool_call_started":
                console.print(f"  Tool: {data.get('name', 'unknown')}")
                if data.get("arguments"):
                    console.print(f"  Args: {str(data['arguments'])[:100]}...")
            elif event_type_str == "bash_executed":
                console.print(f"  Command: {data.get('command', 'unknown')[:80]}...")
                console.print(f"  Exit Code: {data.get('exit_code', '?')}")
            elif event_type_str == "file_created" or event_type_str == "file_modified":
                console.print(f"  Path: {data.get('path', 'unknown')}")
            elif event_type_str == "usage_update":
                console.print(
                    f"  Tokens: {data.get('input_tokens', 0)} in / {data.get('output_tokens', 0)} out"
                )

            console.print()
    else:
        # Show messages only
        for msg in messages:
            role_color = "green" if msg.role == "user" else "cyan"
            console.print(f"[bold {role_color}]{msg.role.upper()}:[/bold {role_color}]")
            if msg.content:
                console.print(msg.content[:500] + ("..." if len(msg.content) > 500 else ""))
            console.print()


@sessions.command("delete")
@click.argument("session_id")
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation")
def sessions_delete(session_id: str, yes: bool) -> None:
    """Delete a session.

    Args:
        session_id: Session ID to delete
        yes: Skip confirmation prompt
    """
    from agentrunner.core.cli_session import CLISession

    if not yes:
        confirm = console.input(f"[yellow]Delete session {session_id}? (y/N):[/yellow] ")
        if confirm.lower() != "y":
            console.print("[dim]Cancelled[/dim]")
            return

    if CLISession.delete_session(session_id):
        console.print(f"[green]✓ Deleted session: {session_id}[/green]")
    else:
        console.print(f"[bold red]Session not found:[/bold red] {session_id}")


@cli.command()
def models() -> None:
    """List all available models and their details.

    Shows all models registered in the model registry with their
    providers, context windows, and pricing information.

    Examples:
        agentrunner models
    """
    from rich.table import Table

    from agentrunner.providers.registry import ModelRegistry, ModelSpec

    console.print("\n[bold]Available Models[/bold]\n")

    # Group models by provider
    providers: dict[str, list[ModelSpec]] = {}
    for model in ModelRegistry.list_models():
        if model.provider_name not in providers:
            providers[model.provider_name] = []
        providers[model.provider_name].append(model)

    # Display each provider's models
    for provider_name in sorted(providers.keys()):
        console.print(f"[bold cyan]{provider_name.upper()}[/bold cyan]")

        table = Table(show_header=True, header_style="bold", box=None, padding=(0, 2))
        table.add_column("Model ID", style="green")
        table.add_column("Display Name")
        table.add_column("Context", justify="right")
        table.add_column("Input $/1K", justify="right")
        table.add_column("Output $/1K", justify="right")

        for model in sorted(providers[provider_name], key=lambda m: m.model_id):
            table.add_row(
                model.model_id,
                model.display_name,
                f"{model.context_window:,}",
                f"${model.input_cost_per_1k:.4f}",
                f"${model.output_cost_per_1k:.4f}",
            )

        console.print(table)
        console.print()


@cli.group()
def config() -> None:
    """Manage configuration profiles."""
    pass


@config.command("list")
def config_list() -> None:
    """List available configuration profiles."""
    profiles_dir = Path.home() / ".agentrunner" / "profiles"

    if not profiles_dir.exists():
        console.print("[yellow]No profiles directory found[/yellow]")
        console.print(f"[dim]Create profiles in: {profiles_dir}[/dim]")
        return

    profiles = list(profiles_dir.glob("*.json"))

    if not profiles:
        console.print("[yellow]No profiles found[/yellow]")
        return

    console.print("[bold]Available Profiles:[/bold]\n")
    for profile_file in sorted(profiles):
        name = profile_file.stem
        console.print(f"  • {name}")


@config.command("show")
@click.argument("profile_name", default="default")
def config_show(profile_name: str) -> None:
    """Show configuration profile details.

    Args:
        profile_name: Profile name to display
    """
    try:
        config_data = load_config(profile_name)
        console.print(f"[bold]Profile: {profile_name}[/bold]\n")
        console.print(f"Max Rounds: {config_data.max_rounds}")
        console.print(f"Tool Timeout: {config_data.tool_timeout_s}s")
        console.print(f"Response Buffer: {config_data.response_buffer_tokens} tokens")
        console.print(f"Streaming: {config_data.allow_streaming}")
    except ConfigurationError as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        sys.exit(1)


@config.command("set-default")
@click.option("--model", "-m", help="Set default model")
@click.option("--temperature", "-t", type=float, help="Set default temperature")
@click.option("--max-tokens", type=int, help="Set default max tokens")
def config_set_default(
    model: str | None, temperature: float | None, max_tokens: int | None
) -> None:
    """Set default environment variables in .env file.

    Creates or updates .env file in current directory with AGENTRUNNER_* variables.

    Examples:
        agentrunner config set-default --model claude-sonnet-4-5-20250929
        agentrunner config set-default --temperature 0.8
        agentrunner config set-default --model gpt-5-codex --temperature 0.7
    """
    from pathlib import Path

    if not any([model, temperature, max_tokens]):
        console.print(
            "[yellow]No options specified. Use --model, --temperature, or --max-tokens[/yellow]"
        )
        console.print(
            "\nExample: agentrunner config set-default --model claude-sonnet-4-5-20250929"
        )
        return

    env_file = Path.cwd() / ".env"

    # Read existing .env content
    existing_lines = []
    if env_file.exists():
        with env_file.open("r") as f:
            existing_lines = f.readlines()

    # Track which variables we're setting
    variables_to_set = {}
    if model:
        variables_to_set["AGENTRUNNER_MODEL"] = model
    if temperature is not None:
        variables_to_set["AGENTRUNNER_TEMPERATURE"] = str(temperature)
    if max_tokens is not None:
        variables_to_set["AGENTRUNNER_MAX_TOKENS"] = str(max_tokens)

    # Update or add variables
    new_lines = []
    updated_vars = set()

    for line in existing_lines:
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            for var_name in variables_to_set:
                if stripped.startswith(f"{var_name}="):
                    new_lines.append(f"{var_name}={variables_to_set[var_name]}\n")
                    updated_vars.add(var_name)
                    break
            else:
                new_lines.append(line)
        else:
            new_lines.append(line)

    # Add new variables that weren't updated
    for var_name, var_value in variables_to_set.items():
        if var_name not in updated_vars:
            new_lines.append(f"{var_name}={var_value}\n")

    # Write back to file
    with env_file.open("w") as f:
        f.writelines(new_lines)

    console.print(f"[green]✓ Updated {env_file}[/green]\n")
    for var_name, var_value in variables_to_set.items():
        console.print(f"  {var_name}={var_value}")
    console.print(
        "\n[dim]Note: Restart your shell or run 'source .env' for changes to take effect[/dim]"
    )


@cli.group()
def session() -> None:
    """Manage agent sessions."""
    pass


@session.command("list")
@click.option("--workspace", "-w", default=".", help="Workspace directory")
def session_list(workspace: str) -> None:
    """List all saved sessions."""
    ws = Workspace(workspace)
    manager = SessionManager(ws)

    sessions = asyncio.run(manager.list())

    if not sessions:
        console.print("[yellow]No saved sessions[/yellow]")
        return

    console.print("[bold]Saved Sessions:[/bold]\n")
    for sess in sessions:
        console.print(f"  • {sess['session_id']}")
        console.print(f"    [dim]Created: {sess.get('created_at', 'Unknown')}[/dim]")
        console.print(f"    [dim]Messages: {sess.get('message_count', 0)}[/dim]")
        console.print()


@session.command("load")
@click.argument("session_id")
@click.option("--workspace", "-w", default=".", help="Workspace directory")
@click.option("--profile", "-p", default="default", help="Configuration profile")
@click.option("--model", "-m", help="Model to use (e.g., gpt-4-turbo)")
@click.option("--temperature", "-t", default=0.7, help="Sampling temperature")
def session_load(
    session_id: str, workspace: str, profile: str, model: str | None, temperature: float
) -> None:
    """Load and resume a saved session.

    Args:
        session_id: Session ID to load
        workspace: Workspace directory
        profile: Configuration profile
        model: Model to use
        temperature: Sampling temperature
    """
    try:
        # Load the session first to check what model was used
        ws = Workspace(workspace)
        manager = SessionManager(ws)
        messages, saved_config, meta = asyncio.run(manager.load(session_id))

        # Use the model from CLI or default to a reasonable one
        model_to_use = model or "gpt-4-turbo"

        # Create provider config
        provider_config = ProviderConfig(
            model=model_to_use,
            temperature=temperature,
        )

        # Create agent
        agent_config = load_config(profile)
        agent = create_agent(
            workspace_path=workspace,
            provider_config=provider_config,
            agent_config=agent_config,
            profile=profile,
            strict_commands=True,  # Session load is always strict
            require_confirmation=True,  # Session load always requires confirmation
        )

        agent.history = MessageHistory()
        for msg in messages:
            agent.history.add(msg)

        # Get model info from agent provider
        model_name = agent.provider.config.model
        console.print(
            Panel(
                f"[bold green]Session Loaded:[/bold green] {session_id}\n"
                f"Messages: {len(messages)}\n"
                f"Model: {model_name}",
                expand=False,
            )
        )

        while True:
            try:
                user_input = console.input("\n[bold green]You:[/bold green] ")

                if not user_input.strip():
                    continue

                if user_input.strip().lower() in ("exit", "quit", "q"):
                    save = console.input("[yellow]Save session before exiting? (y/n):[/yellow] ")
                    if save.lower() in ("y", "yes"):
                        asyncio.run(
                            manager.save(
                                session_id, agent.get_messages(), agent.config, {"rounds": 0}
                            )
                        )
                        console.print("[green]Session saved[/green]")
                    break

                result = asyncio.run(agent.process_message(user_input))

                console.print("\n[bold cyan]Agent:[/bold cyan]")
                if result.content:
                    console.print(Markdown(result.content))

                console.print(
                    f"\n[dim]Rounds: {result.rounds} | Tokens: {result.total_tokens:,}[/dim]"
                )

            except KeyboardInterrupt:
                console.print("\n[dim]Interrupted. Type 'exit' to quit.[/dim]")
                continue

    except ConfigurationError as e:
        console.print(f"[bold red]Configuration error:[/bold red] {e}")
        sys.exit(1)
    except AgentRunnerException as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        sys.exit(1)


@session.command("delete")
@click.argument("session_id")
@click.option("--workspace", "-w", default=".", help="Workspace directory")
def session_delete(session_id: str, workspace: str) -> None:
    """Delete a saved session.

    Args:
        session_id: Session ID to delete
        workspace: Workspace directory
    """
    ws = Workspace(workspace)
    manager = SessionManager(ws)

    try:
        asyncio.run(manager.delete(session_id))
        console.print(f"[green]Deleted session:[/green] {session_id}")
    except FileNotFoundError:
        console.print(f"[yellow]Session not found:[/yellow] {session_id}")
        sys.exit(1)


if __name__ == "__main__":
    cli()
