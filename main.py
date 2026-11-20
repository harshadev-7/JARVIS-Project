"""
JARVIS AI — Interactive Terminal CLI
======================================
A beautiful command-line interface to talk with JARVIS.
Perfect for Phase 1 development and testing before the UI is built.

Features:
- Colored terminal output with Rich
- Multi-turn conversation with session memory
- Commands: /help, /history, /clear, /exit, /status
- Displays token usage and provider info

Usage:
  python main.py              → Start JARVIS CLI
  python main.py --help       → Show options
"""

import asyncio
import sys
import uuid
from datetime import datetime

import typer
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich import print as rprint
from rich.markdown import Markdown

from config.settings import settings
from core.brain.jarvis_brain import brain
from core.memory.memory_manager import memory_manager
from core.logger import log

# Rich console for beautiful output
console = Console()
app_cli = typer.Typer()

JARVIS_ASCII = r"""
     ██╗ █████╗ ██████╗ ██╗   ██╗██╗███████╗
     ██║██╔══██╗██╔══██╗██║   ██║██║██╔════╝
     ██║███████║██████╔╝██║   ██║██║███████╗
██   ██║██╔══██║██╔══██╗╚██╗ ██╔╝██║╚════██║
╚█████╔╝██║  ██║██║  ██║ ╚████╔╝ ██║███████║
 ╚════╝ ╚═╝  ╚═╝╚═╝  ╚═╝  ╚═══╝  ╚═╝╚══════╝
"""

HELP_TEXT = """
[bold cyan]JARVIS CLI Commands:[/bold cyan]

  [yellow]/help[/yellow]       → Show this help menu
  [yellow]/solve <task>[/yellow]→ Execute 10-step reasoning & problem solving
  [yellow]/research <q>[/yellow]→ Deep internet research & knowledge indexing
  [yellow]/history[/yellow]    → Show conversation history
  [yellow]/clear[/yellow]      → Clear conversation memory
  [yellow]/status[/yellow]     → Show system status
  [yellow]/session[/yellow]    → Show current session ID
  [yellow]/exit[/yellow]       → Exit JARVIS
  [yellow]/quit[/yellow]       → Exit JARVIS

Type any message to talk with JARVIS.
"""

COMMANDS = {"/help", "/solve", "/research", "/history", "/clear", "/status", "/session", "/exit", "/quit"}


def display_banner():
    """Display the JARVIS startup banner."""
    console.print(JARVIS_ASCII, style="bold blue")
    console.print(
        Panel(
            f"[bold white]Autonomous Multi-Agent AI System[/bold white]\n"
            f"[dim]v{settings.JARVIS_VERSION} | {settings.AI_PROVIDER.upper()} | "
            f"Autonomy Level {settings.DEFAULT_AUTONOMY_LEVEL}[/dim]",
            border_style="blue",
            expand=False,
        )
    )
    console.print()


def display_time_greeting() -> str:
    """Return time-appropriate greeting."""
    hour = datetime.now().hour
    if 5 <= hour < 12:
        return "Good morning"
    elif 12 <= hour < 17:
        return "Good afternoon"
    elif 17 <= hour < 21:
        return "Good evening"
    else:
        return "Good evening"


async def handle_command(command: str, session_id: str) -> bool:
    """
    Handle CLI commands (starting with /).
    Returns True if should continue, False to exit.
    """
    cmd = command.strip().lower()

    if cmd in ("/exit", "/quit"):
        console.print(
            "\n[bold blue]JARVIS:[/bold blue] Goodbye, sir. Standing by.",
            style="bold"
        )
        return False

    elif cmd == "/help":
        console.print(HELP_TEXT)

    elif cmd == "/status":
        table = Table(title="JARVIS System Status", border_style="blue")
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="white")
        table.add_row("Status", "[green]Operational[/green]")
        table.add_row("Version", settings.JARVIS_VERSION)
        table.add_row("AI Provider", settings.AI_PROVIDER.upper())
        table.add_row("Environment", settings.ENVIRONMENT)
        table.add_row("Autonomy Level", str(settings.DEFAULT_AUTONOMY_LEVEL))
        table.add_row("Session ID", session_id)
        sessions = memory_manager.list_sessions()
        table.add_row("Active Sessions", str(len(sessions)))
        console.print(table)

    elif cmd == "/session":
        console.print(f"\n[dim]Session ID:[/dim] [bold]{session_id}[/bold]\n")

    elif cmd == "/history":
        session = memory_manager.get_session(session_id)
        if not session or session.message_count() == 0:
            console.print("[dim]No conversation history yet.[/dim]\n")
        else:
            console.print(f"\n[bold]Conversation History ({session.message_count()} messages):[/bold]\n")
            for msg in session.messages:
                if msg.role == "system":
                    continue
                role_color = "blue" if msg.role == "assistant" else "green"
                role_name = "JARVIS" if msg.role == "assistant" else "You"
                console.print(
                    f"[{role_color}]{role_name}:[/{role_color}] {msg.content}\n"
                    f"[dim]{msg.timestamp.strftime('%H:%M:%S')}[/dim]\n"
                )

    elif cmd.startswith("/solve"):
        problem = command[6:].strip()
        if not problem:
            console.print("[yellow]Usage:[/yellow] /solve <describe the complex objective or problem>\n")
            return True

        from core.planner.reasoning_engine import reasoning_engine
        console.print(f"\n[bold blue]JARVIS:[/bold blue] Initiating 10-step problem-solving sequence for: [italic]{problem}[/italic]\n")
        
        with console.status("[dim]Analyzing goals, evaluating multiple strategies, planning steps...[/dim]", spinner="dots"):
            res = reasoning_engine.solve(problem)

        console.print(Panel(
            f"[bold cyan]1. Primary Goal:[/bold cyan] {res.step_1_objective_understanding.primary_goal}\n"
            f"[bold cyan]2. Situation Assessment:[/bold cyan] {res.step_2_situation_assessment}\n"
            f"[bold cyan]3. Selected Strategy:[/bold cyan] {res.step_7_selected_solution.name} (Score: {res.step_7_selected_solution.composite_score})\n"
            f"[bold cyan]4. Authorization:[/bold cyan] {res.step_6_evaluation.authorization_note}\n\n"
            f"[bold yellow]Execution Plan:[/bold yellow]\n" +
            "\n".join([f"  • [{t.id}] {t.title}: {t.description}" for t in res.step_8_execution_plan.subtasks]) +
            f"\n\n[bold green]Verification:[/bold green] {res.step_9_verification.actual_outcome}",
            title="[bold blue]JARVIS Reasoning Engine Synthesis[/bold blue]",
            border_style="blue",
        ))
        console.print()

    elif cmd.startswith("/research"):
        topic = command[9:].strip()
        if not topic:
            console.print("[yellow]Usage:[/yellow] /research <topic to investigate>\n")
            return True

        from services.internet.research_pipeline import research_pipeline
        console.print(f"\n[bold blue]JARVIS:[/bold blue] Initiating internet research on: [italic]{topic}[/italic]\n")

        with console.status("[dim]Querying search engines, extracting sources, synthesizing findings...[/dim]", spinner="dots"):
            rep = await research_pipeline.conduct_research(topic)

        console.print(Panel(
            f"[bold cyan]Topic:[/bold cyan] {rep.topic}\n\n"
            f"[bold green]Summary:[/bold green]\n{rep.summary}\n\n"
            f"[bold yellow]Citations & Sources:[/bold yellow]\n" +
            "\n".join([f"  • {c.title} — [dim]{c.url}[/dim]" for c in rep.citations]) +
            f"\n\n[dim]Indexed into JARVIS Knowledge Base: {'Yes (ID: ' + str(rep.knowledge_id) + ')' if rep.knowledge_indexed else 'No'}[/dim]",
            title="[bold blue]JARVIS Universal Research Report[/bold blue]",
            border_style="blue",
        ))
        console.print()

    elif cmd == "/clear":
        memory_manager.delete_session(session_id)
        console.print("[dim]Conversation memory cleared.[/dim]\n")

    return True


async def chat_loop():
    """Main interactive chat loop."""
    display_banner()

    # Initialize AI Brain
    console.print("[dim]Initializing AI systems...[/dim]")
    try:
        await brain.initialize()
    except Exception as e:
        console.print(f"\n[bold red]ERROR:[/bold red] Could not initialize JARVIS Brain.\n{e}")
        console.print(
            "\n[yellow]Ensure local models and checkpoints are ready in checkpoints/local_models.[/yellow]\n"
        )
        sys.exit(1)

    # Create session
    session_id = str(uuid.uuid4())
    greeting_time = display_time_greeting()

    console.print(
        f"\n[bold blue]JARVIS:[/bold blue] {greeting_time}, sir. "
        f"I am online and ready. How may I assist you?\n"
    )
    console.print("[dim]Type /help for commands. Type /exit to quit.[/dim]\n")

    while True:
        try:
            # Get user input
            user_input = console.input("[bold green]You:[/bold green] ").strip()

            if not user_input:
                continue

            # Handle commands
            if user_input.startswith("/"):
                should_continue = await handle_command(user_input, session_id)
                if not should_continue:
                    break
                continue

            # Store user message
            memory_manager.add_user_message(session_id, user_input)

            # Get conversation history
            history = memory_manager.get_history(session_id, max_messages=20)
            history_without_current = history[:-1] if history else []

            # Show thinking indicator
            with console.status("[dim]JARVIS is thinking...[/dim]", spinner="dots"):
                response = await brain.think(
                    user_message=user_input,
                    conversation_history=history_without_current,
                )

            if not response.success:
                console.print(f"\n[bold red]JARVIS ERROR:[/bold red] {response.error}\n")
                continue

            # Store JARVIS response
            memory_manager.add_assistant_message(session_id, response.content)

            # Display JARVIS response
            console.print(f"\n[bold blue]JARVIS:[/bold blue] ", end="")
            console.print(Markdown(response.content))

            # Show token usage if available
            if response.tokens_used:
                console.print(
                    f"[dim]  [{response.provider}/{response.model} | "
                    f"{response.tokens_used} tokens][/dim]"
                )
            console.print()

        except KeyboardInterrupt:
            console.print(
                "\n\n[bold blue]JARVIS:[/bold blue] Pausing. Press Ctrl+C again to exit."
            )
            try:
                await asyncio.sleep(1)
            except KeyboardInterrupt:
                console.print("\n[bold blue]JARVIS:[/bold blue] Goodbye, sir.")
                break
        except Exception as e:
            log.error(f"CLI error: {e}")
            console.print(f"\n[red]Unexpected error: {e}[/red]\n")


@app_cli.command()
def start(
    provider: str = typer.Option(None, "--provider", "-p", help="AI provider: openai|gemini|ollama|anthropic"),
    model: str = typer.Option(None, "--model", "-m", help="Model name override"),
    level: int = typer.Option(None, "--level", "-l", help="Autonomy level 1-5"),
):
    """Start JARVIS AI in interactive CLI mode."""
    # Allow command-line overrides
    if provider:
        settings.AI_PROVIDER = provider
    if level:
        settings.DEFAULT_AUTONOMY_LEVEL = max(1, min(5, level))

    asyncio.run(chat_loop())


@app_cli.command()
def serve(
    host: str = typer.Option("0.0.0.0", "--host", "-h"),
    port: int = typer.Option(8000, "--port", "-p"),
    reload: bool = typer.Option(False, "--reload", "-r", help="Auto-reload on code changes"),
):
    """Start JARVIS API server (FastAPI + uvicorn)."""
    import uvicorn
    console.print(f"\n[bold blue]Starting JARVIS API server at http://{host}:{port}[/bold blue]")
    console.print(f"[dim]API docs: http://localhost:{port}/docs[/dim]\n")
    uvicorn.run(
        "api.app:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info",
    )


if __name__ == "__main__":
    app_cli()
