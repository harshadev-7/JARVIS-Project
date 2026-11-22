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
  [yellow]/autoreel [q][/yellow]→ Generate 100% autonomous 9:16 viral video from scratch
  [yellow]/reel <video>[/yellow]→ Autonomous 9:16 Reel creation with online trending research
  [yellow]/metrics[/yellow]    → Real-time CPU, RAM, & disk hardware telemetry
  [yellow]/memory[/yellow]     → Inspect persistent user profile & episodic memory
  [yellow]/agents[/yellow]     → Inspect all 15 specialized agents in the swarm
  [yellow]/voice <txt>[/yellow]→ Synthesize speech audio using local/neural TTS
  [yellow]/pair[/yellow]        → Generate one-time 6-digit cryptographic pairing PIN (TTL 5m)
  [yellow]/devices[/yellow]     → List connected & authorized devices in the mesh
  [yellow]/scan[/yellow]        → Scan local network (UDP 54321) for companion devices
  [yellow]/folder <p>[/yellow] → Authorize a directory sandbox for verified file transfers
  [yellow]/history[/yellow]    → Show conversation history
  [yellow]/clear[/yellow]      → Clear conversation memory
  [yellow]/status[/yellow]     → Show system status
  [yellow]/session[/yellow]    → Show current session ID
  [yellow]/exit[/yellow]       → Exit JARVIS
  [yellow]/quit[/yellow]       → Exit JARVIS

Type any message to talk with JARVIS.
"""

COMMANDS = {
    "/help", "/solve", "/research", "/metrics", "/memory",
    "/agents", "/voice", "/history", "/clear", "/status",
    "/session", "/exit", "/quit", "/pair", "/devices", "/scan", "/folder"
}


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

    elif any(cmd.startswith(prefix) for prefix in ["/solve", "/slove", "/solv", "/sove", "/calc", "/calculate"]):
        parts = command.split(maxsplit=1)
        problem = parts[1].strip() if len(parts) > 1 else ""
        if not problem:
            console.print("[yellow]Usage:[/yellow] /solve <describe the complex objective or problem>\n")
            return True

        from core.planner.reasoning_engine import reasoning_engine
        console.print(f"\n[bold blue]JARVIS:[/bold blue] Initiating 10-step problem-solving sequence for: [italic]{problem}[/italic]\n")
        
        with console.status("[dim]Analyzing goals, evaluating multiple strategies, planning steps...[/dim]", spinner="dots"):
            res = reasoning_engine.solve(problem)

        console.print(Panel(
            f"[bold cyan]1. Primary Goal:[/bold cyan] {res.step_1_objective_understanding.primary_goal}\n"
            f"[bold cyan]2. Strategy:[/bold cyan] {res.step_7_selected_solution.name} (Score: {res.step_7_selected_solution.composite_score:.2f})\n"
            f"[bold cyan]3. Complexity:[/bold cyan] {res.step_2_situation_assessment}\n"
            f"[bold cyan]4. Authorization:[/bold cyan] {res.step_6_evaluation.authorization_note}\n\n"
            f"[bold yellow]Execution Plan:[/bold yellow]\n" +
            "\n".join([f"  • [{t.id}] {t.title}: {t.description}" for t in res.step_8_execution_plan.subtasks]) +
            f"\n\n[bold green]Verification:[/bold green] {res.step_9_verification.actual_outcome}",
            title="[bold blue]JARVIS Cognitive Reasoning Framework[/bold blue]",
            border_style="blue",
        ))

        # Render the full concrete solution synthesized by the cognitive engine
        if res.detailed_solution:
            console.print(Panel(
                Markdown(res.detailed_solution),
                title="[bold green]JARVIS Verified Solution & Strategy[/bold green]",
                border_style="green",
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

    elif cmd.startswith("/reel"):
        from video_editor.reel_controller import reel_controller
        with console.status("[bold cyan]◈ JARVIS is analyzing footage, composing timeline & rendering...[/bold cyan]", spinner="dots"):
            summary_msg = await reel_controller.execute_command(command)
        try:
            ctx = brain.get_context(session_id)
            ctx.working_variables["last_reel_output"] = summary_msg
        except Exception:
            pass
        console.print(Panel(
            summary_msg,
            title="[bold green]JARVIS Autonomous Video Editing Engine (KANKI.7)[/bold green]",
            border_style="green",
        ))
        console.print()

    elif cmd.startswith("/autoreel") or cmd.startswith("/create_video") or cmd.startswith("/video"):
        parts = command.split(maxsplit=1)
        arg = parts[1].strip() if len(parts) > 1 else ""
        from core.orchestrator.agent_registry import agent_registry
        agent = agent_registry.get_agent("video_editing")
        task_str = arg if arg else "create autonomous viral video on trending tech"
        with console.status("[bold cyan]◈ JARVIS is curating viral content & synthesizing your video...[/bold cyan]", spinner="dots"):
            res = await agent.execute(task_str)
        # Retain media context in Brain for natural language follow-up queries (e.g. "play that video")
        try:
            ctx = brain.get_context(session_id)
            if res.data:
                ctx.working_variables["last_created_media"] = res.data
        except Exception:
            pass
        console.print(Panel(
            res.summary,
            title="[bold green]JARVIS Autonomous Video & Viral Reel Creator[/bold green]",
            border_style="green",
        ))
        console.print()

    elif cmd in ("/play", "/play_video", "/watch"):
        from core.orchestrator.agent_registry import agent_registry
        agent = agent_registry.get_agent("video_editing")
        ctx = brain.get_context(session_id)
        with console.status("[bold cyan]◈ Launching media player...[/bold cyan]", spinner="dots"):
            res = await agent.execute("play that video", context=ctx.working_variables)
        console.print(Panel(
            res.summary,
            title="[bold blue]JARVIS Media Player[/bold blue]",
            border_style="blue",
        ))
        console.print()

    elif cmd in ("/metrics", "/system"):
        from services.automation.system_monitor import system_monitor
        m = system_monitor.get_metrics()
        table = Table(title="JARVIS Hardware & Resource Telemetry", border_style="cyan")
        table.add_column("Resource", style="cyan")
        table.add_column("Telemetry", style="white")
        table.add_row("CPU Utilization", f"{m.cpu_percent}%")
        table.add_row("RAM Active", f"{m.ram_used_gb} GB / {m.ram_total_gb} GB ({m.ram_percent}%)")
        table.add_row("Disk Usage", f"{m.disk_percent}%")
        table.add_row("System Status", "[green]Healthy[/green]" if m.is_healthy else "[red]Warning[/red]")
        console.print(table)
        console.print()

    elif cmd in ("/memory", "/profile"):
        from core.memory.persistent_memory import persistent_memory
        prefs = await persistent_memory.get_all_preferences()
        episodes = await persistent_memory.get_recent_episodes(session_id=session_id, limit=5)
        table = Table(title="JARVIS Persistent Intelligence Memory (jarvis.db)", border_style="magenta")
        table.add_column("Property / Fact", style="magenta")
        table.add_column("Stored Value", style="white")
        if prefs:
            for k, v in prefs.items():
                table.add_row(k.replace("_", " ").title(), str(v))
        else:
            table.add_row("Profile State", "No explicit profile facts committed yet. Tell JARVIS your name, role, or goals!")
        table.add_row("Recent Turns Stored", str(len(episodes)))
        console.print(table)
        console.print()

    elif cmd == "/agents":
        from core.orchestrator.agent_registry import agent_registry
        table = Table(title="JARVIS Multi-Agent Swarm (15 Specialized Agents)", border_style="blue")
        table.add_column("Agent Name", style="cyan")
        table.add_column("Role", style="white")
        table.add_column("Autonomy", style="yellow")
        for ag in agent_registry.list_agents():
            table.add_row(ag.name, ag.role, f"Level {ag.required_autonomy_level}")
        console.print(table)
        console.print()

    elif cmd.startswith("/voice"):
        text = command[6:].strip()
        if not text:
            console.print("[yellow]Usage:[/yellow] /voice <text to synthesize into speech>\n")
            return True
        from services.voice.tts_engine import tts_engine
        with console.status("[dim]Synthesizing speech audio...[/dim]", spinner="dots"):
            audio_path = await tts_engine.synthesize(text)
        console.print(f"[bold green]Voice Audio Generated:[/bold green] {audio_path}\n")

    elif cmd == "/clear":
        memory_manager.delete_session(session_id)
        console.print("[dim]Conversation memory cleared.[/dim]\n")

    elif cmd == "/pair":
        from devices.device_manager import device_manager
        pin = device_manager.generate_pairing_pin()
        console.print(Panel(
            f"[bold green]One-Time Cryptographic Pairing Code:[/bold green] [bold yellow]{pin}[/bold yellow]\n\n"
            f"• [bold]Time-To-Live:[/bold] 5 minutes (300 seconds)\n"
            f"• [bold]Action:[/bold] Enter this PIN on your companion device (Android app or client) to pair.\n"
            f"• [bold]Security:[/bold] Single-use token. Once paired, mutual SHA-256 tokens are provisioned.",
            title="[bold blue]JARVIS Multi-Device Pairing Handshake[/bold blue]",
            border_style="blue",
        ))
        console.print()

    elif cmd == "/devices":
        from devices.device_manager import device_manager
        devs = device_manager.list_devices()
        table = Table(title="Authorized Connected Devices (Mesh Network)", border_style="cyan")
        table.add_column("Device ID", style="cyan")
        table.add_column("Device Name", style="white")
        table.add_column("Type", style="yellow")
        table.add_column("Status", style="green")
        table.add_column("Battery", style="magenta")
        table.add_column("Capabilities", style="dim")
        if devs:
            for d in devs:
                status_color = "green" if d.status.value == "online" else "red"
                battery_str = f"{d.battery_level:.0f}%" if d.battery_level is not None else "--"
                table.add_row(
                    d.device_id,
                    d.name,
                    d.device_type.value,
                    f"[{status_color}]{d.status.value}[/{status_color}]",
                    battery_str,
                    ", ".join(d.capabilities[:3]),
                )
        else:
            table.add_row("dev-host", "This Workstation (Local)", "windows", "[green]online[/green]", "100%", "terminal, automation, video")
        console.print(table)
        console.print()

    elif cmd == "/scan":
        from devices.device_manager import device_manager
        with console.status("[dim]Broadcasting presence beacon & scanning local subnet...[/dim]", spinner="dots"):
            device_manager.broadcast_presence()
            peers = device_manager.scan_network()
        if peers:
            table = Table(title="Local LAN Discovered Devices (JARVIS-DISC-v1)", border_style="green")
            table.add_column("Device ID", style="cyan")
            table.add_column("Name", style="white")
            table.add_column("Type", style="yellow")
            table.add_column("IP Address", style="magenta")
            table.add_column("Service Port", style="green")
            for p in peers:
                table.add_row(p.get("device_id", "--"), p.get("name", "--"), p.get("device_type", "--"), p.get("ip", "--"), str(p.get("port", 8000)))
            console.print(table)
        else:
            console.print("[dim]No companion devices detected broadcasting on UDP port 54321.[/dim]")
            console.print("[dim]Host presence beacon sent to 255.255.255.255:54321 successfully.[/dim]")
        console.print()

    elif cmd.startswith("/folder"):
        parts = command.split(maxsplit=1)
        folder_arg = parts[1].strip() if len(parts) > 1 else ""
        from devices.device_manager import device_manager
        if not folder_arg:
            folders = device_manager.list_authorized_folders()
            table = Table(title="Authorized Directory Sandboxes", border_style="magenta")
            table.add_column("Folder ID", style="cyan")
            table.add_column("Path", style="white")
            table.add_column("Access Level", style="green")
            table.add_column("Authorized At", style="dim")
            if folders:
                for f in folders:
                    table.add_row(f.get("folder_id", "--"), f.get("path", "--"), f.get("access_level", "--"), f.get("authorized_at", "--")[:19])
            else:
                table.add_row("--", "No external folders authorized yet", "read_only", "--")
            console.print(table)
            console.print("[dim]To authorize a folder: /folder <absolute_or_relative_path>[/dim]\n")
        else:
            try:
                rec = device_manager.authorize_folder(folder_arg, access_level="read_write")
                console.print(f"[bold green]Folder Whitelisted:[/bold green] `{rec['path']}` (read_write)\n")
            except Exception as e:
                console.print(f"[bold red]Failed to Authorize Folder:[/bold red] {e}\n")

    else:
        token = command.split()[0]
        console.print(f"[yellow]Unknown command '{token}'. Did you mean /solve or /help?[/yellow]\n")
        console.print("[dim]Type /help to see all available commands.[/dim]\n")

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


@app_cli.command()
def gui():
    """Launch the JARVIS Desktop Arc Reactor HUD Graphical User Interface."""
    console.print("\n[bold cyan]◈ Launching J.A.R.V.I.S. Desktop Arc Reactor HUD...[/bold cyan]\n")
    from interface.hud_app import launch_gui
    launch_gui()


@app_cli.command()
def hud():
    """Alias for 'gui' command: Launch Desktop HUD."""
    gui()


if __name__ == "__main__":
    # If no subcommand specified, default to interactive CLI mode ('start')
    if len(sys.argv) == 1 or (len(sys.argv) > 1 and sys.argv[1].startswith("-") and sys.argv[1] not in ("--help", "-h")):
        sys.argv.insert(1, "start")
    app_cli()
