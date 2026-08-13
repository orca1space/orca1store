"""
Hermes CLI
Main entry point for interacting with Hermes.
Clean, simple, beautiful terminal interface using rich.
"""
import argparse
import json
import sys
import os
from pathlib import Path
from typing import Optional

# Add hermes root to path
HERMES_ROOT = Path(__file__).parent
sys.path.insert(0, str(HERMES_ROOT))

from core.config import ensure_dirs, HERMES_MODEL_FILE
from core.orchestrator import get_orchestrator

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.markdown import Markdown
    from rich.prompt import Prompt
    from rich.table import Table
    from rich import print as rprint
    HAS_RICH = True
except ImportError:
    HAS_RICH = False


console = Console() if HAS_RICH else None


def print_banner():
    banner = """
╔═══════════════════════════════════════════════════════╗
║                                                       ║
║   ⚡  H E R M E S  ⚡                                 ║
║   Local AI Agent — Learns only from you               ║
║                                                       ║
╚═══════════════════════════════════════════════════════╝
"""
    if HAS_RICH:
        console.print(banner, style="bold cyan")
    else:
        print(banner)


def print_status(status: dict):
    if HAS_RICH:
        table = Table(title="📊 Hermes Status", show_header=True, header_style="bold magenta")
        table.add_column("Component", style="cyan")
        table.add_column("Detail", style="white")
        table.add_row("LLM Model", status["llm"]["model"])
        table.add_row("LLM Healthy", "✅ Yes" if status["llm"]["healthy"] else "❌ No")
        table.add_row("Knowledge Chunks", str(status["knowledge_base"]["count"]))
        table.add_row("Total Chars", str(status["knowledge_base"]["total_chars"]))
        table.add_row("Skills", str(status["skills"]["count"]))
        table.add_row("Conversations", str(status["memory"]["conversations"]))
        table.add_row("Training Sessions", str(status["memory"]["training_sessions"]))
        table.add_row("Lessons", str(status["memory"]["lessons"]))
        console.print(table)
    else:
        print(json.dumps(status, indent=2, ensure_ascii=False))


def cmd_chat(args):
    """Start an interactive chat session."""
    print_banner()
    orch = get_orchestrator()
    print_status(orch.status())
    print()
    if HAS_RICH:
        console.print("[bold green]💬 Chat started. Type '/help' for commands, '/exit' to leave.[/bold green]\n")
    else:
        print("Chat started. Type '/help' for commands, '/exit' to leave.\n")

    while True:
        try:
            if HAS_RICH:
                user_input = Prompt.ask("[bold cyan]You[/bold cyan]")
            else:
                user_input = input("You> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break

        user_input = user_input.strip()
        if not user_input:
            continue

        if user_input.startswith("/"):
            handle_command(user_input, orch)
            continue

        # Stream response
        if HAS_RICH:
            console.print("[bold green]Hermes[/bold green]:", end=" ")
        else:
            print("Hermes> ", end="", flush=True)

        try:
            if HAS_RICH:
                with console.status("[dim]thinking...[/dim]"):
                    response = orch.chat(user_input, stream=False)
                console.print(Markdown(response))
            else:
                response = orch.chat(user_input, stream=False)
                print(response)
        except Exception as e:
            if HAS_RICH:
                console.print(f"[red]Error: {e}[/red]")
            else:
                print(f"Error: {e}")
        print()


def handle_command(cmd: str, orch):
    """Handle slash commands."""
    parts = cmd.split(maxsplit=1)
    command = parts[0].lower()
    arg = parts[1] if len(parts) > 1 else ""

    if command in ("/exit", "/quit", "/q"):
        if HAS_RICH:
            console.print("[yellow]👋 Goodbye.[/yellow]")
        else:
            print("Goodbye.")
        sys.exit(0)

    elif command in ("/help", "/h"):
        help_text = """
Available commands:
  /help              Show this help
  /status            Show Hermes status
  /teach <text>      Teach a lesson (stored, will be recalled in future chats)
  /skill             Add a new skill (interactive)
  /skills            List all skills
  /learn <text>      Ingest text into knowledge base
  /learnfile <path>  Ingest a file into knowledge base
  /learndir <path>   Ingest all files in a directory
  /search <query>    Test knowledge base search
  /forget            Clear the knowledge base (with confirmation)
  /lessons           List recorded lessons
  /new               Start a new conversation
  /exit              Exit Hermes
"""
        if HAS_RICH:
            console.print(Panel(help_text, title="Commands", border_style="cyan"))
        else:
            print(help_text)

    elif command == "/status":
        print_status(orch.status())

    elif command == "/teach":
        if not arg:
            if HAS_RICH:
                console.print("[red]Usage: /teach <lesson text>[/red]")
            else:
                print("Usage: /teach <lesson text>")
            return
        lesson_id = orch.teach_lesson(arg)
        if HAS_RICH:
            console.print(f"[green]✓ Lesson recorded (id: {lesson_id[:8]})[/green]")
        else:
            print(f"Lesson recorded (id: {lesson_id[:8]})")

    elif command == "/skills":
        skills = orch.skills.list_all()
        if not skills:
            if HAS_RICH:
                console.print("[yellow]No skills yet. Use /skill to add one.[/yellow]")
            else:
                print("No skills yet.")
            return
        if HAS_RICH:
            t = Table(title=f"📚 {len(skills)} Skill(s)", show_header=True, header_style="bold magenta")
            t.add_column("Name", style="cyan")
            t.add_column("Description")
            t.add_column("Triggers", style="dim")
            t.add_column("Status")
            for s in skills:
                t.add_row(s["name"], s["description"], ", ".join(s["triggers"][:3]),
                           "✅" if s["enabled"] else "❌")
            console.print(t)
        else:
            for s in skills:
                print(f"- {s['name']}: {s['description']}")

    elif command == "/skill":
        add_skill_interactive(orch)

    elif command == "/learn":
        if not arg:
            if HAS_RICH:
                console.print("[red]Usage: /learn <text>[/red]")
            else:
                print("Usage: /learn <text>")
            return
        ids = orch.teach_document(arg, source="cli")
        orch.kb.save()
        if HAS_RICH:
            console.print(f"[green]✓ Added {len(ids)} chunk(s) to knowledge base[/green]")
        else:
            print(f"Added {len(ids)} chunk(s)")

    elif command == "/learnfile":
        if not arg:
            if HAS_RICH:
                console.print("[red]Usage: /learnfile <path>[/red]")
            else:
                print("Usage: /learnfile <path>")
            return
        try:
            ids = orch.kb.add_file(arg)
            orch.kb.save()
            if HAS_RICH:
                console.print(f"[green]✓ Added {len(ids)} chunk(s) from {arg}[/green]")
            else:
                print(f"Added {len(ids)} chunks")
        except Exception as e:
            if HAS_RICH:
                console.print(f"[red]Error: {e}[/red]")
            else:
                print(f"Error: {e}")

    elif command == "/learndir":
        if not arg:
            if HAS_RICH:
                console.print("[red]Usage: /learndir <path>[/red]")
            else:
                print("Usage: /learndir <path>")
            return
        try:
            result = orch.kb.add_directory(arg)
            orch.kb.save()
            if HAS_RICH:
                console.print(f"[green]✓ Processed {result['files_processed']} file(s), added {result['chunks_added']} chunk(s)[/green]")
                if result['errors']:
                    console.print(f"[yellow]⚠ {len(result['errors'])} error(s)[/yellow]")
            else:
                print(json.dumps(result, indent=2))
        except Exception as e:
            if HAS_RICH:
                console.print(f"[red]Error: {e}[/red]")
            else:
                print(f"Error: {e}")

    elif command == "/search":
        if not arg:
            if HAS_RICH:
                console.print("[red]Usage: /search <query>[/red]")
            else:
                print("Usage: /search <query>")
            return
        results = orch.kb.search(arg, top_k=5)
        if not results:
            if HAS_RICH:
                console.print("[yellow]No matches found.[/yellow]")
            else:
                print("No matches.")
            return
        if HAS_RICH:
            for i, r in enumerate(results, 1):
                console.print(Panel(
                    r["content"],
                    title=f"#{i} | score: {r['score']:.3f}",
                    border_style="green" if r["score"] > 0.5 else "yellow"
                ))
        else:
            for i, r in enumerate(results, 1):
                print(f"\n--- Result #{i} (score: {r['score']:.3f}) ---")
                print(r["content"])

    elif command == "/forget":
        if HAS_RICH:
            confirm = Prompt.ask("[red]⚠ Clear all knowledge? (yes/no)[/red]")
        else:
            confirm = input("Clear all knowledge? (yes/no): ")
        if confirm.lower() in ("yes", "y"):
            orch.kb.clear()
            if HAS_RICH:
                console.print("[green]✓ Knowledge base cleared[/green]")
            else:
                print("Knowledge base cleared")
        else:
            if HAS_RICH:
                console.print("[dim]Cancelled[/dim]")
            else:
                print("Cancelled")

    elif command == "/lessons":
        lessons = orch.memory.get_lessons(limit=50)
        if not lessons:
            if HAS_RICH:
                console.print("[yellow]No lessons recorded yet.[/yellow]")
            else:
                print("No lessons.")
            return
        if HAS_RICH:
            for i, l in enumerate(lessons, 1):
                console.print(f"[cyan]{i}.[/cyan] {l['lesson']}")
        else:
            for i, l in enumerate(lessons, 1):
                print(f"{i}. {l['lesson']}")

    elif command == "/new":
        conv_id = orch.start_conversation()
        if HAS_RICH:
            console.print(f"[green]✓ New conversation started ({conv_id[:8]})[/green]")
        else:
            print(f"New conversation: {conv_id[:8]}")

    else:
        if HAS_RICH:
            console.print(f"[red]Unknown command: {command}[/red]. Type /help")
        else:
            print(f"Unknown command: {command}")


def add_skill_interactive(orch):
    """Interactive skill creation."""
    if HAS_RICH:
        console.print(Panel("📚 Add a new skill", border_style="cyan"))
        name = Prompt.ask("[cyan]Skill name (no spaces, snake_case)[/cyan]")
        description = Prompt.ask("[cyan]Description[/cyan]")
        triggers = Prompt.ask("[cyan]Trigger keywords (comma-separated)[/cyan]")
        procedure = Prompt.ask("[cyan]Procedure (step-by-step instructions)[/cyan]")
    else:
        print("--- Add a new skill ---")
        name = input("Name (snake_case): ")
        description = input("Description: ")
        triggers = input("Trigger keywords (comma-separated): ")
        procedure = input("Procedure: ")

    skill_data = {
        "name": name.strip(),
        "description": description.strip(),
        "trigger_keywords": [t.strip() for t in triggers.split(",") if t.strip()],
        "procedure": procedure.strip(),
        "input_schema": {},
        "examples": [],
        "version": "1.0.0",
        "enabled": True,
    }
    try:
        skill = orch.teach_skill(skill_data)
        if HAS_RICH:
            console.print(f"[green]✓ Skill '{skill}' added[/green]")
        else:
            print(f"Skill '{skill}' added")
    except Exception as e:
        if HAS_RICH:
            console.print(f"[red]Error: {e}[/red]")
        else:
            print(f"Error: {e}")


def cmd_status(args):
    print_status(get_orchestrator().status())


def cmd_teach(args):
    orch = get_orchestrator()
    if args.lesson:
        lesson_id = orch.teach_lesson(args.lesson)
        print(f"Lesson recorded: {lesson_id}")
    else:
        # Interactive
        if HAS_RICH:
            lesson = Prompt.ask("[cyan]Enter the lesson[/cyan]")
        else:
            lesson = input("Lesson: ")
        if lesson.strip():
            lesson_id = orch.teach_lesson(lesson.strip())
            print(f"Lesson recorded: {lesson_id}")


def cmd_learn(args):
    orch = get_orchestrator()
    if args.file:
        ids = orch.kb.add_file(args.file)
        print(f"Added {len(ids)} chunks from {args.file}")
    elif args.directory:
        result = orch.kb.add_directory(args.directory)
        print(json.dumps(result, indent=2))
    elif args.text:
        ids = orch.teach_document(args.text, source="cli")
        print(f"Added {len(ids)} chunks")
    else:
        print("Specify --text, --file, or --directory")
    orch.kb.save()


def cmd_skill_add(args):
    """Add a skill from a JSON file."""
    orch = get_orchestrator()
    if not args.file:
        print("--file required")
        return
    with open(args.file, "r", encoding="utf-8") as f:
        data = json.load(f)
    skill = orch.teach_skill(data)
    print(f"Skill added: {skill}")


def cmd_search(args):
    orch = get_orchestrator()
    results = orch.kb.search(args.query, top_k=args.top_k)
    for i, r in enumerate(results, 1):
        print(f"\n--- #{i} (score: {r['score']:.3f}) ---")
        print(r["content"])


def main():
    ensure_dirs()
    parser = argparse.ArgumentParser(
        description="Hermes - Local AI Agent that learns only from you",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command")

    # chat (default)
    parser_chat = subparsers.add_parser("chat", help="Start interactive chat")
    parser_chat.set_defaults(func=cmd_chat)

    # status
    parser_status = subparsers.add_parser("status", help="Show Hermes status")
    parser_status.set_defaults(func=cmd_status)

    # teach (add a lesson)
    parser_teach = subparsers.add_parser("teach", help="Teach a lesson")
    parser_teach.add_argument("--lesson", "-l", help="Lesson text")
    parser_teach.set_defaults(func=cmd_teach)

    # learn (ingest knowledge)
    parser_learn = subparsers.add_parser("learn", help="Ingest knowledge")
    learn_group = parser_learn.add_mutually_exclusive_group(required=True)
    learn_group.add_argument("--text", "-t", help="Text to learn")
    learn_group.add_argument("--file", "-f", help="File to learn from")
    learn_group.add_argument("--directory", "-d", help="Directory of files to learn from")
    parser_learn.set_defaults(func=cmd_learn)

    # skill
    parser_skill = subparsers.add_parser("skill", help="Manage skills")
    sub_skill = parser_skill.add_subparsers(dest="skill_command")
    add_parser = sub_skill.add_parser("add", help="Add a skill from JSON file")
    add_parser.add_argument("--file", "-f", required=True, help="JSON skill file")
    add_parser.set_defaults(func=cmd_skill_add)

    # search
    parser_search = subparsers.add_parser("search", help="Test knowledge search")
    parser_search.add_argument("query", help="Search query")
    parser_search.add_argument("--top-k", "-k", type=int, default=5)
    parser_search.set_defaults(func=cmd_search)

    args = parser.parse_args()
    if args.command is None:
        # Default to chat
        args = parser.parse_args(["chat"])
    args.func(args)


if __name__ == "__main__":
    main()
