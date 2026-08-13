"""
Hermes Training Mode
Interactive, guided training sessions. The user teaches Hermes step by step,
and the lessons are persisted into the knowledge base, skills, and memory.
"""
import sys
import json
from pathlib import Path

HERMES_ROOT = Path(__file__).parent
sys.path.insert(0, str(HERMES_ROOT))

from core.orchestrator import get_orchestrator

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.prompt import Prompt
    from rich import print as rprint
    HAS_RICH = True
except ImportError:
    HAS_RICH = False


def print_banner():
    if HAS_RICH:
        console = Console()
        console.print(Panel(
            "📚 [bold cyan]HERMES TRAINING MODE[/bold cyan]\n\n"
            "Teach Hermes interactively. All your lessons are saved.\n"
            "Type 'done' at any prompt to finish, 'cancel' to abort.",
            border_style="cyan", title="Training Session"
        ))
    else:
        print("=" * 60)
        print(" HERMES TRAINING MODE")
        print(" Teach interactively. 'done'=finish, 'cancel'=abort")
        print("=" * 60)


def main():
    print_banner()
    orch = get_orchestrator()

    if HAS_RICH:
        console = Console()
        topic = Prompt.ask("\n[cyan]📌 Topic of this training session[/cyan]").strip()
    else:
        topic = input("\nTopic of this training session: ").strip()

    if not topic or topic.lower() in ("cancel", "done"):
        print("Aborted.")
        return

    instructions = []
    examples = []

    while True:
        if HAS_RICH:
            console.print("\n[bold]What should Hermes know?[/bold]")
            console.print("[dim](type the instruction, or 'done' to finish, 'cancel' to abort)[/dim]")
            instr = Prompt.ask("[cyan]📝 Instruction/Rule[/cyan]").strip()
        else:
            print("\nWhat should Hermes know?")
            print("(type instruction, 'done' to finish, 'cancel' to abort)")
            instr = input("Instruction: ").strip()

        if instr.lower() == "cancel":
            print("Aborted.")
            return
        if instr.lower() == "done":
            break
        if not instr:
            continue

        instructions.append(instr)

        # Add as lesson
        orch.teach_lesson(instr)
        if HAS_RICH:
            console.print(f"[green]✓ Lesson recorded[/green]")
        else:
            print("  ✓ Lesson recorded")

        # Optionally add as skill
        if HAS_RICH:
            as_skill = Prompt.ask(
                "[cyan]Is this a procedural skill (with steps)? (y/n)[/cyan]",
                default="n"
            ).lower()
        else:
            as_skill = input("Is this a procedural skill with steps? (y/n): ").lower()

        if as_skill.startswith("y"):
            if HAS_RICH:
                steps = Prompt.ask("[cyan]Describe the procedure/steps[/cyan]").strip()
            else:
                steps = input("Procedure: ").strip()
            if steps:
                skill_data = {
                    "name": topic.lower().replace(" ", "_") + f"_rule_{len(orch.skills) + 1}",
                    "description": instr,
                    "trigger_keywords": topic.lower().split(),
                    "procedure": steps,
                    "input_schema": {},
                    "examples": [],
                    "enabled": True,
                }
                skill_name = orch.teach_skill(skill_data)
                if HAS_RICH:
                    console.print(f"[green]✓ Skill '{skill_name}' created[/green]")
                else:
                    print(f"  ✓ Skill '{skill_name}' created")

        # Optional example
        if HAS_RICH:
            add_ex = Prompt.ask(
                "[cyan]Add an example? (y/n)[/cyan]",
                default="n"
            ).lower()
        else:
            add_ex = input("Add an example? (y/n): ").lower()

        if add_ex.startswith("y"):
            if HAS_RICH:
                ex_in = Prompt.ask("[cyan]Example input[/cyan]").strip()
                ex_out = Prompt.ask("[cyan]Example output[/cyan]").strip()
            else:
                ex_in = input("Example input: ").strip()
                ex_out = input("Example output: ").strip()
            if ex_in and ex_out:
                examples.append({"input": ex_in, "output": ex_out})
                # Also ingest as knowledge
                orch.teach_document(
                    f"Example: {ex_in} -> {ex_out}\n(teaching: {instr})",
                    source=f"training:{topic}"
                )
                if HAS_RICH:
                    console.print(f"[green]✓ Example added to knowledge base[/green]")
                else:
                    print("  ✓ Example added")

    # Save training session
    orch.memory.record_training(
        topic=topic,
        instructions=instructions,
        examples=examples,
    )
    if HAS_RICH:
        console.print(f"\n[bold green]✅ Training session on '{topic}' saved.[/bold green]")
        console.print(f"[dim]Lessons: {len(instructions)} | Examples: {len(examples)}[/dim]")
    else:
        print(f"\nTraining session on '{topic}' saved.")
        print(f"Lessons: {len(instructions)} | Examples: {len(examples)}")


if __name__ == "__main__":
    main()
