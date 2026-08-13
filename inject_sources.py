"""
Hermes Source Injector
Injects skills + docs from D:\\Hermes\\sources\\clones\\ into Hermes.
Pure local. No external services.

Reads:
- SKILL.md files (anthropic format) → Hermes skills
- README.md files → Knowledge base

Adds only. Never replaces existing skills (preserves user customizations).
"""
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

HERMES_ROOT = Path(r"D:\Hermes")
sys.path.insert(0, str(HERMES_ROOT))

CLONES_DIR = HERMES_ROOT / "sources" / "clones"
SKILLS_OUT = HERMES_ROOT / "skills"


def parse_skill_md(content: str) -> Optional[Dict]:
    """Parse Anthropic-style SKILL.md with YAML frontmatter."""
    # Frontmatter
    fm_match = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)$", content, re.DOTALL)
    if not fm_match:
        return None
    fm_text = fm_match.group(1)
    body = fm_match.group(2).strip()

    # Parse simple YAML (key: value)
    name = None
    description = None
    for line in fm_text.splitlines():
        line = line.strip()
        if line.startswith("name:"):
            name = line[5:].strip().strip("'\"")
        elif line.startswith("description:"):
            description = line[12:].strip().strip("'\"")

    if not name or not description:
        return None

    # Extract keywords from description and name
    keywords = []
    desc_words = re.findall(r"[a-zA-Z\u0600-\u06FF]{3,}", description.lower())
    keywords.extend(desc_words[:8])
    for w in re.findall(r"[a-zA-Z\u0600-\u06FF]{3,}", name.lower()):
        if w not in keywords:
            keywords.append(w)

    # Use first part of body as procedure
    procedure = body[:3000]

    return {
        "name": f"src_{name}"[:50].replace(" ", "_").replace("-", "_").lower(),
        "description": description[:300],
        "trigger_keywords": keywords[:15],
        "input_schema": {"text": "user input"},
        "procedure": procedure,
        "examples": [],
        "version": "1.0.0",
        "enabled": True,
        "_source": f"cloned: {name}",
    }


def find_skill_files(root: Path) -> List[Path]:
    """Find all SKILL.md files in the clones directory."""
    result = []
    for skill_md in root.rglob("SKILL.md"):
        if ".git" in skill_md.parts:
            continue
        if "node_modules" in skill_md.parts:
            continue
        result.append(skill_md)
    return result


def find_doc_files(root: Path) -> List[Path]:
    """Find README.md and other doc files in the clones directory."""
    docs = []
    for pattern in ["README.md", "*.md", "*.mdx"]:
        for f in root.rglob(pattern):
            if ".git" in f.parts or "node_modules" in f.parts:
                continue
            if f.name.lower() in ("license", "license.md", "code_of_conduct.md",
                                  "contributing.md", "changelog.md", "security.md"):
                continue
            if "skills" in f.parts and f.name == "SKILL.md":
                continue  # already handled
            try:
                if f.stat().st_size > 100_000:  # skip huge files
                    continue
            except OSError:
                continue
            docs.append(f)
    return docs


def inject_skills(skill_files: List[Path]) -> Tuple[int, int]:
    """Inject skills into Hermes. Returns (added, skipped)."""
    SKILLS_OUT.mkdir(parents=True, exist_ok=True)
    added = 0
    skipped = 0
    for sf in skill_files:
        try:
            content = sf.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        parsed = parse_skill_md(content)
        if not parsed:
            continue
        # Sanitize filename
        safe_name = re.sub(r"[^a-z0-9_]", "_", parsed["name"].lower())
        if not safe_name or len(safe_name) < 3:
            continue
        out_path = SKILLS_OUT / f"{safe_name}.json"
        if out_path.exists():
            skipped += 1
            continue
        try:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(parsed, f, ensure_ascii=False, indent=2)
            added += 1
        except Exception as e:
            print(f"  [warn] failed to write {safe_name}: {e}")
    return added, skipped


def inject_docs(doc_files: List[Path], source_tag: str = "cloned_repo") -> int:
    """Ingest documents into the knowledge base."""
    from core.knowledge import get_kb
    kb = get_kb()
    added = 0
    for df in doc_files:
        try:
            content = df.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        if len(content) < 100:
            continue
        # Truncate to avoid huge docs
        if len(content) > 50_000:
            content = content[:50_000] + "\n\n[...truncated...]"
        # Tag with repo name
        try:
            rel = df.relative_to(CLONES_DIR)
            repo = rel.parts[0].replace(".git", "")
            source = f"{source_tag}:{repo}/{df.name}"
        except ValueError:
            source = f"{source_tag}:{df.name}"
        try:
            ids = kb.add_text(
                content,
                source=source,
                metadata={
                    "filename": df.name,
                    "repo": df.parts[-2] if len(df.parts) > 1 else "unknown",
                    "path": str(df),
                    "ingested_at": time.time(),
                    "kind": "cloned_doc",
                },
            )
            if ids:
                added += 1
        except Exception as e:
            print(f"  [warn] failed to ingest {df.name}: {e}")
    kb.save()
    return added


def main():
    print("=" * 60)
    print("  Hermes Source Injector")
    print(f"  Clones: {CLONES_DIR}")
    print("=" * 60)
    print()

    if not CLONES_DIR.exists():
        print(f"ERROR: {CLONES_DIR} not found.")
        print("Run extraction first: unzip the source bundle into sources/extracted/ and git clone each *.git.")
        return 1

    # Discover
    print("Discovering skills...")
    skill_files = find_skill_files(CLONES_DIR)
    print(f"  Found {len(skill_files)} SKILL.md files")

    print("Discovering docs...")
    doc_files = find_doc_files(CLONES_DIR)
    print(f"  Found {len(doc_files)} doc files")

    # Inject skills
    print("\n[1/2] Injecting skills...")
    added, skipped = inject_skills(skill_files)
    print(f"  Added: {added} | Skipped (already exists): {skipped}")

    # Inject docs
    print("\n[2/2] Injecting documents into knowledge base...")
    docs_added = inject_docs(doc_files)
    print(f"  Ingested: {docs_added} documents")

    # Final stats
    from core.skills import get_skills
    from core.knowledge import get_kb
    print("\n=== FINAL STATE ===")
    print(f"  Total skills: {len(get_skills())}")
    print(f"  KB chunks: {get_kb().stats()['count']}")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
