"""
Hermes Local Document Loaders
Load documents from local sources only (no SaaS).
Pure local. No external services.

Loaders:
- FileSystemLoader: load from local directories
- ZipLoader: extract and load from zip files
- GitLoader: load from local git repos
- ArchiveLoader: tar.gz, tar.bz2, rar (best-effort)
"""
import re
import tarfile
import zipfile
from pathlib import Path
from typing import Dict, List, Optional

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.config import HERMES_ROOT
from core.knowledge import chunk_text, SUPPORTED_EXT


class BaseLoader:
    """Base class for all loaders."""

    def __init__(self, path: str, recursive: bool = True):
        self.path = Path(path)
        self.recursive = recursive
        self.loaded: List[Dict] = []

    def load(self) -> List[Dict]:
        """Load documents. Returns list of {content, source, metadata}."""
        raise NotImplementedError

    def _is_supported(self, path: Path) -> bool:
        return path.suffix.lower() in SUPPORTED_EXT


class FileSystemLoader(BaseLoader):
    """Load documents from a local directory."""

    def load(self) -> List[Dict]:
        if not self.path.exists():
            return []
        if not self.path.is_dir():
            if self._is_supported(self.path):
                return [self._read_file(self.path)]
            return []
        results = []
        if self.recursive:
            files = list(self.path.rglob("*"))
        else:
            files = list(self.path.glob("*"))
        for f in files:
            if not f.is_file():
                continue
            if not self._is_supported(f):
                continue
            try:
                results.append(self._read_file(f))
            except Exception as e:
                import logging
                logging.getLogger("hermes.loaders").debug(
                    "Failed to read %s: %s", f, e
                )
                continue
        return results

    def _read_file(self, path: Path) -> Dict:
        try:
            content = path.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            import logging
            logging.getLogger("hermes.loaders").debug(
                "Could not read %s as text: %s", path, e
            )
            content = ""
        return {
            "content": content,
            "source": str(path),
            "metadata": {
                "filename": path.name,
                "loader": "filesystem",
                "size": path.stat().st_size if path.exists() else 0,
            },
        }


class ZipLoader(BaseLoader):
    """Load documents from a zip file."""

    def __init__(self, path: str, member_filter: Optional[callable] = None):
        super().__init__(path, recursive=True)
        self.member_filter = member_filter or (lambda name: True)

    def load(self) -> List[Dict]:
        if not self.path.exists():
            return []
        results = []
        try:
            with zipfile.ZipFile(self.path, "r") as zf:
                for name in zf.namelist():
                    if name.endswith("/"):
                        continue
                    if not self.member_filter(name):
                        continue
                    ext = Path(name).suffix.lower()
                    if ext not in SUPPORTED_EXT:
                        continue
                    try:
                        content_bytes = zf.read(name)
                        try:
                            content = content_bytes.decode("utf-8")
                        except UnicodeDecodeError:
                            content = content_bytes.decode("utf-8", errors="replace")
                        results.append({
                            "content": content,
                            "source": f"{self.path}::{name}",
                            "metadata": {
                                "filename": Path(name).name,
                                "loader": "zip",
                                "archive": str(self.path),
                                "member": name,
                            },
                        })
                    except Exception as e:
                        import logging
                        logging.getLogger("hermes.loaders").debug(
                            "Skipped zip member %s: %s", name, e
                        )
                        continue
        except (zipfile.BadZipFile, OSError) as e:
            import logging
            logging.getLogger("hermes.loaders").warning("Zip load failed: %s", e)
        return results


class GitLoader(BaseLoader):
    """Load documents from a local git repository."""

    def load(self) -> List[Dict]:
        if not self.path.exists():
            return []
        if not (self.path / ".git").exists() and not (self.path / "HEAD").exists():
            # Not a git repo
            return FileSystemLoader(str(self.path), self.recursive).load()
        results = []
        # Use git ls-files to list tracked files
        import subprocess
        try:
            proc = subprocess.run(
                ["git", "ls-files"],
                cwd=str(self.path),
                capture_output=True, text=True, timeout=30,
            )
            if proc.returncode == 0:
                files = [self.path / line.strip() for line in proc.stdout.splitlines() if line.strip()]
            else:
                files = []
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            files = []
        for f in files:
            if not f.is_file() or not self._is_supported(f):
                continue
            try:
                content = f.read_text(encoding="utf-8", errors="replace")
                # Get relative path
                try:
                    rel = f.relative_to(self.path)
                except ValueError:
                    rel = f
                results.append({
                    "content": content,
                    "source": f"git:{self.path.name}/{rel}",
                    "metadata": {
                        "filename": f.name,
                        "loader": "git",
                        "repo": self.path.name,
                        "rel_path": str(rel),
                    },
                })
            except Exception:
                continue
        return results


class ArchiveLoader(BaseLoader):
    """Load from tar/tar.gz/tar.bz2 archives."""

    def load(self) -> List[Dict]:
        if not self.path.exists():
            return []
        results = []
        try:
            with tarfile.open(self.path, "r:*") as tf:
                for member in tf.getmembers():
                    if not member.isfile():
                        continue
                    ext = Path(member.name).suffix.lower()
                    if ext not in SUPPORTED_EXT:
                        continue
                    f = tf.extractfile(member)
                    if f is None:
                        continue
                    try:
                        content_bytes = f.read()
                        try:
                            content = content_bytes.decode("utf-8")
                        except UnicodeDecodeError:
                            content = content_bytes.decode("utf-8", errors="replace")
                        results.append({
                            "content": content,
                            "source": f"{self.path}::{member.name}",
                            "metadata": {
                                "filename": Path(member.name).name,
                                "loader": "archive",
                                "archive": str(self.path),
                                "member": member.name,
                            },
                        })
                    except Exception as e:
                        import logging
                        logging.getLogger("hermes.loaders").debug(
                            "Skipped archive member %s: %s", member.name, e
                        )
                        continue
        except tarfile.TarError as e:
            import logging
            logging.getLogger("hermes.loaders").warning(
                "Archive load failed: %s", e
            )
        return results


# === Unified loader entry point ===

def load_documents(source: str, loader_type: Optional[str] = None,
                  recursive: bool = True) -> List[Dict]:
    """
    Auto-detect loader type and load documents.
    loader_type: 'filesystem' | 'zip' | 'git' | 'archive' (auto-detect if None)
    """
    path = Path(source)
    if not loader_type:
        # Auto-detect
        if path.is_dir():
            # Check for .git
            if (path / ".git").exists() or (path / "HEAD").exists():
                loader_type = "git"
            else:
                loader_type = "filesystem"
        elif path.suffix.lower() == ".zip":
            loader_type = "zip"
        elif path.suffix.lower() in (".tar", ".tar.gz", ".tgz", ".tar.bz2"):
            loader_type = "archive"
        else:
            loader_type = "filesystem"
    if loader_type == "filesystem":
        loader = FileSystemLoader(source, recursive=recursive)
    elif loader_type == "zip":
        loader = ZipLoader(source)
    elif loader_type == "git":
        loader = GitLoader(source, recursive=recursive)
    elif loader_type == "archive":
        loader = ArchiveLoader(source)
    else:
        loader = FileSystemLoader(source, recursive=recursive)
    return loader.load()


def ingest_to_kb(source: str, loader_type: Optional[str] = None,
                  recursive: bool = True) -> Dict:
    """Load + chunk + add to KB. Returns summary."""
    from core.knowledge import get_kb
    docs = load_documents(source, loader_type, recursive)
    if not docs:
        return {"loaded": 0, "chunks": 0, "errors": 0}
    kb = get_kb()
    total_chunks = 0
    errors = 0
    for doc in docs:
        try:
            chunks = chunk_text(doc["content"])
            for i, chunk in enumerate(chunks):
                kb.store.add(
                    content=chunk,
                    metadata={
                        **doc.get("metadata", {}),
                        "source": doc["source"],
                        "chunk_idx": i,
                        "loader_source": source,
                    },
                )
                total_chunks += 1
        except Exception:
            errors += 1
    kb.save()
    return {
        "loaded": len(docs),
        "chunks": total_chunks,
        "errors": errors,
    }


if __name__ == "__main__":
    docs = load_documents(r"D:\Hermes\skills", recursive=True, loader_type="filesystem")
    print(f"Loaded {len(docs)} files from D:\\Hermes\\skills")
    for d in docs[:3]:
        print(f"  - {d['metadata']['filename']} ({len(d['content'])} chars)")
