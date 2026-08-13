"""
Hermes File Uploads
Local file ingestion for the + button in the UI.
Accepts user files, saves them to D:\\Hermes\\uploads\\, and ingests their
content into the knowledge base. Pure local, no external services.

Supported: txt, md, markdown, rst, py, json, csv, log, html, xml, yaml, yml, ini, toml

Returns: {saved: <path>, chunks: <count>, preview: <first 300 chars>}
"""
import shutil
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.config import HERMES_ROOT
from core.logging_setup import get_logger

log = get_logger("file_uploads")

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB hard limit per upload


UPLOADS_DIR = HERMES_ROOT / "uploads"

# Text-like extensions that we can read directly
TEXT_EXTS = {
    ".txt", ".md", ".markdown", ".rst",
    ".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".c", ".cpp", ".h", ".hpp",
    ".cs", ".go", ".rs", ".rb", ".php", ".sh", ".ps1", ".bat", ".cmd",
    ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".conf",
    ".csv", ".tsv", ".log", ".html", ".htm", ".xml", ".css", ".scss",
    ".sql", ".env", ".gitignore", ".mdx",
}
MAX_FILE_BYTES = 10 * 1024 * 1024  # 10 MB per file


class FileUploader:
    """Manages file uploads and ingestion into the knowledge base."""

    def __init__(self, uploads_dir: Path = UPLOADS_DIR):
        self.dir = uploads_dir
        self.dir.mkdir(parents=True, exist_ok=True)

    def _safe_name(self, original_name: str) -> str:
        """Make filename safe: keep alphanum, dot, dash, underscore, CJK chars."""
        stem = Path(original_name).stem
        suffix = Path(original_name).suffix.lower()
        safe = "".join(c if (c.isalnum() or c in "._- ") or ord(c) > 127 else "_"
                       for c in stem).strip("._- ")
        if not safe:
            safe = "file"
        return f"{safe}_{uuid.uuid4().hex[:6]}{suffix}"

    def save_upload(self, original_name: str, content: bytes) -> Tuple[Path, str]:
        """
        Save uploaded bytes to disk with a unique name.
        Returns (absolute path, readable text content).
        Raises ValueError on invalid file.
        """
        if not content:
            raise ValueError("Empty file")
        if len(content) > MAX_FILE_BYTES:
            raise ValueError(
                f"File too large ({len(content)} bytes). Max {MAX_FILE_BYTES} bytes."
            )

        suffix = Path(original_name).suffix.lower()
        if suffix not in TEXT_EXTS:
            raise ValueError(
                f"Unsupported file type: {suffix}. Supported: {sorted(TEXT_EXTS)}"
            )

        # Decode as UTF-8 with replace
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError:
            try:
                text = content.decode("utf-8", errors="replace")
            except Exception:
                raise ValueError("File is not valid UTF-8 text")

        safe_name = self._safe_name(original_name)
        target = self.dir / safe_name
        target.write_bytes(content)
        return target, text

    def ingest(self, original_name: str, content: bytes) -> Dict:
        """
        Full pipeline: save to disk, add to knowledge base.
        Returns summary dict.
        """
        if not isinstance(content, (bytes, bytearray)):
            raise ValueError("content must be bytes")
        if len(content) > MAX_FILE_SIZE:
            raise ValueError(
                f"File too large: {len(content)} bytes (max {MAX_FILE_SIZE})"
            )
        if not original_name or not isinstance(original_name, str):
            raise ValueError("original_name must be a non-empty string")
        from core.knowledge import get_kb
        path, text = self.save_upload(original_name, content)

        kb = get_kb()
        ids = kb.add_text(
            text,
            source=f"upload:{path.name}",
            metadata={
                "filename": path.name,
                "original_name": original_name,
                "uploaded_at": time.time(),
                "size_bytes": len(content),
                "kind": "upload",
            },
        )
        kb.save()
        preview = text[:300].replace("\n", " ").strip()
        if len(text) > 300:
            preview += "..."

        return {
            "saved": str(path),
            "filename": path.name,
            "original_name": original_name,
            "chunks": len(ids),
            "size_bytes": len(content),
            "preview": preview,
            "total_chars": len(text),
        }

    def list_uploads(self, limit: int = 50) -> List[Dict]:
        """List recently uploaded files."""
        items = []
        for p in sorted(self.dir.glob("*"), key=lambda x: x.stat().st_mtime, reverse=True):
            if not p.is_file():
                continue
            st = p.stat()
            items.append({
                "filename": p.name,
                "path": str(p),
                "size_bytes": st.st_size,
                "modified": st.st_mtime,
            })
            if len(items) >= limit:
                break
        return items

    def stats(self) -> Dict:
        files = list(self.dir.glob("*"))
        files = [f for f in files if f.is_file()]
        total_bytes = sum(f.stat().st_size for f in files)
        return {
            "count": len(files),
            "total_bytes": total_bytes,
            "max_per_file": MAX_FILE_BYTES,
            "supported": sorted(TEXT_EXTS),
            "dir": str(self.dir),
        }


_uploader: Optional[FileUploader] = None


def get_uploader() -> FileUploader:
    global _uploader
    if _uploader is None:
        _uploader = FileUploader()
    return _uploader


if __name__ == "__main__":
    u = get_uploader()
    print(u.stats())
