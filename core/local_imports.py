"""Local-only adapters for staged open-source capabilities.

No URL fetching, package auto-installation, npx, or remote provider calls are allowed here.
"""
from __future__ import annotations
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Dict

ROOT = Path(__file__).resolve().parent.parent
SOURCES = ROOT / "imported_sources"


def _local_path(value: Any) -> Path:
    p = Path(str(value)).expanduser().resolve()
    if not p.is_file() or not str(p).startswith(str(ROOT.resolve())):
        raise ValueError("only files inside the Hermes root are allowed")
    return p


def _tool(name: str) -> str | None:
    candidates = [
        shutil.which(name),
        shutil.which(name + ".exe"),
        str(ROOT / "bin" / name),
        str(ROOT / "bin" / (name + ".exe")),
    ]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return str(Path(candidate).resolve())
    return None

def _status(name: str, tool: Any, source: str) -> Dict[str, Any]:
    executable = tool if isinstance(tool, str) else None
    available = bool(tool)
    return {"name": name, "source": source, "available": available, "executable": executable, "local_only": True}


def _local_tool_env() -> Dict[str, str]:
    env = os.environ.copy()
    paths = [
        str(ROOT / "vendor" / "ocrmypdf"),
        r"C:\\Program Files\\Tesseract-OCR",
        r"C:\\Program Files\\gs\\gs10.07.1\\bin",
    ]
    env["PATH"] = os.pathsep.join(paths + [env.get("PATH", "")])
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT / "vendor" / "ocrmypdf"), env.get("PYTHONPATH", "")])
    env["HERMES_OFFLINE_ONLY"] = "1"
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    env["HF_DATASETS_OFFLINE"] = "1"
    return env


def register_local_import_ops(op: Callable[[str], Callable]):
    @op("imports.status")
    def imports_status(params: Dict[str, Any]) -> Any:
        return {
            "local_only": True,
            "sources_root": str(SOURCES),
            "tools": [
                _status("markitdown", "vendor/markitdown" if (ROOT / "vendor" / "markitdown").is_dir() else None, "vendor/markitdown"),
                _status("ocrmypdf", "vendor/ocrmypdf" if (ROOT / "vendor" / "ocrmypdf" / "ocrmypdf").is_dir() else None, "vendor/ocrmypdf"),
                _status("probe", _tool("probe"), "imported_sources/probe"),
                _status("tree-sitter", _tool("probe"), "Probe embedded tree-sitter parser"),
                _status("playwright-mcp", (SOURCES / "playwright-mcp" / "cli.js") if (SOURCES / "playwright-mcp" / "cli.js").is_file() else None, "imported_sources/playwright-mcp"),
            ],
        }

    @op("imports.markdown")
    def imports_markdown(params: Dict[str, Any]) -> Any:
        path = _local_path(params.get("path"))
        try:
            vendor_src = ROOT / "vendor" / "markitdown"
            mark_src = SOURCES / "markitdown" / "packages" / "markitdown" / "src"
            for candidate in (vendor_src, mark_src):
                if candidate.is_dir() and str(candidate) not in sys.path:
                    sys.path.insert(0, str(candidate))
            from markitdown import MarkItDown  # type: ignore
        except Exception as exc:
            return {"ok": False, "error": "markitdown_unavailable", "detail": str(exc), "local_only": True}
        result = MarkItDown().convert(str(path))
        text = getattr(result, "text_content", str(result))
        return {"ok": True, "path": str(path), "markdown": text, "local_only": True}

    @op("imports.ocr")
    def imports_ocr(params: Dict[str, Any]) -> Any:
        src = _local_path(params.get("input"))
        if params.get("output"):
            out = Path(str(params.get("output"))).expanduser().resolve()
            if not str(out).startswith(str(ROOT.resolve())):
                raise ValueError("output must remain inside the Hermes root")
        else:
            out = src.with_name(src.stem + ".ocr.pdf")
        package = ROOT / "vendor" / "ocrmypdf" / "ocrmypdf"
        if not package.is_dir():
            return {"ok": False, "error": "ocrmypdf_unavailable", "detail": "Install locally; no network installation is permitted.", "local_only": True}
        cmd = [sys.executable, "-m", "ocrmypdf"]
        if params.get("redo"):
            cmd.append("--redo-ocr")
        cmd.extend([str(src), str(out)])
        proc = subprocess.run(cmd, env=_local_tool_env(), capture_output=True, text=True, timeout=600, check=False)
        return {"ok": proc.returncode == 0, "returncode": proc.returncode, "output": str(out), "stderr": proc.stderr[-4000:], "local_only": True}

    @op("imports.code_search")
    def imports_code_search(params: Dict[str, Any]) -> Any:
        query = str(params.get("query", "")).strip()
        root = _local_path(params.get("root")) if params.get("root") else ROOT
        if not root.is_dir():
            raise ValueError("root must be a Hermes directory")
        exe = _tool("probe")
        if not exe:
            return {"ok": False, "error": "probe_unavailable", "detail": "Build/install Probe locally; automatic npx download is disabled.", "local_only": True}
        proc = subprocess.run([exe, "search", query, str(root), "--format", "json"], capture_output=True, text=True, timeout=120, check=False)
        return {"ok": proc.returncode == 0, "returncode": proc.returncode, "results": proc.stdout, "stderr": proc.stderr[-4000:], "local_only": True}
    @op("imports.code_symbols")
    def imports_code_symbols(params: Dict[str, Any]) -> Any:
        path = _local_path(params.get("path"))
        exe = _tool("probe")
        if not exe:
            return {"ok": False, "error": "probe_unavailable", "detail": "Probe with embedded Tree-sitter parser is not available.", "local_only": True}
        proc = subprocess.run([exe, "symbols", str(path), "--format", "json"], capture_output=True, text=True, timeout=120, check=False)
        return {"ok": proc.returncode == 0, "returncode": proc.returncode, "symbols": proc.stdout, "stderr": proc.stderr[-4000:], "local_only": True, "parser": "tree-sitter"}

    @op("imports.code_query")
    def imports_code_query(params: Dict[str, Any]) -> Any:
        query = str(params.get("query", "")).strip()
        root = _local_path(params.get("root")) if params.get("root") else ROOT
        if not root.is_dir():
            raise ValueError("root must be a Hermes directory")
        exe = _tool("probe")
        if not exe:
            return {"ok": False, "error": "probe_unavailable", "detail": "Probe with embedded Tree-sitter parser is not available.", "local_only": True}
        proc = subprocess.run([exe, "query", query, str(root), "--format", "json"], capture_output=True, text=True, timeout=120, check=False)
        return {"ok": proc.returncode == 0, "returncode": proc.returncode, "matches": proc.stdout, "stderr": proc.stderr[-4000:], "local_only": True, "parser": "tree-sitter"}
    @op("imports.browser_smoke")
    def imports_browser_smoke(params: Dict[str, Any]) -> Any:
        target = str(params.get("url", "http://127.0.0.1:7777/")).strip()
        helper = ROOT / "core" / "local_browser_smoke.js"
        node = shutil.which("node")
        if not node or not helper.is_file():
            return {"ok": False, "error": "playwright_local_unavailable", "local_only": True}
        parsed = __import__("urllib.parse", fromlist=["urlparse"]).urlparse(target)
        allowed = parsed.scheme == "file" or (parsed.scheme == "http" and parsed.hostname in ("127.0.0.1", "localhost"))
        if not allowed:
            return {"ok": False, "error": "external_urls_blocked", "local_only": True}
        proc = subprocess.run([node, str(helper), target], capture_output=True, text=True, timeout=60, check=False)
        return {"ok": proc.returncode == 0, "returncode": proc.returncode, "result": proc.stdout, "stderr": proc.stderr[-4000:], "local_only": True}



    @op("monetization.score_opportunity")
    def monetization_score_opportunity(params: Dict[str, Any]) -> Any:
        from .monetization_engine import dispatch
        return dispatch("monetization.score_opportunity", params)

    @op("monetization.content_brief")
    def monetization_content_brief(params: Dict[str, Any]) -> Any:
        from .monetization_engine import dispatch
        return dispatch("monetization.content_brief", params)

    @op("monetization.policy_check")
    def monetization_policy_check(params: Dict[str, Any]) -> Any:
        from .monetization_engine import dispatch
        return dispatch("monetization.policy_check", params)

    @op("monetization.metrics")
    def monetization_metrics(params: Dict[str, Any]) -> Any:
        from .monetization_engine import dispatch
        return dispatch("monetization.metrics", params)

    @op("monetization.execution_plan")
    def monetization_execution_plan(params: Dict[str, Any]) -> Any:
        from .monetization_engine import dispatch
        return dispatch("monetization.execution_plan", params)
    @op("research.index_local")
    def research_index_local(params: Dict[str, Any]) -> Any:
        from .research_engine import dispatch
        return dispatch("research.index_local", params)

    @op("research.analyze_corpus")
    def research_analyze_corpus(params: Dict[str, Any]) -> Any:
        from .research_engine import dispatch
        return dispatch("research.analyze_corpus", params)

    @op("research.market_trend")
    def research_market_trend(params: Dict[str, Any]) -> Any:
        from .research_engine import dispatch
        return dispatch("research.market_trend", params)

    @op("research.intelligence_report")
    def research_intelligence_report(params: Dict[str, Any]) -> Any:
        from .research_engine import dispatch
        return dispatch("research.intelligence_report", params)
    @op("content.generate_draft")
    def content_generate_draft(params: Dict[str, Any]) -> Any:
        from .content_engine import dispatch
        return dispatch("content.generate_draft", params)

    @op("content.seo_optimize")
    def content_seo_optimize(params: Dict[str, Any]) -> Any:
        from .content_engine import dispatch
        return dispatch("content.seo_optimize", params)

    @op("content.validate_draft")
    def content_validate_draft(params: Dict[str, Any]) -> Any:
        from .content_engine import dispatch
        return dispatch("content.validate_draft", params)
    @op("approval.request")
    def approval_request(params: Dict[str, Any]) -> Any:
        from .approval_engine import dispatch
        return dispatch("approval.request", params)

    @op("approval.status")
    def approval_status(params: Dict[str, Any]) -> Any:
        from .approval_engine import dispatch
        return dispatch("approval.status", params)

    @op("approval.decide")
    def approval_decide(params: Dict[str, Any]) -> Any:
        from .approval_engine import dispatch
        return dispatch("approval.decide", params)
    @op("workflow.adsense_simulation")
    def workflow_adsense_simulation(params: Dict[str, Any]) -> Any:
        from .workflow_engine import dispatch
        return dispatch("workflow.adsense_simulation", params)

    @op("bridge.record_lesson")
    def bridge_record_lesson(params: Dict[str, Any]) -> Any:
        from .mentor_bridge import dispatch
        return dispatch("bridge.record_lesson", params)

    @op("bridge.promote_lesson")
    def bridge_promote_lesson(params: Dict[str, Any]) -> Any:
        from .mentor_bridge import dispatch
        return dispatch("bridge.promote_lesson", params)

    @op("bridge.create_skill")
    def bridge_create_skill(params: Dict[str, Any]) -> Any:
        from .mentor_bridge import dispatch
        return dispatch("bridge.create_skill", params)

    @op("bridge.record_feedback")
    def bridge_record_feedback(params: Dict[str, Any]) -> Any:
        from .mentor_bridge import dispatch
        return dispatch("bridge.record_feedback", params)

    @op("bridge.list_knowledge")
    def bridge_list_knowledge(params: Dict[str, Any]) -> Any:
        from .mentor_bridge import dispatch
        return dispatch("bridge.list_knowledge", params)

    @op("bridge.ingest_file")
    def bridge_ingest_file(params: Dict[str, Any]) -> Any:
        from .mentor_bridge import dispatch
        return dispatch("bridge.ingest_file", params)

    @op("bridge.snapshot")
    def bridge_snapshot(params: Dict[str, Any]) -> Any:
        from .mentor_bridge import dispatch
        return dispatch("bridge.snapshot", params)

    @op("bridge.status")
    def bridge_status(params: Dict[str, Any]) -> Any:
        from .mentor_bridge import dispatch
        return dispatch("bridge.status", params)
    @op("ledger.record_execution")
    def ledger_record_execution(params: Dict[str, Any]) -> Any:
        from .execution_ledger import dispatch
        return dispatch("ledger.record_execution", params)
    @op("ledger.promote_procedure")
    def ledger_promote_procedure(params: Dict[str, Any]) -> Any:
        from .execution_ledger import dispatch
        return dispatch("ledger.promote_procedure", params)
    @op("ledger.list")
    def ledger_list(params: Dict[str, Any]) -> Any:
        from .execution_ledger import dispatch
        return dispatch("ledger.list", params)
    @op("ledger.status")
    def ledger_status(params: Dict[str, Any]) -> Any:
        from .execution_ledger import dispatch
        return dispatch("ledger.status", params)

    @op("mentor.status")
    def mentor_status(params: Dict[str, Any]) -> Any:
        from .mentor_mode import dispatch
        return dispatch("mentor.status", params)
    @op("mentor.connect")
    def mentor_connect(params: Dict[str, Any]) -> Any:
        from .mentor_mode import dispatch
        return dispatch("mentor.connect", params)
    @op("mentor.disconnect")
    def mentor_disconnect(params: Dict[str, Any]) -> Any:
        from .mentor_mode import dispatch
        return dispatch("mentor.disconnect", params)
