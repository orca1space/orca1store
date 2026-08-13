# OrcaMax Code — Final Session Snapshot

This is the complete deliverable bundle from the build session. Everything
needed to run OrcaMax on a fresh Windows machine is here, except for the
LLM model and the cloned AI source repos (both excluded by size).

## What's in this ZIP

```
OrcaMax_Code_FINAL_*.zip
├── core/                     # 24 Python modules (the entire engine)
│   ├── agent_api.py          # NEW — unified 57-op Agent API
│   ├── api_importer.py       # bulk import from external APIs
│   ├── training_daemon.py    # background self-training (7 task types)
│   ├── logging_setup.py      # central logging with rotation
│   ├── llm.py, knowledge.py, skills.py, memory.py, vector_store.py
│   ├── cache.py, quick_replies.py, file_uploads.py, session_state.py
│   ├── skill_search.py, skill_library.py
│   ├── graph.py, checkpoint.py, time_travel.py
│   ├── human_in_loop.py, multi_agent.py
│   ├── memory_store.py, typed_state.py
│   ├── hybrid_search.py, loaders.py
│   ├── config.py, orchestrator.py
│
├── skills/                   # 97 + 1 = 98 JSON skill files
│   ├── imported_registry.json
│   ├── converse_arabic.json, converse_english.json, ...
│   ├── greet_user.json, phrases_arabic.json
│   └── (95 more)
│
├── reports/                  # 3 markdown reports
│   ├── AGENT_API.md          # NEW — 57-op API reference
│   ├── HERMES_VS_LANGGRAPH.md
│   └── HERMES_V2_GAP_CLOSURE.md
│
├── webui/                    # Browser UI assets
│   ├── index.html
│   └── (css, js, icons)
│
├── webui.py                  # HTTP server + SSE streaming (52 KB)
├── OrcaMax.bat               # Windows launcher (detects running server)
├── cli.py                    # Terminal CLI mode
├── train.py                  # Training entry point
├── requirements.txt          # Python deps
├── README.md                 # Project overview
└── NO_SYNC.md                # Anti-cloud-sync warning
```

## What's NOT in this ZIP (and why)

| Missing | Size | Why | How to recover |
|---------|------|-----|----------------|
| `models/qwen2.5-3b-instruct-q4_k_m.gguf` | 2.0 GB | Too large | Run `python -m llama_cpp.download` or fetch from HF |
| `knowledge/` vector store | 112 MB | Re-build on first KB add | Auto-regenerates on first `kb.add` op |
| `sources/` (11 cloned AI repos) | 7.0 GB | Re-clone from Phase 1 zip | Optional — only used for source injection |
| `data/` runtime state | <1 MB | Auto-regenerates | `session_state.json`, `memory_store.json`, `training/training_state.json` are recreated |
| `assets/` (orcamax.ico) | 162 MB | Optional, webui uses inline icon | Add manually if you want desktop shortcut |
| `__pycache__/` | trivial | Cache | Re-created automatically on first run |
| `models/` etc. | varies | | See above |

## How to run

```powershell
# 1. Extract the ZIP
Expand-Archive OrcaMax_Code_FINAL_*.zip -DestinationPath D:\OrcaMax
cd D:\OrcaMax

# 2. Install Python deps
pip install -r requirements.txt

# 3. Download the model (one-time)
#    Place at: D:\OrcaMax\models\qwen2.5-3b-instruct-q4_k_m.gguf
#    Get it from: https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF
#    Choose file: qwen2.5-3b-instruct-q4_k_m.gguf

# 4. Launch
.\OrcaMax.bat
# → server on http://127.0.0.1:7777
# → native app window opens via Edge --app mode
```

## What OrcaMax does

- **Local AI agent** powered by Qwen 2.5 3B (GGUF, Q4_K_M quantization)
- **No cloud, no telemetry, no external services** at runtime
- **Multi-tab UI** with persistent session state
- **Streaming chat** (SSE — first token in <1s)
- **97 skills** including 5-language conversation, programming help, etc.
- **8,800+ KB chunks** ready for RAG
- **Unified Agent API** (57 ops) at `POST /api/agent/exec` — control
  everything programmatically from any external AI agent
- **Training daemon** that runs 7 task types in the background
- **10 architectural features** that close the gaps vs LangGraph
  (graph executor, checkpointing, time travel, HITL, multi-agent,
  long-term memory, typed state, hybrid search, local loaders, etc.)
- **No auth by default** — OrcaMax is local-only and independent
  (set `HERMES_AGENT_TOKEN` + `HERMES_AGENT_AUTH=1` to opt in)

## Token

The Agent API is **auth-free by default**. The previous auto-generated
bearer token has been removed. To re-enable token auth (only needed if
you ever bind the server to a public interface), launch with:

```powershell
$env:HERMES_AGENT_TOKEN = "your-secret-bearer"
$env:HERMES_AGENT_AUTH = "1"
.\OrcaMax.bat
```

## Verification

Tested and working as of the session close:

- 28/28 Agent API ops return `{"ok": true}` with no auth
- LLM streaming chat functional (Qwen 2.5 3B, 2.1 GB on disk)
- 97 skills listed, create/get/update/delete all work
- KB search across 8,800+ chunks
- 6 Wikipedia articles imported as real KB content
- Training daemon starts/pauses/resumes
- Multi-tab UI persists state across restarts
- Path traversal blocked, input validation on all 17 endpoints

## Session deliverables summary

| Phase | Output |
|-------|--------|
| Core engine | 24 Python modules in `core/` |
| Skills | 98 files in `skills/` (97 created/learned + 1 registry) |
| KB | 8,800+ chunks, 384-dim embeddings, BM25+vector hybrid search |
| Reports | 3 markdown reports in `reports/` |
| WebUI | Native Edge --app window, multi-tab, streaming |
| Performance | Cache (600×), quick replies, optimized LLM params |
| Security | Input validation, path traversal blocked, no auth by default |
| Agent API | 57 unified ops at `POST /api/agent/exec` |
| Training | Daemon with 7 task types, budget-aware, persistent state |
| Imports | Wikipedia (real), API bulk import, HF datasets |

---

**OrcaMax Code** — a local AI agent that learns only from you, built on
D: drive, 100% offline, no cloud, no telemetry. Yours to extend.
