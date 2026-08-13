"""
Hermes Local LLM Client
Direct wrapper for llama-cpp-python. No HTTP, no service.
Loads the GGUF model into memory once, then handles chat + embeddings.

For embeddings, uses sentence-transformers with a small efficient model.
"""
import os
# Hermes is permanently local-only: model libraries may use local caches but must never query a hub.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
import sys
import time
from pathlib import Path
from typing import List, Dict, Optional, Generator

HERMES_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(HERMES_ROOT))
from core.config import (
    HERMES_MODEL_FILE, HERMES_MODEL_REPO, HERMES_MODEL_FILENAME,
    LLM_CONTEXT_WINDOW, LLM_MAX_TOKENS, LLM_MAX_TOKENS_HARD, LLM_TEMPERATURE,
    LLM_N_THREADS, LLM_N_GPU_LAYERS, LLM_N_BATCH,
    LLM_USE_MMAP, LLM_USE_MLOCK, LLM_F16_KV,
    LLM_STOP_TOKENS,
    EMBED_MODEL_NAME, EMBED_DIM,
    QUERY_EMBED_CACHE_SIZE as LLM_EMBED_CACHE_SIZE,
)


class HermesLLM:
    """Local LLM using llama-cpp-python."""

    def __init__(self,
                 model_file: Path = HERMES_MODEL_FILE,
                 model_repo: str = HERMES_MODEL_REPO,
                 model_filename: str = HERMES_MODEL_FILENAME,
                 n_ctx: int = LLM_CONTEXT_WINDOW,
                 n_threads: int = LLM_N_THREADS,
                 n_gpu_layers: int = LLM_N_GPU_LAYERS):
        self.model_file = Path(model_file)
        self.model_repo = model_repo
        self.model_filename = model_filename
        self.n_ctx = n_ctx
        self.n_threads = n_threads if n_threads > 0 else (os.cpu_count() or 4)
        self.n_gpu_layers = n_gpu_layers
        self._llm = None
        self._embedder = None
        self._load_time = None
        # Query embedding cache (LRU, thread-safe)
        self._embed_cache: Dict[str, List[float]] = {}
        self._embed_cache_lock = __import__("threading").RLock()

    def _ensure_model(self):
        """Download the model if not present."""
        if self.model_file.exists():
            return
        print(f"[Hermes] Model not found locally. Downloading from {self.model_repo}...")
        print(f"[Hermes] This may take several minutes depending on connection speed.")
        try:
            from huggingface_hub import hf_hub_download
            downloaded = hf_hub_download(
                repo_id=self.model_repo,
                filename=self.model_filename,
                local_dir=str(self.model_file.parent),
            )
            print(f"[Hermes] Downloaded to: {downloaded}")
        except ImportError:
            print("[Hermes] huggingface_hub not installed. Installing...")
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub"])
            from huggingface_hub import hf_hub_download
            downloaded = hf_hub_download(
                repo_id=self.model_repo,
                filename=self.model_filename,
                local_dir=str(self.model_file.parent),
            )

    def _get_llm(self):
        """Lazy-load the LLM (heavy operation, done once)."""
        if self._llm is not None:
            return self._llm
        self._ensure_model()
        # Resolve n_threads: -1 or 0 = auto-detect physical cores
        n_threads = self.n_threads
        if n_threads <= 0:
            n_threads = max(1, (os.cpu_count() or 4))
        print(f"[Hermes] Loading model: {self.model_file.name}")
        print(f"[Hermes] Threads: {n_threads}, GPU layers: {self.n_gpu_layers}, "
              f"Context: {self.n_ctx}, Batch: {LLM_N_BATCH}, f16_kv: {LLM_F16_KV}")
        t0 = time.time()
        from llama_cpp import Llama
        self._llm = Llama(
            model_path=str(self.model_file),
            n_ctx=self.n_ctx,
            n_threads=n_threads,
            n_gpu_layers=self.n_gpu_layers,
            n_batch=LLM_N_BATCH,
            use_mmap=LLM_USE_MMAP,
            use_mlock=LLM_USE_MLOCK,
            f16_kv=LLM_F16_KV,
            verbose=False,
        )
        self._load_time = time.time() - t0
        print(f"[Hermes] Model loaded in {self._load_time:.1f}s")
        return self._llm

    def _get_embedder(self):
        """Lazy-load the embedding model."""
        if self._embedder is not None:
            return self._embedder
        print(f"[Hermes] Loading embedding model: {EMBED_MODEL_NAME}")
        t0 = time.time()
        from sentence_transformers import SentenceTransformer
        self._embedder = SentenceTransformer(EMBED_MODEL_NAME, device="cpu")
        # Warmup
        self._embedder.encode(["warmup"], show_progress_bar=False)
        print(f"[Hermes] Embedder loaded in {time.time()-t0:.1f}s, dim={EMBED_DIM}")
        return self._embedder

    # === Health & info ===
    def health(self) -> bool:
        """Check if model is loaded and ready."""
        return self._llm is not None or self.model_file.exists()

    def pre_warm(self):
        """Load model + embedder eagerly. Call at startup so first request is instant."""
        self._get_llm()
        self._get_embedder()
        # Warmup the LLM with a tiny completion
        try:
            self._llm.create_chat_completion(
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=4,
                temperature=0.0,
                stream=False,
            )
        except Exception:
            pass
        print(f"[Hermes] Pre-warm complete. Ready for instant responses.")

    def model_info(self) -> Dict:
        return {
            "model_file": str(self.model_file),
            "exists": self.model_file.exists(),
            "size_mb": round(self.model_file.stat().st_size / (1024*1024), 1) if self.model_file.exists() else 0,
            "loaded": self._llm is not None,
            "load_time_s": round(self._load_time, 1) if self._load_time else None,
            "context": self.n_ctx,
            "threads": self.n_threads,
            "gpu_layers": self.n_gpu_layers,
            "embedding_model": EMBED_MODEL_NAME,
            "embedding_dim": EMBED_DIM,
        }

    # === Chat ===
    def chat(self, messages: List[Dict[str, str]],
             temperature: float = LLM_TEMPERATURE,
             max_tokens: int = None) -> Dict:
        """Chat completion. messages = [{"role": ..., "content": ...}, ...]"""
        if max_tokens is None:
            max_tokens = LLM_MAX_TOKENS
        # Hard cap to prevent runaway generation
        if max_tokens > LLM_MAX_TOKENS_HARD:
            max_tokens = LLM_MAX_TOKENS_HARD
        llm = self._get_llm()
        result = llm.create_chat_completion(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stop=LLM_STOP_TOKENS,
            stream=False,
        )
        return {
            "message": result["choices"][0]["message"],
            "usage": result.get("usage", {}),
        }

    def chat_stream(self, messages: List[Dict[str, str]],
                    temperature: float = LLM_TEMPERATURE,
                    max_tokens: int = None) -> Generator[str, None, None]:
        """Stream chat completion token by token."""
        if max_tokens is None:
            max_tokens = LLM_MAX_TOKENS
        if max_tokens > LLM_MAX_TOKENS_HARD:
            max_tokens = LLM_MAX_TOKENS_HARD
        llm = self._get_llm()
        stream = llm.create_chat_completion(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stop=LLM_STOP_TOKENS,
            stream=True,
        )
        for chunk in stream:
            delta = chunk["choices"][0].get("delta", {})
            content = delta.get("content", "")
            if content:
                yield content

    def generate(self, prompt: str,
                 temperature: float = LLM_TEMPERATURE,
                 max_tokens: int = None,
                 system: Optional[str] = None) -> str:
        """Simple text completion (no chat formatting)."""
        if max_tokens is None:
            max_tokens = LLM_MAX_TOKENS
        if max_tokens > LLM_MAX_TOKENS_HARD:
            max_tokens = LLM_MAX_TOKENS_HARD
        llm = self._get_llm()
        result = llm.create_completion(
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            stop=LLM_STOP_TOKENS,
        )
        return result["choices"][0]["text"]

    # === Embeddings ===
    def embed(self, text: str) -> List[float]:
        """Get embedding for a single text. Uses LRU cache."""
        if not isinstance(text, str) or not text:
            return [0.0] * EMBED_DIM
        key = text.strip().lower()
        with self._embed_cache_lock:
            if key in self._embed_cache:
                return self._embed_cache[key]
        result = self.embed_batch([text])[0]
        with self._embed_cache_lock:
            # Evict LRU if full
            if len(self._embed_cache) >= LLM_EMBED_CACHE_SIZE:
                # remove oldest (first item)
                try:
                    oldest = next(iter(self._embed_cache))
                    del self._embed_cache[oldest]
                except (StopIteration, KeyError):
                    pass
            self._embed_cache[key] = result
        return result

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for multiple texts."""
        embedder = self._get_embedder()
        vectors = embedder.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        return [v.tolist() for v in vectors]


_llm_instance: Optional[HermesLLM] = None


def get_llm() -> HermesLLM:
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = HermesLLM()
    return _llm_instance


if __name__ == "__main__":
    llm = get_llm()
    print("Model info:", llm.model_info())
