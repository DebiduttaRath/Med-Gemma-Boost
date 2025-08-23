# components/nemo_rag.py
"""
NeMoRAG: Hybrid retrieval + generation wrapper (full file)

Features:
- Re-uses FAISS index + sentence-transformers embedder (no duplicate KB)
- Guarded NeMo support (only active if nemo_toolkit is installed)
- Supports backends: "faiss", "nemo", "hybrid"
- HuggingFace generator helper (free/open-source models)
- Robust merging, normalization, deduplication, and metadata
- Clear hooks for adapting NeMo retriever/generator loading
"""

from __future__ import annotations
import os
import logging
from typing import List, Dict, Any, Callable, Optional, Tuple, Iterable
import math

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Guarded NeMo imports
try:
    import nemo.collections.nlp as nemo_nlp
    import nemo.collections.nlp.modules.common as nemo_common
    import torch
    NEMO_AVAILABLE = True
except Exception as e:
    nemo_nlp = None
    nemo_common = None
    torch = None
    NEMO_AVAILABLE = False
    _NEMO_IMPORT_ERR = e

# Transformers (HF) imports for fallback generator
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    TRANSFORMERS_AVAILABLE = True
except Exception as e:
    AutoTokenizer = None
    AutoModelForCausalLM = None
    pipeline = None
    TRANSFORMERS_AVAILABLE = False
    _HF_IMPORT_ERR = e

# Utilities
def _normalize_scores(scores: Iterable[float]) -> List[float]:
    """Min-max normalize a list of scores to range [0,1]."""
    s = list(scores)
    if not s:
        return []
    mn, mx = min(s), max(s)
    if mx - mn < 1e-12:
        return [0.5 for _ in s]
    return [(x - mn) / (mx - mn) for x in s]

def _dedupe_keep_best(items: List[Dict[str, Any]], key_fn=lambda x: x.get("text","")) -> List[Dict[str, Any]]:
    """
    Deduplicate by key_fn(text) keeping the highest-scoring entry for identical keys.
    Returns list sorted by score desc.
    """
    best = {}
    for it in items:
        k = key_fn(it).strip()
        if not k:
            # if empty key, include as unique using id
            k = f"_empty_{id(it)}"
        cur_score = float(it.get("score", 0.0))
        if k not in best or cur_score > float(best[k].get("score", 0.0)):
            best[k] = it
    return sorted(list(best.values()), key=lambda x: float(x.get("score", 0.0)), reverse=True)

class NeMoRAG:
    """
    NeMoRAG class:
      - attach FAISS index + passages + embedder via set_faiss_index()
      - retrieve(query, k, backend) with backend in {"faiss","nemo","hybrid"}
      - load_hf_generator() to use HF OSS models for generation
      - generate_with_hf() or generate_with_nemo() to produce answers
    """

    def __init__(self, settings):
        self.settings = settings
        self.nemo_available = NEMO_AVAILABLE
        self.nemo_model = None
        self.nemo_retriever = None
        self.faiss_index = None
        self.passages: List[Dict[str,Any]] = []
        self.embedder = None
        # choose device string
        try:
            self.device = "cuda" if (torch is not None and torch.cuda.is_available()) else "cpu"
        except Exception:
            self.device = "cpu"
        # HF generator
        self.hf_pipeline = None
        self.hf_tokenizer = None
        self.hf_model = None
        # fusion weights
        self.default_faiss_weight = float(os.getenv("HYBRID_FAISS_WEIGHT", "1.0"))
        self.default_nemo_weight = float(os.getenv("HYBRID_NEMO_WEIGHT", "1.0"))
        # active backend default
        self.active_backend = getattr(self.settings, "RAG_BACKEND", "faiss").lower()

        if self.nemo_available:
            logger.info("NeMo detected — NeMoRAG: NeMo features may be enabled.")
        else:
            logger.info("NeMo not detected. NeMo features disabled; using FAISS/HF fallbacks.")

    # -----------------------------
    # Index wiring / reuse methods
    # -----------------------------
    def set_faiss_index(self, index, passages: List[Dict[str,Any]], embedder=None):
        """Attach existing FAISS index + passages + embedder from your RAGSystem."""
        self.faiss_index = index
        self.passages = passages or []
        self.embedder = embedder
        logger.info(f"[NeMoRAG] FAISS index attached (n_passages={len(self.passages)})")

    # -----------------------------
    # NeMo loader scaffolds
    # -----------------------------
    def load_nemo_generator(self, config: Optional[Dict[str,Any]] = None) -> bool:
        """
        Attempt to load a NeMo generator model. This is a scaffold: adapt to the exact NeMo model API you plan to use.
        Example config keys: {'model_path': '/path/to/checkpoint', 'pretrained_name': 'nvidia/...'}
        """
        if not self.nemo_available:
            logger.warning("NeMo not installed; cannot load NeMo generator.")
            return False
        try:
            model_path = None
            if config:
                model_path = config.get("model_path") or config.get("pretrained_name")
            if not model_path:
                logger.warning("No model_path provided in config. Please supply 'model_path' or 'pretrained_name'.")
                return False

            logger.info(f"[NeMoRAG] Loading NeMo generator from {model_path} (scaffold)...")
            # TODO: replace with concrete NeMo loader call appropriate for the chosen NeMo model.
            # e.g. self.nemo_model = nemo_nlp.models.YourModel.restore_from(model_path)
            # for now, keep as scaffold:
            self.nemo_model = None
            logger.warning("NeMo generator loader is scaffolded. Implement loader for your NeMo model.")
            return False
        except Exception as e:
            logger.exception(f"Failed to load NeMo generator: {e}")
            return False

    def load_nemo_retriever(self, config: Optional[Dict[str,Any]] = None) -> bool:
        """
        Attempt to load a NeMo retriever (scaffold). If you have a NeMo retriever, implement here.
        If not implemented, nemo_retrieve will gracefully return [] and hybrid mode will fall back to FAISS.
        """
        if not self.nemo_available:
            logger.warning("NeMo not installed; cannot load NeMo retriever.")
            return False
        try:
            logger.info("[NeMoRAG] load_nemo_retriever scaffold called — implement retriever init here.")
            # TODO: implement depending on your chosen NeMo retriever
            self.nemo_retriever = None
            return False
        except Exception as e:
            logger.exception(f"Failed to load NeMo retriever: {e}")
            return False

    # -----------------------------
    # HuggingFace generator loader (free/open-source)
    # -----------------------------
    def load_hf_generator(self, model_name: str = None, device: Optional[str] = None) -> bool:
        """
        Load a HuggingFace text-generation pipeline for free/open-source models.
        model_name examples: "mistralai/Mistral-7B-Instruct-v0.2", "bigscience/bloomz-7b1", "OmerShah/medgemma-270m" (small)
        Device: "cuda" or "cpu". For CUDA we try to use device_map="auto".
        Returns True on success.
        """
        if not TRANSFORMERS_AVAILABLE:
            logger.error("transformers not available. Install 'transformers' to use HF generator.")
            return False
        device = device or self.device
        if model_name is None:
            # gentle default small model (change if you have others)
            model_name = os.getenv("HF_DEFAULT_GEN", getattr(self.settings, "BASE_MODEL", "facebook/opt-125m"))

        try:
            logger.info(f"[NeMoRAG] Loading HF generator '{model_name}' on device={device}")
            # Load tokenizer + model with safe options
            # Use device_map="auto" if CUDA available, else load on CPU
            from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline as hf_pipeline
            if device == "cuda":
                tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
                model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=None)
                # pipeline will attempt to place model on GPU
                pipe = hf_pipeline("text-generation", model=model, tokenizer=tokenizer, device=0)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
                model = AutoModelForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True)
                pipe = hf_pipeline("text-generation", model=model, tokenizer=tokenizer, device=-1)
            # store references
            self.hf_tokenizer = tokenizer
            self.hf_model = model
            self.hf_pipeline = pipe
            logger.info("[NeMoRAG] HF generator loaded successfully.")
            return True
        except Exception as e:
            logger.exception(f"Failed to load HF generator '{model_name}': {e}")
            return False

    # -----------------------------
    # Retrieval methods
    # -----------------------------
    def faiss_retrieve(self, query: str, k: int = 5) -> List[Dict[str,Any]]:
        """Retrieve from the attached FAISS index using the attached embedder."""
        if self.faiss_index is None or self.embedder is None:
            logger.debug("faiss_retrieve: missing index or embedder; returning [].")
            return []
        try:
            q_emb = self.embedder.encode([query], convert_to_numpy=True)
            # normalize
            try:
                import numpy as np
                q_emb = q_emb / (np.linalg.norm(q_emb, axis=1, keepdims=True) + 1e-12)
            except Exception:
                pass
            D, I = self.faiss_index.search(q_emb, k)
            results = []
            for i, idx in enumerate(I[0]):
                if idx >= 0 and idx < len(self.passages):
                    p = self.passages[idx].copy()
                    score = float(D[0][i]) if D is not None else 0.0
                    p["score"] = score
                    p["_backend"] = "faiss"
                    results.append(p)
            return results
        except Exception as e:
            logger.exception(f"faiss_retrieve error: {e}")
            return []

    def nemo_retrieve(self, query: str, k: int = 5) -> List[Dict[str,Any]]:
        """
        Use NeMo retriever if available. If not implemented or not available, return [].
        This is a scaffold — implement according to the NeMo retriever API you adopt.
        """
        if not self.nemo_available:
            logger.debug("nemo_retrieve: NeMo not available. Returning [].")
            return []
        if self.nemo_retriever is None:
            logger.debug("nemo_retrieve: No NeMo retriever loaded. Returning [].")
            return []

        try:
            # Example pseudocode (adapt to your retriever):
            # hits = self.nemo_retriever.retrieve(query, top_k=k)
            # results = [{"id": h.id, "text": h.text, "source": getattr(h, "source", "nemo"), "score": float(h.score), "_backend": "nemo"} for h in hits]
            logger.warning("nemo_retrieve: NeMo retriever scaffold in use — implement concrete retrieval logic.")
            return []
        except Exception as e:
            logger.exception(f"nemo_retrieve error: {e}")
            return []

    def _merge_and_rerank(self,
                          faiss_docs: List[Dict[str,Any]],
                          nemo_docs: List[Dict[str,Any]],
                          k: int,
                          faiss_weight: float,
                          nemo_weight: float) -> List[Dict[str,Any]]:
        """
        Merge FAISS + NeMo doc lists, normalize scores, apply weights, dedupe, and return top-k.
        """
        combined = []
        faiss_scores = [d.get("score", 0.0) for d in faiss_docs]
        nemo_scores = [d.get("score", 0.0) for d in nemo_docs]

        faiss_norm = _normalize_scores(faiss_scores) if faiss_scores else []
        nemo_norm = _normalize_scores(nemo_scores) if nemo_scores else []

        for i, d in enumerate(faiss_docs):
            d2 = dict(d)
            d2["_orig_score"] = float(d.get("score", 0.0))
            d2["_score_norm"] = faiss_norm[i] if i < len(faiss_norm) else 0.0
            d2["_combined_score"] = faiss_weight * d2["_score_norm"]
            combined.append(d2)

        for i, d in enumerate(nemo_docs):
            d2 = dict(d)
            d2["_orig_score"] = float(d.get("score", 0.0))
            d2["_score_norm"] = nemo_norm[i] if i < len(nemo_norm) else 0.0
            d2["_combined_score"] = nemo_weight * d2["_score_norm"]
            combined.append(d2)

        # Deduplicate and keep highest combined_score
        deduped = _dedupe_keep_best(combined, key_fn=lambda x: x.get("text",""))
        deduped_sorted = sorted(deduped, key=lambda x: float(x.get("_combined_score", 0.0)), reverse=True)
        topk = deduped_sorted[:k]
        for doc in topk:
            doc["score"] = float(doc.get("_combined_score", 0.0))
        return topk

    def retrieve(self, query: str, k: int = 5, backend: Optional[str] = None) -> List[Dict[str,Any]]:
        """
        Unified retrieval entrypoint supporting 'faiss', 'nemo', 'hybrid'.
        - backend None -> uses settings.RAG_BACKEND if available, else 'faiss'
        """
        backend = backend or getattr(self.settings, "RAG_BACKEND", "faiss")
        backend = backend.lower()
        if backend == "faiss":
            return self.faiss_retrieve(query, k)
        elif backend == "nemo":
            # prefer NeMo retriever, but fallback to FAISS if empty
            nemo_docs = self.nemo_retrieve(query, k)
            if nemo_docs:
                return nemo_docs
            logger.debug("nemo_retrieve returned empty — fallback to faiss_retrieve.")
            return self.faiss_retrieve(query, k)
        elif backend == "hybrid":
            faiss_docs = self.faiss_retrieve(query, k*2) if self.faiss_index is not None else []
            nemo_docs = self.nemo_retrieve(query, k*2) if self.nemo_available else []
            if not faiss_docs and not nemo_docs:
                return []
            merged = self._merge_and_rerank(faiss_docs, nemo_docs, k, self.default_faiss_weight, self.default_nemo_weight)
            for d in merged:
                if "_backend" not in d:
                    d["_backend"] = "hybrid"
            return merged
        else:
            logger.warning(f"Unknown retrieval backend '{backend}', falling back to 'faiss'.")
            return self.faiss_retrieve(query, k)

    # -----------------------------
    # Generation helpers
    # -----------------------------
    def _compose_retrieved_block(self, contexts: List[Dict[str,Any]], max_chars_each: int = 1000) -> str:
        """Create a compact retrieved block for prompts with provenance."""
        if not contexts:
            return ""
        parts = []
        for i, c in enumerate(contexts, 1):
            text = c.get("text","")
            text_preview = text if len(text) <= max_chars_each else text[:max_chars_each].rsplit(" ",1)[0] + "..."
            src = c.get("source", c.get("_backend","unknown"))
            score = c.get("score", None)
            if score is not None:
                parts.append(f"[{i}] Source: {src} (score={score:.3f})\n{text_preview}")
            else:
                parts.append(f"[{i}] Source: {src}\n{text_preview}")
        return "\n\n".join(parts)

    def generate_with_hf(self,
                         hf_generator_fn: Optional[Callable[[str, Optional[str], int], str]],
                         question: str,
                         contexts: List[Dict[str,Any]],
                         max_new_tokens: Optional[int] = None,
                         temperature: Optional[float] = None) -> Tuple[str, Dict[str,Any]]:
        """
        Use a HuggingFace generator pipeline (self.hf_pipeline) if loaded, otherwise call hf_generator_fn
        hf_generator_fn signature: (prompt: str, context_block: Optional[str], max_new_tokens: int) -> str
        """
        if max_new_tokens is None:
            max_new_tokens = int(getattr(self.settings, "MAX_NEW_TOKENS", 256))
        if temperature is None:
            temperature = float(getattr(self.settings, "TEMPERATURE", 0.2))

        retrieved_block = self._compose_retrieved_block(contexts)
        system_prompt = getattr(self.settings, "SAFETY_SYSTEM_PROMPT", None) or \
                        "You are a medical-education assistant. Do NOT provide personalized medical advice. Cite sources."
        prompt = f"{system_prompt}\n\nRetrieved Documents:\n{retrieved_block}\n\nUser: {question}\nAssistant:"

        # Prefer pipeline if available
        if self.hf_pipeline is not None:
            try:
                outputs = self.hf_pipeline(prompt, max_new_tokens=max_new_tokens, do_sample=True, temperature=temperature)
                # pipeline returned a list of dicts with 'generated_text'
                gen_text = outputs[0].get("generated_text", "")
                # Remove prompt prefix if model echoes it
                if gen_text.startswith(prompt):
                    response = gen_text[len(prompt):].strip()
                else:
                    response = gen_text.strip()
                meta = {"generator":"hf", "ok":True, "prompt_len": len(prompt), "num_ctx": len(contexts)}
                return response, meta
            except Exception as e:
                logger.exception(f"HF pipeline generation failed: {e}")
                # fallback to user callable if present
        if hf_generator_fn is not None:
            try:
                response = hf_generator_fn(prompt, retrieved_block, max_new_tokens)
                return response, {"generator":"hf", "ok":True}
            except Exception as e:
                logger.exception(f"User-provided HF generator function failed: {e}")
                return (f"HF generation error: {e}", {"generator":"hf", "ok":False})
        return ("No HF generator available. Call load_hf_generator() or pass a hf_generator_fn.", {"generator":"hf", "ok":False})

    def generate_with_nemo(self,
                           question: str,
                           contexts: List[Dict[str,Any]],
                           max_new_tokens: int = 256,
                           generation_config: Optional[Dict[str,Any]] = None) -> Tuple[str, Dict[str,Any]]:
        """
        Attempt to generate using an attached NeMo generator model (self.nemo_model).
        This function is a scaffold and must be adapted to the exact NeMo model API you will use.
        """
        if not self.nemo_available:
            return ("NeMo not installed on this machine.", {"generator":"nemo", "ok":False})
        if self.nemo_model is None:
            return ("NeMo generator not loaded. Call load_nemo_generator() first.", {"generator":"nemo", "ok":False})

        retrieved_block = self._compose_retrieved_block(contexts)
        system_prompt = getattr(self.settings, "SAFETY_SYSTEM_PROMPT", None) or \
                        "You are a medical-education assistant. Do NOT provide personalized medical advice. Cite sources."
        prompt = f"{system_prompt}\n\nRetrieved Documents:\n{retrieved_block}\n\nUser: {question}\nAssistant:"

        try:
            # TODO: replace following pseudocode with real NeMo generation calls
            # outputs = self.nemo_model.generate(prompts=[prompt], max_length=max_new_tokens, **(generation_config or {}))
            # response = outputs[0]
            logger.warning("generate_with_nemo() is scaffolded — implement according to your NeMo model API.")
            return ("NeMo generation scaffold: implement generate_with_nemo() for your NeMo model.", {"generator":"nemo", "ok":False})
        except Exception as e:
            logger.exception(f"NeMo generation error: {e}")
            return (f"NeMo generation error: {e}", {"generator":"nemo", "ok":False})

    # -----------------------------
    # Status & utilities
    # -----------------------------
    def status(self) -> Dict[str,Any]:
        """Return status information for debugging / UI display."""
        return {
            "nemo_available": self.nemo_available,
            "nemo_model_loaded": bool(self.nemo_model),
            "nemo_retriever_loaded": bool(self.nemo_retriever),
            "hf_generator_loaded": bool(self.hf_pipeline),
            "faiss_index_attached": bool(self.faiss_index),
            "num_passages": len(self.passages) if self.passages else 0,
            "device": self.device,
            "active_backend": getattr(self.settings, "RAG_BACKEND", "faiss"),
            "faiss_weight": self.default_faiss_weight,
            "nemo_weight": self.default_nemo_weight
        }

    def info(self) -> str:
        s = self.status()
        return f"NeMoRAG(status={s})"
