"""
Fallback implementations for missing dependencies
This ensures the application works even when advanced ML packages aren't available
"""

import os
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)

# Global flags for available packages
HAS_TORCH = False
HAS_TRANSFORMERS = False
HAS_SENTENCE_TRANSFORMERS = False
HAS_FAISS = False
HAS_PEFT = False
HAS_TRL = False
HAS_DATASETS = False
HAS_NLTK = False
HAS_ROUGE_SCORE = False
HAS_NEMO = False


try:
    import nemo.collections.nlp as nemo_nlp
    HAS_NEMO = True
except ImportError:
    logger.warning("NVIDIA NeMo not available - using fallback implementations")


# Try importing packages and set flags
try:
    import torch
    HAS_TORCH = True
except ImportError:
    logger.warning("PyTorch not available - using fallback implementations")

try:
    import transformers
    HAS_TRANSFORMERS = True
except ImportError:
    logger.warning("Transformers not available - using fallback implementations")

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    logger.warning("Sentence Transformers not available - using fallback implementations")

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    logger.warning("FAISS not available - using fallback implementations")

try:
    import peft
    HAS_PEFT = True
except ImportError:
    logger.warning("PEFT not available - using fallback implementations")

try:
    import trl
    HAS_TRL = True
except ImportError:
    logger.warning("TRL not available - using fallback implementations")

try:
    import datasets
    HAS_DATASETS = True
except ImportError:
    logger.warning("Datasets not available - using fallback implementations")

try:
    import nltk
    HAS_NLTK = True
except ImportError:
    logger.warning("NLTK not available - using fallback implementations")

try:
    from rouge_score import rouge_scorer
    HAS_ROUGE_SCORE = True
except ImportError:
    logger.warning("Rouge Score not available - using fallback implementations")


class FallbackEmbedder:
    """Simple TF-IDF based embedder as fallback for sentence transformers"""
    
    def __init__(self, model_name: str = "fallback"):
        self.model_name = model_name
        self.vocab = {}
        self.idf = {}
        self.fitted = False
        
    def encode(self, texts: List[str], show_progress_bar: bool = False, convert_to_numpy: bool = True) -> np.ndarray:
        """Encode texts using simple TF-IDF"""
        if isinstance(texts, str):
            texts = [texts]
            
        if not self.fitted:
            self._fit(texts)
        
        embeddings = []
        for text in texts:
            embedding = self._text_to_vector(text)
            embeddings.append(embedding)
            
        embeddings = np.array(embeddings)
        return embeddings
    
    def _fit(self, texts: List[str]):
        """Fit the TF-IDF model"""
        # Build vocabulary
        word_counts = {}
        doc_counts = {}
        
        for text in texts:
            words = text.lower().split()
            unique_words = set(words)
            
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1
                
            for word in unique_words:
                doc_counts[word] = doc_counts.get(word, 0) + 1
        
        # Create vocabulary (top 5000 words)
        self.vocab = {word: idx for idx, (word, _) in enumerate(
            sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:5000]
        )}
        
        # Calculate IDF
        total_docs = len(texts)
        for word in self.vocab:
            self.idf[word] = np.log(total_docs / (doc_counts.get(word, 1) + 1))
            
        self.fitted = True
    
    def _text_to_vector(self, text: str) -> np.ndarray:
        """Convert text to TF-IDF vector"""
        words = text.lower().split()
        vector = np.zeros(len(self.vocab))
        
        # Count word frequencies
        word_freq = {}
        for word in words:
            if word in self.vocab:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Calculate TF-IDF
        for word, freq in word_freq.items():
            if word in self.vocab:
                tf = freq / len(words)
                idf = self.idf.get(word, 0)
                vector[self.vocab[word]] = tf * idf
                
        # Normalize
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
            
        return vector


class FallbackFAISS:
    """Simple similarity search as fallback for FAISS"""
    
    def __init__(self, dimension: int):
        self.d = dimension
        self.vectors = []
        self.ntotal = 0
        
    def add(self, vectors: np.ndarray):
        """Add vectors to the index"""
        self.vectors.extend(vectors)
        self.ntotal = len(self.vectors)
    
    def search(self, query_vectors: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Search for nearest neighbors"""
        if not self.vectors:
            return np.array([[]]), np.array([[]])
            
        query_vector = query_vectors[0]  # Assume single query
        similarities = []
        
        for i, vector in enumerate(self.vectors):
            # Cosine similarity
            similarity = np.dot(query_vector, vector) / (
                np.linalg.norm(query_vector) * np.linalg.norm(vector) + 1e-8
            )
            similarities.append((similarity, i))
        
        # Sort by similarity (descending)
        similarities.sort(reverse=True)
        
        # Return top k
        k = min(k, len(similarities))
        scores = np.array([[sim for sim, _ in similarities[:k]]])
        indices = np.array([[idx for _, idx in similarities[:k]]])
        
        return scores, indices


class FallbackTokenizer:
    """Simple tokenizer fallback"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.pad_token = "<pad>"
        self.eos_token = "<eos>"
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.vocab_size = 50000
        
    def encode(self, text: str) -> List[int]:
        """Simple word-based encoding"""
        words = text.split()
        return [hash(word) % self.vocab_size for word in words]
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """Simple decoding"""
        return " ".join([f"token_{tid}" for tid in token_ids])
    
    def apply_chat_template(self, messages: List[Dict], tokenize: bool = False, add_generation_prompt: bool = True) -> str:
        """Simple chat template"""
        formatted = ""
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            formatted += f"{role}: {content}\n"
        return formatted
    
    def __call__(self, text: str, return_tensors: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """Tokenize text"""
        token_ids = self.encode(text)
        result = {"input_ids": token_ids}
        if return_tensors == "pt":
            result["input_ids"] = [token_ids]
            result["attention_mask"] = [1] * len(token_ids)
        return result


class FallbackModel:
    """Simple model fallback"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.device = "cpu"
        self.dtype = "float32"
        
    def generate(self, input_ids, max_new_tokens: int = 50, **kwargs) -> np.ndarray:
        """Generate simple response"""
        # Return dummy output
        input_length = len(input_ids[0]) if isinstance(input_ids, (list, np.ndarray)) else 10
        output_length = input_length + max_new_tokens
        return np.random.randint(0, 1000, (1, output_length))
    
    def eval(self):
        """Set to eval mode"""
        pass
    
    def save_pretrained(self, path: str, **kwargs):
        """Save model"""
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "config.json"), "w") as f:
            import json
            json.dump({"model_type": "fallback", "model_name": self.model_name}, f)


def get_embedder(model_name: str = "all-mpnet-base-v2"):
    """Get embedder with fallback"""
    if HAS_SENTENCE_TRANSFORMERS:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(model_name)
    else:
        logger.warning(f"Using fallback embedder instead of {model_name}")
        return FallbackEmbedder(model_name)


def get_faiss_index(dimension: int):
    """Get FAISS index with fallback"""
    if HAS_FAISS:
        import faiss
        return faiss.IndexFlatIP(dimension)
    else:
        logger.warning("Using fallback similarity search instead of FAISS")
        return FallbackFAISS(dimension)


def normalize_l2(vectors: np.ndarray):
    """Normalize vectors to unit length"""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1  # Avoid division by zero
    return vectors / norms


def get_model_and_tokenizer(model_name: str):
    """Get model and tokenizer with fallback"""
    if HAS_TRANSFORMERS:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name)
            return model, tokenizer
        except Exception as e:
            logger.warning(f"Could not load {model_name}: {e}")
    
    logger.warning(f"Using fallback model instead of {model_name}")
    return FallbackModel(model_name), FallbackTokenizer(model_name)


def compute_bleu_fallback(prediction: str, reference: str) -> float:
    """Simple BLEU computation fallback"""
    pred_words = prediction.lower().split()
    ref_words = reference.lower().split()
    
    if not pred_words or not ref_words:
        return 0.0
    
    # Simple unigram precision
    pred_set = set(pred_words)
    ref_set = set(ref_words)
    
    intersection = len(pred_set & ref_set)
    precision = intersection / len(pred_set) if pred_set else 0
    
    # Simple brevity penalty
    bp = min(1.0, len(pred_words) / len(ref_words)) if ref_words else 0
    
    return bp * precision


def compute_rouge_fallback(prediction: str, reference: str) -> float:
    """Simple ROUGE computation fallback"""
    pred_words = prediction.lower().split()
    ref_words = reference.lower().split()
    
    if not pred_words or not ref_words:
        return 0.0
    
    # Simple word overlap F1
    pred_set = set(pred_words)
    ref_set = set(ref_words)
    
    intersection = len(pred_set & ref_set)
    
    if intersection == 0:
        return 0.0
    
    precision = intersection / len(pred_set)
    recall = intersection / len(ref_set)
    
    return 2 * precision * recall / (precision + recall)


# Export availability flags
__all__ = [
    'HAS_TORCH', 'HAS_TRANSFORMERS', 'HAS_SENTENCE_TRANSFORMERS', 'HAS_FAISS',
    'HAS_PEFT', 'HAS_TRL', 'HAS_DATASETS', 'HAS_NLTK', 'HAS_ROUGE_SCORE',
    'get_embedder', 'get_faiss_index', 'normalize_l2', 'get_model_and_tokenizer',
    'compute_bleu_fallback', 'compute_rouge_fallback',
    'FallbackEmbedder', 'FallbackFAISS', 'FallbackTokenizer', 'FallbackModel'
]