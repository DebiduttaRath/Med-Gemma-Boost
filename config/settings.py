import os
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

class Settings:
    """Application settings and configuration"""
    
    def __init__(self):
        # Model settings
        self.BASE_MODEL = os.getenv("BASE_MODEL", "microsoft/DialoGPT-medium")
        # Allowed base models (OSS + NeMo-compatible)
        self.ALLOWED_MODELS = [
            "meta-llama/Llama-3-8b-instruct",
            "meta-llama/Llama-3-70b-instruct",
            "mistralai/Mistral-7B-Instruct-v0.3",
            "mistralai/Mixtral-8x7B-Instruct-v0.1",
            "microsoft/phi-3-mini-4k-instruct",
            "tiiuae/falcon-7b-instruct",
            "tiiuae/falcon-40b-instruct",
            "microsoft/BioGPT",
            "allenai/biomedlm",
            "StanfordAIMI/MedAlpaca",
            "nvidia/nemo-guardrails",        # Safety / governance
            "nvidia/nv-embed-qa",            # RAG embedding
            "nvidia/megatron-gpt-20b",       # GPT-style LM
            "nvidia/megatron-gpt-530b",      # Very large LM
            "nvidia/stt_en_conformer_transducer_large", # ASR
            "nvidia/tts_en_fastpitch",       # TTS
        ]
        
        # Reset BASE_MODEL if not valid
        if self.BASE_MODEL not in self.ALLOWED_MODELS:
            logger.warning(f"BASE_MODEL={self.BASE_MODEL} not in allowed models, defaulting to {self.ALLOWED_MODELS[0]}")
            self.BASE_MODEL = self.ALLOWED_MODELS[0]
            
        self.MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "256"))
        self.TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
        self.TOP_P = float(os.getenv("TOP_P", "0.9"))
        self.TOP_K = int(os.getenv("TOP_K", "50"))
        self.REPETITION_PENALTY = float(os.getenv("REPETITION_PENALTY", "1.1"))
        
        # RAG settings
        self.EMBED_MODEL = os.getenv("EMBED_MODEL", "all-mpnet-base-v2")
        self.RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "3"))
        self.RETRIEVAL_THRESHOLD = float(os.getenv("RETRIEVAL_THRESHOLD", "0.7"))
        
        # RAG backend selector: faiss | nemo | hybrid
        self.RAG_BACKEND = os.getenv("RAG_BACKEND", "faiss").lower()
        if self.RAG_BACKEND not in ["faiss", "nemo", "hybrid"]:
            logger.warning(f"Unknown RAG_BACKEND={self.RAG_BACKEND}, defaulting to 'faiss'")
            self.RAG_BACKEND = "faiss"

        # NeMo specific configs (free/open-source defaults)
        self.NEMO_MODEL = os.getenv("NEMO_MODEL", "nvidia/nemo-guardrails")  
        self.NEMO_EMBED_MODEL = os.getenv("NEMO_EMBED_MODEL", "nvidia/nv-embed-qa")  
        self.NEMO_TOP_K = int(os.getenv("NEMO_TOP_K", "5"))

        # Hybrid configs
        self.HYBRID_FUSION_METHOD = os.getenv("HYBRID_FUSION_METHOD", "reciprocal_rank_fusion")
                
        # Fine-tuning settings
        self.LORA_R = int(os.getenv("LORA_R", "64"))
        self.LORA_ALPHA = int(os.getenv("LORA_ALPHA", "128"))
        self.LORA_DROPOUT = float(os.getenv("LORA_DROPOUT", "0.1"))
        self.LORA_TARGET_MODULES = os.getenv(
            "LORA_TARGET_MODULES", 
            "q_proj,v_proj,k_proj,o_proj"
        ).split(",")
        
        # Training settings
        self.LEARNING_RATE = float(os.getenv("LEARNING_RATE", "2e-5"))
        self.BATCH_SIZE = int(os.getenv("BATCH_SIZE", "2"))
        self.GRADIENT_ACCUMULATION_STEPS = int(os.getenv("GRADIENT_ACCUMULATION_STEPS", "2"))
        self.NUM_EPOCHS = int(os.getenv("NUM_EPOCHS", "3"))
        self.WEIGHT_DECAY = float(os.getenv("WEIGHT_DECAY", "0.01"))
        self.WARMUP_STEPS = int(os.getenv("WARMUP_STEPS", "100"))
        self.MAX_SEQ_LENGTH = int(os.getenv("MAX_SEQ_LENGTH", "1024"))
        
        # Evaluation settings
        self.EVAL_BATCH_SIZE = int(os.getenv("EVAL_BATCH_SIZE", "4"))
        self.EVAL_STEPS = int(os.getenv("EVAL_STEPS", "100"))
        self.SAVE_STEPS = int(os.getenv("SAVE_STEPS", "500"))
        self.LOGGING_STEPS = int(os.getenv("LOGGING_STEPS", "10"))
        
        # Safety settings
        self.SAFETY_ENABLED = os.getenv("SAFETY_ENABLED", "true").lower() == "true"
        self.CITATION_REQUIRED = os.getenv("CITATION_REQUIRED", "true").lower() == "true"
        self.MEDICAL_DISCLAIMER = os.getenv("MEDICAL_DISCLAIMER", "true").lower() == "true"
        
        # Storage settings
        self.DATA_DIR = os.getenv("DATA_DIR", "data")
        self.MODEL_DIR = os.getenv("MODEL_DIR", "models")
        self.EXPORT_DIR = os.getenv("EXPORT_DIR", "exports")
        self.LOGS_DIR = os.getenv("LOGS_DIR", "logs")
        
        # API settings
        self.HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN", "")
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
        self.ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
        
        # Performance settings
        self.USE_QUANTIZATION = os.getenv("USE_QUANTIZATION", "auto")  # auto, 4bit, 8bit, none
        self.DEVICE_MAP = os.getenv("DEVICE_MAP", "auto")
        self.MAX_MEMORY = os.getenv("MAX_MEMORY", "auto")
        self.LOW_CPU_MEM_USAGE = os.getenv("LOW_CPU_MEM_USAGE", "true").lower() == "true"
        
        # Database settings (for future use)
        self.DATABASE_URL = os.getenv("DATABASE_URL", "")
        self.REDIS_URL = os.getenv("REDIS_URL", "")
        
        # Monitoring settings
        self.ENABLE_TENSORBOARD = os.getenv("ENABLE_TENSORBOARD", "true").lower() == "true"
        self.ENABLE_WANDB = os.getenv("ENABLE_WANDB", "false").lower() == "true"
        self.WANDB_PROJECT = os.getenv("WANDB_PROJECT", "medgemma-ai")
        
        # Security settings
        self.SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-here")
        self.ALLOWED_HOSTS = os.getenv("ALLOWED_HOSTS", "*").split(",")
        
        # Create directories
        self._create_directories()
        
        # Validate settings
        self._validate_settings()
    
    def _create_directories(self):
        """Create necessary directories"""
        directories = [
            self.DATA_DIR,
            self.MODEL_DIR,
            self.EXPORT_DIR,
            self.LOGS_DIR,
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def _validate_settings(self):
        """Validate configuration settings"""
        # Validate temperature range
        if not 0.0 <= self.TEMPERATURE <= 2.0:
            logger.warning(f"Temperature {self.TEMPERATURE} is outside recommended range [0.0, 2.0]")
        
        # Validate top_p range
        if not 0.0 <= self.TOP_P <= 1.0:
            logger.warning(f"Top-p {self.TOP_P} is outside valid range [0.0, 1.0]")
        
        # Validate learning rate
        if not 1e-6 <= self.LEARNING_RATE <= 1e-2:
            logger.warning(f"Learning rate {self.LEARNING_RATE} is outside recommended range [1e-6, 1e-2]")
        
        # Validate LoRA settings
        if self.LORA_R <= 0:
            logger.error("LoRA rank (r) must be positive")
        
        if self.LORA_ALPHA <= 0:
            logger.error("LoRA alpha must be positive")
        
        # Validate batch sizes
        if self.BATCH_SIZE <= 0:
            logger.error("Batch size must be positive")
        
        if self.GRADIENT_ACCUMULATION_STEPS <= 0:
            logger.error("Gradient accumulation steps must be positive")
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration dictionary"""
        return {
            "base_model": self.BASE_MODEL,
            "max_new_tokens": self.MAX_NEW_TOKENS,
            "temperature": self.TEMPERATURE,
            "top_p": self.TOP_P,
            "top_k": self.TOP_K,
            "repetition_penalty": self.REPETITION_PENALTY,
        }
    
    def get_rag_config(self) -> Dict[str, Any]:
        """Get RAG configuration dictionary"""
        return {
            "embed_model": self.EMBED_MODEL,
            "retrieval_top_k": self.RETRIEVAL_TOP_K,
            "retrieval_threshold": self.RETRIEVAL_THRESHOLD,
        }
    
    def get_rag_backend_config(self) -> Dict[str, Any]:
        """Get active RAG backend configuration"""
        return {
            "backend": self.RAG_BACKEND,
            "embed_model": self.NEMO_EMBED_MODEL if self.RAG_BACKEND == "nemo" else self.EMBED_MODEL,
            "retrieval_top_k": self.NEMO_TOP_K if self.RAG_BACKEND == "nemo" else self.RETRIEVAL_TOP_K,
            "retrieval_threshold": self.RETRIEVAL_THRESHOLD,
            "nemo_model": self.NEMO_MODEL,
            "nemo_embed_model": self.NEMO_EMBED_MODEL,
            "nemo_top_k": self.NEMO_TOP_K,
            "hybrid_fusion": self.HYBRID_FUSION_METHOD,
        }



    def get_training_config(self) -> Dict[str, Any]:
        """Get training configuration dictionary"""
        return {
            "learning_rate": self.LEARNING_RATE,
            "batch_size": self.BATCH_SIZE,
            "gradient_accumulation_steps": self.GRADIENT_ACCUMULATION_STEPS,
            "num_epochs": self.NUM_EPOCHS,
            "weight_decay": self.WEIGHT_DECAY,
            "warmup_steps": self.WARMUP_STEPS,
            "max_seq_length": self.MAX_SEQ_LENGTH,
            "lora_r": self.LORA_R,
            "lora_alpha": self.LORA_ALPHA,
            "lora_dropout": self.LORA_DROPOUT,
            "lora_target_modules": self.LORA_TARGET_MODULES,
        }
    
    def get_safety_config(self) -> Dict[str, Any]:
        """Get safety configuration dictionary"""
        return {
            "safety_enabled": self.SAFETY_ENABLED,
            "citation_required": self.CITATION_REQUIRED,
            "medical_disclaimer": self.MEDICAL_DISCLAIMER,
        }
    
    def update_setting(self, key: str, value: Any):
        """Update a specific setting"""
        if hasattr(self, key):
            setattr(self, key, value)
            logger.info(f"Updated setting {key} to {value}")
        else:
            logger.warning(f"Unknown setting: {key}")
    
    def export_config(self) -> Dict[str, Any]:
        """Export all configuration as dictionary"""
        config = {}
        
        for attr_name in dir(self):
            if not attr_name.startswith('_') and not callable(getattr(self, attr_name)):
                config[attr_name] = getattr(self, attr_name)
        
        return config
    
    def import_config(self, config: Dict[str, Any]):
        """Import configuration from dictionary"""
        for key, value in config.items():
            if hasattr(self, key):
                setattr(self, key, value)
                logger.info(f"Imported setting {key}")
            else:
                logger.warning(f"Unknown setting in import: {key}")
        
        # Re-validate after import
        self._validate_settings()
    
    def get_quantization_config(self) -> Optional[str]:
        """Get appropriate quantization setting based on environment"""
        if self.USE_QUANTIZATION == "auto":
            try:
                import torch
                if torch.cuda.is_available():
                    # Check VRAM
                    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                    if vram_gb >= 24:
                        return None  # No quantization needed
                    elif vram_gb >= 12:
                        return "8bit"
                    else:
                        return "4bit"
                else:
                    return "8bit"  # Use 8bit for CPU
            except:
                return "4bit"  # Conservative fallback
        elif self.USE_QUANTIZATION == "none":
            return None
        else:
            return self.USE_QUANTIZATION
    
    def get_device_config(self) -> Dict[str, Any]:
        """Get device configuration"""
        try:
            import torch
            
            config = {
                "device_map": self.DEVICE_MAP,
                "max_memory": self.MAX_MEMORY,
                "low_cpu_mem_usage": self.LOW_CPU_MEM_USAGE,
                "quantization": self.get_quantization_config(),
            }
            
            # Add device-specific settings
            if torch.cuda.is_available():
                config["cuda_available"] = True
                config["cuda_device_count"] = torch.cuda.device_count()
                config["cuda_memory_gb"] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            else:
                config["cuda_available"] = False
            
            return config
            
        except ImportError:
            return {
                "device_map": "cpu",
                "max_memory": None,
                "low_cpu_mem_usage": True,
                "quantization": "8bit",
                "cuda_available": False,
            }
    
    def __str__(self) -> str:
        """String representation of settings"""
        return f"Settings(base_model={self.BASE_MODEL}, safety_enabled={self.SAFETY_ENABLED})"
    
    def __repr__(self) -> str:
        """Detailed representation of settings"""
        return f"Settings({self.export_config()})"

# Global settings instance
settings = Settings()
