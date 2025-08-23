import logging
from typing import Tuple, Optional, Dict, Any
import os
from utils.fallbacks import HAS_TORCH, HAS_TRANSFORMERS, HAS_PEFT, get_model_and_tokenizer

# Import available packages
torch = None
AutoTokenizer = None
AutoModelForCausalLM = None
BitsAndBytesConfig = None
GenerationConfig = None
PeftModel = None
PeftConfig = None

try:
    if HAS_TORCH:
        import torch
except ImportError:
    pass

try:
    if HAS_TRANSFORMERS:
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, GenerationConfig
except ImportError:
    pass

try:
    if HAS_PEFT:
        from peft import PeftModel, PeftConfig
except ImportError:
    pass

logger = logging.getLogger(__name__)

def detect_device() -> str:
    """Detect the best available device for model inference"""
    if HAS_TORCH and torch is not None:
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    else:
        return "cpu"

def get_model_info(model_name: str) -> Dict[str, Any]:
    """Get model information and requirements"""
    model_configs = {
        "google/gemma-2b-it": {
            "size_gb": 4.5,
            "min_ram_gb": 8,
            "recommended_ram_gb": 16,
            "supports_4bit": True,
            "supports_8bit": True,
            "context_length": 8192
        },
        "google/gemma-7b-it": {
            "size_gb": 14,
            "min_ram_gb": 16,
            "recommended_ram_gb": 32,
            "supports_4bit": True,
            "supports_8bit": True,
            "context_length": 8192
        },
        "microsoft/DialoGPT-medium": {
            "size_gb": 1.2,
            "min_ram_gb": 4,
            "recommended_ram_gb": 8,
            "supports_4bit": False,
            "supports_8bit": True,
            "context_length": 1024
        },
        "microsoft/BioGPT": {
            "size_gb": 1.5,
            "min_ram_gb": 4,
            "recommended_ram_gb": 8,
            "supports_4bit": False,
            "supports_8bit": True,
            "context_length": 1024
        }
    }
    
    return model_configs.get(model_name, {
        "size_gb": "unknown",
        "min_ram_gb": 8,
        "recommended_ram_gb": 16,
        "supports_4bit": True,
        "supports_8bit": True,
        "context_length": 2048
    })

def create_quantization_config(quantization_type: str = "4bit") -> Optional[BitsAndBytesConfig]:
    """Create quantization configuration for memory efficiency"""
    
    if quantization_type == "4bit":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    elif quantization_type == "8bit":
        return BitsAndBytesConfig(
            load_in_8bit=True,
        )
    else:
        return None

def load_base_model(
    model_name: str, 
    quantization: Optional[str] = None,
    device_map: str = "auto",
    trust_remote_code: bool = True
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load base model and tokenizer with optimizations
    """
    try:
        # Get model info
        model_info = get_model_info(model_name)
        device = detect_device()
        
        logger.info(f"Loading model {model_name} on device {device}")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
            padding_side="left"
        )
        
        # Add padding token if missing
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        
        # Prepare model loading arguments
        model_kwargs = {
            "trust_remote_code": trust_remote_code,
            "device_map": device_map,
            "torch_dtype": torch.float16 if device != "cpu" else torch.float32,
        }
        
        # Add quantization if specified
        if quantization and device == "cuda":
            quantization_config = create_quantization_config(quantization)
            if quantization_config:
                model_kwargs["quantization_config"] = quantization_config
                logger.info(f"Using {quantization} quantization")
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_kwargs
        )
        
        # Enable gradient checkpointing for memory efficiency
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
        
        logger.info(f"Successfully loaded {model_name}")
        return model, tokenizer
        
    except Exception as e:
        logger.error(f"Error loading model {model_name}: {str(e)}")
        raise e

def load_peft_model(
    base_model_name: str,
    peft_model_path: str,
    quantization: Optional[str] = None
) -> Tuple[PeftModel, AutoTokenizer]:
    """Load a PEFT (LoRA) model"""
    try:
        # Load base model and tokenizer
        base_model, tokenizer = load_base_model(base_model_name, quantization)
        
        # Load PEFT model
        model = PeftModel.from_pretrained(base_model, peft_model_path)
        
        logger.info(f"Successfully loaded PEFT model from {peft_model_path}")
        return model, tokenizer
        
    except Exception as e:
        logger.error(f"Error loading PEFT model: {str(e)}")
        raise e

def merge_peft_model(peft_model: PeftModel) -> AutoModelForCausalLM:
    """Merge PEFT adapters with base model"""
    try:
        merged_model = peft_model.merge_and_unload()
        logger.info("Successfully merged PEFT adapters")
        return merged_model
    except Exception as e:
        logger.error(f"Error merging PEFT model: {str(e)}")
        raise e

def create_generation_config(
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50,
    repetition_penalty: float = 1.1,
    do_sample: bool = True
) -> GenerationConfig:
    """Create optimized generation configuration"""
    
    return GenerationConfig(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        do_sample=do_sample,
        pad_token_id=None,  # Will be set from tokenizer
        eos_token_id=None,  # Will be set from tokenizer
        use_cache=True,
        return_dict_in_generate=True,
        output_scores=True
    )

def estimate_memory_usage(model_name: str, quantization: Optional[str] = None) -> Dict[str, float]:
    """Estimate memory usage for model loading"""
    model_info = get_model_info(model_name)
    base_size_gb = model_info.get("size_gb", 8)
    
    if isinstance(base_size_gb, str):
        base_size_gb = 8  # Default fallback
    
    memory_estimates = {
        "base_model_gb": base_size_gb,
        "recommended_ram_gb": model_info.get("recommended_ram_gb", 16)
    }
    
    if quantization == "4bit":
        memory_estimates["quantized_size_gb"] = base_size_gb * 0.25
        memory_estimates["total_estimated_gb"] = base_size_gb * 0.4  # Include overhead
    elif quantization == "8bit":
        memory_estimates["quantized_size_gb"] = base_size_gb * 0.5
        memory_estimates["total_estimated_gb"] = base_size_gb * 0.7
    else:
        memory_estimates["total_estimated_gb"] = base_size_gb * 1.2  # Include overhead
    
    return memory_estimates

def check_model_compatibility(model_name: str) -> Dict[str, Any]:
    """Check if model is compatible with current environment"""
    device = detect_device()
    model_info = get_model_info(model_name)
    
    # Get available memory
    available_memory_gb = 0
    if device == "cuda":
        try:
            available_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        except:
            available_memory_gb = 8  # Conservative estimate
    else:
        import psutil
        available_memory_gb = psutil.virtual_memory().available / (1024**3)
    
    # Check compatibility
    min_required = model_info.get("min_ram_gb", 8)
    recommended = model_info.get("recommended_ram_gb", 16)
    
    compatibility = {
        "compatible": available_memory_gb >= min_required,
        "optimal": available_memory_gb >= recommended,
        "available_memory_gb": available_memory_gb,
        "min_required_gb": min_required,
        "recommended_gb": recommended,
        "device": device,
        "suggestions": []
    }
    
    if not compatibility["compatible"]:
        compatibility["suggestions"].append("Use 4-bit quantization to reduce memory usage")
        compatibility["suggestions"].append("Consider using a smaller model")
    elif not compatibility["optimal"]:
        compatibility["suggestions"].append("Consider using quantization for better performance")
    
    return compatibility

def optimize_model_for_inference(model: AutoModelForCausalLM) -> AutoModelForCausalLM:
    """Apply optimizations for inference"""
    try:
        # Enable eval mode
        model.eval()
        
        # Compile model if PyTorch 2.0+ is available
        if hasattr(torch, 'compile'):
            try:
                model = torch.compile(model, mode="reduce-overhead")
                logger.info("Model compiled with PyTorch 2.0")
            except Exception as e:
                logger.warning(f"Could not compile model: {str(e)}")
        
        # Enable attention optimizations if available
        if hasattr(model.config, 'use_cache'):
            model.config.use_cache = True
        
        return model
        
    except Exception as e:
        logger.warning(f"Could not apply all optimizations: {str(e)}")
        return model

def get_model_generation_stats(model: AutoModelForCausalLM, tokenizer: AutoTokenizer) -> Dict[str, Any]:
    """Get model statistics and capabilities"""
    try:
        config = model.config
        
        stats = {
            "model_type": getattr(config, 'model_type', 'unknown'),
            "vocab_size": getattr(config, 'vocab_size', len(tokenizer)),
            "hidden_size": getattr(config, 'hidden_size', 'unknown'),
            "num_layers": getattr(config, 'num_hidden_layers', 'unknown'),
            "num_attention_heads": getattr(config, 'num_attention_heads', 'unknown'),
            "max_position_embeddings": getattr(config, 'max_position_embeddings', 'unknown'),
            "torch_dtype": str(model.dtype),
            "device": str(next(model.parameters()).device),
            "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "total_params": sum(p.numel() for p in model.parameters()),
        }
        
        # Calculate model size in MB
        param_size = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
        stats["model_size_mb"] = (param_size + buffer_size) / (1024 * 1024)
        
        return stats
        
    except Exception as e:
        logger.error(f"Error getting model stats: {str(e)}")
        return {"error": str(e)}

def save_model_safely(
    model: AutoModelForCausalLM, 
    tokenizer: AutoTokenizer, 
    save_path: str,
    safe_serialization: bool = True,
    max_shard_size: str = "2GB"
) -> bool:
    """Save model and tokenizer safely"""
    try:
        os.makedirs(save_path, exist_ok=True)
        
        # Save model
        model.save_pretrained(
            save_path,
            safe_serialization=safe_serialization,
            max_shard_size=max_shard_size
        )
        
        # Save tokenizer
        tokenizer.save_pretrained(save_path)
        
        logger.info(f"Model and tokenizer saved to {save_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error saving model: {str(e)}")
        return False
