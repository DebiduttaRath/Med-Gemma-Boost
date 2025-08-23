import streamlit as st
import os
import json
import shutil
import subprocess
from pathlib import Path
import logging
from typing import Dict, Any, Optional
from config.settings import Settings
from utils.fallbacks import HAS_TORCH, HAS_TRANSFORMERS

# Import available packages
torch = None
AutoTokenizer = None
AutoModelForCausalLM = None
HfApi = None
create_repo = None

try:
    if HAS_TORCH:
        import torch
except ImportError:
    pass

try:
    if HAS_TRANSFORMERS:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from huggingface_hub import HfApi, create_repo
except ImportError:
    pass

logger = logging.getLogger(__name__)

class ModelExport:
    def __init__(self):
        self.settings = Settings()
        self.export_formats = {
            "lora_adapters": "LoRA Adapter Weights",
            "merged_fp16": "Merged FP16 Model", 
            "merged_fp32": "Merged FP32 Model",
            "gguf": "GGUF Format (llama.cpp)",
            "onnx": "ONNX Format",
            "tensorrt": "TensorRT Optimized"
        }
        
    def export_lora_adapters(self, model_path: str, output_path: str) -> bool:
        """Export LoRA adapter weights only"""
        try:
            # LoRA adapters are typically already saved during training
            if os.path.exists(model_path):
                if not os.path.exists(output_path):
                    os.makedirs(output_path)
                
                # Copy adapter files
                adapter_files = ["adapter_config.json", "adapter_model.bin", "adapter_model.safetensors"]
                
                for file in adapter_files:
                    src = os.path.join(model_path, file)
                    if os.path.exists(src):
                        dst = os.path.join(output_path, file)
                        shutil.copy2(src, dst)
                
                # Also copy tokenizer files
                tokenizer_files = ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.txt"]
                
                for file in tokenizer_files:
                    src = os.path.join(model_path, file)
                    if os.path.exists(src):
                        dst = os.path.join(output_path, file)
                        shutil.copy2(src, dst)
                
                return True
            return False
            
        except Exception as e:
            logger.error(f"Error exporting LoRA adapters: {str(e)}")
            return False
    
    def export_merged_model(self, model_path: str, output_path: str, precision: str = "fp16") -> bool:
        """Export merged model with base weights + LoRA adapters"""
        try:
            from peft import PeftModel, PeftConfig
            
            # Load the configuration
            peft_config = PeftConfig.from_pretrained(model_path)
            
            # Load base model
            base_model = AutoModelForCausalLM.from_pretrained(
                peft_config.base_model_name_or_path,
                torch_dtype=torch.float16 if precision == "fp16" else torch.float32,
                device_map="auto"
            )
            
            # Load LoRA model
            model = PeftModel.from_pretrained(base_model, model_path)
            
            # Merge and unload
            merged_model = model.merge_and_unload()
            
            # Save merged model
            merged_model.save_pretrained(
                output_path,
                safe_serialization=True,
                max_shard_size="2GB"
            )
            
            # Save tokenizer
            tokenizer = AutoTokenizer.from_pretrained(peft_config.base_model_name_or_path)
            tokenizer.save_pretrained(output_path)
            
            return True
            
        except Exception as e:
            logger.error(f"Error exporting merged model: {str(e)}")
            return False
    
    def export_to_gguf(self, model_path: str, output_path: str, quantization: str = "q4_0") -> bool:
        """Export model to GGUF format for llama.cpp"""
        try:
            # This requires llama.cpp tools to be installed
            convert_script = "convert.py"  # from llama.cpp
            
            if not shutil.which("python") or not os.path.exists(convert_script):
                st.warning("llama.cpp conversion tools not found. Please install llama.cpp and ensure convert.py is available.")
                return False
            
            # Convert to GGUF
            cmd = [
                "python", convert_script,
                model_path,
                "--outfile", output_path,
                "--outtype", quantization
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                return True
            else:
                logger.error(f"GGUF conversion failed: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Error exporting to GGUF: {str(e)}")
            return False
    
    def export_to_onnx(self, model_path: str, output_path: str) -> bool:
        """Export model to ONNX format"""
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch
            
            # Load model and tokenizer
            model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float32)
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            
            # Create dummy input
            dummy_input = tokenizer("Hello world", return_tensors="pt")
            
            # Export to ONNX
            torch.onnx.export(
                model,
                (dummy_input["input_ids"], dummy_input["attention_mask"]),
                output_path,
                input_names=["input_ids", "attention_mask"],
                output_names=["logits"],
                dynamic_axes={
                    "input_ids": {0: "batch_size", 1: "sequence_length"},
                    "attention_mask": {0: "batch_size", 1: "sequence_length"},
                    "logits": {0: "batch_size", 1: "sequence_length"}
                },
                opset_version=11
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Error exporting to ONNX: {str(e)}")
            return False
    
    def create_model_card(self, model_info: Dict[str, Any]) -> str:
        """Create a model card with metadata"""
        card_content = f"""
    # {model_info.get('model_name', 'Medical AI Model')}

    ## Model Description

    This is a fine-tuned medical AI model based on {model_info.get('base_model', 'Unknown')} for educational purposes.

    ## Training Details

    - **Base Model**: {model_info.get('base_model', 'Unknown')}
    - **Training Data**: {model_info.get('training_examples', 0)} examples
    - **Fine-tuning Method**: LoRA (Low-Rank Adaptation)
    - **Training Date**: {model_info.get('training_date', 'Unknown')}

    ## Performance Metrics

    {self.format_metrics(model_info.get('metrics', {}))}

    ## Usage

    This model is intended for educational purposes only and should not be used for actual medical diagnosis or treatment recommendations.

    ```python
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel

    # Load the model
    base_model = AutoModelForCausalLM.from_pretrained("{model_info.get('base_model', '')}")
    model = PeftModel.from_pretrained(base_model, "path/to/adapter")
    tokenizer = AutoTokenizer.from_pretrained("{model_info.get('base_model', '')}")

    # Generate response
    inputs = tokenizer("Your medical question here", return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=256)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    ```

    ## Safety Considerations

    ⚠️ **Important**: This model is for educational purposes only. Do not use for actual medical diagnosis or treatment decisions.

    ## License

    Please refer to the base model's license for usage terms.
        """
        return card_content.strip()
    
    def format_metrics(self, metrics: Dict[str, Any]) -> str:
        """Format metrics for model card"""
        if not metrics:
            return "No evaluation metrics available."
        
        formatted = []
        for metric_name, value in metrics.items():
            if isinstance(value, float):
                formatted.append(f"- **{metric_name.replace('_', ' ').title()}**: {value:.3f}")
            else:
                formatted.append(f"- **{metric_name.replace('_', ' ').title()}**: {value}")
        
        return "\n".join(formatted)
    
    def render(self):
        """Render the model export interface"""
        st.header("📦 Model Export & Deployment")
        
        # Check if model exists
        if not st.session_state.model_trained:
            st.warning("⚠️ No trained model available. Please complete fine-tuning first.")
            return
        
        st.subheader("Export Options")
        
        # Export format selection
        export_format = st.selectbox(
            "Select Export Format",
            list(self.export_formats.keys()),
            format_func=lambda x: self.export_formats[x]
        )
        
        # Export settings
        col1, col2 = st.columns(2)
        
        with col1:
            model_path = st.text_input(
                "Source Model Path", 
                value="./fine_tuned_model",
                help="Path to the trained model directory"
            )
        
        with col2:
            output_path = st.text_input(
                "Export Path",
                value=f"./exports/{export_format}_model",
                help="Where to save the exported model"
            )
        
        # Format-specific options
        if export_format == "merged_fp16" or export_format == "merged_fp32":
            st.info("💡 Merged models include both base weights and LoRA adapters for easier deployment.")
        
        elif export_format == "gguf":
            st.info("💡 GGUF format is optimized for llama.cpp inference on CPU and edge devices.")
            quantization = st.selectbox(
                "GGUF Quantization",
                ["q4_0", "q4_1", "q5_0", "q5_1", "q8_0", "f16", "f32"],
                index=0
            )
        
        elif export_format == "onnx":
            st.info("💡 ONNX format provides cross-platform compatibility and optimization.")
        
        elif export_format == "lora_adapters":
            st.info("💡 LoRA adapters are lightweight and can be easily shared or swapped.")
        
        # Model information for card
        st.subheader("Model Information")
        
        with st.form("model_info_form"):
            model_name = st.text_input("Model Name", value="Medical-AI-Assistant")
            model_description = st.text_area(
                "Model Description",
                value="A fine-tuned medical AI assistant for educational purposes."
            )
            training_date = st.date_input("Training Date")
            
            submit_export = st.form_submit_button("Export Model", type="primary")
        
        # Export execution
        if submit_export:
            model_info = {
                "model_name": model_name,
                "base_model": self.settings.BASE_MODEL,
                "training_date": training_date.isoformat(),
                "description": model_description,
                "training_examples": getattr(st.session_state, 'training_examples', 0),
                "metrics": getattr(st.session_state, 'evaluation_metrics', {})
            }
            
            with st.spinner(f"Exporting model in {self.export_formats[export_format]} format..."):
                success = False
                
                try:
                    if export_format == "lora_adapters":
                        success = self.export_lora_adapters(model_path, output_path)
                    
                    elif export_format in ["merged_fp16", "merged_fp32"]:
                        precision = "fp16" if export_format == "merged_fp16" else "fp32"
                        success = self.export_merged_model(model_path, output_path, precision)
                    
                    elif export_format == "gguf":
                        success = self.export_to_gguf(model_path, output_path, quantization)
                    
                    elif export_format == "onnx":
                        success = self.export_to_onnx(model_path, output_path)
                    
                    else:
                        st.error(f"Export format {export_format} not yet implemented.")
                        success = False
                    
                    if success:
                        st.success(f"✅ Model exported successfully to {output_path}")
                        
                        # Create and save model card
                        model_card = self.create_model_card(model_info)
                        card_path = f"{output_path}/README.md"
                        
                        try:
                            os.makedirs(output_path, exist_ok=True)
                            with open(card_path, 'w') as f:
                                f.write(model_card)
                            st.success(f"📄 Model card saved to {card_path}")
                        except Exception as e:
                            st.warning(f"Could not save model card: {str(e)}")
                        
                        # Show download option
                        if os.path.exists(output_path):
                            st.subheader("Download Options")
                            st.info(f"Model exported to: `{output_path}`")
                            
                            # Display model card preview
                            with st.expander("📄 Model Card Preview"):
                                st.markdown(model_card)
                    
                    else:
                        st.error("❌ Export failed. Please check the logs for details.")
                        
                except Exception as e:
                    st.error(f"❌ Export failed: {str(e)}")
        
        # Deployment guidance
        st.subheader("🚀 Deployment Options")
        
        tab1, tab2, tab3 = st.tabs(["Local Serving", "Cloud Deployment", "Edge Deployment"])
        
        with tab1:
            st.markdown("""
            ### Local Serving Options
            
            **For LoRA Adapters:**
            ```python
            from transformers import AutoTokenizer, AutoModelForCausalLM
            from peft import PeftModel
            
            # Load base model
            base_model = AutoModelForCausalLM.from_pretrained("google/gemma-2b-it")
            tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b-it")
            
            # Load LoRA adapters
            model = PeftModel.from_pretrained(base_model, "./exports/lora_adapters")
            ```
            """)
        
        with tab2:
            st.markdown("""
            ### Cloud Deployment
            
            **AWS SageMaker:**
            1. Upload model to S3
            2. Create SageMaker endpoint
            3. Deploy with auto-scaling
            """)
        
        with tab3:
            st.markdown("""
            ### Edge Deployment
            
            **For GGUF Models (CPU/Edge):**
            ```bash
            # Install llama.cpp
            git clone https://github.com/ggerganov/llama.cpp
            cd llama.cpp && make
            
            # Run inference
            ./main -m ./exports/gguf_model/model.gguf -p "Your medical question"
            ```
            """)
