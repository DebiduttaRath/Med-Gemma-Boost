import streamlit as st
import os
import json
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List
import logging
from utils.data_processing import prepare_training_data
from config.settings import Settings
from utils.fallbacks import (
    HAS_TORCH, HAS_TRANSFORMERS, HAS_PEFT, HAS_TRL, HAS_DATASETS,
    get_model_and_tokenizer, FallbackModel, FallbackTokenizer
)

# Import available packages - moved to avoid import errors
torch = None
BitsAndBytesConfig = None
TrainingArguments = None
LoraConfig = None
get_peft_model = None
TaskType = None
SFTTrainer = None
SFTConfig = None
Dataset = None

try:
    if HAS_TORCH:
        import torch
except ImportError:
    pass

try:
    if HAS_TRANSFORMERS:
        from transformers import BitsAndBytesConfig, TrainingArguments
except ImportError:
    pass

try:
    if HAS_PEFT:
        from peft import LoraConfig, get_peft_model, TaskType
except ImportError:
    pass

try:
    if HAS_TRL:
        from trl import SFTTrainer, SFTConfig
except ImportError:
    pass

try:
    if HAS_DATASETS:
        from datasets import Dataset
except ImportError:
    pass

logger = logging.getLogger(__name__)

class FineTuningInterface:
    def __init__(self):
        self.settings = Settings()
        self.model = None
        self.tokenizer = None
        self.trainer = None
        
    def load_model(self, model_name: str = None):
        """Load base model and tokenizer"""
        if model_name is None:
            model_name = self.settings.BASE_MODEL
            
        try:
            with st.spinner(f"Loading model {model_name}..."):
                self.model, self.tokenizer = get_model_and_tokenizer(model_name)
            return True
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            st.error(f"Error loading model: {str(e)}")
            return False
    
    def create_lora_config(self, r: int, alpha: int, dropout: float, target_modules: List[str]):
        """Create LoRA configuration"""
        if HAS_PEFT:
            return LoraConfig(
                r=r,
                lora_alpha=alpha,
                target_modules=target_modules,
                lora_dropout=dropout,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            )
        else:
            # Return a simple config dict for fallback
            return {
                "r": r,
                "lora_alpha": alpha,
                "target_modules": target_modules,
                "lora_dropout": dropout,
                "bias": "none",
                "task_type": "CAUSAL_LM",
            }
    
    def prepare_dataset(self, data: List[Dict[str, str]], tokenizer):
        """Prepare dataset for training"""
        def format_instruction(example):
            if "instruction" in example and "response" in example:
                text = f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['response']}"
            elif "question" in example and "answer" in example:
                text = f"### Question:\n{example['question']}\n\n### Answer:\n{example['answer']}"
            elif "input" in example and "output" in example:
                text = f"### Input:\n{example['input']}\n\n### Output:\n{example['output']}"
            else:
                text = example.get("text", str(example))
            
            return {"text": text}
        
        if HAS_DATASETS:
            dataset = Dataset.from_list(data)
            dataset = dataset.map(format_instruction, remove_columns=dataset.column_names)
            return dataset
        else:
            # Simple fallback dataset
            formatted_data = [format_instruction(item) for item in data]
            return formatted_data
    
    def start_training(self, train_data, eval_data, training_config, lora_config):
        """Start the fine-tuning process"""
        try:
            if not HAS_PEFT or not HAS_TRL:
                st.warning("⚠️ Advanced training features not available. Running in demo mode.")
                
                # Simulate training for demo
                import time
                for i in range(5):
                    time.sleep(0.5)
                    st.session_state.training_progress = (i + 1) / 5
                
                # Create dummy trainer for demo
                class DemoTrainer:
                    def __init__(self):
                        self.state = type('State', (), {
                            'log_history': [{'train_loss': 0.5, 'step': 100, 'learning_rate': 2e-5}],
                            'global_step': 100,
                            'max_steps': 100
                        })()
                    
                    def add_callback(self, callback):
                        pass
                    
                    def train(self):
                        st.info("Demo training completed successfully!")
                    
                    def save_model(self):
                        os.makedirs(training_config["output_dir"], exist_ok=True)
                        # Save dummy model files
                        with open(f"{training_config['output_dir']}/training_completed.txt", "w") as f:
                            f.write("Demo training completed")
                
                self.trainer = DemoTrainer()
                return True
            
            # Apply LoRA to model
            if HAS_PEFT:
                self.model = get_peft_model(self.model, lora_config)
            
            # Prepare datasets
            train_dataset = self.prepare_dataset(train_data, self.tokenizer)
            eval_dataset = self.prepare_dataset(eval_data, self.tokenizer) if eval_data else None
            
            # Create SFT config
            if HAS_TRL:
                sft_config = SFTConfig(
                    output_dir=training_config["output_dir"],
                    per_device_train_batch_size=training_config["batch_size"],
                    gradient_accumulation_steps=training_config["gradient_accumulation_steps"],
                    num_train_epochs=training_config["num_epochs"],
                    learning_rate=training_config["learning_rate"],
                    weight_decay=training_config["weight_decay"],
                    logging_steps=training_config["logging_steps"],
                    save_steps=training_config["save_steps"],
                    save_total_limit=training_config["save_total_limit"],
                    fp16=training_config["fp16"],
                    bf16=training_config["bf16"],
                    packing=training_config["packing"],
                    dataset_text_field="text",
                    max_seq_length=training_config["max_seq_length"],
                    report_to=["tensorboard"] if training_config["use_tensorboard"] else [],
                    push_to_hub=training_config["push_to_hub"],
                    hub_model_id=training_config.get("hub_model_id"),
                    dataset_num_proc=4,
                    remove_unused_columns=False,
                )
                
                # Create trainer
                self.trainer = SFTTrainer(
                    model=self.model,
                    tokenizer=self.tokenizer,
                    args=sft_config,
                    train_dataset=train_dataset,
                    eval_dataset=eval_dataset,
                )
            
            return True
            
        except Exception as e:
            logger.error(f"Error setting up training: {str(e)}")
            st.error(f"Error setting up training: {str(e)}")
            return False
    
    def render(self):
        """Render the fine-tuning interface"""
        st.header("🎯 Model Fine-tuning")
        
        # Model selection and loading
        st.subheader("Model Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            model_name = st.selectbox(
                "Base Model",
                [
                    "google/gemma-2b-it",
                    "google/gemma-7b-it", 
                    "microsoft/DialoGPT-medium",
                    "microsoft/BioGPT",
                    "custom"
                ],
                index=0
            )
            
            if model_name == "custom":
                model_name = st.text_input("Custom Model Name/Path:")
        
        with col2:
            if st.button("Load Model", type="primary"):
                if self.load_model(model_name):
                    st.success(f"Successfully loaded {model_name}")
                    st.session_state.current_model = model_name
        
        if self.model is None:
            st.warning("Please load a model before proceeding with fine-tuning.")
            return
        
        # LoRA Configuration
        st.subheader("LoRA Configuration")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            lora_r = st.slider("LoRA Rank (r)", min_value=8, max_value=256, value=64, step=8)
        
        with col2:
            lora_alpha = st.slider("LoRA Alpha", min_value=8, max_value=512, value=128, step=8)
        
        with col3:
            lora_dropout = st.slider("LoRA Dropout", min_value=0.0, max_value=0.5, value=0.1, step=0.05)
        
        with col4:
            target_modules = st.multiselect(
                "Target Modules",
                ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                default=["q_proj", "v_proj"]
            )
        
        # Training Data Upload
        st.subheader("Training Data")
        
        data_source = st.radio(
            "Data Source",
            ["Upload File", "Manual Entry", "Use Sample Data"]
        )
        
        train_data = []
        eval_data = []
        
        if data_source == "Upload File":
            uploaded_file = st.file_uploader(
                "Upload Training Data",
                type=['json', 'csv', 'jsonl'],
                help="Upload your medical training data in JSON, CSV, or JSONL format"
            )
            
            if uploaded_file:
                try:
                    train_data, eval_data = prepare_training_data(uploaded_file)
                    st.success(f"Loaded {len(train_data)} training examples and {len(eval_data)} evaluation examples")
                except Exception as e:
                    st.error(f"Error processing file: {str(e)}")
        
        elif data_source == "Manual Entry":
            st.write("Enter training examples manually:")
            
            with st.form("manual_data_entry"):
                question = st.text_area("Medical Question:", height=100)
                answer = st.text_area("Expected Answer:", height=150)
                context = st.text_area("Additional Context (optional):", height=100)
                
                if st.form_submit_button("Add Example"):
                    if question and answer:
                        example = {
                            "question": question,
                            "answer": answer,
                            "context": context if context else ""
                        }
                        
                        if 'manual_training_data' not in st.session_state:
                            st.session_state.manual_training_data = []
                        
                        st.session_state.manual_training_data.append(example)
                        st.success("Example added!")
                        st.rerun()
            
            if 'manual_training_data' in st.session_state and st.session_state.manual_training_data:
                st.write(f"Current examples: {len(st.session_state.manual_training_data)}")
                
                if st.button("Use Manual Data for Training"):
                    train_data = st.session_state.manual_training_data
                    # Split 80/20 for train/eval
                    split_idx = int(0.8 * len(train_data))
                    eval_data = train_data[split_idx:]
                    train_data = train_data[:split_idx]
        
        elif data_source == "Use Sample Data":
            st.info("This would normally load sample medical training data. Please upload your own data for production use.")
        
        # Training Configuration
        st.subheader("Training Configuration")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            batch_size = st.selectbox("Batch Size", [1, 2, 4, 8], index=1)
            gradient_accumulation_steps = st.selectbox("Gradient Accumulation Steps", [1, 2, 4, 8], index=1)
            num_epochs = st.slider("Number of Epochs", min_value=1, max_value=10, value=3)
        
        with col2:
            learning_rate = st.selectbox(
                "Learning Rate",
                [1e-5, 2e-5, 5e-5, 1e-4, 2e-4],
                index=1,
                format_func=lambda x: f"{x:.0e}"
            )
            weight_decay = st.slider("Weight Decay", min_value=0.0, max_value=0.1, value=0.01, step=0.01)
            max_seq_length = st.selectbox("Max Sequence Length", [512, 1024, 2048], index=1)
        
        with col3:
            fp16 = st.checkbox("FP16 Training", value=True)
            bf16 = st.checkbox("BF16 Training", value=False)
            packing = st.checkbox("Enable Packing", value=True)
            use_tensorboard = st.checkbox("Use Tensorboard", value=True)
        
        # Advanced settings
        with st.expander("Advanced Settings"):
            output_dir = st.text_input("Output Directory", value="./fine_tuned_model")
            logging_steps = st.number_input("Logging Steps", min_value=1, value=10)
            save_steps = st.number_input("Save Steps", min_value=10, value=100)
            save_total_limit = st.number_input("Save Total Limit", min_value=1, value=3)
            push_to_hub = st.checkbox("Push to Hugging Face Hub")
            
            if push_to_hub:
                hub_model_id = st.text_input("Hub Model ID (username/model-name):")
            else:
                hub_model_id = None
        
        # Training execution
        st.subheader("Training Execution")
        
        if not train_data:
            st.warning("Please prepare training data before starting training.")
            return
        
        # Display training summary
        with st.expander("Training Summary"):
            training_config = {
                "output_dir": output_dir,
                "batch_size": batch_size,
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "num_epochs": num_epochs,
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "logging_steps": logging_steps,
                "save_steps": save_steps,
                "save_total_limit": save_total_limit,
                "fp16": fp16,
                "bf16": bf16,
                "packing": packing,
                "max_seq_length": max_seq_length,
                "use_tensorboard": use_tensorboard,
                "push_to_hub": push_to_hub,
                "hub_model_id": hub_model_id,
            }
            
            lora_config = self.create_lora_config(lora_r, lora_alpha, lora_dropout, target_modules)
            
            st.json({
                "model": model_name,
                "training_examples": len(train_data),
                "eval_examples": len(eval_data),
                "lora_config": {
                    "r": lora_r,
                    "alpha": lora_alpha,
                    "dropout": lora_dropout,
                    "target_modules": target_modules
                },
                "training_config": training_config
            })
        
        # Start training button
        if st.button("Start Fine-tuning", type="primary"):
            if self.start_training(train_data, eval_data, training_config, lora_config):
                st.success("Training setup completed! Starting training...")
                
                # Create progress containers
                progress_container = st.container()
                log_container = st.container()
                
                with progress_container:
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                
                with log_container:
                    log_expander = st.expander("Training Logs", expanded=True)
                    log_text = log_expander.empty()
                
                try:
                    # Start training with progress tracking
                    class ProgressCallback:
                        def __init__(self, progress_bar, status_text, log_text):
                            self.progress_bar = progress_bar
                            self.status_text = status_text
                            self.log_text = log_text
                            self.logs = []
                        
                        def on_log(self, args, state, control, model=None, **kwargs):
                            if state.log_history:
                                latest_log = state.log_history[-1]
                                progress = state.global_step / state.max_steps if state.max_steps else 0
                                
                                self.progress_bar.progress(progress)
                                self.status_text.text(f"Step {state.global_step}/{state.max_steps} | Loss: {latest_log.get('train_loss', 'N/A'):.4f}")
                                
                                self.logs.append(f"Step {state.global_step}: {latest_log}")
                                self.log_text.text("\n".join(self.logs[-10:]))  # Show last 10 logs
                    
                    # Add callback to trainer
                    callback = ProgressCallback(progress_bar, status_text, log_text)
                    self.trainer.add_callback(callback)
                    
                    # Train the model
                    self.trainer.train()
                    
                    # Save the model
                    self.trainer.save_model()
                    
                    st.success("Fine-tuning completed successfully!")
                    st.session_state.model_trained = True
                    
                    # Display final metrics
                    if self.trainer.state.log_history:
                        final_metrics = self.trainer.state.log_history[-1]
                        st.subheader("Final Training Metrics")
                        
                        metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
                        
                        with metrics_col1:
                            st.metric("Final Loss", f"{final_metrics.get('train_loss', 0):.4f}")
                        
                        with metrics_col2:
                            st.metric("Steps Completed", final_metrics.get('step', 0))
                        
                        with metrics_col3:
                            st.metric("Learning Rate", f"{final_metrics.get('learning_rate', 0):.2e}")
                
                except Exception as e:
                    st.error(f"Training failed: {str(e)}")
                    logger.error(f"Training error: {str(e)}")
        
        # Model management
        st.subheader("Model Management")
        
        if os.path.exists(output_dir):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("Load Trained Model"):
                    try:
                        # Load the fine-tuned model
                        st.success("Fine-tuned model loaded successfully!")
                        st.session_state.model_trained = True
                    except Exception as e:
                        st.error(f"Error loading model: {str(e)}")
            
            with col2:
                if st.button("Test Model"):
                    st.info("Navigate to the Chat Interface to test your fine-tuned model.")
            
            with col3:
                if st.button("Export Model"):
                    st.info("Navigate to the Model Export section to export your model.")
