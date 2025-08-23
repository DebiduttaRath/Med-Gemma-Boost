import streamlit as st
import time
import json
import logging
from typing import List, Dict, Any, Tuple, Optional
from components.safety import SafetySystem
from config.settings import Settings
from utils.fallbacks import (
    HAS_TORCH, HAS_TRANSFORMERS, HAS_NEMO,
    get_model_and_tokenizer, 
    FallbackModel, FallbackTokenizer
)

# Import available packages - moved to avoid import errors
torch = None
pipeline = None

try:
    if HAS_TORCH:
        import torch
except ImportError:
    pass

try:
    if HAS_TRANSFORMERS:
        from transformers import pipeline
except ImportError:
    pass

logger = logging.getLogger(__name__)

class ChatInterface:
    def __init__(self):
        self.settings = Settings()
        self.safety_system = SafetySystem()
        self.model = None
        self.tokenizer = None
        self.chat_pipeline = None
        
    def load_model(self, model_name: str = None):
        """Unified model loading"""
        if model_name is None:
            model_name = self.settings.BASE_MODEL
            
        try:
            if self.model is None:
                from utils.model_utils import load_base_model_unified
                self.model, self.tokenizer = load_base_model_unified(model_name)
                logger.info(f"Loaded {model_name} using unified loader")
                
                # Additional setup for existing functionality
                if hasattr(self.tokenizer, 'pad_token') and self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                
                if HAS_TRANSFORMERS and not isinstance(self.model, FallbackModel):
                    self.chat_pipeline = pipeline(
                        "text-generation",
                        model=self.model,
                        tokenizer=self.tokenizer,
                        torch_dtype=torch.float16 if HAS_TORCH else None,
                        device_map="auto"
                    )
            return True
        except Exception as e:
            logger.error(f"Model loading failed: {e}")
            st.error(f"Error loading model: {str(e)}")
            
            # Fallback to original approach if unified loading fails
            try:
                if self.model is None or self.tokenizer is None:
                    with st.spinner(f"Loading model {model_name}..."):
                        
                        # ---- NeMo loader ----
                        if "nvidia/" in model_name and HAS_NEMO:
                            import nemo.collections.nlp as nemo_nlp
                            from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
                            self.model = MegatronGPTModel.from_pretrained(model_name)
                            self.tokenizer = self.model.tokenizer
                            logger.info(f"Loaded NeMo model: {model_name}")
                        
                        # ---- Default HuggingFace loader ----
                        else:
                            self.model, self.tokenizer = get_model_and_tokenizer(model_name)
                            
                            if hasattr(self.tokenizer, 'pad_token') and self.tokenizer.pad_token is None:
                                self.tokenizer.pad_token = self.tokenizer.eos_token
                            
                            if HAS_TRANSFORMERS and not isinstance(self.model, FallbackModel):
                                self.chat_pipeline = pipeline(
                                    "text-generation",
                                    model=self.model,
                                    tokenizer=self.tokenizer,
                                    torch_dtype=torch.float16 if HAS_TORCH else None,
                                    device_map="auto"
                                )
                return True
            except Exception as e2:
                logger.error(f"Fallback model loading also failed: {e2}")
                self.model = FallbackModel(model_name)
                self.tokenizer = FallbackTokenizer(model_name)
                return True
    
    def generate_response(self, question: str, context: str = "", max_tokens: int = None) -> str:
        """Unified response generation"""
        if max_tokens is None:
            max_tokens = self.settings.MAX_NEW_TOKENS
            
        if not self.model or not self.tokenizer:
            if not self.load_model():
                return "Error: Model not available. Please check model loading."
        
        # Use the unified model interface
        if hasattr(self.model, 'generate'):
            # Proper transformer model
            return self._generate_with_transformers(question, context, max_tokens)
        else:
            # Fallback model
            return self._generate_with_fallback(question, context)
    
    def _generate_with_transformers(self, question: str, context: str, max_tokens: int) -> str:
        """Generate with proper transformer model"""
        # Safety check
        is_safe, rule_triggered, safe_response = self.safety_system.check_input_safety(question)
        if not is_safe:
            return safe_response
        
        # Build messages with safety system
        system_prompt = self.safety_system.get_safety_system_prompt()
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context: {context}\n\nQuestion: {question}"} if context else 
            {"role": "user", "content": question}
        ]
        
        # Apply chat template
        if hasattr(self.tokenizer, 'apply_chat_template'):
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            prompt = f"{system_prompt}\n\nUser: {question}"
            if context:
                prompt = f"{system_prompt}\n\nContext: {context}\n\nQuestion: {question}"
        
        # Generate
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        
        try:
            if HAS_TORCH and hasattr(self.model, 'device'):
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            if HAS_TORCH:
                with torch.inference_mode():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=max_tokens,
                        temperature=self.settings.TEMPERATURE,
                        do_sample=True,
                        top_p=0.9,
                        repetition_penalty=1.1,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id
                    )
            else:
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=self.settings.TEMPERATURE,
                    do_sample=True,
                    top_p=0.9,
                    repetition_penalty=1.1,
                    pad_token_id=getattr(self.tokenizer, 'pad_token_id', 0),
                    eos_token_id=getattr(self.tokenizer, 'eos_token_id', 1)
                )
            
            # Decode
            if hasattr(self.tokenizer, 'decode'):
                if HAS_TORCH and hasattr(inputs, 'input_ids'):
                    response = self.tokenizer.decode(
                        outputs[0][inputs['input_ids'].shape[-1]:], 
                        skip_special_tokens=True
                    ).strip()
                else:
                    response = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
            else:
                response = f"Generated response for: {question}"
            
            return self.safety_system.sanitize_response(response)
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return f"Error generating response: {str(e)}"
    
    def _generate_with_fallback(self, question: str, context: str) -> str:
        """Generate with fallback model"""
        # Safety check
        is_safe, rule_triggered, safe_response = self.safety_system.check_input_safety(question)
        if not is_safe:
            return safe_response
            
        # Handle NeMo models
        if "nvidia/" in self.settings.BASE_MODEL and HAS_NEMO:
            system_prompt = self.safety_system.get_safety_system_prompt()
            prompt = f"{system_prompt}\n\nUser: {question}"
            if context:
                prompt = f"{system_prompt}\n\nContext: {context}\n\nQuestion: {question}"
            
            try:
                response = self.model.complete(prompt, tokens_to_generate=self.settings.MAX_NEW_TOKENS)
                if isinstance(response, list):
                    response = response[0] if response else ""
                return self.safety_system.sanitize_response(response)
            except Exception as e:
                logger.error(f"NeMo generation error: {str(e)}")
        
        # Handle other fallback models
        return f"I understand you're asking about: {question}. As an educational medical AI, I can provide general information, but please consult healthcare professionals for specific medical advice."
    
    def generate_response_with_rag(self, question: str, context: str, rag_system) -> Tuple[str, List[Dict[str, Any]]]:
        """Generate response using RAG system (faiss | nemo | hybrid supported)"""
        try:
            retrieved_docs = rag_system.retrieve_documents(
                question + (" " + context if context else ""), 
                k=self.settings.RETRIEVAL_TOP_K
            )
            
            if retrieved_docs:
                retrieved_context = "\n\n".join([
                    f"[{i+1}] Source: {doc.get('source', 'unknown')}\n{doc['text'][:500]}..."
                    for i, doc in enumerate(retrieved_docs)
                ])
                full_context = f"{context}\n\nRetrieved Information:\n{retrieved_context}" if context else f"Retrieved Information:\n{retrieved_context}"
            else:
                full_context = context
                retrieved_docs = []
            
            response = self.generate_response(question, full_context)
            
            if retrieved_docs:
                response = self.safety_system.enforce_citation_format(response, retrieved_docs)
            
            return response, retrieved_docs
            
        except Exception as e:
            logger.error(f"Error in RAG generation: {str(e)}")
            return f"Error in RAG generation: {str(e)}", []

    
    def format_chat_message(self, message: str, sender: str, timestamp: str = None) -> Dict[str, Any]:
        """Format a chat message for display"""
        if timestamp is None:
            timestamp = time.strftime("%H:%M:%S")
        
        return {
            "sender": sender,
            "message": message,
            "timestamp": timestamp
        }
    
    def display_chat_history(self, chat_history: List[Dict[str, Any]]):
        """Display chat history in the interface"""
        for msg in chat_history:
            if msg["sender"] == "user":
                with st.chat_message("user"):
                    st.write(msg["message"])
                    st.caption(f"Sent at {msg['timestamp']}")
            else:
                with st.chat_message("assistant"):
                    st.markdown(msg["message"])
                    st.caption(f"Generated at {msg['timestamp']}")
    
    def display_retrieved_context(self, retrieved_docs: List[Dict[str, Any]]):
        """Display retrieved context in an expandable section"""
        if not retrieved_docs:
            return
        
        with st.expander(f"📚 Retrieved Sources ({len(retrieved_docs)} documents)", expanded=False):
            for i, doc in enumerate(retrieved_docs, 1):
                st.write(f"**Source {i}: {doc.get('source', 'Unknown')}**")
                st.write(f"Relevance Score: {doc.get('score', 0):.3f}")
                st.write(f"Content: {doc['text'][:300]}...")
                if i < len(retrieved_docs):
                    st.divider()
    
    def export_chat_session(self, chat_history: List[Dict[str, Any]]) -> str:
        """Export chat session as JSON"""
        export_data = {
            "session_id": f"chat_{int(time.time())}",
            "export_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model_info": {
                "base_model": self.settings.BASE_MODEL,
                "rag_enabled": st.session_state.rag_system is not None,
                "safety_enabled": True
            },
            "chat_history": chat_history,
            "settings": {
                "max_tokens": self.settings.MAX_NEW_TOKENS,
                "temperature": self.settings.TEMPERATURE,
                "retrieval_top_k": self.settings.RETRIEVAL_TOP_K
            }
        }
        
        return json.dumps(export_data, indent=2)
    
    def clear_conversation(self):
        """Clear the current conversation"""
        if 'chat_history' in st.session_state:
            st.session_state.chat_history = []
        if 'current_conversation' in st.session_state:
            st.session_state.current_conversation = []
    
    def render(self):
        """Render the enhanced chat interface with model selection"""
        st.header("💬 Medical AI Chat Interface")
        
        # Initialize session state
        if 'current_conversation' not in st.session_state:
            st.session_state.current_conversation = []
        
        # Import and initialize AI agent
        from components.ai_agent import healthcare_ai_agent
        
        # AI Agent Model Selection
        selected_model = healthcare_ai_agent.render_agent_interface()
        
        # Chat configuration
        st.subheader("💬 Intelligent Chat Configuration")
        
        
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            use_rag = st.checkbox(
                "Enable RAG", 
                value=True, 
                disabled=not st.session_state.rag_system,
                help="Use Retrieval-Augmented Generation for enhanced responses"
            )
        
        with col2:
            max_tokens = st.slider(
                "Max Response Tokens", 
                min_value=50, 
                max_value=1000, 
                value=self.settings.MAX_NEW_TOKENS,
                step=50
            )
        
        with col3:
            temperature = st.slider(
                "Temperature", 
                min_value=0.1, 
                max_value=1.0, 
                value=self.settings.TEMPERATURE,
                step=0.1
            )
        
        # Model status
        if not self.model:
            st.warning("⚠️ Model not loaded. Loading default model...")
            if st.button("Load Model"):
                if self.load_model():
                    st.success("✅ Model loaded successfully!")
                    st.rerun()
        else:
            st.success("✅ Model ready for conversation")
        
        # Main chat interface
        st.subheader("Conversation")
        
        # Display current conversation
        chat_container = st.container()
        with chat_container:
            if st.session_state.current_conversation:
                for msg in st.session_state.current_conversation:
                    if msg["type"] == "user":
                        with st.chat_message("user"):
                            st.write(msg["content"])
                    elif msg["type"] == "assistant":
                        with st.chat_message("assistant"):
                            st.markdown(msg["content"])
                            if "retrieved_docs" in msg and msg["retrieved_docs"]:
                                self.display_retrieved_context(msg["retrieved_docs"])
            else:
                st.info("👋 Welcome! Ask me any medical question for educational purposes.")
        
        # Chat input - FIXED: Separate forms for input and buttons
        with st.form("chat_input_form", clear_on_submit=True):
            user_input = st.text_area(
                "Your question:",
                placeholder="Ask a medical question...",
                height=120,  # Fixed: minimum 68px requirement
                label_visibility="collapsed"
            )
            
            context_input = st.text_area(
                "Additional context (optional):",
                placeholder="Provide any relevant context or background information...",
                height=100,  # Fixed: minimum 68px requirement
                label_visibility="collapsed"
            )
            
            # Submit button inside the form
            submitted = st.form_submit_button("Send", type="primary", use_container_width=True)
        
        # Separate form for clear button to avoid conflicts
        with st.form("chat_management_form"):
            clear_submitted = st.form_submit_button("Clear Chat", use_container_width=True)
        
        # Process user input
        if submitted and user_input.strip():
            if not self.model:
                st.error("Please load a model first.")
                return
            
            # Add user message to conversation
            user_msg = {
                "type": "user",
                "content": user_input,
                "timestamp": time.strftime("%H:%M:%S")
            }
            st.session_state.current_conversation.append(user_msg)
            
            # Generate intelligent response
            with st.spinner("🤖 AI Agent thinking..."):
                start_time = time.time()
                
                # Use AI agent for intelligent response generation
                from components.ai_agent import healthcare_ai_agent
                current_model = st.session_state.get('selected_ai_model', selected_model)
                
                if use_rag and st.session_state.rag_system:
                    response, retrieved_docs = self.generate_response_with_rag(
                        user_input, context_input, st.session_state.rag_system
                    )
                    # Enhance with AI agent intelligence
                    response = healthcare_ai_agent.generate_intelligent_response(
                        user_input, f"{context_input}\n\nRetrieved Information: {response}", current_model
                    )
                else:
                    response = healthcare_ai_agent.generate_intelligent_response(
                        user_input, context_input, current_model
                    )
                    retrieved_docs = []
                
                response_time = time.time() - start_time
            
            # Add assistant response to conversation
            assistant_msg = {
                "type": "assistant",
                "content": response,
                "timestamp": time.strftime("%H:%M:%S"),
                "response_time": response_time,
                "retrieved_docs": retrieved_docs
            }
            st.session_state.current_conversation.append(assistant_msg)
            
            # Update global chat history
            if 'chat_history' not in st.session_state:
                st.session_state.chat_history = []
            
            st.session_state.chat_history.extend([
                self.format_chat_message(user_input, "user"),
                self.format_chat_message(response, "assistant")
            ])
            
            st.rerun()
        
        # Clear conversation
        if clear_submitted:
            self.clear_conversation()
            st.rerun()
        
        # Conversation management
        if st.session_state.current_conversation:
            st.subheader("Conversation Management")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("📊 Analyze Conversation"):
                    self.analyze_conversation()
            
            with col2:
                conversation_data = self.export_chat_session(st.session_state.current_conversation)
                st.download_button(
                    label="💾 Save Conversation",
                    data=conversation_data,
                    file_name=f"medical_chat_{int(time.time())}.json",
                    mime="application/json",
                    use_container_width=True
                )
            
            with col3:
                if st.button("🔄 New Conversation", use_container_width=True):
                    self.clear_conversation()
                    st.rerun()
    
    def analyze_conversation(self):
        """Analyze the current conversation for insights"""
        if not st.session_state.current_conversation:
            st.warning("No conversation to analyze.")
            return
        
        # Count messages and calculate metrics
        user_messages = [msg for msg in st.session_state.current_conversation if msg["type"] == "user"]
        assistant_messages = [msg for msg in st.session_state.current_conversation if msg["type"] == "assistant"]
        
        # Calculate average response time
        response_times = [msg.get("response_time", 0) for msg in assistant_messages]
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0
        
        # Count RAG usage
        rag_used = sum(1 for msg in assistant_messages if msg.get("retrieved_docs"))
        
        # Display analysis
        with st.expander("📊 Conversation Analysis", expanded=True):
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Messages", len(st.session_state.current_conversation))
            
            with col2:
                st.metric("User Questions", len(user_messages))
            
            with col3:
                st.metric("Avg Response Time", f"{avg_response_time:.2f}s")
            
            with col4:
                st.metric("RAG Usage", f"{rag_used}/{len(assistant_messages)}")
            
            # Safety analysis
            safety_checks = 0
            for msg in user_messages:
                is_safe, _, _ = self.safety_system.check_input_safety(msg["content"])
                if not is_safe:
                    safety_checks += 1
            
            if safety_checks > 0:
                st.warning(f"⚠️ {safety_checks} messages triggered safety guidelines.")
            else:
                st.success("✅ All messages passed safety checks.")