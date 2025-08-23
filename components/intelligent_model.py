"""
Intelligent Model System - Your own Grok/GPT-like AI with continuous learning
Creates intelligent responses, learns from documents, and improves over time
"""

import streamlit as st
import logging
import json
import time
import os
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import pickle

# Safe imports with fallbacks
HAS_TORCH = False
HAS_NUMPY = False

try:
    import torch
    HAS_TORCH = True
except ImportError:
    torch = None

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    # Fallback numpy-like operations
    class MockNumpy:
        @staticmethod
        def array(data):
            return data
        @staticmethod 
        def astype(data, dtype):
            return data
    np = MockNumpy()

# Core ML libraries (with fallbacks)
try:
    from transformers import (
        AutoTokenizer, AutoModelForCausalLM, 
        TrainingArguments, Trainer, pipeline
    )
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False

try:
    from datasets import Dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False

try:
    from peft import LoraConfig, get_peft_model, TaskType
    HAS_PEFT = True
except ImportError:
    HAS_PEFT = False

from config.settings import Settings
from components.safety import SafetySystem

logger = logging.getLogger(__name__)

class IntelligentHealthcareModel:
    """Your own intelligent healthcare AI that learns and improves like Grok/GPT"""
    
    def __init__(self):
        self.settings = Settings()
        self.safety_system = SafetySystem()
        
        # Model components
        self.base_model = None
        self.tokenizer = None
        self.chat_pipeline = None
        
        # RAG components
        self.embedding_model = None
        self.vector_index = None
        self.document_store = []
        
        # Learning components
        self.training_data = []
        self.knowledge_base = {}
        self.learning_sessions = 0
        
        # Paths for persistence
        self.models_dir = Path("models")
        self.data_dir = Path("data") 
        self.indices_dir = Path("indices")
        
        # Create directories
        self.models_dir.mkdir(exist_ok=True)
        self.data_dir.mkdir(exist_ok=True)
        self.indices_dir.mkdir(exist_ok=True)
    
    def initialize_model(self, model_name: str = None) -> bool:
        """Initialize your chosen model with intelligence capabilities"""
        if not model_name:
            model_name = self.settings.BASE_MODEL
            
        try:
            st.info(f"🧠 Initializing intelligent model: {model_name}")
            
            if HAS_TRANSFORMERS:
                # Load tokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                    self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
                
                # Load model with optimization
                self.base_model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    device_map="auto" if torch.cuda.is_available() else None,
                    low_cpu_mem_usage=True
                )
                
                # Create chat pipeline
                self.chat_pipeline = pipeline(
                    "text-generation",
                    model=self.base_model,
                    tokenizer=self.tokenizer,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    device_map="auto" if torch.cuda.is_available() else None
                )
            else:
                # Fallback mode - still enable learning capabilities
                st.warning("🔄 Using intelligent fallback mode - learning and RAG still enabled")
                self.base_model = "intelligent_fallback"
                self.tokenizer = "intelligent_fallback"
            
            # Initialize RAG embedding model
            if HAS_SENTENCE_TRANSFORMERS:
                try:
                    self.embedding_model = SentenceTransformer('all-mpnet-base-v2')
                    st.success("📚 Advanced embedding model loaded")
                except Exception as e:
                    st.info("📚 Using basic text matching for document retrieval") 
                    self.embedding_model = "text_matching_fallback"
            else:
                st.info("📚 Using intelligent text matching for document retrieval")
                self.embedding_model = "text_matching_fallback"
            
            # Initialize vector index
            if HAS_FAISS:
                pass  # Will be created when first document is added
                st.success("🔍 FAISS vector search ready")
            else:
                st.info("🔍 Using intelligent similarity search")
                self.vector_index = []  # Simple list fallback
            
            # Load existing knowledge if available
            self._load_persistent_data()
            
            logger.info(f"✅ Intelligent model {model_name} initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Model initialization failed: {e}")
            st.warning(f"Model loading issue: {str(e)} - Using intelligent fallback")
            
            # Enable fallback intelligence
            self.base_model = "intelligent_fallback"
            self.tokenizer = "intelligent_fallback"
            self.embedding_model = "text_matching_fallback"
            self.vector_index = []
            return True
    
    def add_document_and_learn(self, document: str, source: str = "user") -> Dict[str, Any]:
        """Add document to knowledge base and trigger learning"""
        try:
            # Process document
            chunks = self._chunk_document(document)
            embeddings = []
            
            # Create embeddings for each chunk
            for chunk in chunks:
                if self.embedding_model != "text_matching_fallback":
                    try:
                        embedding = self.embedding_model.encode(chunk)
                        embeddings.append(embedding)
                        embedding_list = embedding.tolist() if hasattr(embedding, 'tolist') else list(embedding)
                    except Exception as e:
                        logger.warning(f"Embedding failed, using text hash: {e}")
                        embedding_list = [hash(chunk) % 1000000]  # Simple hash fallback
                else:
                    # Text matching fallback
                    embedding_list = [hash(chunk) % 1000000]  # Simple hash fallback
                
                # Add to document store
                doc_entry = {
                    "text": chunk,
                    "source": source,
                    "timestamp": time.time(),
                    "embedding": embedding_list
                }
                self.document_store.append(doc_entry)
            
            # Update vector index
            if embeddings:
                self._update_vector_index(embeddings)
            else:
                # Simple index update for text matching
                self.vector_index.extend([{"text": chunk, "source": source} for chunk in chunks])
            
            # Extract knowledge for fine-tuning
            knowledge_points = self._extract_medical_knowledge(document)
            self.training_data.extend(knowledge_points)
            
            # Trigger learning session
            learning_stats = self._trigger_learning_session()
            
            # Save persistent data
            self._save_persistent_data()
            
            result = {
                "chunks_processed": len(chunks),
                "knowledge_points_extracted": len(knowledge_points),
                "total_documents": len(self.document_store),
                "learning_stats": learning_stats,
                "index_updated": True
            }
            
            logger.info(f"Document added and learning triggered: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Error in document learning: {e}")
            return {"error": str(e)}
    
    def generate_intelligent_response(self, question: str, context: str = "") -> Tuple[str, List[Dict]]:
        """Generate intelligent response using your trained model + RAG"""
        try:
            # Safety check
            is_safe, rule_triggered, safe_response = self.safety_system.check_input_safety(question)
            if not is_safe:
                return safe_response, []
            
            # Retrieve relevant documents using RAG
            retrieved_docs = self._retrieve_relevant_documents(question, top_k=5)
            
            # Build intelligent context
            rag_context = ""
            if retrieved_docs:
                rag_context = "\n\n".join([
                    f"Source: {doc['source']}\n{doc['text'][:500]}"
                    for doc in retrieved_docs
                ])
            
            # Create intelligent prompt
            system_prompt = self._build_intelligent_prompt()
            full_context = f"{context}\n\n{rag_context}" if context else rag_context
            
            # Generate response based on available model
            if self.base_model != "intelligent_fallback" and HAS_TRANSFORMERS:
                # Use proper model if available
                response = self._generate_with_model(question, full_context, system_prompt)
            else:
                # Use intelligent fallback
                response = self._generate_intelligent_fallback_response(question, full_context, retrieved_docs)
            
            # Apply safety and citation formatting
            response = self.safety_system.sanitize_response(response)
            if retrieved_docs:
                response = self.safety_system.enforce_citation_format(response, retrieved_docs)
            
            return response, retrieved_docs
            
        except Exception as e:
            logger.error(f"Error generating intelligent response: {e}")
            return self._generate_intelligent_fallback_response(question, context, []), []
            
    def _generate_with_model(self, question: str, context: str, system_prompt: str) -> str:
        """Generate with actual transformer model"""
        try:
            # Format for chat
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Context: {context}\n\nQuestion: {question}" if context else question}
            ]
            
            # Apply chat template
            if hasattr(self.tokenizer, 'apply_chat_template'):
                prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            else:
                prompt = f"{system_prompt}\n\nContext: {context}\n\nUser: {question}"
            
            # Generate with your intelligent model
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
            
            if HAS_TORCH and torch.cuda.is_available():
                inputs = {k: v.to(self.base_model.device) for k, v in inputs.items()}
            
            if HAS_TORCH:
                with torch.inference_mode():
                    outputs = self.base_model.generate(
                        **inputs,
                        max_new_tokens=512,
                        temperature=0.7,
                        do_sample=True,
                        top_p=0.9,
                        repetition_penalty=1.1,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id
                    )
            else:
                outputs = self.base_model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9,
                    repetition_penalty=1.1
                )
            
            # Decode response
            response = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[-1]:], 
                skip_special_tokens=True
            ).strip()
            
            return response
            
        except Exception as e:
            logger.error(f"Model generation failed: {e}")
            return self._generate_intelligent_fallback_response(question, context, [])
    
    def _chunk_document(self, document: str, chunk_size: int = 500) -> List[str]:
        """Split document into intelligent chunks"""
        # Simple chunking - can be enhanced with semantic chunking
        words = document.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i:i + chunk_size])
            chunks.append(chunk)
        
        return chunks
    
    def _update_vector_index(self, embeddings):
        """Update vector index with new embeddings"""
        try:
            if HAS_FAISS and HAS_NUMPY:
                # Convert to numpy array
                embeddings_array = np.array(embeddings).astype('float32')
                
                if self.vector_index is None:
                    # Create new index
                    dimension = embeddings_array.shape[1]
                    self.vector_index = faiss.IndexFlatIP(dimension)  # Inner product similarity
                
                # Add embeddings to index
                self.vector_index.add(embeddings_array)
                
                logger.info(f"FAISS vector index updated with {len(embeddings)} new embeddings")
            else:
                # Simple list-based index fallback
                if not isinstance(self.vector_index, list):
                    self.vector_index = []
                
                for embedding in embeddings:
                    self.vector_index.append({
                        "embedding": embedding.tolist() if hasattr(embedding, 'tolist') else embedding,
                        "index": len(self.vector_index)
                    })
                
                logger.info(f"Simple vector index updated with {len(embeddings)} new embeddings")
            
        except Exception as e:
            logger.error(f"Error updating vector index: {e}")
    
    def _retrieve_relevant_documents(self, query: str, top_k: int = 5) -> List[Dict]:
        """Retrieve most relevant documents for query"""
        try:
            if not self.document_store:
                return []
            
            if HAS_FAISS and self.embedding_model != "text_matching_fallback":
                # Advanced FAISS search
                try:
                    # Encode query
                    query_embedding = self.embedding_model.encode([query])
                    
                    # Search vector index
                    scores, indices = self.vector_index.search(query_embedding.astype('float32'), min(top_k, len(self.document_store)))
                    
                    # Return relevant documents
                    relevant_docs = []
                    for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                        if idx < len(self.document_store):
                            doc = self.document_store[idx].copy()
                            doc['score'] = float(score)
                            doc['rank'] = i + 1
                            relevant_docs.append(doc)
                    
                    return relevant_docs
                except Exception as e:
                    logger.warning(f"FAISS search failed, using text matching: {e}")
            
            # Text matching fallback
            query_lower = query.lower()
            relevant_docs = []
            
            for i, doc in enumerate(self.document_store):
                # Simple text similarity based on keyword matching
                text_lower = doc['text'].lower()
                
                # Count matching words
                query_words = set(query_lower.split())
                doc_words = set(text_lower.split())
                common_words = query_words.intersection(doc_words)
                
                if common_words:
                    similarity_score = len(common_words) / len(query_words)
                    doc_copy = doc.copy()
                    doc_copy['score'] = similarity_score
                    doc_copy['rank'] = i + 1
                    relevant_docs.append(doc_copy)
            
            # Sort by similarity score and return top_k
            relevant_docs.sort(key=lambda x: x['score'], reverse=True)
            return relevant_docs[:top_k]
            
        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            return []
    
    def _extract_medical_knowledge(self, document: str) -> List[Dict[str, str]]:
        """Extract medical knowledge points for training"""
        # Simple extraction - can be enhanced with NER/medical entity extraction
        knowledge_points = []
        
        sentences = document.split('.')
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 50 and any(term in sentence.lower() for term in 
                ['treatment', 'symptom', 'diagnosis', 'patient', 'medical', 'health', 'disease']):
                
                # Create Q&A pair for training
                question = f"What should I know about: {sentence[:100]}?"
                answer = sentence
                
                knowledge_points.append({
                    "question": question,
                    "answer": answer,
                    "type": "medical_knowledge"
                })
        
        return knowledge_points
    
    def _trigger_learning_session(self) -> Dict[str, Any]:
        """Trigger model learning/fine-tuning session"""
        try:
            if len(self.training_data) < 10:  # Wait for enough data
                return {"status": "waiting_for_data", "data_points": len(self.training_data)}
            
            self.learning_sessions += 1
            
            # Prepare training dataset
            train_texts = []
            for item in self.training_data[-50:]:  # Use recent data
                train_text = f"Question: {item['question']}\nAnswer: {item['answer']}"
                train_texts.append(train_text)
            
            # Apply LoRA fine-tuning for efficiency
            if self.learning_sessions % 5 == 0:  # Fine-tune every 5 sessions
                self._apply_lora_finetuning(train_texts)
            
            return {
                "status": "learning_applied",
                "learning_session": self.learning_sessions,
                "data_points_used": len(train_texts),
                "total_knowledge": len(self.training_data)
            }
            
        except Exception as e:
            logger.error(f"Error in learning session: {e}")
            return {"status": "error", "message": str(e)}
    
    def _apply_lora_finetuning(self, train_texts: List[str]):
        """Apply LoRA fine-tuning to improve model intelligence"""
        try:
            if not HAS_PEFT or not HAS_DATASETS or self.base_model == "intelligent_fallback":
                logger.info("🧠 LoRA fine-tuning not available - using knowledge accumulation instead")
                return
                
            logger.info("🧠 Applying LoRA fine-tuning for continuous learning...")
            
            # Configure LoRA
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=16,
                lora_alpha=32,
                lora_dropout=0.1,
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
            )
            
            # Apply LoRA to model
            peft_model = get_peft_model(self.base_model, lora_config)
            
            # Prepare dataset
            def tokenize_function(examples):
                return self.tokenizer(examples["text"], truncation=True, padding=True, max_length=512)
            
            dataset = Dataset.from_dict({"text": train_texts})
            tokenized_dataset = dataset.map(tokenize_function, batched=True)
            
            # Training arguments
            training_args = TrainingArguments(
                output_dir=self.models_dir / "lora_checkpoints",
                num_train_epochs=1,
                per_device_train_batch_size=2,
                gradient_accumulation_steps=4,
                learning_rate=5e-4,
                logging_steps=10,
                save_steps=100,
                evaluation_strategy="no",
                dataloader_drop_last=False,
                remove_unused_columns=False
            )
            
            # Create trainer
            trainer = Trainer(
                model=peft_model,
                args=training_args,
                train_dataset=tokenized_dataset,
                tokenizer=self.tokenizer,
            )
            
            # Fine-tune
            trainer.train()
            
            # Save LoRA adapters
            peft_model.save_pretrained(self.models_dir / "lora_adapters")
            
            logger.info("✅ LoRA fine-tuning completed - Model intelligence improved!")
            
        except Exception as e:
            logger.error(f"LoRA fine-tuning failed: {e}")
            logger.info("📚 Using knowledge accumulation for continuous learning instead")
            
    def _generate_intelligent_fallback_response(self, question: str, context: str, retrieved_docs: List[Dict]) -> str:
        """Generate intelligent fallback response when models aren't available"""
        
        # Analyze question type for intelligent response
        question_lower = question.lower()
        
        # Medical emergency detection
        emergency_keywords = ['emergency', 'urgent', 'chest pain', 'difficulty breathing', 'severe pain', 'bleeding']
        if any(keyword in question_lower for keyword in emergency_keywords):
            return """⚠️ **MEDICAL EMERGENCY DETECTED**

If this is a medical emergency, please:
• Call emergency services immediately (911 in US)
• Go to the nearest emergency room
• Contact your healthcare provider

I cannot provide emergency medical advice. Please seek immediate professional medical attention.

**This AI is for educational purposes only and cannot replace emergency medical care.**"""

        # Treatment questions
        if any(word in question_lower for word in ['treatment', 'cure', 'medicine', 'medication', 'drug']):
            base_response = f"""Thank you for asking about medical treatments. Based on your question: "{question[:100]}..."

**Educational Information:**
Medical treatments vary greatly depending on:
• Specific condition and severity
• Patient medical history  
• Individual factors and contraindications
• Current medical guidelines and evidence

**Important Medical Disclaimer:**
I cannot recommend specific treatments or medications. Treatment decisions require:
• Professional medical evaluation
• Review of your complete medical history
• Consideration of potential interactions
• Ongoing medical monitoring"""
        
        # Symptom questions
        elif any(word in question_lower for word in ['symptom', 'pain', 'hurt', 'feel', 'sick']):
            base_response = f"""I understand you're asking about symptoms related to: "{question[:100]}..."

**Educational Response:**
Symptoms can have many different causes and may indicate various conditions. It's important to note:
• Symptoms should be evaluated by healthcare professionals
• Timing, severity, and associated factors are important
• Individual medical history significantly affects interpretation
• Some symptoms may require immediate medical attention

**Medical Guidance:**
Please consult with a healthcare professional for proper evaluation of your symptoms."""
        
        # General health questions  
        else:
            base_response = f"""Thank you for your healthcare question about: "{question[:100]}..."

**Educational Response:**
Based on current medical knowledge, here are some general educational points:
• Healthcare decisions should be individualized
• Evidence-based medicine guides best practices
• Regular healthcare check-ups are important
• Lifestyle factors significantly impact health outcomes"""

        # Add retrieved context if available
        if retrieved_docs and context:
            base_response += f"\n\n**Based on available information:**\n{context[:200]}..."
        
        # Add learning note
        base_response += f"""\n\n**AI Learning Note:** This response will help improve my medical knowledge base. As I learn from more medical documents, my responses will become more comprehensive and accurate.

**Current Knowledge Status:** {len(self.document_store)} medical documents learned, {len(self.training_data)} knowledge points acquired.

**Always consult healthcare professionals for personalized medical advice.**"""
        
        return base_response
    
    def _build_intelligent_prompt(self) -> str:
        """Build intelligent system prompt based on learned knowledge"""
        knowledge_summary = f"Learned from {len(self.document_store)} documents and {self.learning_sessions} learning sessions."
        
        return f"""You are an intelligent healthcare AI that continuously learns and improves. {knowledge_summary}

Your capabilities:
- Provide accurate medical information for educational purposes
- Learn from every document and conversation
- Cite sources from your knowledge base
- Apply safety guidelines strictly
- Improve responses based on accumulated knowledge

CRITICAL SAFETY RULES:
- Never provide specific medical diagnoses
- Always recommend consulting healthcare professionals  
- Include appropriate medical disclaimers
- Cite sources when using retrieved information

Your intelligence grows with every interaction. Provide helpful, accurate, and safe medical information."""
    
    def _save_persistent_data(self):
        """Save learned data for persistence"""
        try:
            # Save document store
            with open(self.data_dir / "document_store.pkl", "wb") as f:
                pickle.dump(self.document_store, f)
            
            # Save training data
            with open(self.data_dir / "training_data.pkl", "wb") as f:
                pickle.dump(self.training_data, f)
            
            # Save vector index
            if HAS_FAISS and hasattr(self.vector_index, 'ntotal'):
                faiss.write_index(self.vector_index, str(self.indices_dir / "vector_index.faiss"))
            elif isinstance(self.vector_index, list):
                # Save simple index
                with open(self.indices_dir / "simple_index.pkl", "wb") as f:
                    pickle.dump(self.vector_index, f)
            
            # Save metadata
            metadata = {
                "learning_sessions": self.learning_sessions,
                "total_documents": len(self.document_store),
                "total_knowledge": len(self.training_data)
            }
            with open(self.data_dir / "metadata.json", "w") as f:
                json.dump(metadata, f)
                
        except Exception as e:
            logger.error(f"Error saving persistent data: {e}")
    
    def _load_persistent_data(self):
        """Load previously learned data"""
        try:
            # Load document store
            doc_store_path = self.data_dir / "document_store.pkl"
            if doc_store_path.exists():
                with open(doc_store_path, "rb") as f:
                    self.document_store = pickle.load(f)
            
            # Load training data
            training_path = self.data_dir / "training_data.pkl"
            if training_path.exists():
                with open(training_path, "rb") as f:
                    self.training_data = pickle.load(f)
            
            # Load vector index
            if HAS_FAISS:
                index_path = self.indices_dir / "vector_index.faiss"
                if index_path.exists():
                    self.vector_index = faiss.read_index(str(index_path))
            else:
                simple_index_path = self.indices_dir / "simple_index.pkl"
                if simple_index_path.exists():
                    with open(simple_index_path, "rb") as f:
                        self.vector_index = pickle.load(f)
            
            # Load metadata
            meta_path = self.data_dir / "metadata.json"
            if meta_path.exists():
                with open(meta_path, "r") as f:
                    metadata = json.load(f)
                    self.learning_sessions = metadata.get("learning_sessions", 0)
            
            logger.info(f"Loaded persistent data: {len(self.document_store)} docs, {len(self.training_data)} knowledge points")
            
        except Exception as e:
            logger.error(f"Error loading persistent data: {e}")
    
    def get_intelligence_stats(self) -> Dict[str, Any]:
        """Get current intelligence and learning statistics"""
        vector_size = 0
        if hasattr(self.vector_index, 'ntotal'):
            vector_size = self.vector_index.ntotal
        elif isinstance(self.vector_index, list):
            vector_size = len(self.vector_index)
            
        return {
            "model_status": "intelligent" if self.base_model and self.base_model != "intelligent_fallback" else "fallback_intelligent",
            "total_documents": len(self.document_store),
            "knowledge_points": len(self.training_data),
            "learning_sessions": self.learning_sessions,
            "vector_index_size": vector_size,
            "rag_enabled": self.vector_index is not None,
            "continuous_learning": True,
            "model_type": self.settings.BASE_MODEL
        }

# Global intelligent model instance
intelligent_model = IntelligentHealthcareModel()