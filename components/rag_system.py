import sys
from pathlib import Path
import os

current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.append(str(project_root))

import streamlit as st
import pickle
import json
import numpy as np
import pandas as pd
import logging
from typing import List, Dict, Any
from utils.data_processing import process_medical_documents
from config.settings import Settings
from utils.fallbacks import (
    get_embedder,
    get_faiss_index,
    normalize_l2,
    HAS_SENTENCE_TRANSFORMERS,
    HAS_FAISS,
)
from components.nemo_rag import NeMoRAG  # 🔑 New import

logger = logging.getLogger(__name__)

class RAGSystem:
    def __init__(self):
        self.settings = Settings()
        self.embedder = None
        self.index = None
        self.passages = []
        self.index_path = "data/rag_index.faiss"
        self.meta_path = "data/rag_meta.pkl"

        # 🔑 Integrate NeMoRAG wrapper
        self.nemo_rag = NeMoRAG(self.settings)

    def load_embedder(self):
        """Load embedding model based on backend (HF | NeMo)"""
        if self.embedder is None:
            with st.spinner("Loading embedding model..."):
                if self.settings.RAG_BACKEND == "nemo":
                    # Load NeMo embedding model
                    self.embedder = get_embedder(self.settings.NEMO_EMBED_MODEL)
                else:
                    # Default HuggingFace embedder
                    self.embedder = get_embedder(self.settings.EMBED_MODEL)
        return self.embedder


    def load_medical_dataset(self) -> List[Dict[str, Any]]:
        """Load medical dataset from HuggingFace"""
        try:
            from datasets import load_dataset

            with st.spinner("Loading medical dataset from HuggingFace..."):
                ds = load_dataset("FreedomIntelligence/medical-o1-reasoning-SFT", "en")

                passages = []
                if "train" in ds:
                    for i, example in enumerate(ds["train"]):
                        if "input" in example and "output" in example:
                            passage_text = (
                                f"Medical Case: {example['input']}\n\n"
                                f"Medical Reasoning: {example['output']}"
                            )
                            passages.append(
                                {
                                    "id": f"medical_dataset_{i}",
                                    "text": passage_text,
                                    "source": "medical-o1-reasoning-SFT",
                                    "metadata": {
                                        "type": "medical_reasoning",
                                        "dataset": "medical-o1-reasoning-SFT",
                                        "index": i,
                                    },
                                }
                            )

                logger.info(f"Loaded {len(passages)} medical reasoning examples")
                return passages

        except Exception as e:
            logger.error(f"Error loading medical dataset: {str(e)}")
            st.warning(
                f"Could not load medical dataset: {str(e)}. Using fallback medical data."
            )
            return self.get_fallback_medical_data()

    def get_fallback_medical_data(self) -> List[Dict[str, Any]]:
        """Provide fallback medical data if dataset loading fails"""
        return [
            {
                "id": "fallback_1",
                "text": "Diabetes is a chronic condition that affects how your body processes blood sugar. Symptoms include increased thirst, frequent urination, extreme hunger, unexplained weight loss, fatigue, and blurred vision. Treatment involves lifestyle changes, medication, and regular monitoring.",
                "source": "fallback_medical_knowledge",
                "metadata": {"type": "condition", "condition": "diabetes"},
            },
            {
                "id": "fallback_2",
                "text": "Hypertension (high blood pressure) is a common condition where the force of blood against artery walls is too high. It can lead to heart disease, stroke, and other complications. Management includes diet, exercise, and medication.",
                "source": "fallback_medical_knowledge",
                "metadata": {"type": "condition", "condition": "hypertension"},
            },
            {
                "id": "fallback_3",
                "text": "COVID-19 symptoms range from mild to severe and may appear 2-14 days after exposure. Common symptoms include fever, cough, shortness of breath, fatigue, muscle aches, loss of taste or smell. Prevention includes vaccination, masking, and social distancing.",
                "source": "fallback_medical_knowledge",
                "metadata": {"type": "condition", "condition": "covid-19"},
            },
        ]

    def build_index(self, passages: List[Dict[str, Any]]) -> bool:
        """Build FAISS index from passages"""
        try:
            os.makedirs("data", exist_ok=True)
            embedder = self.load_embedder()
            texts = [p["text"] for p in passages]

            with st.spinner(f"Generating embeddings for {len(texts)} documents..."):
                embeddings = embedder.encode(
                    texts, show_progress_bar=False, convert_to_numpy=True
                )

            dim = embeddings.shape[1]
            self.index = get_faiss_index(dim)
            embeddings = normalize_l2(embeddings)
            self.index.add(embeddings)

            if HAS_FAISS:
                import faiss

                faiss.write_index(index, self.index_path)
            else:
                with open(self.index_path + ".fallback", "wb") as f:
                    pickle.dump(index, f)

            with open(self.meta_path, "wb") as f:
                pickle.dump(passages, f)

            #self.index = index
            self.passages = passages

            # 🔑 Wire FAISS into NeMoRAG
            self.nemo_rag.set_faiss_index(self.index, self.passages, self.embedder)
            
            
            # Update session state
            st.session_state.rag_initialized = True


            logger.info(f"Built index with {len(passages)} documents")
            return True

        except Exception as e:
            logger.error(f"Error building index: {str(e)}")
            st.error(f"Error building index: {str(e)}")
            return False

    def load_index(self) -> bool:
        """Load existing FAISS index and metadata"""
        try:
            fallback_path = self.index_path + ".fallback"

            if HAS_FAISS and os.path.exists(self.index_path) and os.path.exists(
                self.meta_path
            ):
                import faiss

                self.index = faiss.read_index(self.index_path)
                with open(self.meta_path, "rb") as f:
                    self.passages = pickle.load(f)

                self.load_embedder()
                self.nemo_rag.set_faiss_index(
                    self.index, self.passages, self.embedder
                )  # 🔑
                logger.info(f"Loaded FAISS index with {len(self.passages)} documents")
                return True

            elif os.path.exists(fallback_path) and os.path.exists(self.meta_path):
                with open(fallback_path, "rb") as f:
                    self.index = pickle.load(f)
                with open(self.meta_path, "rb") as f:
                    self.passages = pickle.load(f)

                self.load_embedder()
                self.nemo_rag.set_faiss_index(
                    self.index, self.passages, self.embedder
                )  # 🔑
                logger.info(f"Loaded fallback index with {len(self.passages)} documents")
                return True

            return False

        except Exception as e:
            logger.error(f"Error loading index: {str(e)}")
            st.error(f"Error loading index: {str(e)}")
            return False

    def retrieve_documents(self, query: str, k: int = None) -> List[Dict[str, Any]]:
        if k is None:
            k = self.settings.RETRIEVAL_TOP_K
        if not self.index or not self.embedder:
            st.warning("⚠️ No index loaded. Please initialize the RAG system first.")
            return []

        try:
            if self.settings.RAG_BACKEND in ["faiss", "nemo"]:
                return self.nemo_rag.retrieve(query, k=k, backend=self.settings.RAG_BACKEND)
            elif self.settings.RAG_BACKEND == "hybrid":
                faiss_docs = self.nemo_rag.retrieve(query, k=k, backend="faiss")
                nemo_docs = self.nemo_rag.retrieve(query, k=k, backend="nemo")
                return self.nemo_rag.fuse_results(faiss_docs, nemo_docs, method=self.settings.HYBRID_FUSION_METHOD)
        except Exception as e:
            logger.error(f"Error retrieving documents: {str(e)}")
            st.error(f"Error during retrieval: {str(e)}")
            return []


    # 🔽 render() remains identical except now retrieves via NeMoRAG
    def render(self):
        """Render the RAG system interface"""
        st.header("🔍 Retrieval-Augmented Generation System")

        # Backend selector synced with Settings + session_state
        st.subheader("🔽 RAG Backend Selection")
        backend_choice = st.radio(
            "Choose RAG Backend:",
            options=["faiss", "nemo", "hybrid"],
            index=["faiss", "nemo", "hybrid"].index(self.settings.RAG_BACKEND),
            help="Select FAISS (local), NeMo (dense retriever), or Hybrid (fusion of both)"
        )
        if backend_choice != self.settings.RAG_BACKEND:
            self.settings.RAG_BACKEND = backend_choice
            st.session_state["rag_backend"] = backend_choice
            st.success(f"✅ Switched RAG backend to **{backend_choice}**")

        # Check if index exists
        fallback_exists = os.path.exists(self.index_path + ".fallback") and os.path.exists(self.meta_path)
        index_exists = (os.path.exists(self.index_path) and os.path.exists(self.meta_path)) or fallback_exists

        # Load existing index if available but not loaded
        if index_exists and not self.index:
            if st.button("📂 Load Existing Index"):
                if self.load_index():
                    st.success(f"Successfully loaded index with {len(self.passages)} documents!")
                    st.session_state.rag_system = self
                    st.rerun()

        # Document upload and indexing section
        st.subheader("📚 Document Management")
        
        # Load medical dataset option
        if st.button("📖 Load Medical Dataset", help="Load pre-built medical reasoning dataset"):
            with st.spinner("Loading medical dataset..."):
                medical_passages = self.load_medical_dataset()
                
                if medical_passages:
                    if self.build_index(medical_passages):
                        st.success(f"Successfully loaded medical dataset with {len(medical_passages)} examples!")
                        st.session_state.rag_system = self
                        st.rerun()

        # File upload section
        uploaded_files = st.file_uploader(
            "Upload Medical Documents",
            accept_multiple_files=True,
            type=['txt', 'pdf', 'json', 'csv'],
            help="Upload medical documents, research papers, or structured medical data"
        )

        # Manual text input
        manual_text = st.text_area(
            "Or paste medical text directly:",
            height=200,
            help="Paste medical guidelines, protocols, or reference text"
        )

        # Build index button
        if st.button("🏗️ Build Knowledge Base", type="primary"):
            passages = []
            
            # Process uploaded files
            if uploaded_files:
                for file in uploaded_files:
                    try:
                        file_passages = process_medical_documents(file)
                        passages.extend(file_passages)
                        st.success(f"Processed {file.name}: {len(file_passages)} passages")
                    except Exception as e:
                        st.error(f"Error processing {file.name}: {str(e)}")
            
            # Process manual text
            if manual_text.strip():
                passages.append({
                    "id": f"manual_input_{len(passages)}",
                    "text": manual_text.strip(),
                    "source": "manual_input",
                    "metadata": {"type": "manual"}
                })
            
            if passages:
                if self.build_index(passages):
                    st.success(f"Successfully built knowledge base with {len(passages)} documents!")
                    st.session_state.rag_system = self
                    st.rerun()
            else:
                st.warning("No documents to process. Please upload files or enter text.")

        # Display current index status
        if self.index and self.passages:
            st.subheader("📊 Current Knowledge Base")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Documents", len(self.passages))
            with col2:
                st.metric("Index Dimension", self.index.d if hasattr(self.index, 'd') else "N/A")
            with col3:
                st.metric("Total Vectors", self.index.ntotal if hasattr(self.index, 'ntotal') else len(self.passages))
            
            # Document preview
            if st.checkbox("📋 Show Document Preview"):
                df = pd.DataFrame([
                    {
                        "ID": p.get("id", "unknown"),
                        "Source": p.get("source", "unknown"),
                        "Type": p.get("metadata", {}).get("type", "unknown"),
                        "Text Preview": p["text"][:200] + "..." if len(p["text"]) > 200 else p["text"]
                    }
                    for p in self.passages[:10]
                ])
                st.dataframe(df, use_container_width=True)

        # Test retrieval section
        st.subheader("🔍 Test Retrieval")
        test_query = st.text_input("Enter a medical query to test retrieval:")
        
        if test_query and self.index:
            with st.spinner(f"Searching knowledge base using {self.settings.RAG_BACKEND}..."):
                results = self.retrieve_documents(test_query)
            
            if results:
                st.write(f"Found {len(results)} relevant documents:")
                
                for i, doc in enumerate(results):
                    with st.expander(f"Document {i+1} (Score: {doc.get('score', 0):.3f})"):
                        st.write(f"**Source:** {doc.get('source', 'unknown')}")
                        st.write(f"**Type:** {doc.get('metadata', {}).get('type', 'unknown')}")
                        st.write(f"**Text:** {doc['text']}")
            else:
                st.warning("No relevant documents found.")

        # Export/Import functionality
        st.subheader("💾 Index Management")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📤 Export Index") and self.index:
                try:
                    export_data = {
                        "passages": self.passages,
                        "settings": {
                            "embed_model": self.settings.EMBED_MODEL,
                            "created_at": pd.Timestamp.now().isoformat()
                        }
                    }
                    
                    export_json = json.dumps(export_data, indent=2)
                    st.download_button(
                        label="💾 Download Knowledge Base",
                        data=export_json,
                        file_name="medical_knowledge_base.json",
                        mime="application/json"
                    )
                except Exception as e:
                    st.error(f"Export failed: {str(e)}")
        
        with col2:
            import_file = st.file_uploader(
                "📥 Import Knowledge Base",
                type=['json'],
                help="Import previously exported knowledge base"
            )
            
            if import_file and st.button("📥 Import"):
                try:
                    import_data = json.load(import_file)
                    if self.build_index(import_data["passages"]):
                        st.success("Successfully imported knowledge base!")
                        st.session_state.rag_system = self
                        st.rerun()
                except Exception as e:
                    st.error(f"Import failed: {str(e)}")