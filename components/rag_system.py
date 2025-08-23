import streamlit as st
import os
import pickle
import json
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from typing import List, Dict, Any
from utils.data_processing import process_medical_documents
from config.settings import Settings
from utils.fallbacks import get_embedder, get_faiss_index, normalize_l2, HAS_SENTENCE_TRANSFORMERS, HAS_FAISS

logger = logging.getLogger(__name__)

class RAGSystem:
    def __init__(self):
        self.settings = Settings()
        self.embedder = None
        self.index = None
        self.passages = []
        self.index_path = "data/rag_index.faiss"
        self.meta_path = "data/rag_meta.pkl"
        
    def load_embedder(self):
        """Load the sentence transformer model"""
        if self.embedder is None:
            with st.spinner("Loading embedding model..."):
                self.embedder = get_embedder(self.settings.EMBED_MODEL)
        return self.embedder
    
    def build_index(self, passages: List[Dict[str, Any]]) -> bool:
        """Build FAISS index from passages"""
        try:
            # Ensure data directory exists
            os.makedirs("data", exist_ok=True)
            
            embedder = self.load_embedder()
            texts = [p["text"] for p in passages]
            
            # Generate embeddings
            with st.spinner(f"Generating embeddings for {len(texts)} documents..."):
                embeddings = embedder.encode(texts, show_progress_bar=False, convert_to_numpy=True)
            
            # Create FAISS index
            dim = embeddings.shape[1]
            index = get_faiss_index(dim)  # Inner product (cosine after normalizing)
            embeddings = normalize_l2(embeddings)
            index.add(embeddings)
            
            # Save index and metadata
            if HAS_FAISS:
                import faiss
                faiss.write_index(index, self.index_path)
            else:
                # Save fallback index
                with open(self.index_path + ".fallback", "wb") as f:
                    pickle.dump(index, f)
            
            with open(self.meta_path, "wb") as f:
                pickle.dump(passages, f)
            
            self.index = index
            self.passages = passages
            
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
            
            if HAS_FAISS and os.path.exists(self.index_path) and os.path.exists(self.meta_path):
                import faiss
                self.index = faiss.read_index(self.index_path)
                with open(self.meta_path, "rb") as f:
                    self.passages = pickle.load(f)
                
                self.load_embedder()  # Ensure embedder is loaded
                logger.info(f"Loaded FAISS index with {len(self.passages)} documents")
                return True
            elif os.path.exists(fallback_path) and os.path.exists(self.meta_path):
                with open(fallback_path, "rb") as f:
                    self.index = pickle.load(f)
                with open(self.meta_path, "rb") as f:
                    self.passages = pickle.load(f)
                
                self.load_embedder()  # Ensure embedder is loaded
                logger.info(f"Loaded fallback index with {len(self.passages)} documents")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Error loading index: {str(e)}")
            st.error(f"Error loading index: {str(e)}")
            return False
    
    def retrieve_documents(self, query: str, k: int = None) -> List[Dict[str, Any]]:
        """Retrieve top-k most relevant documents"""
        if k is None:
            k = self.settings.RETRIEVAL_TOP_K
            
        if not self.index or not self.embedder:
            return []
        
        try:
            # Encode query
            q_emb = self.embedder.encode([query], convert_to_numpy=True)
            q_emb = normalize_l2(q_emb)
            
            # Search
            scores, indices = self.index.search(q_emb, k)
            
            results = []
            for i, idx in enumerate(indices[0]):
                if idx >= 0:  # Valid index
                    passage = self.passages[idx].copy()
                    passage['score'] = float(scores[0][i])
                    results.append(passage)
            
            return results
            
        except Exception as e:
            logger.error(f"Error retrieving documents: {str(e)}")
            return []
    
    def render(self):
        """Render the RAG system interface"""
        st.header("🔍 Retrieval-Augmented Generation System")
        
        # Check if index exists
        fallback_exists = os.path.exists(self.index_path + ".fallback") and os.path.exists(self.meta_path)
        index_exists = (os.path.exists(self.index_path) and os.path.exists(self.meta_path)) or fallback_exists
        
        if index_exists and not self.index:
            if st.button("Load Existing Index"):
                if self.load_index():
                    st.success(f"Successfully loaded index with {len(self.passages)} documents!")
                    st.session_state.rag_system = self
                    st.rerun()
        
        # Document upload and indexing
        st.subheader("Document Management")
        
        # File upload
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
        
        # URL input for medical databases
        url_input = st.text_input(
            "Medical Database URL (optional):",
            help="Enter URL to fetch medical documents from trusted sources"
        )
        
        # Build index button
        if st.button("Build Knowledge Base", type="primary"):
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
            
            # Process URL (basic implementation)
            if url_input.strip():
                st.warning("URL processing not implemented in this version. Please upload files directly.")
            
            if passages:
                if self.build_index(passages):
                    st.success(f"Successfully built knowledge base with {len(passages)} documents!")
                    st.session_state.rag_system = self
                    st.rerun()
            else:
                st.warning("No documents to process. Please upload files or enter text.")
        
        # Display current index status
        if self.index and self.passages:
            st.subheader("Current Knowledge Base")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Documents", len(self.passages))
            with col2:
                st.metric("Index Dimension", self.index.d if self.index else 0)
            with col3:
                st.metric("Total Vectors", self.index.ntotal if self.index else 0)
            
            # Document preview
            if st.checkbox("Show Document Preview"):
                df = pd.DataFrame([
                    {
                        "ID": p.get("id", "unknown"),
                        "Source": p.get("source", "unknown"),
                        "Text Preview": p["text"][:200] + "..." if len(p["text"]) > 200 else p["text"]
                    }
                    for p in self.passages[:10]  # Show first 10
                ])
                st.dataframe(df, use_container_width=True)
        
        # Test retrieval
        st.subheader("Test Retrieval")
        test_query = st.text_input("Enter a medical query to test retrieval:")
        
        if test_query and self.index:
            with st.spinner("Searching knowledge base..."):
                results = self.retrieve_documents(test_query)
            
            if results:
                st.write(f"Found {len(results)} relevant documents:")
                
                for i, doc in enumerate(results):
                    with st.expander(f"Document {i+1} (Score: {doc['score']:.3f})"):
                        st.write(f"**Source:** {doc.get('source', 'unknown')}")
                        st.write(f"**Text:** {doc['text']}")
                        if 'metadata' in doc:
                            st.write(f"**Metadata:** {doc['metadata']}")
            else:
                st.warning("No relevant documents found.")
        
        # Export/Import functionality
        st.subheader("Index Management")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Export Index") and self.index:
                try:
                    # Create export data
                    export_data = {
                        "passages": self.passages,
                        "settings": {
                            "embed_model": self.settings.EMBED_MODEL,
                            "created_at": pd.Timestamp.now().isoformat()
                        }
                    }
                    
                    export_json = json.dumps(export_data, indent=2)
                    st.download_button(
                        label="Download Knowledge Base",
                        data=export_json,
                        file_name="medical_knowledge_base.json",
                        mime="application/json"
                    )
                except Exception as e:
                    st.error(f"Export failed: {str(e)}")
        
        with col2:
            import_file = st.file_uploader(
                "Import Knowledge Base",
                type=['json'],
                help="Import previously exported knowledge base"
            )
            
            if import_file and st.button("Import"):
                try:
                    import_data = json.load(import_file)
                    if self.build_index(import_data["passages"]):
                        st.success("Successfully imported knowledge base!")
                        st.session_state.rag_system = self
                        st.rerun()
                except Exception as e:
                    st.error(f"Import failed: {str(e)}")
