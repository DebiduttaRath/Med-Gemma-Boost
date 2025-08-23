import streamlit as st
import os
import logging
from pathlib import Path

# Import custom components
from components.rag_system import RAGSystem
from components.fine_tuning import FineTuningInterface
from components.evaluation import EvaluationDashboard
from components.safety import SafetySystem
from components.model_export import ModelExport
from components.chat_interface import ChatInterface
from config.settings import Settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize settings
settings = Settings()

def initialize_session_state():
    """Initialize session state variables"""
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = None
    if 'model_trained' not in st.session_state:
        st.session_state.model_trained = False
    if 'current_model' not in st.session_state:
        st.session_state.current_model = None
    if 'safety_system' not in st.session_state:
        st.session_state.safety_system = SafetySystem()
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

def main():
    st.set_page_config(
        page_title="MedGemma AI Platform",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    initialize_session_state()
    
    # Main title
    st.title("🏥 MedGemma AI Platform")
    st.markdown("*Advanced Medical AI with RAG, Fine-tuning, and Safety Systems*")
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Select Module",
        [
            "🏠 Dashboard",
            "🔍 RAG System",
            "🎯 Fine-tuning",
            "📊 Evaluation",
            "🛡️ Safety & Guardrails",
            "📦 Model Export",
            "💬 Chat Interface"
        ]
    )
    
    # System status in sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("### System Status")
    
    # Check RAG system status
    rag_status = "✅ Ready" if st.session_state.rag_system else "❌ Not Initialized"
    st.sidebar.markdown(f"**RAG System:** {rag_status}")
    
    # Check model status
    model_status = "✅ Trained" if st.session_state.model_trained else "❌ Not Trained"
    st.sidebar.markdown(f"**Model:** {model_status}")
    
    # Display current model info
    if st.session_state.current_model:
        st.sidebar.markdown(f"**Current Model:** {st.session_state.current_model}")
    
    # Main content based on selected page
    if page == "🏠 Dashboard":
        show_dashboard()
    elif page == "🔍 RAG System":
        rag_interface = RAGSystem()
        rag_interface.render()
    elif page == "🎯 Fine-tuning":
        ft_interface = FineTuningInterface()
        ft_interface.render()
    elif page == "📊 Evaluation":
        eval_interface = EvaluationDashboard()
        eval_interface.render()
    elif page == "🛡️ Safety & Guardrails":
        st.session_state.safety_system.render()
    elif page == "📦 Model Export":
        export_interface = ModelExport()
        export_interface.render()
    elif page == "💬 Chat Interface":
        chat_interface = ChatInterface()
        chat_interface.render()

def show_dashboard():
    """Display main dashboard with system overview"""
    st.header("System Overview")
    
    # Create metrics columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="RAG Documents",
            value=len(st.session_state.rag_system.passages) if st.session_state.rag_system else 0,
            delta="Ready" if st.session_state.rag_system else "Not loaded"
        )
    
    with col2:
        st.metric(
            label="Model Status",
            value="Trained" if st.session_state.model_trained else "Base",
            delta="Fine-tuned" if st.session_state.model_trained else "Not trained"
        )
    
    with col3:
        st.metric(
            label="Safety Checks",
            value="Active",
            delta="Monitoring"
        )
    
    with col4:
        st.metric(
            label="Chat Sessions",
            value=len(st.session_state.chat_history),
            delta="Active"
        )
    
    # Quick start guide
    st.markdown("---")
    st.header("Quick Start Guide")
    
    steps = [
        {
            "title": "1. Initialize RAG System",
            "description": "Upload medical documents and build the knowledge base",
            "status": "✅ Complete" if st.session_state.rag_system else "⏳ Pending"
        },
        {
            "title": "2. Fine-tune Model",
            "description": "Train the model on your specific medical data",
            "status": "✅ Complete" if st.session_state.model_trained else "⏳ Pending"
        },
        {
            "title": "3. Evaluate Performance",
            "description": "Run evaluation metrics and review results",
            "status": "🔄 Available"
        },
        {
            "title": "4. Deploy for Chat",
            "description": "Start using the medical AI assistant",
            "status": "🔄 Available"
        }
    ]
    
    for step in steps:
        with st.expander(step["title"]):
            st.markdown(f"**Status:** {step['status']}")
            st.markdown(step["description"])
    
    # Recent activity
    st.markdown("---")
    st.header("Recent Activity")
    
    if st.session_state.chat_history:
        st.markdown("**Latest Chat Sessions:**")
        for i, chat in enumerate(st.session_state.chat_history[-5:]):
            st.markdown(f"- Session {i+1}: {chat.get('timestamp', 'Unknown time')}")
    else:
        st.info("No recent activity. Start by initializing the RAG system or begin a chat session.")
    
    # System configuration
    st.markdown("---")
    st.header("System Configuration")
    
    config_col1, config_col2 = st.columns(2)
    
    with config_col1:
        st.subheader("Model Settings")
        st.code(f"""
Base Model: {settings.BASE_MODEL}
Max Tokens: {settings.MAX_NEW_TOKENS}
Temperature: {settings.TEMPERATURE}
        """)
    
    with config_col2:
        st.subheader("RAG Settings")
        st.code(f"""
Embedding Model: {settings.EMBED_MODEL}
Top-K Retrieval: {settings.RETRIEVAL_TOP_K}
Index Type: FAISS
        """)

if __name__ == "__main__":
    main()
