# Overview

MedGemma AI Platform is a comprehensive medical AI system built with Streamlit that combines Retrieval-Augmented Generation (RAG), fine-tuning capabilities, and safety systems specifically designed for medical applications. The platform enables users to train and deploy medical AI models using the Gemma architecture while ensuring safety guardrails and proper citation handling for medical information.

The system provides a complete workflow from data ingestion and model fine-tuning to evaluation and deployment, with specialized components for medical document processing, safety validation, and model export capabilities.

# User Preferences

Preferred communication style: Simple, everyday language.

# System Architecture

## Frontend Architecture
- **Streamlit-based Web Interface**: Single-page application with sidebar navigation providing access to different modules including RAG system, fine-tuning, evaluation, safety checks, model export, and chat interface
- **Session State Management**: Persistent state across components using Streamlit's session state for maintaining model instances, training status, and chat history
- **Modular Component Design**: Each major functionality (RAG, fine-tuning, evaluation, etc.) is implemented as a separate component class for maintainability and separation of concerns

## Backend Architecture
- **Transformer-based Models**: Built around Hugging Face transformers with primary support for Google's Gemma models (2B and 7B variants)
- **LoRA Fine-tuning**: Parameter-Efficient Fine-Tuning using Low-Rank Adaptation (LoRA) with configurable rank, alpha, and dropout parameters
- **Mixed Precision Training**: Support for both FP16 and BF16 training with automatic device detection (CUDA, MPS, CPU)
- **Memory Optimization**: 4-bit and 8-bit quantization support using BitsAndBytesConfig for resource-constrained environments

## Data Storage Solutions
- **FAISS Vector Database**: High-performance similarity search using Facebook AI Similarity Search for document retrieval
- **Local File System**: Models, indices, and metadata stored locally with configurable paths
- **Pickle Serialization**: Metadata and embeddings cached using Python pickle for fast loading
- **Multiple Data Formats**: Support for TXT, JSON, CSV, and PDF document ingestion

## Authentication and Authorization
- **Environment-based Configuration**: Settings managed through environment variables with sensible defaults
- **No Built-in Authentication**: Currently designed for single-user deployment without authentication layer

## RAG System Architecture
- **Sentence Transformers**: Uses all-mpnet-base-v2 model for generating document embeddings
- **Document Processing Pipeline**: Automated extraction and chunking of medical documents with metadata preservation
- **Retrieval Pipeline**: Top-K similarity search with configurable threshold filtering
- **Context Injection**: Retrieved passages automatically prepended to model prompts for grounding

## Safety and Compliance
- **Medical Safety Rules**: Hardcoded safety patterns to prevent personalized medical advice, diagnosis, or treatment recommendations
- **Citation Requirements**: Enforced citation formatting from retrieved sources
- **Content Filtering**: Pattern-based blocking of potentially harmful medical content
- **Emergency Response**: Automatic detection and appropriate responses for medical emergency queries

## Model Training and Evaluation
- **SFT Trainer Integration**: Supervised Fine-Tuning using TRL (Transformer Reinforcement Learning) library
- **Comprehensive Metrics**: Exact Match, F1-score, BLEU, and ROUGE evaluation metrics
- **Training Monitoring**: Real-time training progress with configurable logging and checkpointing
- **Gradient Accumulation**: Support for effective large batch training on limited hardware

## Model Export and Deployment
- **Multiple Export Formats**: LoRA adapters, merged FP16/FP32 models, GGUF for llama.cpp, ONNX, and TensorRT formats
- **Hugging Face Integration**: Direct model publishing to Hugging Face Hub
- **Local Serving Options**: Support for various deployment scenarios from local inference to production serving

# External Dependencies

## Core ML Libraries
- **Hugging Face Ecosystem**: transformers, peft, trl, datasets for model handling and training
- **PyTorch**: Primary deep learning framework with CUDA support
- **Sentence Transformers**: Document embedding generation for RAG system
- **FAISS**: Facebook AI Similarity Search for efficient vector similarity search

## Data Processing and Visualization
- **Streamlit**: Web application framework for the user interface
- **Pandas**: Data manipulation and analysis for training data processing
- **NumPy**: Numerical computations and array operations
- **Plotly**: Interactive visualizations for evaluation dashboards and metrics

## Model Training and Optimization
- **BitsAndBytesConfig**: 4-bit and 8-bit model quantization for memory efficiency
- **LoRA (Low-Rank Adaptation)**: Parameter-efficient fine-tuning approach
- **Mixed Precision Training**: Automatic Mixed Precision (AMP) for faster training

## File Processing and Utilities
- **Pathlib**: Modern path handling for cross-platform compatibility
- **JSON/CSV Processing**: Built-in Python libraries for data format handling
- **Pickle**: Object serialization for caching embeddings and metadata
- **Regular Expressions**: Text processing and safety pattern matching

## Optional Integrations
- **Hugging Face Hub**: Model sharing and deployment platform
- **PDF Processing**: Document extraction capabilities (implementation pending)
- **ONNX Runtime**: Model optimization and deployment format
- **TensorRT**: NVIDIA GPU optimization for inference acceleration

## Development and Monitoring
- **Python Logging**: Comprehensive logging throughout the application
- **Environment Variables**: Configuration management via OS environment
- **Git/Version Control**: Standard development workflow support