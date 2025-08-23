import pandas as pd
import json
import csv
import re
import logging
from typing import List, Dict, Any, Tuple
from pathlib import Path
import streamlit as st

logger = logging.getLogger(__name__)

def process_medical_documents(uploaded_file) -> List[Dict[str, Any]]:
    """
    Process uploaded medical documents and extract passages
    Returns list of document passages with metadata
    """
    passages = []
    
    try:
        file_extension = Path(uploaded_file.name).suffix.lower()
        
        if file_extension == '.txt':
            passages = process_text_file(uploaded_file)
        elif file_extension == '.json':
            passages = process_json_file(uploaded_file)
        elif file_extension == '.csv':
            passages = process_csv_file(uploaded_file)
        elif file_extension == '.pdf':
            passages = process_pdf_file(uploaded_file)
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")
            
    except Exception as e:
        logger.error(f"Error processing file {uploaded_file.name}: {str(e)}")
        raise e
    
    return passages

def process_text_file(uploaded_file) -> List[Dict[str, Any]]:
    """Process plain text files and split into passages"""
    content = uploaded_file.read().decode('utf-8')
    
    # Split text into paragraphs or sentences
    paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
    
    passages = []
    for i, paragraph in enumerate(paragraphs):
        if len(paragraph) > 50:  # Filter out very short paragraphs
            passages.append({
                "id": f"{uploaded_file.name}_{i}",
                "text": paragraph,
                "source": uploaded_file.name,
                "metadata": {
                    "type": "text_file",
                    "paragraph_index": i,
                    "length": len(paragraph)
                }
            })
    
    return passages

def process_json_file(uploaded_file) -> List[Dict[str, Any]]:
    """Process JSON files containing medical data"""
    data = json.load(uploaded_file)
    passages = []
    
    if isinstance(data, list):
        # List of documents/passages
        for i, item in enumerate(data):
            passage = extract_passage_from_dict(item, f"{uploaded_file.name}_{i}", uploaded_file.name)
            if passage:
                passages.append(passage)
    elif isinstance(data, dict):
        # Single document or structured data
        if "passages" in data:
            # Structured format with passages array
            for i, passage_data in enumerate(data["passages"]):
                passage = extract_passage_from_dict(passage_data, f"{uploaded_file.name}_{i}", uploaded_file.name)
                if passage:
                    passages.append(passage)
        else:
            # Single document
            passage = extract_passage_from_dict(data, uploaded_file.name, uploaded_file.name)
            if passage:
                passages.append(passage)
    
    return passages

def process_csv_file(uploaded_file) -> List[Dict[str, Any]]:
    """Process CSV files containing medical data"""
    df = pd.read_csv(uploaded_file)
    passages = []
    
    # Try to identify text columns
    text_columns = []
    for col in df.columns:
        if any(keyword in col.lower() for keyword in ['text', 'content', 'passage', 'document', 'description', 'answer', 'response']):
            text_columns.append(col)
    
    if not text_columns:
        # If no obvious text columns, use all string columns
        text_columns = [col for col in df.columns if df[col].dtype == 'object']
    
    for idx, row in df.iterrows():
        text_parts = []
        metadata = {}
        
        for col in df.columns:
            if col in text_columns and pd.notna(row[col]):
                text_parts.append(str(row[col]))
            else:
                metadata[col] = row[col] if pd.notna(row[col]) else None
        
        if text_parts:
            combined_text = " ".join(text_parts)
            if len(combined_text.strip()) > 50:
                passages.append({
                    "id": f"{uploaded_file.name}_{idx}",
                    "text": combined_text.strip(),
                    "source": uploaded_file.name,
                    "metadata": {
                        "type": "csv_file",
                        "row_index": idx,
                        "columns_used": text_columns,
                        **metadata
                    }
                })
    
    return passages

def process_pdf_file(uploaded_file) -> List[Dict[str, Any]]:
    """Process PDF files (basic text extraction)"""
    try:
        import PyPDF2
        
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        passages = []
        
        for page_num, page in enumerate(pdf_reader.pages):
            text = page.extract_text()
            
            # Split page text into paragraphs
            paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
            
            for para_idx, paragraph in enumerate(paragraphs):
                if len(paragraph) > 100:  # Filter out headers/footers
                    passages.append({
                        "id": f"{uploaded_file.name}_page{page_num}_{para_idx}",
                        "text": clean_text(paragraph),
                        "source": uploaded_file.name,
                        "metadata": {
                            "type": "pdf_file",
                            "page_number": page_num + 1,
                            "paragraph_index": para_idx
                        }
                    })
        
        return passages
        
    except ImportError:
        st.error("PyPDF2 library not available. Please install it to process PDF files.")
        return []
    except Exception as e:
        logger.error(f"Error processing PDF: {str(e)}")
        return []

def extract_passage_from_dict(data: Dict[str, Any], passage_id: str, source: str) -> Dict[str, Any]:
    """Extract passage from dictionary data"""
    # Try different possible text fields
    text_fields = ['text', 'content', 'passage', 'document', 'description', 'answer', 'response', 'body']
    
    text_content = None
    for field in text_fields:
        if field in data and data[field]:
            text_content = str(data[field])
            break
    
    if not text_content or len(text_content.strip()) < 20:
        return None
    
    # Extract metadata (all other fields)
    metadata = {k: v for k, v in data.items() if k not in text_fields}
    metadata["type"] = "json_file"
    
    return {
        "id": passage_id,
        "text": clean_text(text_content),
        "source": source,
        "metadata": metadata
    }

def clean_text(text: str) -> str:
    """Clean and normalize text content"""
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters that might interfere
    text = re.sub(r'[^\w\s\.,!?;:()\[\]"\'%-]', '', text)
    
    # Normalize quotes
    text = re.sub(r'[""]', '"', text)
    text = re.sub(r'['']', "'", text)
    
    return text.strip()

def prepare_training_data(uploaded_file) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    """
    Prepare training data from uploaded file
    Returns (train_data, eval_data) as tuple of lists
    """
    try:
        file_extension = Path(uploaded_file.name).suffix.lower()
        
        if file_extension == '.json':
            data = json.load(uploaded_file)
        elif file_extension == '.csv':
            df = pd.read_csv(uploaded_file)
            data = df.to_dict('records')
        elif file_extension == '.jsonl':
            data = []
            for line in uploaded_file:
                data.append(json.loads(line.decode('utf-8')))
        else:
            raise ValueError(f"Unsupported training data format: {file_extension}")
        
        # Normalize data format
        normalized_data = []
        for item in data:
            normalized_item = normalize_training_example(item)
            if normalized_item:
                normalized_data.append(normalized_item)
        
        # Split into train/eval (80/20)
        split_index = int(0.8 * len(normalized_data))
        train_data = normalized_data[:split_index]
        eval_data = normalized_data[split_index:]
        
        return train_data, eval_data
        
    except Exception as e:
        logger.error(f"Error preparing training data: {str(e)}")
        raise e

def normalize_training_example(item: Dict[str, Any]) -> Dict[str, str]:
    """Normalize a training example to standard format"""
    
    # Try different field combinations for question-answer pairs
    question_fields = ['question', 'input', 'prompt', 'instruction', 'query']
    answer_fields = ['answer', 'output', 'response', 'completion', 'target']
    context_fields = ['context', 'passage', 'background', 'document']
    
    question = None
    answer = None
    context = ""
    
    # Find question
    for field in question_fields:
        if field in item and item[field]:
            question = str(item[field]).strip()
            break
    
    # Find answer
    for field in answer_fields:
        if field in item and item[field]:
            answer = str(item[field]).strip()
            break
    
    # Find context
    for field in context_fields:
        if field in item and item[field]:
            context = str(item[field]).strip()
            break
    
    if not question or not answer:
        return None
    
    return {
        "question": question,
        "answer": answer,
        "context": context
    }

def validate_medical_content(text: str) -> Tuple[bool, List[str]]:
    """
    Validate if content appears to be medical in nature
    Returns (is_medical, validation_notes)
    """
    medical_keywords = [
        # General medical terms
        'medical', 'patient', 'diagnosis', 'treatment', 'therapy', 'clinical',
        'hospital', 'doctor', 'physician', 'nurse', 'healthcare',
        
        # Body systems
        'cardiovascular', 'respiratory', 'neurological', 'gastrointestinal',
        'musculoskeletal', 'endocrine', 'immune', 'reproductive',
        
        # Common conditions
        'diabetes', 'hypertension', 'cancer', 'infection', 'inflammation',
        'disease', 'syndrome', 'disorder', 'condition',
        
        # Medical procedures
        'surgery', 'biopsy', 'scan', 'test', 'examination', 'procedure',
        'medication', 'drug', 'prescription', 'dose', 'dosage',
        
        # Anatomy
        'heart', 'lung', 'brain', 'liver', 'kidney', 'blood', 'bone',
        'muscle', 'tissue', 'organ', 'cell', 'vessel'
    ]
    
    text_lower = text.lower()
    found_keywords = [kw for kw in medical_keywords if kw in text_lower]
    
    is_medical = len(found_keywords) >= 2  # Require at least 2 medical keywords
    
    validation_notes = []
    if is_medical:
        validation_notes.append(f"Found medical keywords: {', '.join(found_keywords[:5])}")
    else:
        validation_notes.append("Low medical content confidence - consider reviewing")
    
    return is_medical, validation_notes

def extract_medical_entities(text: str) -> Dict[str, List[str]]:
    """
    Extract medical entities from text using rule-based approach
    Returns dictionary of entity types and their values
    """
    entities = {
        'medications': [],
        'conditions': [],
        'procedures': [],
        'anatomy': [],
        'symptoms': []
    }
    
    # Simple rule-based entity extraction
    medication_patterns = [
        r'\b\w+mycin\b',  # antibiotics ending in -mycin
        r'\b\w+cillin\b',  # antibiotics ending in -cillin
        r'\b\w+pril\b',   # ACE inhibitors
        r'\b\w+olol\b',   # beta blockers
        r'\binsulin\b', r'\baspirin\b', r'\bibuprofen\b'
    ]
    
    condition_patterns = [
        r'\bdiabetes\b', r'\bhypertension\b', r'\bcancer\b',
        r'\binfection\b', r'\bpneumonia\b', r'\basthma\b'
    ]
    
    # Extract entities using patterns
    text_lower = text.lower()
    
    for pattern in medication_patterns:
        matches = re.findall(pattern, text_lower)
        entities['medications'].extend(matches)
    
    for pattern in condition_patterns:
        matches = re.findall(pattern, text_lower)
        entities['conditions'].extend(matches)
    
    # Remove duplicates
    for key in entities:
        entities[key] = list(set(entities[key]))
    
    return entities

def create_data_summary(passages: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Create a summary of the processed data"""
    if not passages:
        return {"error": "No passages to summarize"}
    
    # Basic statistics
    total_passages = len(passages)
    total_text_length = sum(len(p['text']) for p in passages)
    avg_passage_length = total_text_length / total_passages
    
    # Source distribution
    sources = {}
    for passage in passages:
        source = passage.get('source', 'unknown')
        sources[source] = sources.get(source, 0) + 1
    
    # Metadata analysis
    metadata_fields = set()
    for passage in passages:
        if 'metadata' in passage:
            metadata_fields.update(passage['metadata'].keys())
    
    # Medical content validation
    medical_passages = 0
    for passage in passages:
        is_medical, _ = validate_medical_content(passage['text'])
        if is_medical:
            medical_passages += 1
    
    return {
        "total_passages": total_passages,
        "total_text_length": total_text_length,
        "avg_passage_length": avg_passage_length,
        "medical_content_ratio": medical_passages / total_passages,
        "source_distribution": sources,
        "metadata_fields": list(metadata_fields),
        "quality_score": calculate_quality_score(passages)
    }

def calculate_quality_score(passages: List[Dict[str, Any]]) -> float:
    """Calculate a quality score for the dataset"""
    if not passages:
        return 0.0
    
    scores = []
    for passage in passages:
        score = 0.0
        text = passage['text']
        
        # Length score (prefer moderate length)
        if 50 <= len(text) <= 2000:
            score += 0.3
        elif len(text) > 2000:
            score += 0.2
        
        # Medical content score
        is_medical, _ = validate_medical_content(text)
        if is_medical:
            score += 0.4
        
        # Structure score (has proper sentences)
        sentences = text.split('.')
        if len(sentences) >= 2:
            score += 0.2
        
        # Metadata score
        if 'metadata' in passage and passage['metadata']:
            score += 0.1
        
        scores.append(score)
    
    return sum(scores) / len(scores)
