import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
from collections import Counter
import json
import logging
from typing import List, Dict, Any, Tuple
from utils.metrics import compute_exact_match, compute_f1_score, compute_bleu_score, compute_rouge_score
from components.chat_interface import ChatInterface
from config.settings import Settings

logger = logging.getLogger(__name__)

class EvaluationDashboard:
    def __init__(self):
        self.settings = Settings()
        self.chat_interface = ChatInterface()
        
    def normalize_answer(self, text: str) -> str:
        """Normalize answer for comparison"""
        text = text.lower()
        text = re.sub(r'\b(a|an|the)\b', ' ', text)
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        text = ' '.join(text.split())
        return text
    
    def compute_metrics(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """Compute evaluation metrics"""
        metrics = {
            'exact_match': [],
            'f1_score': [],
            'bleu_score': [],
            'rouge_l': []
        }
        
        for pred, ref in zip(predictions, references):
            # Exact Match
            em = compute_exact_match(pred, ref)
            metrics['exact_match'].append(em)
            
            # F1 Score
            f1 = compute_f1_score(pred, ref)
            metrics['f1_score'].append(f1)
            
            # BLEU Score
            bleu = compute_bleu_score(pred, ref)
            metrics['bleu_score'].append(bleu)
            
            # ROUGE-L Score
            rouge = compute_rouge_score(pred, ref)
            metrics['rouge_l'].append(rouge)
        
        # Compute averages
        avg_metrics = {
            'exact_match': np.mean(metrics['exact_match']) * 100,
            'f1_score': np.mean(metrics['f1_score']) * 100,
            'bleu_score': np.mean(metrics['bleu_score']) * 100,
            'rouge_l': np.mean(metrics['rouge_l']) * 100
        }
        
        return avg_metrics, metrics
    
    def compute_confidence_calibration(self, predictions: List[str], references: List[str], 
                                     confidences: List[float]) -> Dict[str, Any]:
        """Compute confidence calibration metrics"""
        if not confidences:
            return {}
        
        accuracies = [compute_exact_match(pred, ref) for pred, ref in zip(predictions, references)]
        
        # Bin predictions by confidence
        bins = np.linspace(0, 1, 11)
        bin_boundaries = list(zip(bins[:-1], bins[1:]))
        
        calibration_data = []
        for low, high in bin_boundaries:
            mask = (np.array(confidences) >= low) & (np.array(confidences) < high)
            if np.sum(mask) > 0:
                avg_confidence = np.mean(np.array(confidences)[mask])
                avg_accuracy = np.mean(np.array(accuracies)[mask])
                count = np.sum(mask)
                
                calibration_data.append({
                    'bin_lower': low,
                    'bin_upper': high,
                    'avg_confidence': avg_confidence,
                    'avg_accuracy': avg_accuracy,
                    'count': count
                })
        
        # Expected Calibration Error (ECE)
        ece = 0
        total_samples = len(predictions)
        for data in calibration_data:
            ece += (data['count'] / total_samples) * abs(data['avg_confidence'] - data['avg_accuracy'])
        
        return {
            'calibration_data': calibration_data,
            'ece': ece
        }
    
    def run_evaluation(self, test_data: List[Dict[str, str]], model_name: str = "current") -> Dict[str, Any]:
        """Run comprehensive evaluation"""
        predictions = []
        references = []
        confidences = []
        response_times = []
        retrieved_contexts = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, example in enumerate(test_data):
            progress = (i + 1) / len(test_data)
            progress_bar.progress(progress)
            status_text.text(f"Evaluating: {i + 1}/{len(test_data)}")
            
            question = example.get('question', example.get('input', ''))
            reference = example.get('answer', example.get('output', example.get('reference', '')))
            context = example.get('context', example.get('passage', ''))
            
            try:
                # Get model prediction
                import time
                start_time = time.time()
                
                if st.session_state.rag_system:
                    prediction, contexts = self.chat_interface.generate_response_with_rag(
                        question, context, st.session_state.rag_system
                    )
                    retrieved_contexts.append(contexts)
                else:
                    prediction = self.chat_interface.generate_response(question, context)
                    retrieved_contexts.append([])
                
                response_time = time.time() - start_time
                
                predictions.append(prediction)
                references.append(reference)
                response_times.append(response_time)
                
                # Placeholder confidence (would need proper implementation)
                confidences.append(0.8)  # Mock confidence
                
            except Exception as e:
                logger.error(f"Error evaluating example {i}: {str(e)}")
                predictions.append("")
                references.append(reference)
                response_times.append(0)
                confidences.append(0)
                retrieved_contexts.append([])
        
        progress_bar.empty()
        status_text.empty()
        
        # Compute metrics
        avg_metrics, detailed_metrics = self.compute_metrics(predictions, references)
        calibration_metrics = self.compute_confidence_calibration(predictions, references, confidences)
        
        # Compile results
        results = {
            'model_name': model_name,
            'num_examples': len(test_data),
            'avg_metrics': avg_metrics,
            'detailed_metrics': detailed_metrics,
            'calibration_metrics': calibration_metrics,
            'avg_response_time': np.mean(response_times),
            'predictions': predictions,
            'references': references,
            'confidences': confidences,
            'response_times': response_times,
            'retrieved_contexts': retrieved_contexts,
            'test_data': test_data
        }
        
        return results
    
    def visualize_metrics(self, results: Dict[str, Any]):
        """Create visualizations for evaluation metrics"""
        
        # Metrics overview
        st.subheader("📊 Metrics Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Exact Match",
                f"{results['avg_metrics']['exact_match']:.1f}%",
                delta=None
            )
        
        with col2:
            st.metric(
                "F1 Score",
                f"{results['avg_metrics']['f1_score']:.1f}%",
                delta=None
            )
        
        with col3:
            st.metric(
                "BLEU Score",
                f"{results['avg_metrics']['bleu_score']:.1f}%",
                delta=None
            )
        
        with col4:
            st.metric(
                "ROUGE-L",
                f"{results['avg_metrics']['rouge_l']:.1f}%",
                delta=None
            )
        
        # Performance metrics
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                "Avg Response Time",
                f"{results['avg_response_time']:.2f}s",
                delta=None
            )
        
        with col2:
            if results['calibration_metrics']:
                st.metric(
                    "Calibration Error (ECE)",
                    f"{results['calibration_metrics']['ece']:.3f}",
                    delta=None
                )
        
        # Detailed visualizations
        tab1, tab2, tab3, tab4 = st.tabs(["Score Distribution", "Calibration", "Performance Analysis", "Error Analysis"])
        
        with tab1:
            self.plot_score_distribution(results)
        
        with tab2:
            self.plot_calibration(results)
        
        with tab3:
            self.plot_performance_analysis(results)
        
        with tab4:
            self.show_error_analysis(results)
    
    def plot_score_distribution(self, results: Dict[str, Any]):
        """Plot score distribution histograms"""
        st.subheader("Score Distribution")
        
        metrics_data = results['detailed_metrics']
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Exact Match', 'F1 Score', 'BLEU Score', 'ROUGE-L'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Plot histograms
        metrics = ['exact_match', 'f1_score', 'bleu_score', 'rouge_l']
        positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
        
        for metric, (row, col) in zip(metrics, positions):
            values = np.array(metrics_data[metric]) * 100 if metric == 'exact_match' else np.array(metrics_data[metric]) * 100
            
            fig.add_trace(
                go.Histogram(
                    x=values,
                    name=metric.replace('_', ' ').title(),
                    nbinsx=20,
                    showlegend=False
                ),
                row=row, col=col
            )
        
        fig.update_layout(height=600, title_text="Metric Score Distributions")
        st.plotly_chart(fig, use_container_width=True)
    
    def plot_calibration(self, results: Dict[str, Any]):
        """Plot calibration curve"""
        st.subheader("Model Calibration")
        
        if not results['calibration_metrics']:
            st.warning("Calibration data not available.")
            return
        
        calibration_data = results['calibration_metrics']['calibration_data']
        
        if not calibration_data:
            st.warning("Insufficient data for calibration analysis.")
            return
        
        # Prepare data
        confidences = [d['avg_confidence'] for d in calibration_data]
        accuracies = [d['avg_accuracy'] for d in calibration_data]
        counts = [d['count'] for d in calibration_data]
        
        # Create calibration plot
        fig = go.Figure()
        
        # Perfect calibration line
        fig.add_trace(go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode='lines',
            name='Perfect Calibration',
            line=dict(dash='dash', color='gray')
        ))
        
        # Actual calibration
        fig.add_trace(go.Scatter(
            x=confidences,
            y=accuracies,
            mode='markers+lines',
            name='Model Calibration',
            marker=dict(
                size=[c/2 for c in counts],  # Size proportional to count
                sizemode='diameter',
                sizeref=2.*max(counts)/(20.**2),
                sizemin=4
            ),
            text=[f'Count: {c}' for c in counts],
            hovertemplate='<b>Confidence:</b> %{x:.2f}<br><b>Accuracy:</b> %{y:.2f}<br>%{text}<extra></extra>'
        ))
        
        fig.update_layout(
            title='Confidence vs Accuracy (Reliability Diagram)',
            xaxis_title='Mean Predicted Confidence',
            yaxis_title='Mean Accuracy',
            width=700,
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # ECE display
        ece = results['calibration_metrics']['ece']
        st.info(f"**Expected Calibration Error (ECE):** {ece:.3f}")
        st.caption("Lower ECE indicates better calibration. ECE < 0.1 is generally considered well-calibrated.")
    
    def plot_performance_analysis(self, results: Dict[str, Any]):
        """Plot performance analysis charts"""
        st.subheader("Performance Analysis")
        
        # Response time analysis
        response_times = results['response_times']
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Response time histogram
            fig = px.histogram(
                x=response_times,
                nbins=20,
                title="Response Time Distribution",
                labels={'x': 'Response Time (seconds)', 'y': 'Count'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Response time vs accuracy
            accuracies = [1 if em else 0 for em in results['detailed_metrics']['exact_match']]
            
            fig = px.scatter(
                x=response_times,
                y=accuracies,
                title="Response Time vs Accuracy",
                labels={'x': 'Response Time (seconds)', 'y': 'Accuracy (Exact Match)'},
                trendline="ols"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Performance summary
        st.subheader("Performance Summary")
        perf_col1, perf_col2, perf_col3 = st.columns(3)
        
        with perf_col1:
            st.metric("Min Response Time", f"{min(response_times):.2f}s")
        
        with perf_col2:
            st.metric("Max Response Time", f"{max(response_times):.2f}s")
        
        with perf_col3:
            st.metric("Std Response Time", f"{np.std(response_times):.2f}s")
    
    def show_error_analysis(self, results: Dict[str, Any]):
        """Show detailed error analysis"""
        st.subheader("Error Analysis")
        
        # Find examples with low scores
        f1_scores = results['detailed_metrics']['f1_score']
        em_scores = results['detailed_metrics']['exact_match']
        
        # Sort by F1 score (ascending) to show worst examples first
        sorted_indices = np.argsort(f1_scores)
        
        # Show worst performing examples
        st.subheader("Lowest Scoring Examples")
        
        num_examples = min(10, len(sorted_indices))
        
        for i in range(num_examples):
            idx = sorted_indices[i]
            
            with st.expander(f"Example {idx + 1} (F1: {f1_scores[idx]:.2f}, EM: {em_scores[idx]:.0f})"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Question:**")
                    st.write(results['test_data'][idx].get('question', 'N/A'))
                    
                    st.write("**Context:**")
                    st.write(results['test_data'][idx].get('context', 'N/A')[:200] + "...")
                
                with col2:
                    st.write("**Reference Answer:**")
                    st.write(results['references'][idx])
                    
                    st.write("**Model Prediction:**")
                    st.write(results['predictions'][idx])
                
                # Show retrieved contexts if available
                if results['retrieved_contexts'][idx]:
                    st.write("**Retrieved Contexts:**")
                    for j, ctx in enumerate(results['retrieved_contexts'][idx][:2]):  # Show top 2
                        st.write(f"Context {j+1}: {ctx.get('text', '')[:150]}...")
        
        # Error statistics
        st.subheader("Error Statistics")
        
        # Count different types of errors
        zero_f1 = sum(1 for score in f1_scores if score == 0)
        low_f1 = sum(1 for score in f1_scores if 0 < score < 0.5)
        med_f1 = sum(1 for score in f1_scores if 0.5 <= score < 0.8)
        high_f1 = sum(1 for score in f1_scores if score >= 0.8)
        
        error_stats = {
            'No Match (F1=0)': zero_f1,
            'Low Match (0<F1<0.5)': low_f1,
            'Medium Match (0.5≤F1<0.8)': med_f1,
            'High Match (F1≥0.8)': high_f1
        }
        
        # Create pie chart
        fig = px.pie(
            values=list(error_stats.values()),
            names=list(error_stats.keys()),
            title="Distribution of F1 Score Ranges"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    def export_results(self, results: Dict[str, Any]):
        """Export evaluation results"""
        export_data = {
            'model_name': results['model_name'],
            'evaluation_date': pd.Timestamp.now().isoformat(),
            'metrics': results['avg_metrics'],
            'num_examples': results['num_examples'],
            'detailed_results': [
                {
                    'question': results['test_data'][i].get('question', ''),
                    'reference': results['references'][i],
                    'prediction': results['predictions'][i],
                    'exact_match': results['detailed_metrics']['exact_match'][i],
                    'f1_score': results['detailed_metrics']['f1_score'][i],
                    'bleu_score': results['detailed_metrics']['bleu_score'][i],
                    'rouge_l': results['detailed_metrics']['rouge_l'][i],
                    'response_time': results['response_times'][i],
                    'confidence': results['confidences'][i]
                }
                for i in range(len(results['predictions']))
            ]
        }
        
        return json.dumps(export_data, indent=2)
    
    def render(self):
        """Render the evaluation dashboard"""
        st.header("📊 Evaluation Dashboard")
        
        # Load test data
        st.subheader("Test Data")
        
        data_source = st.radio(
            "Select test data source:",
            ["Upload Test File", "Use Manual Examples", "Generate Synthetic Data"]
        )
        
        test_data = []
        
        if data_source == "Upload Test File":
            uploaded_file = st.file_uploader(
                "Upload test data",
                type=['json', 'csv', 'jsonl'],
                help="Upload test data in JSON, CSV, or JSONL format with questions and reference answers"
            )
            
            if uploaded_file:
                try:
                    if uploaded_file.type == 'application/json':
                        test_data = json.load(uploaded_file)
                    elif uploaded_file.type == 'text/csv':
                        df = pd.read_csv(uploaded_file)
                        test_data = df.to_dict('records')
                    
                    st.success(f"Loaded {len(test_data)} test examples")
                    
                    # Show preview
                    if st.checkbox("Show data preview"):
                        st.dataframe(pd.DataFrame(test_data[:5]))
                        
                except Exception as e:
                    st.error(f"Error loading test data: {str(e)}")
        
        elif data_source == "Use Manual Examples":
            if 'manual_test_data' not in st.session_state:
                st.session_state.manual_test_data = []
            
            with st.form("manual_test_entry"):
                question = st.text_area("Test Question:")
                answer = st.text_area("Reference Answer:")
                context = st.text_area("Context (optional):")
                
                if st.form_submit_button("Add Test Example"):
                    if question and answer:
                        example = {
                            "question": question,
                            "answer": answer,
                            "context": context if context else ""
                        }
                        st.session_state.manual_test_data.append(example)
                        st.success("Test example added!")
                        st.rerun()
            
            if st.session_state.manual_test_data:
                st.write(f"Current test examples: {len(st.session_state.manual_test_data)}")
                test_data = st.session_state.manual_test_data
        
        elif data_source == "Generate Synthetic Data":
            st.info("Synthetic data generation is not implemented. Please upload real test data.")
        
        # Model selection
        st.subheader("Model Selection")
        
        model_options = ["Current Loaded Model"]
        if st.session_state.model_trained:
            model_options.append("Fine-tuned Model")
        
        selected_model = st.selectbox("Select model to evaluate:", model_options)
        
        # Evaluation settings
        st.subheader("Evaluation Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            include_rag = st.checkbox(
                "Include RAG Retrieval", 
                value=True, 
                disabled=not st.session_state.rag_system,
                help="Use RAG system for enhanced responses"
            )
        
        with col2:
            batch_size = st.selectbox("Evaluation Batch Size", [1, 5, 10], index=1)
        
        # Run evaluation
        if test_data and st.button("Run Evaluation", type="primary"):
            with st.spinner("Running evaluation..."):
                results = self.run_evaluation(test_data, selected_model)
                st.session_state.evaluation_results = results
            
            st.success("Evaluation completed!")
        
        # Display results
        if 'evaluation_results' in st.session_state:
            results = st.session_state.evaluation_results
            
            st.markdown("---")
            st.header("Evaluation Results")
            
            # Visualize metrics
            self.visualize_metrics(results)
            
            # Export results
            st.subheader("Export Results")
            export_data = self.export_results(results)
            
            st.download_button(
                label="Download Evaluation Report",
                data=export_data,
                file_name=f"evaluation_report_{results['model_name']}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
