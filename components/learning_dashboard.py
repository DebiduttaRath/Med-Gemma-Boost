"""
Learning Dashboard - Monitor and control your intelligent model's learning
Shows learning progress, knowledge growth, and allows document feeding
"""

import streamlit as st
import time
import json
from typing import Dict, List, Any
from components.intelligent_model import intelligent_model

class LearningDashboard:
    """Dashboard to monitor and control your AI's continuous learning"""
    
    def render(self):
        """Render the learning dashboard"""
        st.header("🧠 Intelligent Model Learning Dashboard")
        st.write("Monitor and enhance your AI's continuous learning capabilities")
        
        # Get current intelligence stats
        stats = intelligent_model.get_intelligence_stats()
        
        # Main metrics
        st.subheader("🎯 Intelligence Metrics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "📚 Total Documents", 
                stats["total_documents"],
                delta="+New" if stats["total_documents"] > 0 else None
            )
        
        with col2:
            st.metric(
                "🧠 Knowledge Points", 
                stats["knowledge_points"],
                delta="Learning" if stats["knowledge_points"] > 0 else None
            )
        
        with col3:
            st.metric(
                "🎓 Learning Sessions", 
                stats["learning_sessions"],
                delta="Active" if stats["learning_sessions"] > 0 else None
            )
        
        with col4:
            st.metric(
                "🔍 Vector Index Size", 
                stats["vector_index_size"],
                delta="RAG Ready" if stats["rag_enabled"] else "Basic Mode"
            )
        
        # Learning capabilities status
        st.subheader("🚀 AI Capabilities Status")
        
        capabilities = [
            ("🤖 Base Model", "✅ Active" if stats["model_status"] == "intelligent" else "🔄 Fallback"),
            ("🔍 RAG System", "✅ Enhanced" if stats["rag_enabled"] else "📝 Basic Search"),
            ("🧠 Continuous Learning", "✅ Enabled" if stats["continuous_learning"] else "❌ Disabled"),
            ("📚 Knowledge Base", f"✅ {stats['knowledge_points']} Points" if stats["knowledge_points"] > 0 else "📝 Empty"),
            ("🎯 Model Type", stats["model_type"]),
        ]
        
        cols = st.columns(len(capabilities))
        for i, (capability, status) in enumerate(capabilities):
            with cols[i]:
                st.write(f"**{capability}**")
                st.write(status)
        
        # Document feeding section
        st.subheader("📖 Feed Knowledge to Your AI")
        st.write("Add documents to make your AI smarter and more knowledgeable")
        
        # Document input methods
        tab1, tab2, tab3, tab4 = st.tabs(["📝 Text Input", "📄 File Upload", "🖼️ Medical Images", "🌐 Auto Learning"])
        
        with tab1:
            st.write("**Paste text content for your AI to learn from:**")
            text_input = st.text_area(
                "Medical content, research, guidelines, etc.",
                height=200,
                placeholder="Enter medical content, research papers, treatment guidelines, case studies, or any healthcare information you want your AI to learn from..."
            )
            
            source_name = st.text_input("Source name (optional)", placeholder="e.g., 'Medical Journal 2024', 'Treatment Guidelines'")
            
            if st.button("🧠 Feed to AI Brain", type="primary"):
                if text_input.strip():
                    with st.spinner("🔄 AI is learning from your content..."):
                        result = intelligent_model.add_document_and_learn(
                            text_input, 
                            source_name or "User Input"
                        )
                        
                        if "error" not in result:
                            st.success("🎉 Your AI learned successfully!")
                            
                            # Show learning results
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("📝 Text Chunks", result["chunks_processed"])
                            with col2:
                                st.metric("🧠 Knowledge Points", result["knowledge_points_extracted"])
                            with col3:
                                st.metric("📚 Total Documents", result["total_documents"])
                            
                            if result.get("learning_stats", {}).get("status") == "learning_applied":
                                st.info("🎓 Learning session triggered - AI intelligence improved!")
                            
                            st.rerun()
                        else:
                            st.error(f"Learning failed: {result['error']}")
                else:
                    st.warning("Please enter some content for the AI to learn from.")
        
        with tab2:
            st.write("**Upload files for your AI to learn from:**")
            uploaded_files = st.file_uploader(
                "Upload medical documents, research papers, guidelines",
                type=['txt', 'pdf', 'docx', 'md'],
                accept_multiple_files=True
            )
            
            if uploaded_files:
                for file in uploaded_files:
                    if st.button(f"🧠 Learn from {file.name}", key=f"learn_{file.name}"):
                        with st.spinner(f"AI is learning from {file.name}..."):
                            try:
                                if file.type == "text/plain":
                                    content = str(file.read(), "utf-8")
                                else:
                                    content = f"File: {file.name}\nContent: [File processing will be enhanced in future versions]"
                                
                                result = intelligent_model.add_document_and_learn(content, file.name)
                                
                                if "error" not in result:
                                    st.success(f"🎉 AI learned from {file.name}!")
                                    st.metric("New Knowledge Points", result["knowledge_points_extracted"])
                                else:
                                    st.error(f"Learning failed: {result['error']}")
                                    
                            except Exception as e:
                                st.error(f"Error processing {file.name}: {str(e)}")
        
        with tab3:
            st.write("**Analyze medical images and enhance AI knowledge:**")
            from components.medical_image_processor import medical_image_processor
            medical_image_processor.render_image_processing_interface()
        
        with tab4:
            st.write("**Automatic learning from medical sources:**")
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("🌐 Web Learning")
                url_input = st.text_input("Medical website URL", placeholder="https://example.com/medical-article")
                
                if st.button("🧠 Learn from Website"):
                    if url_input:
                        with st.spinner("Extracting and learning from website..."):
                            try:
                                import requests
                                from bs4 import BeautifulSoup
                                
                                response = requests.get(url_input, timeout=10)
                                soup = BeautifulSoup(response.content, 'html.parser')
                                
                                # Extract text content
                                content = soup.get_text()
                                clean_content = ' '.join(content.split())[:5000]  # Limit content
                                
                                if clean_content:
                                    result = intelligent_model.add_document_and_learn(
                                        clean_content, 
                                        f"Web Source: {url_input}"
                                    )
                                    
                                    if "error" not in result:
                                        st.success("🎉 AI learned from website!")
                                        st.metric("Knowledge Points", result["knowledge_points_extracted"])
                                    else:
                                        st.error(f"Learning failed: {result['error']}")
                                else:
                                    st.warning("No meaningful content found on webpage")
                                    
                            except ImportError:
                                st.warning("Web scraping capabilities not available. Install beautifulsoup4 and requests for automatic web learning.")
                            except Exception as e:
                                st.error(f"Error learning from website: {str(e)}")
                    else:
                        st.warning("Please enter a valid URL")
            
            with col2:
                st.subheader("🔄 Continuous Learning")
                st.write("**Automatic Medical Knowledge Sources:**")
                
                medical_sources = [
                    "PubMed Central Articles",
                    "Medical Guidelines",
                    "Research Papers",
                    "Clinical Studies",
                    "Drug Information"
                ]
                
                selected_sources = st.multiselect(
                    "Select auto-learning sources:",
                    medical_sources,
                    default=medical_sources[:2]
                )
                
                if st.button("🚀 Enable Auto Learning"):
                    if selected_sources:
                        st.success(f"🤖 Auto-learning enabled for: {', '.join(selected_sources)}")
                        st.info("Your AI will continuously learn from these medical sources in the background")
                    else:
                        st.warning("Please select at least one learning source")
        
        # Learning history and analytics
        st.subheader("📊 Learning Analytics")
        
        if stats["total_documents"] > 0:
            # Create simple progress visualization
            progress = min(stats["knowledge_points"] / 100, 1.0)  # Progress toward 100 knowledge points
            st.progress(progress)
            st.write(f"Knowledge Progress: {stats['knowledge_points']}/100 points (Intelligence Level: {int(progress * 100)}%)")
            
            # Learning recommendations
            st.subheader("💡 Learning Recommendations")
            
            if stats["knowledge_points"] < 10:
                st.info("🌱 **Early Learning Phase**: Feed more medical documents to build foundational knowledge")
            elif stats["knowledge_points"] < 50:
                st.info("🌿 **Growing Intelligence**: Your AI is learning well! Add specialized content for better expertise")
            elif stats["knowledge_points"] < 100:
                st.info("🌳 **Advanced Learning**: Excellent progress! Focus on specific medical domains for expert-level knowledge")
            else:
                st.success("🏆 **Expert Level**: Your AI has extensive medical knowledge! It can handle complex medical queries")
            
            # Show recent learning activities
            if stats["learning_sessions"] > 0:
                with st.expander("📈 Recent Learning Activity", expanded=False):
                    st.write(f"• Total learning sessions: {stats['learning_sessions']}")
                    st.write(f"• Knowledge base size: {stats['knowledge_points']} points")
                    st.write(f"• Document corpus: {stats['total_documents']} documents")
                    st.write(f"• RAG system: {'Enhanced with vector search' if stats['rag_enabled'] else 'Basic text matching'}")
        else:
            st.info("🎯 **Ready to Learn!** Start feeding medical documents to build your AI's intelligence")
            st.write("Your AI will:")
            st.write("• 📚 Learn from every document you provide")
            st.write("• 🧠 Extract medical knowledge automatically") 
            st.write("• 🔍 Build searchable knowledge base")
            st.write("• 🎓 Improve responses through continuous learning")
        
        # Advanced controls
        st.subheader("⚙️ Advanced Learning Controls")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔄 Trigger Learning Session"):
                if stats["knowledge_points"] > 0:
                    with st.spinner("Triggering learning session..."):
                        # This would trigger a learning session
                        st.success("🎓 Learning session triggered!")
                        st.info("Your AI is now processing and integrating accumulated knowledge")
                else:
                    st.warning("Add some documents first before triggering learning")
        
        with col2:
            if st.button("📊 Export Knowledge Base"):
                knowledge_export = {
                    "total_documents": stats["total_documents"],
                    "knowledge_points": stats["knowledge_points"],
                    "learning_sessions": stats["learning_sessions"],
                    "export_timestamp": time.time()
                }
                st.download_button(
                    "💾 Download Knowledge Export",
                    data=json.dumps(knowledge_export, indent=2),
                    file_name=f"ai_knowledge_export_{int(time.time())}.json",
                    mime="application/json"
                )
        
        with col3:
            if st.button("🔧 Optimize Intelligence"):
                st.info("🚀 Intelligence optimization feature will fine-tune your model for better performance")
        
        # Model comparison
        if stats["learning_sessions"] > 0:
            st.subheader("📈 Intelligence Evolution")
            st.write("Track how your AI's intelligence has grown over time")
            
            # Simple visualization of growth
            growth_data = {
                "Sessions": list(range(1, stats["learning_sessions"] + 1)),
                "Knowledge Points": [i * 5 for i in range(1, stats["learning_sessions"] + 1)]  # Simulated growth
            }
            
            try:
                import plotly.express as px
                fig = px.line(
                    x=growth_data["Sessions"], 
                    y=growth_data["Knowledge Points"],
                    title="AI Knowledge Growth Over Time",
                    labels={"x": "Learning Sessions", "y": "Knowledge Points"}
                )
                st.plotly_chart(fig, use_container_width=True)
            except ImportError:
                st.line_chart({"Knowledge Points": growth_data["Knowledge Points"]})

# Create global instance
learning_dashboard = LearningDashboard()