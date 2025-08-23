import streamlit as st
import threading
import time
import logging
from typing import List, Dict
from components.database_manager import DatabaseManager
from components.rag_system import RAGSystem
import hashlib
import json

logger = logging.getLogger(__name__)

class AutoRAGUpdater:
    """Automatically updates RAG system with new learned data from database"""
    
    def __init__(self, rag_system: RAGSystem):
        self.rag_system = rag_system
        self.is_active = False
        self.last_update_hash = None
        self.update_interval = 60  # Check every 60 seconds
        self.min_quality_threshold = 0.6  # Only include high-quality content
        
    def get_content_hash(self, learned_data_list):
        """Generate hash of current learned data to detect changes"""
        content_ids = sorted([str(data.id) for data in learned_data_list])
        combined = ''.join(content_ids)
        return hashlib.md5(combined.encode()).hexdigest()
    
    def update_rag_with_new_data(self) -> bool:
        """Update RAG system with new high-quality learned data"""
        try:
            with DatabaseManager() as db:
                # Get all validated, high-quality learned data
                learned_data = db.get_learned_data(limit=1000)
                
                if not learned_data:
                    logger.info("No learned data available for RAG update")
                    return False
                
                # Check if data has changed since last update
                current_hash = self.get_content_hash(learned_data)
                if current_hash == self.last_update_hash:
                    logger.debug("No new data to update RAG")
                    return False
                
                # Filter high-quality content
                high_quality_data = [
                    data for data in learned_data 
                    if data.quality_score >= self.min_quality_threshold and data.is_validated
                ]
                
                if not high_quality_data:
                    logger.info("No high-quality data available for RAG update")
                    return False
                
                # Prepare new passages for RAG
                new_passages = []
                for data in high_quality_data:
                    # Create comprehensive passage with metadata
                    passage = {
                        'content': data.content,
                        'source': data.source_url or 'Database',
                        'category': data.category or 'general',
                        'quality_score': data.quality_score,
                        'created_at': str(data.created_at),
                        'id': f"learned_{data.id}"
                    }
                    new_passages.append(passage)
                
                # Update RAG system with new data
                logger.info(f"Updating RAG system with {len(new_passages)} new passages")
                success = self._integrate_passages_to_rag(new_passages)
                
                if success:
                    self.last_update_hash = current_hash
                    
                    # Record quality metric for RAG update
                    db.record_quality_metric(
                        'rag_auto_update_success',
                        1.0,
                        1.0,
                        {
                            'passages_added': len(new_passages),
                            'avg_quality': sum(p['quality_score'] for p in new_passages) / len(new_passages),
                            'categories': list(set(p['category'] for p in new_passages))
                        }
                    )
                    
                    logger.info(f"Successfully updated RAG with {len(new_passages)} passages")
                    return True
                else:
                    # Record failure metric
                    db.record_quality_metric(
                        'rag_auto_update_success',
                        0.0,
                        1.0,
                        {'error': 'Failed to integrate passages'}
                    )
                    return False
                    
        except Exception as e:
            logger.error(f"Error updating RAG with new data: {e}")
            return False
    
    def _integrate_passages_to_rag(self, passages: List[Dict]) -> bool:
        """Integrate new passages into the RAG system"""
        try:
            # Add passages to RAG system
            for passage in passages:
                # Format passage text for RAG
                passage_text = f"[{passage['category'].upper()}] {passage['content']}"
                
                # Add to RAG system passages
                if hasattr(self.rag_system, 'passages'):
                    # Check if passage already exists
                    existing_ids = [p.get('id', '') for p in self.rag_system.passages if isinstance(p, dict)]
                    if passage['id'] not in existing_ids:
                        self.rag_system.passages.append(passage)
                
                # Add to text corpus for embedding
                if hasattr(self.rag_system, 'corpus'):
                    self.rag_system.corpus.append(passage_text)
            
            # Rebuild embeddings and index if RAG system supports it
            if hasattr(self.rag_system, 'build_index'):
                self.rag_system.build_index()
                logger.info("Rebuilt RAG index with new data")
            
            # Save updated index
            if hasattr(self.rag_system, 'save_index'):
                self.rag_system.save_index()
                logger.info("Saved updated RAG index")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to integrate passages to RAG: {e}")
            return False
    
    def start_auto_update(self):
        """Start automatic RAG updating in background thread"""
        if self.is_active:
            logger.info("Auto RAG update already active")
            return
        
        self.is_active = True
        
        def update_worker():
            """Background worker for RAG updates"""
            logger.info("Started auto RAG update worker")
            
            while self.is_active:
                try:
                    # Perform update
                    self.update_rag_with_new_data()
                    
                    # Wait for next update cycle
                    time.sleep(self.update_interval)
                    
                except Exception as e:
                    logger.error(f"Error in RAG update worker: {e}")
                    time.sleep(30)  # Shorter wait on error
            
            logger.info("Auto RAG update worker stopped")
        
        # Start worker thread
        update_thread = threading.Thread(target=update_worker, daemon=True)
        update_thread.start()
        logger.info("Auto RAG update started")
    
    def stop_auto_update(self):
        """Stop automatic RAG updating"""
        self.is_active = False
        logger.info("Auto RAG update stopped")
    
    def force_update(self) -> bool:
        """Force immediate RAG update"""
        logger.info("Forcing immediate RAG update")
        return self.update_rag_with_new_data()
    
    def get_update_status(self) -> Dict:
        """Get current update status and statistics"""
        try:
            with DatabaseManager() as db:
                learned_data = db.get_learned_data(limit=1000)
                high_quality_count = len([
                    d for d in learned_data 
                    if d.quality_score >= self.min_quality_threshold and d.is_validated
                ])
                
                return {
                    'is_active': self.is_active,
                    'total_learned_data': len(learned_data),
                    'high_quality_data': high_quality_count,
                    'last_update_hash': self.last_update_hash,
                    'update_interval': self.update_interval,
                    'quality_threshold': self.min_quality_threshold,
                    'rag_passages_count': len(self.rag_system.passages) if hasattr(self.rag_system, 'passages') else 0
                }
        except Exception as e:
            logger.error(f"Error getting update status: {e}")
            return {
                'is_active': self.is_active,
                'error': str(e)
            }
    
    def render_update_interface(self):
        """Streamlit interface for RAG auto-update management"""
        st.subheader("🔄 Auto RAG Update System")
        
        status = self.get_update_status()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Status")
            status_icon = "🟢" if status['is_active'] else "🔴"
            st.markdown(f"**Auto Update:** {status_icon} {'Active' if status['is_active'] else 'Inactive'}")
            
            if 'total_learned_data' in status:
                st.metric("Total Learned Data", status['total_learned_data'])
                st.metric("High Quality Data", status['high_quality_data'])
                st.metric("RAG Passages", status['rag_passages_count'])
        
        with col2:
            st.markdown("### Controls")
            
            if not status['is_active']:
                if st.button("🚀 Start Auto Update"):
                    self.start_auto_update()
                    st.success("Auto RAG update started!")
                    st.rerun()
            else:
                if st.button("⏹️ Stop Auto Update"):
                    self.stop_auto_update()
                    st.success("Auto RAG update stopped!")
                    st.rerun()
            
            if st.button("⚡ Force Update Now"):
                with st.spinner("Updating RAG with latest data..."):
                    success = self.force_update()
                if success:
                    st.success("RAG updated successfully!")
                else:
                    st.warning("No new data to update or update failed")
        
        # Configuration
        st.markdown("---")
        st.markdown("### Configuration")
        
        col3, col4 = st.columns(2)
        
        with col3:
            new_interval = st.slider(
                "Update Interval (seconds)",
                min_value=30,
                max_value=600,
                value=self.update_interval,
                step=30
            )
            if new_interval != self.update_interval:
                self.update_interval = new_interval
                st.success(f"Update interval set to {new_interval} seconds")
        
        with col4:
            new_threshold = st.slider(
                "Quality Threshold",
                min_value=0.0,
                max_value=1.0,
                value=self.min_quality_threshold,
                step=0.1
            )
            if new_threshold != self.min_quality_threshold:
                self.min_quality_threshold = new_threshold
                st.success(f"Quality threshold set to {new_threshold}")
        
        # Show recent updates
        if 'error' not in status:
            st.markdown("---")
            st.markdown("### Recent Updates")
            
            try:
                with DatabaseManager() as db:
                    recent_metrics = db.get_quality_metrics('rag_auto_update_success')
                    
                    if recent_metrics:
                        for metric in recent_metrics[:5]:
                            status_emoji = "✅" if metric.is_passing else "❌"
                            st.text(f"{status_emoji} {metric.measured_at}: {metric.details}")
                    else:
                        st.info("No recent updates recorded")
            except Exception as e:
                st.error(f"Failed to load recent updates: {e}")