import streamlit as st
import json
import hashlib
from datetime import datetime
from typing import Dict, List, Optional, Any
from database.models import (
    UserSession, LearnedData, ModelTraining, WebCrawlLog, QualityMetrics,
    get_session, create_tables
)
import logging

logger = logging.getLogger(__name__)

class DatabaseManager:
    def __init__(self):
        try:
            create_tables()
            self.session = get_session()
            logger.info("Database connection established successfully")
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            self.session = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            self.session.close()
    
    def get_or_create_user_session(self, session_id: str) -> UserSession:
        """Get or create user session for data persistence"""
        if not self.session:
            return None
            
        user_session = self.session.query(UserSession).filter_by(session_id=session_id).first()
        
        if not user_session:
            user_session = UserSession(
                session_id=session_id,
                user_preferences={},
                chat_history=[]
            )
            self.session.add(user_session)
            self.session.commit()
            logger.info(f"Created new user session: {session_id}")
        
        return user_session
    
    def save_user_preferences(self, session_id: str, preferences: Dict[str, Any]):
        """Save user preferences for persistence between visits"""
        if not self.session:
            return False
            
        try:
            user_session = self.get_or_create_user_session(session_id)
            user_session.user_preferences = preferences
            user_session.last_active = datetime.utcnow()
            self.session.commit()
            return True
        except Exception as e:
            logger.error(f"Failed to save user preferences: {e}")
            return False
    
    def get_user_preferences(self, session_id: str) -> Dict[str, Any]:
        """Get user preferences for persistence"""
        if not self.session:
            return {}
            
        try:
            user_session = self.get_or_create_user_session(session_id)
            return user_session.user_preferences or {}
        except Exception as e:
            logger.error(f"Failed to get user preferences: {e}")
            return {}
    
    def save_chat_history(self, session_id: str, chat_history: List[Dict]):
        """Save chat history for persistence"""
        if not self.session:
            return False
            
        try:
            user_session = self.get_or_create_user_session(session_id)
            user_session.chat_history = chat_history
            user_session.last_active = datetime.utcnow()
            self.session.commit()
            return True
        except Exception as e:
            logger.error(f"Failed to save chat history: {e}")
            return False
    
    def get_chat_history(self, session_id: str) -> List[Dict]:
        """Get chat history for persistence"""
        if not self.session:
            return []
            
        try:
            user_session = self.get_or_create_user_session(session_id)
            return user_session.chat_history or []
        except Exception as e:
            logger.error(f"Failed to get chat history: {e}")
            return []
    
    def store_learned_data(self, content: str, source_url: str = None, 
                          category: str = None, embedding: List[float] = None) -> bool:
        """Store auto-learned data from web sources"""
        if not self.session:
            return False
            
        try:
            content_hash = hashlib.md5(content.encode()).hexdigest()
            
            # Check if content already exists
            existing = self.session.query(LearnedData).filter_by(
                content=content[:500]  # Check first 500 chars
            ).first()
            
            if existing:
                logger.info("Content already exists in database")
                return True
            
            learned_data = LearnedData(
                content=content,
                source_url=source_url,
                category=category,
                embedding=embedding,
                quality_score=self._calculate_quality_score(content),
                is_validated=True  # Auto-validate for now
            )
            
            self.session.add(learned_data)
            self.session.commit()
            logger.info(f"Stored learned data from {source_url}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store learned data: {e}")
            return False
    
    def _calculate_quality_score(self, content: str) -> float:
        """Calculate quality score for learned content"""
        # Simple quality scoring based on content characteristics
        score = 0.5  # Base score
        
        # Length factor
        if len(content) > 100:
            score += 0.2
        if len(content) > 500:
            score += 0.1
            
        # Contains multiple sentences
        if content.count('.') > 2:
            score += 0.1
            
        # Contains structured information
        if any(char in content for char in [':', '-', '*']):
            score += 0.1
            
        return min(score, 1.0)
    
    def get_learned_data(self, category: str = None, limit: int = 100) -> List[LearnedData]:
        """Retrieve learned data for training"""
        if not self.session:
            return []
            
        try:
            query = self.session.query(LearnedData)
            
            if category:
                query = query.filter_by(category=category)
            
            query = query.filter_by(is_validated=True)
            query = query.order_by(LearnedData.quality_score.desc())
            query = query.limit(limit)
            
            return query.all()
            
        except Exception as e:
            logger.error(f"Failed to get learned data: {e}")
            return []
    
    def save_model_training(self, model_name: str, model_path: str, 
                           metrics: Dict[str, Any], accuracy: float) -> bool:
        """Save model training results"""
        if not self.session:
            return False
            
        try:
            training_record = ModelTraining(
                model_name=model_name,
                model_path=model_path,
                performance_metrics=metrics,
                training_status='completed',
                completed_at=datetime.utcnow(),
                accuracy_score=accuracy
            )
            
            self.session.add(training_record)
            self.session.commit()
            logger.info(f"Saved model training: {model_name} with accuracy: {accuracy}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save model training: {e}")
            return False
    
    def get_best_model(self) -> Optional[ModelTraining]:
        """Get the best trained model based on accuracy"""
        if not self.session:
            return None
            
        try:
            return self.session.query(ModelTraining)\
                .filter_by(training_status='completed')\
                .order_by(ModelTraining.accuracy_score.desc())\
                .first()
        except Exception as e:
            logger.error(f"Failed to get best model: {e}")
            return None
    
    def log_web_crawl(self, url: str, status: str, content_length: int = 0, 
                     error_message: str = None) -> bool:
        """Log web crawling activities"""
        if not self.session:
            return False
            
        try:
            crawl_log = WebCrawlLog(
                url=url,
                status=status,
                content_length=content_length,
                error_message=error_message,
                content_hash=hashlib.md5(url.encode()).hexdigest()
            )
            
            self.session.add(crawl_log)
            self.session.commit()
            return True
            
        except Exception as e:
            logger.error(f"Failed to log web crawl: {e}")
            return False
    
    def record_quality_metric(self, metric_type: str, value: float, 
                            target_value: float, details: Dict = None) -> bool:
        """Record quality metrics for zero-defect validation"""
        if not self.session:
            return False
            
        try:
            is_passing = value >= target_value
            
            metric = QualityMetrics(
                metric_type=metric_type,
                value=value,
                target_value=target_value,
                is_passing=is_passing,
                details=details or {}
            )
            
            self.session.add(metric)
            self.session.commit()
            
            if not is_passing:
                logger.warning(f"Quality metric {metric_type} failed: {value} < {target_value}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to record quality metric: {e}")
            return False
    
    def get_quality_metrics(self, metric_type: str = None) -> List[QualityMetrics]:
        """Get quality metrics for monitoring"""
        if not self.session:
            return []
            
        try:
            query = self.session.query(QualityMetrics)
            
            if metric_type:
                query = query.filter_by(metric_type=metric_type)
            
            return query.order_by(QualityMetrics.measured_at.desc()).limit(100).all()
            
        except Exception as e:
            logger.error(f"Failed to get quality metrics: {e}")
            return []