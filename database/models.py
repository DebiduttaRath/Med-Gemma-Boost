from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, Boolean, Float, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.sql import func
import os

Base = declarative_base()

class UserSession(Base):
    __tablename__ = 'user_sessions'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(255), unique=True, nullable=False)
    user_preferences = Column(JSON, default=dict)
    created_at = Column(DateTime, server_default=func.now())
    last_active = Column(DateTime, server_default=func.now())
    chat_history = Column(JSON, default=list)

class LearnedData(Base):
    __tablename__ = 'learned_data'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    source_url = Column(String(500))
    content = Column(Text, nullable=False)
    summary = Column(Text)
    embedding = Column(JSON)  # Store embeddings as JSON
    category = Column(String(100))
    quality_score = Column(Float, default=0.0)
    created_at = Column(DateTime, server_default=func.now())
    last_updated = Column(DateTime, server_default=func.now())
    is_validated = Column(Boolean, default=False)

class ModelTraining(Base):
    __tablename__ = 'model_training'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    model_name = Column(String(255), nullable=False)
    training_data_hash = Column(String(255))
    model_path = Column(String(500))
    performance_metrics = Column(JSON)
    training_status = Column(String(50), default='pending')
    created_at = Column(DateTime, server_default=func.now())
    completed_at = Column(DateTime)
    accuracy_score = Column(Float)

class WebCrawlLog(Base):
    __tablename__ = 'web_crawl_logs'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    url = Column(String(500), nullable=False)
    status = Column(String(50))
    content_length = Column(Integer)
    crawled_at = Column(DateTime, server_default=func.now())
    error_message = Column(Text)
    content_hash = Column(String(255))

class QualityMetrics(Base):
    __tablename__ = 'quality_metrics'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    metric_type = Column(String(100))
    value = Column(Float)
    target_value = Column(Float)
    measured_at = Column(DateTime, server_default=func.now())
    is_passing = Column(Boolean)
    details = Column(JSON)

# Database connection and session management
def get_engine():
    database_url = os.getenv('DATABASE_URL')
    if not database_url:
        raise ValueError("DATABASE_URL environment variable not set")
    return create_engine(database_url)

def get_session():
    engine = get_engine()
    Session = sessionmaker(bind=engine)
    return Session()

def create_tables():
    engine = get_engine()
    Base.metadata.create_all(engine)

def drop_tables():
    engine = get_engine()
    Base.metadata.drop_all(engine)