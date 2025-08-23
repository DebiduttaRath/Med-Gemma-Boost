import streamlit as st
import requests
from bs4 import BeautifulSoup
import trafilatura
import logging
from typing import List, Dict, Optional
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from components.database_manager import DatabaseManager
import hashlib
import re

logger = logging.getLogger(__name__)

class AutoWebLearner:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        self.learning_active = False
        self.processed_urls = set()
        
    def extract_content(self, url: str) -> Optional[Dict[str, str]]:
        """Extract clean content from a web page"""
        try:
            with DatabaseManager() as db:
                db.log_web_crawl(url, 'attempting', 0)
            
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            # Use trafilatura for clean content extraction
            content = trafilatura.extract(response.text, include_links=False, include_images=False)
            
            if not content or len(content.strip()) < 100:
                logger.warning(f"Insufficient content from {url}")
                return None
            
            # Extract title using BeautifulSoup as fallback
            soup = BeautifulSoup(response.text, 'html.parser')
            title = soup.find('title')
            title_text = title.get_text().strip() if title else ""
            
            # Clean and validate content
            clean_content = self._clean_content(content)
            
            if self._is_quality_content(clean_content):
                with DatabaseManager() as db:
                    db.log_web_crawl(url, 'success', len(clean_content))
                
                return {
                    'url': url,
                    'title': title_text,
                    'content': clean_content,
                    'category': self._categorize_content(clean_content)
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to extract content from {url}: {e}")
            with DatabaseManager() as db:
                db.log_web_crawl(url, 'failed', 0, str(e))
            return None
    
    def _clean_content(self, content: str) -> str:
        """Clean and normalize extracted content"""
        # Remove extra whitespace
        content = re.sub(r'\s+', ' ', content)
        
        # Remove special characters that might cause issues
        content = re.sub(r'[^\w\s\.\,\!\?\:\;\-\(\)]', '', content)
        
        # Limit content length for storage efficiency
        if len(content) > 5000:
            # Try to cut at sentence boundary
            sentences = content[:5000].split('.')
            content = '.'.join(sentences[:-1]) + '.'
        
        return content.strip()
    
    def _is_quality_content(self, content: str) -> bool:
        """Determine if content meets quality standards for learning"""
        if len(content) < 200:
            return False
        
        # Check for minimum information density
        word_count = len(content.split())
        if word_count < 50:
            return False
        
        # Check for structured content indicators
        has_structure = any([
            '.' in content,  # Sentences
            ':' in content,  # Lists or definitions
            content.count('\n') > 2,  # Multiple paragraphs
        ])
        
        # Avoid purely promotional or low-value content
        promotional_indicators = ['buy now', 'click here', 'subscribe', 'advertisement']
        if sum(indicator in content.lower() for indicator in promotional_indicators) > 2:
            return False
        
        return has_structure
    
    def _categorize_content(self, content: str) -> str:
        """Automatically categorize content for better organization"""
        content_lower = content.lower()
        
        # Medical/Health content
        medical_keywords = ['health', 'medical', 'disease', 'treatment', 'diagnosis', 'patient', 'doctor', 'medicine']
        if sum(keyword in content_lower for keyword in medical_keywords) >= 2:
            return 'medical'
        
        # Technology content
        tech_keywords = ['technology', 'software', 'programming', 'computer', 'algorithm', 'data', 'artificial intelligence']
        if sum(keyword in content_lower for keyword in tech_keywords) >= 2:
            return 'technology'
        
        # Science content
        science_keywords = ['research', 'study', 'experiment', 'scientific', 'analysis', 'hypothesis']
        if sum(keyword in content_lower for keyword in science_keywords) >= 2:
            return 'science'
        
        # Education content
        education_keywords = ['learn', 'education', 'tutorial', 'guide', 'instruction', 'course']
        if sum(keyword in content_lower for keyword in education_keywords) >= 2:
            return 'education'
        
        return 'general'
    
    def learn_from_urls(self, urls: List[str], max_workers: int = 3) -> Dict[str, int]:
        """Learn from multiple URLs concurrently"""
        results = {'success': 0, 'failed': 0, 'skipped': 0}
        
        # Filter out already processed URLs
        new_urls = [url for url in urls if url not in self.processed_urls]
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self.extract_content, url): url for url in new_urls}
            
            for future in futures:
                url = futures[future]
                try:
                    content_data = future.result()
                    
                    if content_data:
                        # Store in database
                        with DatabaseManager() as db:
                            success = db.store_learned_data(
                                content=content_data['content'],
                                source_url=content_data['url'],
                                category=content_data['category']
                            )
                        
                        if success:
                            results['success'] += 1
                            self.processed_urls.add(url)
                            logger.info(f"Successfully learned from: {url}")
                        else:
                            results['failed'] += 1
                    else:
                        results['skipped'] += 1
                        
                except Exception as e:
                    logger.error(f"Error processing {url}: {e}")
                    results['failed'] += 1
                
                # Small delay to be respectful to servers
                time.sleep(0.5)
        
        return results
    
    def auto_discover_urls(self, seed_urls: List[str], max_depth: int = 2) -> List[str]:
        """Auto-discover related URLs for learning"""
        discovered_urls = set()
        urls_to_process = set(seed_urls)
        processed = set()
        
        for depth in range(max_depth):
            current_level_urls = urls_to_process - processed
            if not current_level_urls:
                break
            
            for url in current_level_urls:
                try:
                    response = self.session.get(url, timeout=5)
                    soup = BeautifulSoup(response.text, 'html.parser')
                    
                    # Find relevant links
                    links = soup.find_all('a', href=True)
                    for link in links[:10]:  # Limit to prevent explosion
                        href = link['href']
                        if href.startswith('http') and self._is_relevant_url(href):
                            discovered_urls.add(href)
                    
                    processed.add(url)
                    
                except Exception as e:
                    logger.warning(f"Failed to discover URLs from {url}: {e}")
                    processed.add(url)
            
            urls_to_process.update(discovered_urls)
            
            if len(discovered_urls) > 100:  # Prevent too many URLs
                break
        
        return list(discovered_urls)[:50]  # Return top 50
    
    def _is_relevant_url(self, url: str) -> bool:
        """Check if URL is relevant for learning"""
        # Skip common irrelevant patterns
        skip_patterns = [
            'login', 'register', 'cart', 'checkout', 'privacy', 'terms',
            'contact', 'about', 'javascript:', 'mailto:', '#'
        ]
        
        return not any(pattern in url.lower() for pattern in skip_patterns)
    
    def start_continuous_learning(self, seed_urls: List[str]):
        """Start continuous learning process in background"""
        if self.learning_active:
            return
        
        self.learning_active = True
        
        def learning_worker():
            while self.learning_active:
                try:
                    # Discover new URLs
                    discovered_urls = self.auto_discover_urls(seed_urls, max_depth=1)
                    
                    # Learn from discovered URLs
                    if discovered_urls:
                        results = self.learn_from_urls(discovered_urls)
                        
                        # Record quality metrics
                        with DatabaseManager() as db:
                            total_processed = sum(results.values())
                            if total_processed > 0:
                                success_rate = results['success'] / total_processed
                                db.record_quality_metric(
                                    'learning_success_rate',
                                    success_rate,
                                    0.7,  # Target 70% success rate
                                    {'results': results}
                                )
                    
                    # Sleep before next learning cycle
                    time.sleep(300)  # 5 minutes
                    
                except Exception as e:
                    logger.error(f"Error in continuous learning: {e}")
                    time.sleep(60)  # Wait 1 minute on error
        
        # Start learning in background thread
        learning_thread = threading.Thread(target=learning_worker, daemon=True)
        learning_thread.start()
        logger.info("Started continuous learning process")
    
    def stop_continuous_learning(self):
        """Stop continuous learning process"""
        self.learning_active = False
        logger.info("Stopped continuous learning process")
    
    def render_learning_interface(self):
        """Streamlit interface for web learning management"""
        st.header("🌐 Auto Web Learning System")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Learning Status")
            status = "🟢 Active" if self.learning_active else "🔴 Inactive"
            st.markdown(f"**Status:** {status}")
            
            # Show learning statistics
            with DatabaseManager() as db:
                learned_data = db.get_learned_data(limit=1000)
                st.metric("Total Learned Articles", len(learned_data))
                
                # Show category breakdown
                categories = {}
                for data in learned_data:
                    cat = data.category or 'general'
                    categories[cat] = categories.get(cat, 0) + 1
                
                if categories:
                    st.subheader("Content Categories")
                    for category, count in categories.items():
                        st.text(f"{category.title()}: {count}")
        
        with col2:
            st.subheader("Learning Controls")
            
            # Manual URL input
            manual_urls = st.text_area(
                "Add URLs to learn from (one per line):",
                height=100,
                placeholder="https://example.com/article1\nhttps://example.com/article2"
            )
            
            if st.button("Learn from URLs"):
                if manual_urls.strip():
                    urls = [url.strip() for url in manual_urls.split('\n') if url.strip()]
                    with st.spinner("Learning from URLs..."):
                        results = self.learn_from_urls(urls)
                    
                    st.success(f"Learning completed: {results['success']} success, {results['failed']} failed, {results['skipped']} skipped")
            
            # Continuous learning controls
            if not self.learning_active:
                if st.button("Start Continuous Learning"):
                    default_seeds = [
                        "https://en.wikipedia.org/wiki/Artificial_intelligence",
                        "https://arxiv.org/list/cs.AI/recent",
                        "https://www.nature.com/subjects/machine-learning"
                    ]
                    self.start_continuous_learning(default_seeds)
                    st.success("Started continuous learning!")
                    st.rerun()
            else:
                if st.button("Stop Continuous Learning"):
                    self.stop_continuous_learning()
                    st.success("Stopped continuous learning!")
                    st.rerun()
        
        # Show recent learning activities
        st.subheader("Recent Learning Activities")
        with DatabaseManager() as db:
            recent_data = db.get_learned_data(limit=10)
            
            if recent_data:
                for data in recent_data:
                    with st.expander(f"{data.category.title()} - Quality: {data.quality_score:.1f}"):
                        st.text(f"Source: {data.source_url}")
                        st.text(f"Length: {len(data.content)} characters")
                        st.text(f"Created: {data.created_at}")
                        if len(data.content) > 200:
                            st.text(data.content[:200] + "...")
                        else:
                            st.text(data.content)
            else:
                st.info("No learned data yet. Add some URLs to start learning!")