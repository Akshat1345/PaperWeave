# modules/utils.py - Enhanced Utility Functions
import os
import re
import hashlib
import logging
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler
from typing import List, Dict, Optional
from config import config

# ========== LOGGING SETUP ==========

def setup_logging() -> logging.Logger:
    """
    Configure comprehensive logging system.
    
    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger('ResearchAssistant')
    logger.setLevel(getattr(logging, config.LOG_LEVEL.upper()))
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    # Detailed formatter
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
    )
    
    # Simple formatter for console
    simple_formatter = logging.Formatter(
        '%(levelname)s - %(message)s'
    )
    
    # Console handler (INFO and above)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(simple_formatter)
    
    # File handler with rotation (DEBUG and above)
    file_handler = RotatingFileHandler(
        config.LOG_FILE,
        maxBytes=config.LOG_MAX_BYTES,
        backupCount=config.LOG_BACKUP_COUNT
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)
    
    # Add handlers
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger

# Global logger instance
logger = setup_logging()

# ========== DIRECTORY MANAGEMENT ==========

def ensure_directories(directories: List[str]):
    """
    Create directories if they don't exist.
    
    Args:
        directories: List of directory paths to create
    """
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.debug(f"Ensured directory exists: {directory}")

def get_organized_pdf_path(topic: str, arxiv_id: str) -> str:
    """
    Generate organized path for PDF storage.
    
    Args:
        topic: Research topic
        arxiv_id: ArXiv paper ID
    
    Returns:
        Full path for PDF storage
    """
    # Create topic slug (safe for filesystem)
    topic_slug = re.sub(r'[^\w\s-]', '', topic).strip().replace(' ', '_')[:50]
    
    # Create date-based folder
    date_folder = datetime.now().strftime('%Y%m%d')
    
    # Construct path
    pdf_dir = os.path.join(
        config.DATA_DIR,
        'pdfs',
        topic_slug,
        date_folder
    )
    
    os.makedirs(pdf_dir, exist_ok=True)
    
    # Sanitize arxiv_id for filename
    safe_id = arxiv_id.replace('/', '_').replace(':', '_')
    
    return os.path.join(pdf_dir, f"{safe_id}.pdf")

# ========== FILE HASHING ==========

def get_file_hash(filepath: str, algorithm: str = 'md5') -> str:
    """
    Calculate hash of file for caching/deduplication.
    
    Args:
        filepath: Path to file
        algorithm: Hash algorithm (md5, sha256, etc.)
    
    Returns:
        Hexadecimal hash string
    """
    hash_func = hashlib.new(algorithm)
    
    try:
        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                hash_func.update(chunk)
        return hash_func.hexdigest()
    except Exception as e:
        logger.error(f"Error hashing file {filepath}: {e}")
        return None

# ========== TEXT PROCESSING ==========

def clean_text(text: str) -> str:
    """
    Clean and normalize text.
    
    Args:
        text: Raw text string
    
    Returns:
        Cleaned text
    """
    if not text:
        return ""
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove page numbers (standalone digits)
    text = re.sub(r'\b\d+\b(?=\s|$)', '', text)
    
    # Remove reference markers [1], [2], etc.
    text = re.sub(r'\[\d+\]', '', text)
    
    # Fix hyphenation
    text = re.sub(r'-\s+', '', text)
    
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

def extract_title_from_text(text: str, max_length: int = 200) -> Optional[str]:
    """
    Extract paper title from text (first substantial line).
    
    Args:
        text: Full text
        max_length: Maximum title length
    
    Returns:
        Extracted title or None
    """
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    
    for line in lines[:10]:  # Check first 10 lines
        if 10 < len(line) < max_length and not line.startswith('http'):
            return line
    
    return None

def split_into_sentences(text: str) -> List[str]:
    """
    Split text into sentences.
    
    Args:
        text: Input text
    
    Returns:
        List of sentences
    """
    # Simple sentence splitter (can be improved with nltk)
    sentences = re.split(r'[.!?]+\s+', text)
    return [s.strip() for s in sentences if s.strip()]

# ========== VALIDATION ==========

def is_valid_arxiv_id(arxiv_id: str) -> bool:
    """
    Validate arXiv ID format.
    
    Formats:
    - Old: cs.AI/0001234
    - New: 2301.12345
    
    Args:
        arxiv_id: ArXiv ID to validate
    
    Returns:
        True if valid format
    """
    old_format = r'^[a-z\-]+(\.[A-Z]{2})?/\d{7}$'
    new_format = r'^\d{4}\.\d{4,5}(v\d+)?$'
    
    return bool(re.match(old_format, arxiv_id) or re.match(new_format, arxiv_id))

def is_valid_pdf(filepath: str) -> bool:
    """
    Validate PDF file.
    
    Args:
        filepath: Path to PDF file
    
    Returns:
        True if valid PDF
    """
    if not os.path.exists(filepath):
        return False
    
    # Check file size
    if os.path.getsize(filepath) < 1000:  # Less than 1KB
        return False
    
    # Check PDF header
    try:
        with open(filepath, 'rb') as f:
            header = f.read(8)
            return header.startswith(b'%PDF')
    except Exception:
        return False

# ========== FORMAT CONVERSION ==========

def format_authors(authors: List[str], max_authors: int = 3) -> str:
    """
    Format author list for display.
    
    Args:
        authors: List of author names
        max_authors: Maximum authors to show before "et al."
    
    Returns:
        Formatted author string
    """
    if not authors:
        return "Unknown Authors"
    
    if len(authors) <= max_authors:
        return ", ".join(authors)
    else:
        return f"{', '.join(authors[:max_authors])} et al."

def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format.
    
    Args:
        size_bytes: Size in bytes
    
    Returns:
        Formatted string (e.g., "2.5 MB")
    """
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"

def format_duration(seconds: float) -> str:
    """
    Format duration in human-readable format.
    
    Args:
        seconds: Duration in seconds
    
    Returns:
        Formatted string (e.g., "2m 30s")
    """
    if seconds < 60:
        return f"{int(seconds)}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"

# ========== ERROR HANDLING ==========

class ScraperError(Exception):
    """Custom exception for scraper errors."""
    pass

class CompilationError(Exception):
    """Custom exception for compilation errors."""
    pass

class DatabaseError(Exception):
    """Custom exception for database errors."""
    pass

def safe_execute(func, default=None, error_msg: str = "Error executing function"):
    """
    Execute function with error handling.
    
    Args:
        func: Function to execute
        default: Default return value on error
        error_msg: Error message prefix
    
    Returns:
        Function result or default value
    """
    try:
        return func()
    except Exception as e:
        logger.error(f"{error_msg}: {e}")
        return default

# ========== PROGRESS TRACKING ==========

class ProgressTracker:
    """Simple progress tracking utility."""
    
    def __init__(self, total: int, description: str = "Processing"):
        self.total = total
        self.current = 0
        self.description = description
        self.start_time = datetime.now()
    
    def update(self, increment: int = 1):
        """Update progress."""
        self.current += increment
        percentage = (self.current / self.total) * 100 if self.total > 0 else 0
        
        elapsed = (datetime.now() - self.start_time).total_seconds()
        if self.current > 0:
            eta = (elapsed / self.current) * (self.total - self.current)
            logger.info(f"{self.description}: {self.current}/{self.total} ({percentage:.1f}%) - ETA: {format_duration(eta)}")
        else:
            logger.info(f"{self.description}: {self.current}/{self.total} ({percentage:.1f}%)")
    
    def complete(self):
        """Mark as complete."""
        elapsed = (datetime.now() - self.start_time).total_seconds()
        logger.info(f"{self.description}: Completed {self.total} items in {format_duration(elapsed)}")

# ========== CACHE MANAGEMENT ==========

def get_cache_path(identifier: str, cache_type: str = 'compilation') -> str:
    """
    Get cache file path for an identifier.
    
    Args:
        identifier: Unique identifier (e.g., arxiv_id or file hash)
        cache_type: Type of cache (compilation, summary, etc.)
    
    Returns:
        Path to cache file
    """
    cache_subdir = os.path.join(config.CACHE_DIR, cache_type)
    os.makedirs(cache_subdir, exist_ok=True)
    
    safe_id = re.sub(r'[^\w\-]', '_', identifier)
    return os.path.join(cache_subdir, f"{safe_id}.json")

def cache_exists(identifier: str, cache_type: str = 'compilation') -> bool:
    """Check if cache exists for identifier."""
    cache_path = get_cache_path(identifier, cache_type)
    return os.path.exists(cache_path)