# config.py - Centralized Configuration Management
import os
from dataclasses import dataclass, field
from typing import List

@dataclass
class Config:
    """Application configuration with sensible defaults."""
    
    # ========== SCRAPER SETTINGS ==========
    ARXIV_BASE_URL: str = "http://export.arxiv.org/api/query"
    ARXIV_MAX_RESULTS: int = 20
    ARXIV_RATE_LIMIT_DELAY: float = 3.0  # seconds between requests
    PDF_DOWNLOAD_TIMEOUT: int = 60  # seconds
    PDF_MAX_RETRIES: int = 3
    
    # Semantic Scholar API (for citation metrics)
    SEMANTIC_SCHOLAR_API: str = "https://api.semanticscholar.org/graph/v1"
    ENABLE_CITATION_FETCH: bool = True
    
    # ========== COMPILER SETTINGS ==========
    OLLAMA_MODEL: str = "llama3.2:latest"
    OLLAMA_TIMEOUT: int = 120  # seconds for LLM responses
    
    # PDF Processing limits
    PAGE_LIMIT: int = 30
    WORD_LIMIT: int = 20000
    
    # Feature flags
    ENABLE_EQUATIONS: bool = True
    ENABLE_REFERENCES: bool = True
    ENABLE_CAPTIONS: bool = True
    ENABLE_CACHING: bool = True
    
    # Summarization settings
    CHUNK_SIZE_WORDS: int = 500
    ENABLE_SECTION_SPECIFIC_PROMPTS: bool = True
    
    # ========== STORAGE SETTINGS ==========
    DATA_DIR: str = "data"
    PROCESSED_DIR: str = "processed"
    CACHE_DIR: str = "processed/cache"
    COMPILED_DIR: str = "processed/compiled"
    IMAGES_DIR: str = "processed/images"
    
    # Database
    DATABASE_PATH: str = "research_assistant.db"
    
    # ========== FUTURE: RAG SETTINGS ==========
    # (Will be used in Phase 2)
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    RAG_CHUNK_SIZE: int = 512
    RAG_CHUNK_OVERLAP: int = 50
    RAG_TOP_K_RESULTS: int = 5
    
    # ========== FUTURE: KNOWLEDGE GRAPH ==========
    # (Will be used in Phase 3)
    NEO4J_URI: str = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
    NEO4J_USER: str = os.getenv('NEO4J_USER', 'neo4j')
    NEO4J_PASSWORD: str = os.getenv('NEO4J_PASSWORD', 'password')
    
    # ========== LOGGING ==========
    LOG_FILE: str = "research_assistant.log"
    LOG_MAX_BYTES: int = 10 * 1024 * 1024  # 10MB
    LOG_BACKUP_COUNT: int = 5
    LOG_LEVEL: str = "INFO"
    
    def __post_init__(self):
        """Create necessary directories on initialization."""
        for directory in [
            self.DATA_DIR,
            self.PROCESSED_DIR,
            self.CACHE_DIR,
            self.COMPILED_DIR,
            self.IMAGES_DIR,
            os.path.join(self.DATA_DIR, 'pdfs')
        ]:
            os.makedirs(directory, exist_ok=True)

# Global configuration instance
config = Config()