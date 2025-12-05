# config.py - Enhanced Configuration for Better RAG Performance
import os
from dataclasses import dataclass

@dataclass
class Config:
    """Enhanced configuration with optimal RAG settings."""
    
    # ========== SCRAPER SETTINGS ==========
    ARXIV_BASE_URL: str = "http://export.arxiv.org/api/query"
    ARXIV_MAX_RESULTS: int = 20
    ARXIV_RATE_LIMIT_DELAY: float = 3.0
    PDF_DOWNLOAD_TIMEOUT: int = 60
    PDF_MAX_RETRIES: int = 3
    
    SEMANTIC_SCHOLAR_API: str = "https://api.semanticscholar.org/graph/v1"
    ENABLE_CITATION_FETCH: bool = True
    
    # ========== COMPILER SETTINGS ==========
    OLLAMA_MODEL: str = "llama3.2:latest"
    OLLAMA_TIMEOUT: int = 120
    
    PAGE_LIMIT: int = 30
    WORD_LIMIT: int = 20000
    
    ENABLE_EQUATIONS: bool = True
    ENABLE_REFERENCES: bool = True
    ENABLE_CAPTIONS: bool = True
    ENABLE_CACHING: bool = True
    
    CHUNK_SIZE_WORDS: int = 500
    ENABLE_SECTION_SPECIFIC_PROMPTS: bool = True
    
    # ========== STORAGE SETTINGS ==========
    DATA_DIR: str = "data"
    PROCESSED_DIR: str = "processed"
    CACHE_DIR: str = "processed/cache"
    COMPILED_DIR: str = "processed/compiled"
    IMAGES_DIR: str = "processed/images"
    DATABASE_PATH: str = "research_assistant.db"
    
    # ========== ENHANCED RAG SETTINGS ==========
    # Embedding Model - using a better model for research papers
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"  # Fast and accurate
    EMBEDDING_DEVICE: str = "cpu"
    
    # Vector Database
    CHROMA_PERSIST_DIR: str = "processed/chroma_db"
    CHROMA_COLLECTION_NAME: str = "research_papers"
    
    # Chunking Strategy (OPTIMIZED for research papers)
    CHUNK_SIZE: int = 600  # Increased from 512 for better context
    CHUNK_OVERLAP: int = 100  # Increased overlap to prevent context loss
    
    # Retrieval (CRITICAL CHANGES)
    RAG_TOP_K_RESULTS: int = 15  # Increased from 10 to get more context
    RAG_SIMILARITY_THRESHOLD: float = 0.35  # INCREASED from 0.10 to filter noise
    RAG_INITIAL_RETRIEVAL: int = 30  # Get more results before filtering
    RAG_FILTER_BY_JOB: bool = False  # Whether to filter by job context
    
    # Answer Generation
    RAG_MAX_CONTEXT_LENGTH: int = 4000  # Increased from 3000
    RAG_TEMPERATURE: float = 0.2
    
    # Query Expansion
    RAG_ENABLE_QUERY_EXPANSION: bool = True  # NEW
    MAX_EXPANDED_QUERIES: int = 3  # NEW
    
    # Re-ranking
    RAG_ENABLE_RERANKING: bool = True  # NEW
    BOOST_CONTRIBUTIONS: float = 1.5  # NEW
    BOOST_ABSTRACTS: float = 1.3  # NEW
    
    # ========== KNOWLEDGE GRAPH SETTINGS ==========
    GRAPH_DB_PATH: str = "processed/knowledge_graph.pkl"  # Changed to .pkl
    ENABLE_GRAPH_VISUALIZATION: bool = True
    GRAPH_EXPORT_DIR: str = "processed/graph_exports"
    
    MIN_CITATION_SIMILARITY: float = 0.7
    EXTRACT_CONCEPTS: bool = True
    MAX_CONCEPTS_PER_PAPER: int = 15  # Increased from 10
    
    ENABLE_CENTRALITY_ANALYSIS: bool = True
    ENABLE_COMMUNITY_DETECTION: bool = True
    
    # ========== LOGGING ==========
    LOG_FILE: str = "research_assistant.log"
    LOG_MAX_BYTES: int = 10 * 1024 * 1024
    LOG_BACKUP_COUNT: int = 5
    LOG_LEVEL: str = "INFO"
    
    def __post_init__(self):
        """Create necessary directories."""
        for directory in [
            self.DATA_DIR,
            self.PROCESSED_DIR,
            self.CACHE_DIR,
            self.COMPILED_DIR,
            self.IMAGES_DIR,
            self.GRAPH_EXPORT_DIR,
            os.path.join(self.DATA_DIR, 'pdfs')
        ]:
            os.makedirs(directory, exist_ok=True)

config = Config()