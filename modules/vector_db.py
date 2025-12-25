# modules/vector_db.py - Vector Database Management
import os
import json
from typing import List, Dict, Optional, Tuple
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from config import config
from modules.utils import logger
from modules.database import db

class VectorDatabase:
    """
    Manages vector embeddings and semantic search for research papers.
    Uses ChromaDB for efficient similarity search.
    """
    
    def __init__(self):
        """Initialize vector database and embedding model."""
        self.persist_dir = config.CHROMA_PERSIST_DIR
        os.makedirs(self.persist_dir, exist_ok=True)
        
        # Initialize ChromaDB client with proper persistence
        self.client = chromadb.PersistentClient(
            path=self.persist_dir,
            settings=chromadb.Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Get or create collection
        try:
            self.collection = self.client.get_collection(
                name=config.CHROMA_COLLECTION_NAME
            )
            logger.info(f"Loaded existing collection: {self.collection.count()} documents")
        except:
            self.collection = self.client.create_collection(
                name=config.CHROMA_COLLECTION_NAME,
                metadata={"description": "Research paper embeddings"}
            )
            logger.info("Created new collection")
        
        # Initialize embedding model
        logger.info(f"Loading embedding model: {config.EMBEDDING_MODEL}")
        self.embedder = SentenceTransformer(
            config.EMBEDDING_MODEL,
            device=config.EMBEDDING_DEVICE
        )
        
        logger.info(f"VectorDatabase initialized with {self.collection.count()} documents")

    def refresh(self):
        """Refresh collection handle so newly indexed documents are queryable without restart."""
        try:
            self.collection = self.client.get_collection(name=config.CHROMA_COLLECTION_NAME)
            logger.info(f"Vector DB refreshed: {self.collection.count()} documents")
        except Exception as e:
            logger.warning(f"Vector DB refresh failed: {e}")
    
    def chunk_text(self, text: str, chunk_size: int = None, overlap: int = None) -> List[str]:
        """
        Split text into overlapping chunks.
        
        Args:
            text: Text to chunk
            chunk_size: Size of each chunk (words)
            overlap: Overlap between chunks (words)
        
        Returns:
            List of text chunks
        """
        chunk_size = chunk_size or config.CHUNK_SIZE
        overlap = overlap or config.CHUNK_OVERLAP
        
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            if len(chunk.split()) > 50:  # Minimum chunk size
                chunks.append(chunk)
        
        return chunks
    
    def index_paper(self, paper_id: int, paper_data: Dict) -> int:
        """
        Index a complete paper into the vector database.
        
        Args:
            paper_id: Database paper ID
            paper_data: Complete paper data including sections
        
        Returns:
            Number of chunks indexed
        """
        try:
            # FIRST: Delete existing data for this paper to avoid duplicates
            try:
                self.delete_paper(paper_id)
            except:
                pass  # Paper might not exist yet
            
            # Get job_id from database for filtering
            job_id = None
            try:
                with db.get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute('SELECT job_id FROM papers WHERE id = ?', (paper_id,))
                    row = cursor.fetchone()
                    if row:
                        job_id = row['job_id']
            except Exception as e:
                logger.warning(f"Could not fetch job_id for paper {paper_id}: {e}")
            
            metadata = paper_data.get('metadata', {})
            sections = paper_data.get('sections_text', {})
            contributions = paper_data.get('contributions', {})
            
            documents = []
            metadatas = []
            ids = []
            
            # Index abstract separately (high priority)
            abstract = metadata.get('abstract', '')
            if abstract and len(abstract.strip()) > 50:
                documents.append(abstract)
                metadatas.append({
                    'paper_id': paper_id,
                    'job_id': job_id if job_id is not None else 0,  # Use 0 as default if None
                    'arxiv_id': metadata.get('arxiv_id', 'unknown'),
                    'title': metadata.get('title', 'unknown'),
                    'section_type': 'abstract',
                    'chunk_index': 0,
                    'priority': 'high'
                })
                ids.append(f"paper_{paper_id}_abstract")
            
            # Index key contributions (high priority)
            if contributions:
                contrib_text = f"""
                Problem: {contributions.get('main_problem', '')}
                Innovation: {contributions.get('key_innovation', '')}
                Methodology: {contributions.get('core_methodology', '')}
                Results: {contributions.get('major_results', '')}
                """
                if len(contrib_text.strip()) > 50:
                    documents.append(contrib_text.strip())
                    metadatas.append({
                        'paper_id': paper_id,
                        'job_id': job_id if job_id is not None else 0,  # Use 0 as default if None
                        'arxiv_id': metadata.get('arxiv_id', 'unknown'),
                        'title': metadata.get('title', 'unknown'),
                        'section_type': 'contributions',
                        'chunk_index': 0,
                        'priority': 'high'
                    })
                    ids.append(f"paper_{paper_id}_contributions")
            
            # Index sections with chunking
            for section_name, section_text in sections.items():
                if not section_text or len(section_text.split()) < 50:
                    continue
                
                # Skip references section
                if 'reference' in section_name.lower():
                    continue
                
                # Clean section name for ID
                safe_section = section_name.replace(' ', '_').replace('.', '').replace(',', '')[:30]
                
                chunks = self.chunk_text(section_text)
                
                for chunk_idx, chunk in enumerate(chunks):
                    if len(chunk.strip()) < 50:  # Skip very short chunks
                        continue
                    
                    # Create truly unique ID with hash for safety
                    import hashlib
                    chunk_hash = hashlib.md5(chunk.encode()).hexdigest()[:8]
                    unique_id = f"paper_{paper_id}_{safe_section}_{chunk_idx}_{chunk_hash}"
                        
                    documents.append(chunk)
                    metadatas.append({
                        'paper_id': paper_id,
                        'job_id': job_id if job_id is not None else 0,  # Use 0 as default if None
                        'arxiv_id': metadata.get('arxiv_id', 'unknown'),
                        'title': metadata.get('title', 'unknown'),
                        'section_type': section_name,
                        'chunk_index': chunk_idx,
                        'priority': 'normal'
                    })
                    ids.append(unique_id)
            
            # Add to collection
            if documents:
                self.collection.add(
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids
                )
                
                logger.info(f"Indexed {len(documents)} chunks for paper {paper_id}")
                return len(documents)
            
            return 0
            
        except Exception as e:
            logger.error(f"Error indexing paper {paper_id}: {e}", exc_info=True)
            return 0
    
    def search(self, query: str, top_k: int = None, 
              filter_job_id: Optional[int] = None,
              filter_paper_id: Optional[int] = None) -> List[Dict]:
        """
        Semantic search across all indexed papers.
        
        Args:
            query: Search query
            top_k: Number of results to return
            filter_job_id: Optional job ID to restrict search (for isolation)
            filter_paper_id: Optional paper ID to restrict search
        
        Returns:
            List of search results with metadata
        """
        top_k = top_k or config.RAG_TOP_K_RESULTS
        
        try:
            # Check if collection has data
            total_docs = self.collection.count()
            if total_docs == 0:
                logger.warning("Collection is empty!")
                return []
            
            logger.debug(f"Searching {total_docs} documents with query: '{query[:50]}' (job_id={filter_job_id})")
            
            # First, try to search with filter if job_id is specified
            # If no results, fall back to unfiltered search (handles old documents)
            where = None
            n_results = min(top_k * 2, total_docs)
            
            # Try with filter first (for newly indexed documents)
            if filter_job_id is not None or filter_paper_id is not None:
                where = {}
                # Only add job_id filter if provided AND is greater than 0
                if filter_job_id is not None and filter_job_id > 0:
                    where["job_id"] = filter_job_id
                if filter_paper_id is not None:
                    where["paper_id"] = filter_paper_id
                where = where if where else None
            
            # Perform search - get more results than needed
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                where=where
            )
            
            # If no results with filter, try without filter to get old documents
            if (not results or not results.get('ids') or not results['ids'][0]) and where is not None:
                logger.debug(f"No results with filter, trying without filter...")
                results = self.collection.query(
                    query_texts=[query],
                    n_results=n_results,
                    where=None
                )
            
            if not results or not results.get('ids') or not results['ids'][0]:
                logger.warning(f"Search returned no results")
                return []
            
            # Format results
            formatted_results = []
            
            for i in range(len(results['ids'][0])):
                distance = results['distances'][0][i] if 'distances' in results else 0
                relevance_score = 1 - distance  # Convert distance to similarity
                
                # Only include results above threshold
                if relevance_score >= config.RAG_SIMILARITY_THRESHOLD:
                    result = {
                        'id': results['ids'][0][i],
                        'text': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i],
                        'distance': distance,
                        'relevance_score': relevance_score
                    }
                    formatted_results.append(result)
            
            # Sort by relevance and limit
            formatted_results.sort(key=lambda x: x['relevance_score'], reverse=True)
            formatted_results = formatted_results[:top_k]
            
            logger.info(f"Search for '{query[:50]}...' returned {len(formatted_results)} results")
            
            if formatted_results:
                logger.debug(f"Top result: {formatted_results[0]['metadata'].get('title', 'Unknown')[:50]} (score: {formatted_results[0]['relevance_score']:.3f})")
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Search error: {e}", exc_info=True)
            return []
    
    def get_paper_context(self, paper_id: int) -> str:
        """
        Get all indexed content for a specific paper.
        
        Args:
            paper_id: Database paper ID
        
        Returns:
            Combined text of all chunks for the paper
        """
        try:
            results = self.collection.get(
                where={"paper_id": paper_id}
            )
            
            if results['documents']:
                return "\n\n".join(results['documents'])
            
            return ""
            
        except Exception as e:
            logger.error(f"Error getting paper context: {e}")
            return ""
    
    def delete_paper(self, paper_id: int) -> bool:
        """
        Remove all chunks for a specific paper.
        
        Args:
            paper_id: Database paper ID
        
        Returns:
            True if successful
        """
        try:
            results = self.collection.get(
                where={"paper_id": paper_id}
            )
            
            if results['ids']:
                self.collection.delete(ids=results['ids'])
                logger.info(f"Deleted {len(results['ids'])} chunks for paper {paper_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error deleting paper: {e}")
            return False
    
    def get_statistics(self) -> Dict:
        """Get database statistics."""
        try:
            total_docs = self.collection.count()
            
            if total_docs == 0:
                return {
                    'total_chunks': 0,
                    'unique_papers': 0,
                    'avg_chunks_per_paper': 0,
                    'collection_name': config.CHROMA_COLLECTION_NAME
                }
            
            # Get papers indexed
            all_docs = self.collection.get(limit=total_docs)
            unique_papers = len(set(m['paper_id'] for m in all_docs['metadatas']))
            
            return {
                'total_chunks': total_docs,
                'unique_papers': unique_papers,
                'avg_chunks_per_paper': total_docs / unique_papers if unique_papers > 0 else 0,
                'collection_name': config.CHROMA_COLLECTION_NAME
            }
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return {
                'total_chunks': 0,
                'unique_papers': 0,
                'avg_chunks_per_paper': 0,
                'collection_name': config.CHROMA_COLLECTION_NAME
            }
    
    def clear_collection(self):
        """Clear all data from collection. USE WITH CAUTION!"""
        try:
            self.client.delete_collection(config.CHROMA_COLLECTION_NAME)
            self.collection = self.client.create_collection(
                name=config.CHROMA_COLLECTION_NAME
            )
            logger.warning("Collection cleared!")
        except Exception as e:
            logger.error(f"Error clearing collection: {e}")

# Global vector database instance
vector_db = VectorDatabase()