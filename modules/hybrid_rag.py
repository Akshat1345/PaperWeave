# modules/hybrid_rag.py - Hybrid RAG with BM25 + Semantic Search
import json
import re
import math
from typing import List, Dict, Optional, Set, Tuple
from collections import defaultdict, Counter
import ollama
from config import config
from modules.utils import logger
from modules.vector_db import vector_db
from modules.knowledge_graph import knowledge_graph
from modules.database import db

class BM25Retriever:
    """
    BM25 (Best Matching 25) - Probabilistic keyword-based retrieval.
    Excellent for finding papers with specific technical terms and keywords.
    """
    
    def __init__(self):
        self.documents = []  # Store documents for indexing
        self.document_metadata = {}
        self.inverted_index = defaultdict(list)  # word -> [doc_ids]
        self.doc_lengths = {}
        self.avg_doc_length = 0
        self.word_freqs = defaultdict(lambda: defaultdict(int))  # doc_id -> {word: count}
        
        # BM25 parameters
        self.k1 = 1.5  # Term saturation parameter
        self.b = 0.75   # Length normalization parameter
        
        self._load_index()
        logger.info("BM25 Retriever initialized")
    
    def _load_index(self):
        """Load BM25 index from database."""
        try:
            with db.get_connection() as conn:
                cursor = conn.cursor()
                # Get all papers and sections with job_id for filtering
                cursor.execute('''
                    SELECT ps.id, p.id as paper_id, p.job_id, ps.section_name, ps.content, p.title, p.arxiv_id
                    FROM paper_sections ps
                    JOIN papers p ON ps.paper_id = p.id
                ''')
                
                for row in cursor.fetchall():
                    doc_id = row['id']
                    paper_id = row['paper_id']
                    job_id = row['job_id']
                    content = row['content'] or ''
                    section_name = row['section_name']
                    
                    self.documents.append(content)
                    self.document_metadata[doc_id] = {
                        'paper_id': paper_id,
                        'job_id': job_id,  # Store job_id for filtering
                        'section': section_name,
                        'title': row['title'],
                        'arxiv_id': row['arxiv_id']
                    }
                    
                    # Tokenize and index
                    tokens = self._tokenize(content)
                    self.doc_lengths[doc_id] = len(tokens)
                    
                    for token in set(tokens):  # Unique tokens only for inverted index
                        self.inverted_index[token].append(doc_id)
                    
                    for token in tokens:
                        self.word_freqs[doc_id][token] += 1
                
                if self.documents:
                    self.avg_doc_length = sum(self.doc_lengths.values()) / len(self.documents)
                    logger.info(f"BM25 index loaded: {len(self.documents)} documents, {len(self.inverted_index)} unique terms")
        
        except Exception as e:
            logger.warning(f"Could not load BM25 index: {e}")

    def refresh(self):
        """Rebuild BM25 index from the database to pick up newly processed papers."""
        try:
            self.documents = []
            self.document_metadata = {}
            self.inverted_index = defaultdict(list)
            self.doc_lengths = {}
            self.word_freqs = defaultdict(lambda: defaultdict(int))
            self.avg_doc_length = 0
            self._load_index()
            logger.info(f"BM25 index refreshed: {len(self.documents)} documents, {len(self.inverted_index)} unique terms")
        except Exception as e:
            logger.warning(f"BM25 refresh failed: {e}")
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text for BM25."""
        # Lowercase and remove special characters
        text = text.lower()
        # Keep alphanumeric and underscores
        tokens = re.findall(r'\b[a-z0-9_]+\b', text)
        # Remove very short tokens
        tokens = [t for t in tokens if len(t) > 2]
        return tokens
    
    def search(self, query: str, top_k: int = 20, job_id: Optional[int] = None) -> List[Dict]:
        """
        BM25 search for documents.
        
        Args:
            query: Search query
            top_k: Number of results to return
            job_id: Optional job ID to filter results
        
        Returns:
            List of search results with BM25 scores
        """
        if not self.documents:
            return []
        
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []
        
        scores = {}
        
        for token in query_tokens:
            if token not in self.inverted_index:
                continue
            
            idf = math.log(
                (len(self.documents) - len(self.inverted_index[token]) + 0.5) /
                (len(self.inverted_index[token]) + 0.5) + 1
            )
            
            for doc_id in self.inverted_index[token]:
                tf = self.word_freqs[doc_id][token]
                doc_length = self.doc_lengths[doc_id]
                
                # BM25 formula
                numerator = idf * tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * (doc_length / self.avg_doc_length))
                score = numerator / denominator
                
                scores[doc_id] = scores.get(doc_id, 0) + score
        
        # Sort by score
        sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        # Filter by job_id and return top_k
        results = []
        for doc_id, score in sorted_results:
            metadata = self.document_metadata.get(doc_id, {})
            
            # Apply job_id filter for isolation (if job_id provided and metadata has real job_id)
            if job_id is not None:
                doc_job_id = metadata.get('job_id', 0)
                # Only filter if document has a non-zero job_id (meaning it was explicitly set)
                # If doc_job_id is 0, it's an old document, include it
                if doc_job_id > 0 and doc_job_id != job_id:
                    continue
            
            results.append({
                'id': f"bm25_{doc_id}",
                'doc_id': doc_id,
                'metadata': metadata,
                'bm25_score': score,
                'relevance_score': min(score / (max(scores.values()) + 1e-10), 1.0)  # Normalize
            })
            
            if len(results) >= top_k:
                break
        
        return results

class HybridRAGEngine:
    """
    Hybrid RAG combining:
    1. BM25 (keyword matching)
    2. Semantic Search (dense embeddings)
    3. RRF (Reciprocal Rank Fusion)
    4. Cross-encoder Reranking
    5. Knowledge Graph enrichment
    """
    
    def __init__(self):
        self.model = config.OLLAMA_MODEL
        self.bm25 = BM25Retriever()
        
        # Query preprocessing patterns
        self.keywords_pattern = r'\b(method|approach|model|algorithm|technique|framework|system|network|dataset|metric)\b'
        self.technical_terms_pattern = r'\b([a-z]+(?:_[a-z]+)*|[A-Z]{2,})\b'
        
        logger.info("Hybrid RAG Engine initialized with BM25 + Semantic Search")
    
    def query(self, question: str, job_id: Optional[int] = None, 
             specific_paper_id: Optional[int] = None) -> Dict:
        """
        Hybrid RAG query with BM25 + Semantic + Reranking.
        
        Args:
            question: User's question
            job_id: Optional job filter
            specific_paper_id: Optional paper filter
        
        Returns:
            Comprehensive answer with sources
        """
        try:
            logger.info(f"🔍 Hybrid RAG Query: {question}")
            
            # Step 1: Query preprocessing
            processed_query = self._preprocess_query(question)
            logger.debug(f"Processed query: {processed_query}")
            
            # Step 2: Multi-stage retrieval with job_id isolation
            logger.info("Retrieving BM25 results...")
            bm25_results = self._retrieve_bm25(processed_query, top_k=20, job_id=job_id)
            logger.info(f"BM25 retrieved: {len(bm25_results)} results")
            
            logger.info("Retrieving semantic results...")
            semantic_results = self._retrieve_semantic(question, top_k=20, job_id=job_id, paper_id=specific_paper_id)
            logger.info(f"Semantic retrieved: {len(semantic_results)} results")
            
            # If no results from either method, refresh indexes and retry once
            if not bm25_results and not semantic_results:
                logger.warning("No results found initially. Refreshing indexes and retrying once...")
                try:
                    vector_db.refresh()
                except Exception as e:
                    logger.debug(f"Vector DB refresh failed: {e}")
                try:
                    self.bm25.refresh()
                except Exception as e:
                    logger.debug(f"BM25 refresh failed: {e}")

                bm25_results = self._retrieve_bm25(processed_query, top_k=20, job_id=job_id)
                semantic_results = self._retrieve_semantic(question, top_k=20, job_id=job_id, paper_id=specific_paper_id)
                logger.info(f"Post-refresh retrieval -> BM25: {len(bm25_results)}, Semantic: {len(semantic_results)}")

                if not bm25_results and not semantic_results:
                    logger.warning(f"No results found for query after refresh: {question}")
                    return {
                        'answer': 'No relevant information found in the research papers. Try a different question or rephrase your query.',
                        'sources': [],
                        'confidence': 'low',
                        'method': 'hybrid_rag',
                        'retrieval_methods': {
                            'bm25_count': 0,
                            'semantic_count': 0,
                            'after_fusion': 0,
                            'after_dedup': 0
                        }
                    }
            
            # Step 3: Reciprocal Rank Fusion (RRF)
            logger.info("Performing RRF fusion...")
            fused_results = self._reciprocal_rank_fusion(bm25_results, semantic_results)
            logger.info(f"After fusion: {len(fused_results)} unique results")
            
            # Step 4: Reranking with cross-encoder (with timeout protection)
            logger.info("Reranking with cross-encoder...")
            reranked = self._rerank_with_cross_encoder(question, fused_results)
            logger.info(f"After reranking: {len(reranked)} results")
            
            # Step 5: Deduplication
            logger.info("Deduplicating results...")
            unique_results = self._deduplicate_results(reranked)
            final_results = unique_results[:config.RAG_TOP_K_RESULTS]
            
            logger.info(f"Final results: {len(final_results)} unique documents")
            
            if not final_results:
                logger.warning("No results after deduplication")
                return {
                    'answer': 'No relevant information found. Try rephrasing your question.',
                    'sources': [],
                    'confidence': 'low',
                    'method': 'hybrid_rag'
                }
            
            # Step 6: Enrich with knowledge graph
            logger.info("Enriching with knowledge graph...")
            enriched = self._enrich_with_context(final_results, job_id=job_id)
            
            # Count knowledge graph enrichments
            kg_enrichments = sum(1 for r in enriched if r.get('related_papers'))
            
            # Step 7: Build context
            logger.info("Building context...")
            context = self._build_context(enriched)
            
            # Step 8: Generate answer
            logger.info("Generating answer with LLM...")
            answer_data = self._generate_answer(question, context, enriched)
            answer_data['sources'] = self._format_sources(enriched)
            answer_data['method'] = 'hybrid_rag'
            answer_data['retrieval_methods'] = {
                'bm25_count': len(bm25_results),
                'semantic_count': len(semantic_results),
                'after_fusion': len(fused_results),
                'after_dedup': len(unique_results)
            }
            
            # Add knowledge graph usage info
            answer_data['knowledge_graph'] = {
                'papers_enriched': kg_enrichments,
                'total_papers': len(final_results),
                'graph_used': kg_enrichments > 0,
                'total_nodes': knowledge_graph.graph.number_of_nodes(),
                'total_edges': knowledge_graph.graph.number_of_edges()
            }
            
            if kg_enrichments > 0:
                logger.info(f"🔗 Knowledge graph enriched {kg_enrichments}/{len(final_results)} papers")
            
            logger.info(f"✅ Hybrid RAG completed: {answer_data['confidence']} confidence")
            return answer_data
            
        except Exception as e:
            logger.error(f"❌ Hybrid RAG error: {e}", exc_info=True)
            return {
                'answer': f"Error processing query: {str(e)}",
                'sources': [],
                'confidence': 'error',
                'error': str(e),
                'method': 'hybrid_rag'
            }
    
    def _preprocess_query(self, query: str) -> str:
        """Preprocess query for better keyword matching."""
        # Extract key technical terms
        keywords = re.findall(self.keywords_pattern, query, re.IGNORECASE)
        technical = re.findall(self.technical_terms_pattern, query)
        
        # Boost query with important terms
        if keywords:
            query = query + " " + " ".join(keywords)
        
        return query
    
    def _retrieve_bm25(self, query: str, top_k: int = 20, job_id: Optional[int] = None) -> List[Dict]:
        """Retrieve using BM25 keyword matching."""
        try:
            results = self.bm25.search(query, top_k=top_k, job_id=job_id)
            logger.debug(f"BM25 retrieved {len(results)} results (job_id={job_id})")
            return results
        except Exception as e:
            logger.warning(f"BM25 retrieval failed: {e}")
            return []
    
    def _retrieve_semantic(self, query: str, top_k: int = 20, 
                          job_id: Optional[int] = None, paper_id: Optional[int] = None) -> List[Dict]:
        """Retrieve using semantic similarity."""
        try:
            results = vector_db.search(
                query=query,
                top_k=top_k,
                filter_job_id=job_id,
                filter_paper_id=paper_id
            )
            logger.debug(f"Semantic search retrieved {len(results)} results (job_id={job_id})")
            return results
        except Exception as e:
            logger.warning(f"Semantic retrieval failed: {e}")
            return []
    
    def _reciprocal_rank_fusion(self, bm25_results: List[Dict], 
                               semantic_results: List[Dict]) -> List[Dict]:
        """
        Combine BM25 and semantic results using Reciprocal Rank Fusion (RRF).
        RRF formula: score = sum(1 / (k + rank))
        where k is typically 60
        """
        rrf_scores = defaultdict(float)
        doc_info = {}
        k = 60
        
        # Score BM25 results
        for rank, result in enumerate(bm25_results, 1):
            doc_key = self._get_doc_key(result)
            rrf_scores[doc_key] += 1.0 / (k + rank)
            
            if doc_key not in doc_info:
                doc_info[doc_key] = {
                    'bm25_result': result,
                    'bm25_rank': rank,
                    'bm25_score': result.get('bm25_score', 0),
                    'semantic_rank': None,
                    'semantic_score': None
                }
        
        # Score semantic results
        for rank, result in enumerate(semantic_results, 1):
            doc_key = self._get_doc_key(result)
            rrf_scores[doc_key] += 1.0 / (k + rank)
            
            if doc_key in doc_info:
                doc_info[doc_key]['semantic_rank'] = rank
                doc_info[doc_key]['semantic_score'] = result.get('relevance_score', 0)
            else:
                doc_info[doc_key] = {
                    'semantic_result': result,
                    'semantic_rank': rank,
                    'semantic_score': result.get('relevance_score', 0),
                    'bm25_rank': None,
                    'bm25_score': None
                }
        
        # Sort by RRF score
        sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        
        results = []
        for doc_key, rrf_score in sorted_docs:
            info = doc_info[doc_key]
            
            # Prefer the best available source
            if 'semantic_result' in info:
                result = info['semantic_result']
            else:
                result = info['bm25_result']
            
            result['rrf_score'] = rrf_score
            result['fusion_info'] = {
                'bm25_rank': info.get('bm25_rank'),
                'semantic_rank': info.get('semantic_rank'),
                'combined_score': rrf_score
            }
            results.append(result)
        
        logger.info(f"RRF fusion: {len(rrf_scores)} unique documents")
        return results
    
    def _get_doc_key(self, result: Dict) -> str:
        """Get unique document key for fusion."""
        # Handle both semantic and BM25 results
        if 'doc_id' in result:
            return f"bm25_{result['doc_id']}"
        elif 'id' in result:
            return result['id']
        else:
            return str(result.get('metadata', {}).get('paper_id', ''))
    
    def _rerank_with_cross_encoder(self, query: str, results: List[Dict]) -> List[Dict]:
        """
        Rerank results using LLM-based cross-encoder scoring.
        More accurate but slower - only use for top candidates.
        Falls back to RRF order if LLM call fails.
        """
        try:
            if len(results) <= 5:
                # If few results, return as-is
                return results
            
            # Take top candidates for reranking
            candidates = results[:min(10, len(results))]
            
            # Build comparison text
            comparison_text = ""
            for i, result in enumerate(candidates):
                metadata = result.get('metadata', {})
                section = metadata.get('section_type', metadata.get('section', 'Unknown'))
                title = metadata.get('title', 'Unknown')[:50]
                
                text = result.get('text', '')[:150] if 'text' in result else \
                       result.get('bm25_score', 0) and "BM25 result" or "Semantic result"
                
                comparison_text += f"\n[Result {i+1}] {title} ({section})\n{text}\n"
            
            prompt = f"""Rerank these search results by relevance. Return ONLY the ranking.

QUERY: {query}

RESULTS:
{comparison_text}

Respond with ONLY result numbers (1-indexed), comma-separated, most relevant first.
Example: "3,1,5,2"
NUMBERS ONLY:"""

            logger.debug("Starting cross-encoder reranking...")
            response = ollama.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.1, "num_predict": 30}
            )
            
            ranking_str = response['message']['content'].strip().replace(" ", "")
            logger.debug(f"Cross-encoder response: {ranking_str}")
            
            # Parse rankings
            ranks = []
            for part in ranking_str.split(','):
                try:
                    rank = int(part.strip()) - 1
                    if 0 <= rank < len(candidates):
                        ranks.append(rank)
                except ValueError:
                    continue
            
            if not ranks:
                logger.warning("Failed to parse cross-encoder ranking, using RRF order")
                return results
            
            # Reorder results
            reranked = []
            for rank in ranks:
                if 0 <= rank < len(candidates):
                    candidates[rank]['cross_encoder_rank'] = len(reranked) + 1
                    reranked.append(candidates[rank])
            
            # Add remaining results
            reranked.extend(candidates[len(reranked):])
            reranked.extend(results[len(candidates):])
            
            logger.info(f"Cross-encoder reranking applied to {len(candidates)} results")
            return reranked
            
        except Exception as e:
            logger.warning(f"Cross-encoder reranking failed ({type(e).__name__}: {e}), using RRF order")
            return results
    
    def _deduplicate_results(self, results: List[Dict]) -> List[Dict]:
        """Remove duplicate chunks, keeping highest scoring ones."""
        seen_ids = {}
        unique = []
        
        for result in results:
            # Get paper ID
            paper_id = result.get('metadata', {}).get('paper_id')
            section = result.get('metadata', {}).get('section_type')
            
            key = (paper_id, section)
            
            if key not in seen_ids:
                seen_ids[key] = True
                unique.append(result)
        
        logger.info(f"Deduplicated: {len(results)} -> {len(unique)} results")
        return unique
    
    def _enrich_with_context(self, results: List[Dict], job_id: Optional[int] = None) -> List[Dict]:
        """Enrich results with knowledge graph and additional context."""
        enriched = []
        
        for result in results:
            paper_id = result.get('metadata', {}).get('paper_id')
            
            if paper_id:
                try:
                    # Get related papers from knowledge graph, filtered by job_id
                    related = knowledge_graph.find_related_papers(paper_id, max_results=3, job_id=job_id)
                    result['related_papers'] = related
                except:
                    result['related_papers'] = []
            
            enriched.append(result)
        
        return enriched
    
    def _build_context(self, results: List[Dict]) -> str:
        """Build final context from results, enhanced with knowledge graph relationships."""
        from collections import defaultdict
        
        papers = defaultdict(list)
        for result in results:
            paper_id = result['metadata'].get('paper_id')
            papers[paper_id].append(result)
        
        context_parts = []
        
        for paper_id, chunks in list(papers.items())[:10]:  # Limit to 10 papers
            if not chunks:
                continue
            
            first = chunks[0]
            metadata = first['metadata']
            
            paper_header = f"""
═══════════════════════════════════════
PAPER: {metadata.get('title', 'Unknown')[:80]}
ArXiv: {metadata.get('arxiv_id', 'N/A')} | Section: {metadata.get('section_type', 'N/A')}
═══════════════════════════════════════
"""
            content = "\n".join([c.get('text', '') for c in chunks][:3])
            
            # Add knowledge graph context if available
            kg_context = ""
            if first.get('related_papers'):
                related = first['related_papers']
                if related:
                    kg_context = "\n[Related Papers in Graph]: "
                    for rel in related[:3]:  # Show top 3 related
                        kg_context += f"\n  • {rel.get('title', 'Unknown')[:60]} ({rel.get('relationship', 'related')})"
            
            context_parts.append(paper_header + content + kg_context)
        
        full_context = "\n".join(context_parts)
        
        # Truncate if needed
        words = full_context.split()
        if len(words) > config.RAG_MAX_CONTEXT_LENGTH:
            full_context = " ".join(words[:config.RAG_MAX_CONTEXT_LENGTH]) + "\n[Context truncated...]"
        
        return full_context
    
    def _generate_answer(self, question: str, context: str, sources: List[Dict]) -> Dict:
        """Generate answer using LLM, leveraging knowledge graph relationships."""
        try:
            # Extract knowledge graph relationship information from sources
            kg_info = []
            for source in sources:
                if source.get('related_papers'):
                    for rel in source['related_papers'][:2]:
                        kg_info.append(f"• {source['metadata'].get('title', 'Paper')[:40]} connects to {rel.get('title', 'Unknown')[:40]} via {rel.get('relationship', 'relationship')}")
            
            kg_section = ""
            if kg_info:
                kg_section = f"\n\nKNOWLEDGE GRAPH CONNECTIONS:\n" + "\n".join(kg_info[:5])
            
            prompt = f"""You are a research expert analyzing scientific papers. Answer the following question based ONLY on the provided research context.

CONTEXT FROM RESEARCH PAPERS:
{context}{kg_section}

QUESTION: {question}

INSTRUCTIONS:
1. Answer comprehensively using information from the papers
2. Cite specific papers: [Paper Title - ArXiv ID]
3. If papers provide different perspectives, mention all
4. Use knowledge graph connections to show how papers relate to each other
5. Be specific about methods, results, and findings
6. If information is insufficient, clearly state it
7. When multiple papers address the same topic, discuss their relationships and differences

ANSWER:"""

            response = ollama.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                options={
                    "temperature": config.RAG_TEMPERATURE,
                    "num_predict": 800
                }
            )
            
            answer = response['message']['content'].strip()
            
            # Determine confidence
            papers_cited = len(set(s['metadata'].get('paper_id') for s in sources))
            avg_score = sum(s.get('relevance_score', 0) for s in sources) / len(sources) if sources else 0
            
            if papers_cited >= 3 and avg_score >= 0.5:
                confidence = 'high'
            elif papers_cited >= 1 and avg_score >= 0.3:
                confidence = 'medium'
            else:
                confidence = 'low'
            
            return {
                'answer': answer,
                'confidence': confidence,
                'papers_analyzed': papers_cited,
                'average_score': avg_score,
                'relationships_used': len(kg_info) > 0
            }
            
        except Exception as e:
            logger.error(f"Answer generation error: {e}")
            return {
                'answer': f"Error: {str(e)}",
                'confidence': 'error'
            }
    
    def _format_sources(self, results: List[Dict]) -> List[Dict]:
        """Format sources for display."""
        papers_seen = {}
        sources = []
        
        for i, result in enumerate(results):
            metadata = result['metadata']
            paper_id = metadata.get('paper_id')
            
            if paper_id not in papers_seen:
                papers_seen[paper_id] = len(papers_seen) + 1
            
            paper_num = papers_seen[paper_id]
            
            sources.append({
                'source_number': i + 1,
                'paper_number': paper_num,
                'paper_id': paper_id,
                'title': metadata.get('title', 'Unknown'),
                'arxiv_id': metadata.get('arxiv_id', 'Unknown'),
                'section': metadata.get('section_type', metadata.get('section', 'Unknown')),
                'relevance_score': result.get('relevance_score', result.get('bm25_score', 0)),
                'retrieval_method': 'semantic' if 'relevance_score' in result else 'bm25',
                'rrf_score': result.get('rrf_score'),
                'fusion_info': result.get('fusion_info')
            })
        
        return sources

# Global instance
hybrid_rag_engine = HybridRAGEngine()
