# modules/rag_engine.py - ENHANCED RAG Query Engine
import ollama
import re
from typing import List, Dict, Optional
from collections import defaultdict
from config import config
from modules.utils import logger
from modules.vector_db import vector_db
from modules.knowledge_graph import knowledge_graph
from modules.database import db

class EnhancedRAGEngine:
    """
    Enhanced RAG Engine with:
    - Query classification and expansion
    - Multi-stage retrieval with reranking
    - Specialized prompts for comparative/gap analysis
    - Topic-based filtering
    """
    
    def __init__(self):
        self.model = config.OLLAMA_MODEL
        
        # Query type patterns
        self.query_patterns = {
            'comparison': r'\b(compare|contrast|difference|versus|vs|across|between)\b',
            'methodology': r'\b(method|approach|technique|algorithm|implementation|how)\b',
            'gap': r'\b(gap|limitation|challenge|problem|issue|future work|missing)\b',
            'result': r'\b(result|finding|outcome|performance|accuracy|metric)\b',
            'summary': r'\b(summarize|overview|main|key|important)\b'
        }
        
        logger.info("Enhanced RAG Engine initialized")
    
    def classify_query(self, question: str) -> List[str]:
        """
        Classify query type for specialized handling.
        
        Returns list of query types (can be multiple)
        """
        question_lower = question.lower()
        types = []
        
        for query_type, pattern in self.query_patterns.items():
            if re.search(pattern, question_lower):
                types.append(query_type)
        
        if not types:
            types = ['general']
        
        logger.info(f"Query classified as: {types}")
        return types
    
    def expand_query(self, question: str, query_types: List[str]) -> List[str]:
        """
        Expand query with related terms for better retrieval.
        """
        expansions = [question]  # Original query
        
        # Add type-specific expansions
        if 'methodology' in query_types:
            expansions.append(f"describe the approach and techniques used: {question}")
            expansions.append(f"implementation details and methods: {question}")
        
        if 'gap' in query_types:
            expansions.append(f"limitations and future work: {question}")
            expansions.append(f"challenges and open problems: {question}")
        
        if 'comparison' in query_types:
            expansions.append(f"similarities and differences: {question}")
        
        return expansions
    
    def query(self, question: str, job_id: Optional[int] = None, 
             specific_paper_id: Optional[int] = None) -> Dict:
        """
        Answer a question using enhanced RAG pipeline.
        
        Args:
            question: User's question
            job_id: Filter by specific job/topic
            specific_paper_id: Filter by specific paper
        """
        try:
            logger.info(f"🔍 Enhanced RAG Query: {question}")
            
            # Step 1: Classify query type
            query_types = self.classify_query(question)
            
            # Step 2: Check if data exists
            stats = vector_db.get_statistics()
            if stats.get('total_chunks', 0) == 0:
                return {
                    'answer': 'No papers have been indexed yet. Please click "Reindex All Papers" first.',
                    'sources': [],
                    'confidence': 'low',
                    'error': 'vector_db_empty'
                }
            
            # Step 3: Multi-query retrieval
            all_results = []
            
            if config.RAG_ENABLE_QUERY_EXPANSION:
                queries = self.expand_query(question, query_types)
                logger.info(f"📝 Expanded to {len(queries)} queries")
            else:
                queries = [question]
            
            for query in queries:
                results = vector_db.search(
                    query=query,
                    top_k=config.RAG_INITIAL_RETRIEVAL,
                    filter_paper_id=specific_paper_id
                )
                all_results.extend(results)
            
            # Step 4: Deduplicate and rerank
            unique_results = self._deduplicate_results(all_results)
            
            if not unique_results:
                return {
                    'answer': f"No relevant information found for this question. The indexed papers might not cover '{question[:50]}...'",
                    'sources': [],
                    'confidence': 'low'
                }
            
            logger.info(f"✅ Retrieved {len(unique_results)} unique chunks")
            
            # Step 5: Rerank by relevance
            if config.RAG_ENABLE_RERANKING:
                reranked = self._rerank_results(unique_results, question)
                final_results = reranked[:config.RAG_TOP_K_RESULTS]
            else:
                final_results = unique_results[:config.RAG_TOP_K_RESULTS]
            
            # Step 6: Enrich with knowledge graph
            enriched = self._enrich_with_graph(final_results)
            
            # Step 7: Build specialized context
            context = self._build_enhanced_context(enriched, query_types)
            
            # Step 8: Generate answer with specialized prompt
            answer_data = self._generate_enhanced_answer(
                question, context, query_types, enriched
            )
            
            answer_data['sources'] = self._format_sources(enriched)
            answer_data['query_types'] = query_types
            
            logger.info(f"✅ Enhanced RAG completed: {answer_data['confidence']} confidence")
            return answer_data
            
        except Exception as e:
            logger.error(f"❌ Enhanced RAG error: {e}", exc_info=True)
            return {
                'answer': f"Error processing query: {str(e)}",
                'sources': [],
                'confidence': 'error',
                'error': str(e)
            }
    
    def _deduplicate_results(self, results: List[Dict]) -> List[Dict]:
        """Remove duplicate chunks, keeping highest scoring ones."""
        seen_ids = set()
        unique = []
        
        # Sort by relevance first
        sorted_results = sorted(results, key=lambda x: x.get('relevance_score', 0), reverse=True)
        
        for result in sorted_results:
            chunk_id = result['id']
            if chunk_id not in seen_ids:
                seen_ids.add(chunk_id)
                unique.append(result)
        
        return unique
    
    def _rerank_results(self, results: List[Dict], question: str) -> List[Dict]:
        """
        Rerank results by considering multiple factors:
        - Relevance score
        - Section priority (abstract, contributions > body)
        - Chunk position (earlier chunks often more relevant)
        """
        for result in results:
            base_score = result.get('relevance_score', 0)
            metadata = result.get('metadata', {})
            
            # Priority boost
            priority = metadata.get('priority', 'normal')
            if priority == 'high':
                base_score *= 1.3
            
            # Section type boost
            section = metadata.get('section_type', '').lower()
            if any(kw in section for kw in ['abstract', 'contribution', 'conclusion']):
                base_score *= 1.2
            elif any(kw in section for kw in ['method', 'approach', 'implementation']):
                base_score *= 1.15
            
            result['reranked_score'] = base_score
        
        return sorted(results, key=lambda x: x.get('reranked_score', 0), reverse=True)
    
    def _enrich_with_graph(self, results: List[Dict]) -> List[Dict]:
        """Enrich with knowledge graph relationships."""
        enriched = []
        
        for result in results:
            paper_id = result['metadata'].get('paper_id')
            
            if paper_id:
                try:
                    related = knowledge_graph.find_related_papers(paper_id, max_results=2)
                    result['related_papers'] = related
                except Exception as e:
                    logger.debug(f"Could not get related papers: {e}")
                    result['related_papers'] = []
            
            enriched.append(result)
        
        return enriched
    
    def _build_enhanced_context(self, results: List[Dict], query_types: List[str]) -> str:
        """
        Build context optimized for query type.
        Groups chunks by paper for better comparison.
        """
        # Group by paper
        papers = defaultdict(list)
        for result in results:
            paper_id = result['metadata'].get('paper_id')
            papers[paper_id].append(result)
        
        context_parts = []
        
        for paper_id, chunks in papers.items():
            if not chunks:
                continue
            
            # Get paper info from first chunk
            first_chunk = chunks[0]
            metadata = first_chunk['metadata']
            
            paper_section = f"""
═══════════════════════════════════════
PAPER: {metadata.get('title', 'Unknown')}
ArXiv ID: {metadata.get('arxiv_id', 'Unknown')}
═══════════════════════════════════════
"""
            
            # Add chunks grouped by section
            sections = defaultdict(list)
            for chunk in chunks:
                section = chunk['metadata'].get('section_type', 'Body')
                sections[section].append(chunk['text'])
            
            for section, texts in sections.items():
                paper_section += f"\n[{section}]\n"
                paper_section += "\n".join(texts)
                paper_section += "\n"
            
            context_parts.append(paper_section)
        
        full_context = "\n\n".join(context_parts)
        
        # Truncate if needed
        words = full_context.split()
        if len(words) > config.RAG_MAX_CONTEXT_LENGTH:
            full_context = " ".join(words[:config.RAG_MAX_CONTEXT_LENGTH]) + "\n[Context truncated...]"
        
        return full_context
    
    def _generate_enhanced_answer(self, question: str, context: str, 
                                  query_types: List[str], sources: List[Dict]) -> Dict:
        """
        Generate answer with specialized prompts based on query type.
        """
        
        # Select specialized prompt
        if 'comparison' in query_types:
            prompt = self._comparison_prompt(question, context, sources)
        elif 'gap' in query_types:
            prompt = self._gap_analysis_prompt(question, context, sources)
        elif 'methodology' in query_types:
            prompt = self._methodology_prompt(question, context, sources)
        else:
            prompt = self._general_prompt(question, context)
        
        try:
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
            citation_count = answer.count('[Paper') + answer.count('[Source')
            papers_cited = len(set(s['metadata'].get('paper_id') for s in sources))
            
            if citation_count >= 3 and papers_cited >= 2:
                confidence = 'high'
            elif citation_count >= 1:
                confidence = 'medium'
            else:
                confidence = 'low'
            
            return {
                'answer': answer,
                'confidence': confidence,
                'model_used': self.model,
                'papers_analyzed': papers_cited
            }
            
        except Exception as e:
            logger.error(f"Answer generation error: {e}")
            return {
                'answer': f"Error generating answer: {str(e)}",
                'confidence': 'error',
                'error': str(e)
            }
    
    def _comparison_prompt(self, question: str, context: str, sources: List[Dict]) -> str:
        """Specialized prompt for comparative analysis."""
        paper_titles = [s['metadata'].get('title', 'Unknown')[:60] for s in sources[:5]]
        
        return f"""You are analyzing multiple research papers to answer a comparative question.

PAPERS ANALYZED:
{chr(10).join(f'{i+1}. {title}' for i, title in enumerate(set(paper_titles)))}

CONTEXT:
{context}

QUESTION: {question}

INSTRUCTIONS FOR COMPARISON:
1. Create a structured comparison across ALL papers
2. Identify similarities and differences explicitly
3. Use format: "[Paper X] uses approach A, while [Paper Y] uses approach B"
4. Cite paper numbers: [Paper 1], [Paper 2], etc.
5. Create a summary table if comparing multiple dimensions
6. Highlight consensus vs disagreements
7. Note any papers that stand out

PROVIDE YOUR COMPARATIVE ANALYSIS:"""
    
    def _gap_analysis_prompt(self, question: str, context: str, sources: List[Dict]) -> str:
        """Specialized prompt for research gap analysis."""
        return f"""You are analyzing research papers to identify gaps, limitations, and future directions.

CONTEXT:
{context}

QUESTION: {question}

INSTRUCTIONS FOR GAP ANALYSIS:
1. Extract explicitly stated limitations from each paper
2. Identify implicit gaps by analyzing what's missing
3. Synthesize common challenges mentioned across papers
4. List specific future work directions suggested
5. Identify methodological limitations
6. Note data/resource constraints mentioned
7. Cite papers when describing gaps: [Paper X]

FORMAT YOUR ANSWER AS:
**Stated Limitations:**
- [list from papers]

**Common Challenges:**
- [synthesize across papers]

**Future Research Directions:**
- [compile suggestions]

**Methodological Gaps:**
- [identify missing approaches]

YOUR GAP ANALYSIS:"""
    
    def _methodology_prompt(self, question: str, context: str, sources: List[Dict]) -> str:
        """Specialized prompt for methodology questions."""
        return f"""You are explaining research methodologies and approaches from multiple papers.

CONTEXT:
{context}

QUESTION: {question}

INSTRUCTIONS FOR METHODOLOGY EXPLANATION:
1. Describe the core approach of each relevant paper
2. Break down algorithms/techniques step-by-step
3. Explain key innovations or modifications
4. Compare implementation details across papers
5. Note dataset choices and experimental setup
6. Cite papers for each methodology: [Paper X]
7. Highlight what makes each approach unique

STRUCTURE:
- Overview of approaches
- Detailed methodology breakdown per paper
- Key differences and innovations
- Implementation considerations

YOUR METHODOLOGY EXPLANATION:"""
    
    def _general_prompt(self, question: str, context: str) -> str:
        """General prompt for other question types."""
        return f"""You are a research assistant analyzing scientific papers.

CONTEXT:
{context}

QUESTION: {question}

INSTRUCTIONS:
1. Answer based ONLY on the provided context
2. Synthesize information across papers
3. Cite sources using [Paper X] notation
4. Be specific and technical
5. If papers disagree, mention both perspectives
6. If information is incomplete, say so clearly

YOUR ANSWER:"""
    
    def _format_sources(self, results: List[Dict]) -> List[Dict]:
        """Format sources with paper grouping."""
        # Group by paper
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
                'section': metadata.get('section_type', 'Unknown'),
                'relevance_score': result.get('relevance_score', 0),
                'related_papers': result.get('related_papers', [])
            })
        
        return sources
    
    def generate_research_summary(self, job_id: Optional[int] = None) -> Dict:
        """Generate comprehensive research summary."""
        try:
            # Get overview from knowledge graph
            overview = knowledge_graph.get_research_overview()
            
            total_papers = overview.get('total_papers', 0)
            if total_papers == 0:
                return {
                    'summary': 'No papers indexed yet.',
                    'statistics': overview,
                    'error': 'no_papers'
                }
            
            # Filter by job if specified
            if job_id:
                papers = db.get_papers_by_job(job_id)
                logger.info(f"Generating summary for job {job_id}: {len(papers)} papers")
            
            # Build comprehensive prompt
            top_concepts = overview.get('top_concepts', [])[:10]
            
            prompt = f"""Analyze this collection of {total_papers} research papers and provide a comprehensive landscape overview.

TOP RESEARCH CONCEPTS:
{chr(10).join(f'- {name} (mentioned {count} times)' for name, count in top_concepts)}

PROVIDE A COMPREHENSIVE ANALYSIS:

1. **Main Research Themes** (2-3 dominant themes)
2. **Common Methodologies** (what approaches are popular?)
3. **Key Findings & Consensus** (areas of agreement)
4. **Open Challenges** (what problems remain?)
5. **Future Directions** (where is field heading?)

Write in clear paragraphs with specific references to the concepts above."""

            response = ollama.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.3, "num_predict": 1000}
            )
            
            return {
                'summary': response['message']['content'].strip(),
                'statistics': overview,
                'total_papers': total_papers
            }
            
        except Exception as e:
            logger.error(f"Summary error: {e}", exc_info=True)
            return {
                'summary': f'Error: {str(e)}',
                'statistics': {},
                'error': str(e)
            }

# Global instance
rag_engine = EnhancedRAGEngine()