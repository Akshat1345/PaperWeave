# modules/rag_engine.py - RAG Query Engine
import ollama
from typing import List, Dict, Optional
from config import config
from modules.utils import logger
from modules.vector_db import vector_db
from modules.knowledge_graph import knowledge_graph
from modules.database import db

class RAGEngine:
    """
    Retrieval-Augmented Generation Engine.
    
    Combines vector search, knowledge graph, and LLM to answer questions
    about research papers with proper citations.
    """
    
    def __init__(self):
        """Initialize RAG engine."""
        self.model = config.OLLAMA_MODEL
        logger.info("RAG Engine initialized")
    
    def query(self, question: str, specific_paper_id: Optional[int] = None) -> Dict:
        """
        Answer a question using RAG.
        
        Args:
            question: User's question
            specific_paper_id: Optional paper ID to restrict search
        
        Returns:
            Dictionary with answer and sources
        """
        try:
            logger.info(f"🔍 RAG Query: {question[:100]}")
            
            # Step 1: Retrieve relevant chunks
            logger.info("📚 Step 1: Searching vector database...")
            search_results = vector_db.search(
                query=question,
                top_k=config.RAG_TOP_K_RESULTS,
                filter_paper_id=specific_paper_id
            )
            
            logger.info(f"✅ Found {len(search_results)} relevant chunks")
            
            if not search_results:
                logger.warning("⚠️  No search results found")
                return {
                    'answer': "I couldn't find relevant information in the indexed papers to answer this question. The papers may not cover this topic, or try rephrasing your question.",
                    'sources': [],
                    'confidence': 'low',
                    'debug': {
                        'search_results': 0,
                        'question': question
                    }
                }
            
            # Log what was found
            for i, result in enumerate(search_results[:3], 1):
                logger.debug(f"  [{i}] {result['metadata'].get('title', 'Unknown')[:50]} (score: {result.get('relevance_score', 0):.2f})")
            
            # Step 2: Enrich with knowledge graph context
            logger.info("🕸️  Step 2: Enriching with knowledge graph...")
            context_with_graph = self._enrich_with_graph(search_results)
            
            # Step 3: Build context for LLM
            logger.info("📝 Step 3: Building context for LLM...")
            context = self._build_context(context_with_graph)
            logger.info(f"📊 Context length: {len(context.split())} words")
            
            # Step 4: Generate answer
            logger.info("🤖 Step 4: Generating answer with LLM...")
            answer_data = self._generate_answer(question, context)
            
            # Step 5: Add sources
            answer_data['sources'] = self._format_sources(context_with_graph)
            answer_data['debug'] = {
                'search_results': len(search_results),
                'context_length': len(context.split()),
                'question': question
            }
            
            logger.info(f"✅ RAG query completed successfully")
            
            return answer_data
            
        except Exception as e:
            logger.error(f"❌ RAG query error: {e}", exc_info=True)
            return {
                'answer': f"Error processing query: {str(e)}",
                'sources': [],
                'confidence': 'error',
                'error': str(e)
            }
    
    def _enrich_with_graph(self, search_results: List[Dict]) -> List[Dict]:
        """
        Enrich search results with knowledge graph context.
        
        Args:
            search_results: Results from vector search
        
        Returns:
            Enriched results with graph context
        """
        enriched = []
        
        for result in search_results:
            paper_id = result['metadata'].get('paper_id')
            
            # Get related papers from knowledge graph
            try:
                related_papers = knowledge_graph.find_related_papers(paper_id, max_results=3)
                result['related_papers'] = related_papers
            except Exception as e:
                logger.debug(f"Could not get related papers: {e}")
                result['related_papers'] = []
            
            enriched.append(result)
        
        return enriched
    
    def _build_context(self, results: List[Dict]) -> str:
        """
        Build context string for LLM from search results.
        
        Args:
            results: Search results with metadata
        
        Returns:
            Formatted context string
        """
        context_parts = []
        
        for i, result in enumerate(results, 1):
            metadata = result['metadata']
            
            context_part = f"""
[Source {i}]
Paper: {metadata.get('title', 'Unknown')}
ArXiv ID: {metadata.get('arxiv_id', 'Unknown')}
Section: {metadata.get('section_type', 'Unknown')}
Relevance: {result.get('relevance_score', 0):.2f}

Content:
{result['text']}

---
"""
            context_parts.append(context_part)
        
        # Limit context length
        full_context = "\n".join(context_parts)
        
        # Truncate if too long
        words = full_context.split()
        if len(words) > config.RAG_MAX_CONTEXT_LENGTH:
            full_context = " ".join(words[:config.RAG_MAX_CONTEXT_LENGTH]) + "\n[Context truncated...]"
        
        return full_context
    
    def _generate_answer(self, question: str, context: str) -> Dict:
        """
        Generate answer using LLM.
        
        Args:
            question: User's question
            context: Context from retrieved documents
        
        Returns:
            Dictionary with answer and metadata
        """
        prompt = f"""You are a research assistant analyzing scientific papers. Answer the question based on the provided context.

CRITICAL RULES:
1. Only use information from the provided context
2. Cite sources using [Source N] notation
3. If information spans multiple papers, synthesize and cite all
4. If papers disagree, mention both perspectives
5. Be specific and technical
6. If the context doesn't contain enough information, say so

Context:
{context}

Question: {question}

Instructions:
- Provide a clear, detailed answer
- Use citations like [Source 1], [Source 2]
- Highlight areas of consensus or disagreement
- Be precise with technical terms

Answer:"""

        try:
            response = ollama.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                options={
                    "temperature": config.RAG_TEMPERATURE,
                    "num_predict": 500
                }
            )
            
            answer = response['message']['content'].strip()
            
            # Determine confidence based on citation usage
            citation_count = answer.count('[Source')
            confidence = 'high' if citation_count >= 2 else 'medium' if citation_count >= 1 else 'low'
            
            return {
                'answer': answer,
                'confidence': confidence,
                'model_used': self.model
            }
            
        except Exception as e:
            logger.error(f"Answer generation error: {e}")
            return {
                'answer': "Error generating answer. Please try again.",
                'confidence': 'error',
                'error': str(e)
            }
    
    def _format_sources(self, results: List[Dict]) -> List[Dict]:
        """
        Format source information for response.
        
        Args:
            results: Search results
        
        Returns:
            List of formatted source info
        """
        sources = []
        
        for i, result in enumerate(results, 1):
            metadata = result['metadata']
            
            source = {
                'source_number': i,
                'paper_id': metadata.get('paper_id'),
                'title': metadata.get('title', 'Unknown'),
                'arxiv_id': metadata.get('arxiv_id', 'Unknown'),
                'section': metadata.get('section_type', 'Unknown'),
                'relevance_score': result.get('relevance_score', 0),
                'related_papers': result.get('related_papers', [])
            }
            
            sources.append(source)
        
        return sources
    
    def generate_research_summary(self) -> Dict:
        """
        Generate a comprehensive summary of all research papers.
        
        Returns:
            Dictionary with overall insights
        """
        try:
            # Get overview from knowledge graph
            overview = knowledge_graph.get_research_overview()
            
            # Check if we have any data
            total_papers = overview.get('total_papers', 0)
            
            if total_papers == 0:
                return {
                    'summary': 'No papers have been indexed yet. Please process some papers first.',
                    'statistics': overview,
                    'error': 'no_papers'
                }
            
            # Get actual content from papers for better summary
            top_concepts = overview.get('top_concepts', [])[:5]
            common_problems = overview.get('common_problems', [])[:3]
            key_innovations = overview.get('key_innovations', [])[:3]
            research_gaps = overview.get('research_gaps', [])[:3]
            
            # Build detailed prompt
            prompt = f"""You are analyzing a collection of {total_papers} research papers. Based on the extracted information, provide a comprehensive research landscape overview.

KEY INFORMATION EXTRACTED:

Top Research Concepts (frequency):
{self._format_list(top_concepts)}

Common Research Problems:
{chr(10).join(f'{i+1}. {p[:200]}' for i, p in enumerate(common_problems) if p)}

Key Innovations:
{chr(10).join(f'{i+1}. {inn[:200]}' for i, inn in enumerate(key_innovations) if inn)}

Research Gaps Identified:
{chr(10).join(f'{i+1}. {g[:200]}' for i, g in enumerate(research_gaps) if g)}

TASK: Provide a comprehensive summary covering:

1. **Overall Research Themes** (2-3 main themes based on the concepts and problems)
2. **Common Methodologies** (infer from innovations and approaches mentioned)
3. **Key Findings and Consensus** (synthesize from innovations and results)
4. **Open Challenges** (based on research gaps and limitations)
5. **Future Directions** (what the field should focus on next)

Be specific and refer to the actual information provided above. Write in clear paragraphs.

Research Landscape Summary:"""

            response = ollama.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.3, "num_predict": 800}
            )
            
            summary = response['message']['content'].strip()
            
            return {
                'summary': summary,
                'statistics': overview,
                'total_papers': total_papers
            }
            
        except Exception as e:
            logger.error(f"Summary generation error: {e}", exc_info=True)
            return {
                'summary': f'Error generating summary: {str(e)}',
                'statistics': {},
                'error': str(e)
            }
    
    def _format_list(self, items: List) -> str:
        """Format list for prompt."""
        if not items:
            return "None available"
        
        if isinstance(items[0], tuple):
            # Concept frequency tuples
            return "\n".join(f"- {name} (mentioned {count} times)" for name, count in items)
        else:
            return "\n".join(f"- {item}" for item in items)

# Global RAG engine instance
rag_engine = RAGEngine()