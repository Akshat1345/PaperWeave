# modules/knowledge_graph.py - Knowledge Graph Management
import os
import json
import pickle
from typing import List, Dict, Set, Tuple, Optional
import networkx as nx
from collections import Counter, defaultdict
from config import config
from modules.utils import logger
from modules.database import db

class KnowledgeGraph:
    """
    Builds and manages a knowledge graph of research papers.
    
    Graph Structure:
    - Nodes: Papers, Authors, Concepts
    - Edges: Citations, Co-authorship, Shared Concepts
    """
    
    def __init__(self):
        """Initialize knowledge graph."""
        self.graph = nx.MultiDiGraph()  # Directed graph with multiple edges
        self.graph_path = config.GRAPH_DB_PATH
        
        # Load existing graph if available
        if os.path.exists(self.graph_path):
            self.load_graph()
            logger.info(f"Loaded knowledge graph: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")
        else:
            logger.info("Initialized new knowledge graph")
    
    def add_paper(self, paper_id: int, paper_data: Dict) -> bool:
        """
        Add a paper node to the graph.
        
        Args:
            paper_id: Database paper ID
            paper_data: Complete paper data
        
        Returns:
            True if successful
        """
        try:
            metadata = paper_data.get('metadata', {})
            contributions = paper_data.get('contributions', {})
            
            # Get job_id from database for isolation
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
            
            # Add paper node
            self.graph.add_node(
                f"paper_{paper_id}",
                type='paper',
                paper_id=paper_id,
                job_id=job_id if job_id is not None else 0,  # Use 0 if None
                arxiv_id=metadata.get('arxiv_id', ''),
                title=metadata.get('title', ''),
                year=self._extract_year(metadata.get('published', '')),
                citation_count=metadata.get('citation_count', 0),
                abstract=metadata.get('abstract', ''),
                main_problem=contributions.get('main_problem', ''),
                key_innovation=contributions.get('key_innovation', ''),
                limitations=contributions.get('limitations', ''),
                research_gaps=contributions.get('research_gaps', '')
            )
            
            # Add author nodes and relationships
            authors = metadata.get('authors', [])
            for author in authors:
                author_id = self._normalize_author_name(author)
                
                if not self.graph.has_node(author_id):
                    self.graph.add_node(
                        author_id,
                        type='author',
                        name=author
                    )
                
                self.graph.add_edge(
                    author_id,
                    f"paper_{paper_id}",
                    relationship='authored'
                )
            
            # Extract and add concept nodes
            if config.EXTRACT_CONCEPTS:
                concepts = self._extract_concepts(paper_data)
                for concept in concepts:
                    concept_id = f"concept_{concept.lower().replace(' ', '_')}"
                    
                    if not self.graph.has_node(concept_id):
                        self.graph.add_node(
                            concept_id,
                            type='concept',
                            name=concept
                        )
                    
                    self.graph.add_edge(
                        f"paper_{paper_id}",
                        concept_id,
                        relationship='discusses'
                    )
            
            logger.info(f"Added paper {paper_id} to knowledge graph")
            return True
            
        except Exception as e:
            logger.error(f"Error adding paper to graph: {e}", exc_info=True)
            return False
    
    def link_citations(self, paper_id: int, references: List[Dict]) -> int:
        """
        Create citation links between papers.
        
        Args:
            paper_id: Source paper ID
            references: List of referenced papers
        
        Returns:
            Number of citation links created
        """
        links_created = 0
        
        try:
            source_node = f"paper_{paper_id}"
            
            if not self.graph.has_node(source_node):
                return 0
            
            # Try to match references with papers in database
            for ref in references:
                # Simple matching by title similarity
                ref_title = ref.get('title', '').lower()
                
                for node in self.graph.nodes():
                    if node.startswith('paper_'):
                        node_data = self.graph.nodes[node]
                        node_title = node_data.get('title', '').lower()
                        
                        # Simple string matching (can be improved)
                        if ref_title and node_title and self._title_similarity(ref_title, node_title) > 0.8:
                            self.graph.add_edge(
                                source_node,
                                node,
                                relationship='cites',
                                reference_info=ref
                            )
                            links_created += 1
                            break
            
            if links_created > 0:
                logger.info(f"Created {links_created} citation links for paper {paper_id}")
            
            return links_created
            
        except Exception as e:
            logger.error(f"Error linking citations: {e}")
            return 0
    
    def find_related_papers(self, paper_id: int, max_results: int = 5, job_id: Optional[int] = None) -> List[Dict]:
        """
        Find papers related to a given paper through various relationships.
        
        Args:
            paper_id: Source paper ID
            max_results: Maximum number of related papers
            job_id: Optional job_id to filter related papers by
        
        Returns:
            List of related papers with relationship info
        """
        try:
            source_node = f"paper_{paper_id}"
            
            if not self.graph.has_node(source_node):
                return []
            
            # Get source paper's job_id if not provided
            if job_id is None:
                job_id = self.graph.nodes[source_node].get('job_id')
            
            related = []
            
            # Papers that cite this paper
            for predecessor in self.graph.predecessors(source_node):
                if predecessor.startswith('paper_'):
                    # Filter by job_id if specified
                    if job_id is not None and self.graph.nodes[predecessor].get('job_id') != job_id:
                        continue
                    
                    edge_data = self.graph.get_edge_data(predecessor, source_node)
                    if edge_data and any(e.get('relationship') == 'cites' for e in edge_data.values()):
                        related.append({
                            'paper_id': int(predecessor.split('_')[1]),
                            'relationship': 'cites_this',
                            'title': self.graph.nodes[predecessor].get('title', '')
                        })
            
            # Papers cited by this paper
            for successor in self.graph.successors(source_node):
                if successor.startswith('paper_'):
                    # Filter by job_id if specified
                    if job_id is not None and self.graph.nodes[successor].get('job_id') != job_id:
                        continue
                    
                    edge_data = self.graph.get_edge_data(source_node, successor)
                    if edge_data and any(e.get('relationship') == 'cites' for e in edge_data.values()):
                        related.append({
                            'paper_id': int(successor.split('_')[1]),
                            'relationship': 'cited_by_this',
                            'title': self.graph.nodes[successor].get('title', '')
                        })
            
            # Papers by same authors
            authors = [n for n in self.graph.predecessors(source_node) if n.startswith('author_')]
            for author in authors:
                for paper in self.graph.successors(author):
                    if paper.startswith('paper_') and paper != source_node:
                        # Filter by job_id if specified
                        if job_id is not None and self.graph.nodes[paper].get('job_id') != job_id:
                            continue
                        
                        related.append({
                            'paper_id': int(paper.split('_')[1]),
                            'relationship': 'same_author',
                            'title': self.graph.nodes[paper].get('title', '')
                        })
            
            # Papers with shared concepts
            concepts = [n for n in self.graph.successors(source_node) if n.startswith('concept_')]
            for concept in concepts:
                for paper in self.graph.predecessors(concept):
                    if paper.startswith('paper_') and paper != source_node:
                        # Filter by job_id if specified
                        if job_id is not None and self.graph.nodes[paper].get('job_id') != job_id:
                            continue
                        
                        related.append({
                            'paper_id': int(paper.split('_')[1]),
                            'relationship': 'shared_concept',
                            'concept': self.graph.nodes[concept].get('name', ''),
                            'title': self.graph.nodes[paper].get('title', '')
                        })
            
            # Remove duplicates and limit
            seen = set()
            unique_related = []
            for item in related:
                if item['paper_id'] not in seen:
                    seen.add(item['paper_id'])
                    unique_related.append(item)
            
            return unique_related[:max_results]
            
        except Exception as e:
            logger.error(f"Error finding related papers: {e}")
            return []
    
    def get_research_overview(self) -> Dict:
        """
        Generate an overview of the research landscape.
        
        Returns:
            Dictionary with aggregated insights
        """
        try:
            papers = [n for n in self.graph.nodes() if n.startswith('paper_')]
            
            # Aggregate research problems
            problems = []
            innovations = []
            gaps = []
            
            for paper in papers:
                node_data = self.graph.nodes[paper]
                if node_data.get('main_problem'):
                    problems.append(node_data['main_problem'])
                if node_data.get('key_innovation'):
                    innovations.append(node_data['key_innovation'])
                if node_data.get('research_gaps'):
                    gaps.append(node_data['research_gaps'])
            
            # Most common concepts
            concepts = [n for n in self.graph.nodes() if n.startswith('concept_')]
            concept_frequency = {}
            for concept in concepts:
                # Count papers discussing this concept
                papers_count = len(list(self.graph.predecessors(concept)))
                concept_frequency[self.graph.nodes[concept]['name']] = papers_count
            
            top_concepts = sorted(concept_frequency.items(), key=lambda x: x[1], reverse=True)[:10]
            
            # Most influential papers (by citations)
            influential_papers = []
            for paper in papers:
                node_data = self.graph.nodes[paper]
                influential_papers.append({
                    'paper_id': node_data.get('paper_id'),
                    'title': node_data.get('title', ''),
                    'citation_count': node_data.get('citation_count', 0)
                })
            
            influential_papers.sort(key=lambda x: x['citation_count'], reverse=True)
            
            return {
                'total_papers': len(papers),
                'total_concepts': len(concepts),
                'top_concepts': top_concepts,
                'most_influential_papers': influential_papers[:5],
                'common_problems': problems[:5],
                'key_innovations': innovations[:5],
                'research_gaps': gaps[:5]
            }
            
        except Exception as e:
            logger.error(f"Error generating overview: {e}")
            return {}
    
    def _extract_concepts(self, paper_data: Dict) -> List[str]:
        """Extract key concepts from paper."""
        concepts = set()
        
        # From contributions
        contributions = paper_data.get('contributions', {})
        for key in ['key_innovation', 'core_methodology']:
            text = contributions.get(key, '')
            # Simple concept extraction (can be improved with NLP)
            keywords = self._extract_keywords(text)
            concepts.update(keywords[:5])
        
        # From categories
        metadata = paper_data.get('metadata', {})
        categories = metadata.get('categories', [])
        concepts.update(categories)
        
        return list(concepts)[:config.MAX_CONCEPTS_PER_PAPER]
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Simple keyword extraction."""
        if not text:
            return []
        
        # Common technical terms (simplified - can use NLP libraries)
        keywords = []
        terms = text.lower().split()
        
        important_terms = [
            'neural', 'network', 'learning', 'deep', 'machine', 'model',
            'algorithm', 'optimization', 'training', 'architecture', 
            'transformer', 'attention', 'convolution', 'lstm', 'gru'
        ]
        
        for term in terms:
            if term in important_terms and term not in keywords:
                keywords.append(term)
        
        return keywords
    
    def _extract_year(self, date_str: str) -> int:
        """Extract year from date string."""
        try:
            if date_str:
                return int(date_str[:4])
        except:
            pass
        return 0
    
    def _normalize_author_name(self, name: str) -> str:
        """Normalize author name for matching."""
        normalized = name.lower().strip()
        return f"author_{normalized.replace(' ', '_')}"
    
    def _title_similarity(self, title1: str, title2: str) -> float:
        """Simple title similarity (can be improved)."""
        words1 = set(title1.lower().split())
        words2 = set(title2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    def save_graph(self):
        """Save graph to disk."""
        try:
            os.makedirs(os.path.dirname(self.graph_path), exist_ok=True)
            with open(self.graph_path, 'wb') as f:
                pickle.dump(self.graph, f)
            logger.info(f"Saved knowledge graph to {self.graph_path}")
        except Exception as e:
            logger.error(f"Error saving graph: {e}")
    
    def load_graph(self):
        """Load graph from disk."""
        try:
            with open(self.graph_path, 'rb') as f:
                self.graph = pickle.load(f)
            logger.info(f"Loaded knowledge graph from {self.graph_path}")
        except Exception as e:
            logger.error(f"Error loading graph: {e}")
    
    def get_statistics(self) -> Dict:
        """Get graph statistics."""
        return {
            'total_nodes': self.graph.number_of_nodes(),
            'total_edges': self.graph.number_of_edges(),
            'paper_nodes': len([n for n in self.graph.nodes() if n.startswith('paper_')]),
            'author_nodes': len([n for n in self.graph.nodes() if n.startswith('author_')]),
            'concept_nodes': len([n for n in self.graph.nodes() if n.startswith('concept_')])
        }

# Global knowledge graph instance
knowledge_graph = KnowledgeGraph()