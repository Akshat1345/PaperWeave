#!/usr/bin/env python3
# visualize_graph.py - Visualize Knowledge Graph
"""
Create beautiful visualizations of the knowledge graph.

Usage:
    python visualize_graph.py
"""

import matplotlib.pyplot as plt
import networkx as nx
from modules.knowledge_graph import knowledge_graph
import os

def visualize_full_graph():
    """Create a full visualization of the knowledge graph."""
    print("🎨 Creating knowledge graph visualization...\n")
    
    graph = knowledge_graph.graph
    
    if graph.number_of_nodes() == 0:
        print("❌ Knowledge graph is empty!")
        print("   Run: python diagnose_rag.py reindex")
        return
    
    print(f"📊 Graph Statistics:")
    print(f"   Nodes: {graph.number_of_nodes()}")
    print(f"   Edges: {graph.number_of_edges()}")
    
    # Separate nodes by type
    paper_nodes = [n for n in graph.nodes() if n.startswith('paper_')]
    author_nodes = [n for n in graph.nodes() if n.startswith('author_')]
    concept_nodes = [n for n in graph.nodes() if n.startswith('concept_')]
    
    print(f"   Papers: {len(paper_nodes)}")
    print(f"   Authors: {len(author_nodes)}")
    print(f"   Concepts: {len(concept_nodes)}\n")
    
    # Create figure
    plt.figure(figsize=(20, 16))
    
    # Use spring layout for better visualization
    print("📐 Computing layout...")
    pos = nx.spring_layout(graph, k=2, iterations=50, seed=42)
    
    # Draw different node types with different colors
    print("🎨 Drawing nodes...")
    
    # Papers (large, blue)
    nx.draw_networkx_nodes(graph, pos,
                          nodelist=paper_nodes,
                          node_color='#667eea',
                          node_size=1500,
                          label='Papers',
                          alpha=0.9)
    
    # Authors (medium, green)
    nx.draw_networkx_nodes(graph, pos,
                          nodelist=author_nodes,
                          node_color='#48bb78',
                          node_size=800,
                          label='Authors',
                          alpha=0.8)
    
    # Concepts (small, orange)
    nx.draw_networkx_nodes(graph, pos,
                          nodelist=concept_nodes,
                          node_color='#f6ad55',
                          node_size=600,
                          label='Concepts',
                          alpha=0.8)
    
    # Draw edges
    print("🔗 Drawing edges...")
    nx.draw_networkx_edges(graph, pos,
                          alpha=0.2,
                          arrows=True,
                          arrowsize=10,
                          width=0.5)
    
    # Add labels for papers only (to avoid clutter)
    print("📝 Adding labels...")
    paper_labels = {}
    for node in paper_nodes:
        title = graph.nodes[node].get('title', node)
        # Truncate long titles
        paper_labels[node] = title[:30] + '...' if len(title) > 30 else title
    
    nx.draw_networkx_labels(graph, pos,
                           labels=paper_labels,
                           font_size=8,
                           font_weight='bold')
    
    plt.title("Research Paper Knowledge Graph", fontsize=20, fontweight='bold')
    plt.legend(loc='upper left', fontsize=12)
    plt.axis('off')
    plt.tight_layout()
    
    # Save
    output_dir = "processed/graph_exports"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "knowledge_graph_full.png")
    
    print(f"💾 Saving to {output_path}...")
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Saved!\n")
    
    plt.show()

def visualize_paper_network():
    """Visualize only papers and their citations."""
    print("🎨 Creating paper citation network...\n")
    
    graph = knowledge_graph.graph
    
    # Create subgraph with only papers
    paper_nodes = [n for n in graph.nodes() if n.startswith('paper_')]
    subgraph = graph.subgraph(paper_nodes).copy()
    
    if subgraph.number_of_nodes() == 0:
        print("❌ No papers in graph!")
        return
    
    plt.figure(figsize=(16, 12))
    
    # Layout
    pos = nx.spring_layout(subgraph, k=3, iterations=50, seed=42)
    
    # Node sizes based on citations
    node_sizes = []
    for node in subgraph.nodes():
        citation_count = graph.nodes[node].get('citation_count', 0)
        # Size between 1000 and 3000 based on citations
        size = 1000 + min(citation_count * 10, 2000)
        node_sizes.append(size)
    
    # Draw nodes
    nx.draw_networkx_nodes(subgraph, pos,
                          node_color='#667eea',
                          node_size=node_sizes,
                          alpha=0.8)
    
    # Draw edges (citations)
    nx.draw_networkx_edges(subgraph, pos,
                          alpha=0.3,
                          arrows=True,
                          arrowsize=15,
                          width=2,
                          edge_color='#764ba2')
    
    # Labels
    labels = {}
    for node in subgraph.nodes():
        title = graph.nodes[node].get('title', node)
        labels[node] = title[:40] + '...' if len(title) > 40 else title
    
    nx.draw_networkx_labels(subgraph, pos,
                           labels=labels,
                           font_size=9,
                           font_weight='bold')
    
    plt.title("Paper Citation Network", fontsize=18, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    
    output_path = "processed/graph_exports/paper_network.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Saved to {output_path}\n")
    
    plt.show()

def visualize_author_network():
    """Visualize author collaboration network."""
    print("🎨 Creating author collaboration network...\n")
    
    graph = knowledge_graph.graph
    
    # Create author collaboration graph
    author_collab = nx.Graph()
    
    # Find authors and papers
    paper_nodes = [n for n in graph.nodes() if n.startswith('paper_')]
    
    for paper in paper_nodes:
        # Get authors of this paper
        authors = [pred for pred in graph.predecessors(paper) 
                  if pred.startswith('author_')]
        
        # Create edges between co-authors
        for i, author1 in enumerate(authors):
            for author2 in authors[i+1:]:
                if author_collab.has_edge(author1, author2):
                    author_collab[author1][author2]['weight'] += 1
                else:
                    author_collab.add_edge(author1, author2, weight=1)
                    # Add author names
                    if not author_collab.nodes[author1].get('name'):
                        author_collab.nodes[author1]['name'] = graph.nodes[author1].get('name', author1)
                    if not author_collab.nodes[author2].get('name'):
                        author_collab.nodes[author2]['name'] = graph.nodes[author2].get('name', author2)
    
    if author_collab.number_of_nodes() == 0:
        print("❌ No author collaborations found!")
        return
    
    plt.figure(figsize=(16, 12))
    
    pos = nx.spring_layout(author_collab, k=2, iterations=50, seed=42)
    
    # Node sizes based on number of collaborations
    node_sizes = [300 + author_collab.degree(n) * 200 for n in author_collab.nodes()]
    
    nx.draw_networkx_nodes(author_collab, pos,
                          node_color='#48bb78',
                          node_size=node_sizes,
                          alpha=0.8)
    
    # Edge width based on collaboration strength
    edges = author_collab.edges()
    weights = [author_collab[u][v]['weight'] for u, v in edges]
    
    nx.draw_networkx_edges(author_collab, pos,
                          width=[w * 0.5 for w in weights],
                          alpha=0.4)
    
    # Labels with author names
    labels = {node: data.get('name', node).split('_')[-1] 
             for node, data in author_collab.nodes(data=True)}
    
    nx.draw_networkx_labels(author_collab, pos,
                           labels=labels,
                           font_size=8)
    
    plt.title("Author Collaboration Network", fontsize=18, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    
    output_path = "processed/graph_exports/author_network.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Saved to {output_path}\n")
    
    plt.show()

def print_graph_stats():
    """Print detailed graph statistics."""
    graph = knowledge_graph.graph
    
    print("\n" + "="*70)
    print("📊 DETAILED KNOWLEDGE GRAPH STATISTICS")
    print("="*70 + "\n")
    
    # Basic stats
    print("Basic Statistics:")
    print(f"  Total Nodes: {graph.number_of_nodes()}")
    print(f"  Total Edges: {graph.number_of_edges()}")
    print(f"  Graph Density: {nx.density(graph):.4f}")
    
    # Nodes by type
    paper_nodes = [n for n in graph.nodes() if n.startswith('paper_')]
    author_nodes = [n for n in graph.nodes() if n.startswith('author_')]
    concept_nodes = [n for n in graph.nodes() if n.startswith('concept_')]
    
    print(f"\nNode Types:")
    print(f"  Papers: {len(paper_nodes)}")
    print(f"  Authors: {len(author_nodes)}")
    print(f"  Concepts: {len(concept_nodes)}")
    
    # Top concepts
    concept_freq = {}
    for concept in concept_nodes:
        # Count papers discussing this concept
        papers_count = len(list(graph.predecessors(concept)))
        concept_freq[graph.nodes[concept].get('name', concept)] = papers_count
    
    print(f"\nTop 10 Concepts:")
    for i, (concept, count) in enumerate(sorted(concept_freq.items(), 
                                                key=lambda x: x[1], 
                                                reverse=True)[:10], 1):
        print(f"  {i}. {concept}: {count} papers")
    
    # Most cited papers
    print(f"\nMost Cited Papers:")
    cited_papers = []
    for paper in paper_nodes:
        citations = graph.nodes[paper].get('citation_count', 0)
        title = graph.nodes[paper].get('title', 'Unknown')
        cited_papers.append((title, citations))
    
    for i, (title, citations) in enumerate(sorted(cited_papers, 
                                                  key=lambda x: x[1], 
                                                  reverse=True)[:5], 1):
        print(f"  {i}. {title[:60]}: {citations} citations")
    
    print("\n" + "="*70 + "\n")

def main():
    """Main visualization menu."""
    print("\n" + "="*70)
    print("🎨 KNOWLEDGE GRAPH VISUALIZER")
    print("="*70 + "\n")
    
    print("Options:")
    print("  1. Visualize full graph (papers, authors, concepts)")
    print("  2. Visualize paper citation network")
    print("  3. Visualize author collaboration network")
    print("  4. Show graph statistics")
    print("  5. Generate all visualizations")
    print("  6. Exit")
    
    choice = input("\nEnter choice (1-6): ").strip()
    
    if choice == '1':
        visualize_full_graph()
    elif choice == '2':
        visualize_paper_network()
    elif choice == '3':
        visualize_author_network()
    elif choice == '4':
        print_graph_stats()
    elif choice == '5':
        print_graph_stats()
        visualize_full_graph()
        visualize_paper_network()
        visualize_author_network()
    elif choice == '6':
        print("\n👋 Goodbye!\n")
    else:
        print("\n❌ Invalid choice!")

if __name__ == "__main__":
    main()