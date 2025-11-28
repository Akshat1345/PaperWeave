# diagnose_rag.py - RAG System Diagnostic
"""
Diagnose RAG indexing issues and manually reindex if needed.

Usage:
    python diagnose_rag.py
"""

import os
import json
from modules.database import db
from modules.vector_db import vector_db
from modules.knowledge_graph import knowledge_graph
from modules.utils import logger

def diagnose():
    """Run comprehensive RAG diagnostics."""
    print("\n" + "="*70)
    print("🔍 RAG SYSTEM DIAGNOSTIC")
    print("="*70 + "\n")
    
    # 1. Check database
    print("1️⃣ Checking Database...")
    db_stats = db.get_database_stats()
    print(f"   Total papers in DB: {db_stats.get('total_papers', 0)}")
    print(f"   Total sections: {db_stats.get('total_sections', 0)}")
    
    if db_stats.get('total_papers', 0) == 0:
        print("   ❌ No papers in database!")
        print("   → Process some papers first\n")
        return
    
    # 2. Check vector database
    print("\n2️⃣ Checking Vector Database...")
    vector_stats = vector_db.get_statistics()
    print(f"   Total chunks indexed: {vector_stats.get('total_chunks', 0)}")
    print(f"   Unique papers indexed: {vector_stats.get('unique_papers', 0)}")
    
    # 3. Check knowledge graph
    print("\n3️⃣ Checking Knowledge Graph...")
    graph_stats = knowledge_graph.get_statistics()
    print(f"   Total nodes: {graph_stats.get('total_nodes', 0)}")
    print(f"   Paper nodes: {graph_stats.get('paper_nodes', 0)}")
    print(f"   Author nodes: {graph_stats.get('author_nodes', 0)}")
    print(f"   Concept nodes: {graph_stats.get('concept_nodes', 0)}")
    
    # 4. Compare indexing status
    print("\n4️⃣ Indexing Status...")
    papers_in_db = db_stats.get('total_papers', 0)
    papers_in_vector = vector_stats.get('unique_papers', 0)
    papers_in_graph = graph_stats.get('paper_nodes', 0)
    
    print(f"   Papers in Database: {papers_in_db}")
    print(f"   Papers in Vector DB: {papers_in_vector}")
    print(f"   Papers in Knowledge Graph: {papers_in_graph}")
    
    if papers_in_vector < papers_in_db:
        print(f"\n   ⚠️  WARNING: {papers_in_db - papers_in_vector} papers not indexed in Vector DB!")
        print("   → Run reindexing\n")
    
    if papers_in_graph < papers_in_db:
        print(f"\n   ⚠️  WARNING: {papers_in_db - papers_in_graph} papers not in Knowledge Graph!")
        print("   → Run reindexing\n")
    
    # 5. Test search
    print("\n5️⃣ Testing Search...")
    if papers_in_vector > 0:
        try:
            results = vector_db.search("machine learning", top_k=3)
            print(f"   ✅ Search returned {len(results)} results")
            
            if results:
                print(f"\n   Sample result:")
                print(f"   Title: {results[0]['metadata'].get('title', 'N/A')[:60]}...")
                print(f"   Relevance: {results[0].get('relevance_score', 0):.2f}")
        except Exception as e:
            print(f"   ❌ Search failed: {e}")
    else:
        print("   ⚠️  No papers indexed, skipping search test")
    
    # 6. List papers that need indexing
    print("\n6️⃣ Papers Status...")
    jobs = db.get_recent_jobs(limit=10)
    
    for job in jobs:
        papers = db.get_papers_by_job(job['id'])
        print(f"\n   Job #{job['id']}: {job['topic']}")
        print(f"   Total papers: {len(papers)}")
        
        for paper in papers:
            compiled_path = paper.get('compiled_json_path')
            status = "✅" if compiled_path and os.path.exists(compiled_path) else "❌"
            print(f"      {status} Paper {paper['id']}: {paper['title'][:50]}")
    
    print("\n" + "="*70)
    print("RECOMMENDATIONS:")
    print("="*70)
    
    if papers_in_vector == 0 and papers_in_db > 0:
        print("❗ No papers indexed in vector database")
        print("   Run: python reindex_all.py")
    elif papers_in_vector < papers_in_db:
        print("⚠️  Some papers not indexed")
        print("   Run: python reindex_all.py")
    else:
        print("✅ All systems nominal")
        print("   Try asking questions in the web interface!")
    
    print()

def reindex_all_papers():
    """Manually reindex all papers."""
    print("\n" + "="*70)
    print("🔄 REINDEXING ALL PAPERS")
    print("="*70 + "\n")
    
    # Get all papers
    jobs = db.get_recent_jobs(limit=100)
    all_papers = []
    
    for job in jobs:
        papers = db.get_papers_by_job(job['id'])
        all_papers.extend(papers)
    
    print(f"Found {len(all_papers)} papers to index\n")
    
    indexed_count = 0
    skipped_count = 0
    error_count = 0
    
    for i, paper in enumerate(all_papers, 1):
        print(f"[{i}/{len(all_papers)}] Processing: {paper['title'][:50]}...")
        
        compiled_path = paper.get('compiled_json_path')
        
        if not compiled_path or not os.path.exists(compiled_path):
            print(f"   ⚠️  No compiled data found, skipping")
            skipped_count += 1
            continue
        
        try:
            # Load compiled data
            with open(compiled_path, 'r', encoding='utf-8') as f:
                paper_data = json.load(f)
            
            # Index in vector database
            chunks = vector_db.index_paper(paper['id'], paper_data)
            print(f"   ✅ Indexed {chunks} chunks in Vector DB")
            
            # Add to knowledge graph
            knowledge_graph.add_paper(paper['id'], paper_data)
            print(f"   ✅ Added to Knowledge Graph")
            
            # Link citations
            if paper_data.get('references'):
                links = knowledge_graph.link_citations(paper['id'], paper_data['references'])
                if links > 0:
                    print(f"   ✅ Created {links} citation links")
            
            indexed_count += 1
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            error_count += 1
    
    # Save knowledge graph
    try:
        knowledge_graph.save_graph()
        print("\n✅ Knowledge graph saved")
    except Exception as e:
        print(f"\n❌ Error saving graph: {e}")
    
    print("\n" + "="*70)
    print("REINDEXING COMPLETE")
    print("="*70)
    print(f"✅ Successfully indexed: {indexed_count}")
    print(f"⚠️  Skipped: {skipped_count}")
    print(f"❌ Errors: {error_count}")
    print()
    
    # Show final stats
    vector_stats = vector_db.get_statistics()
    graph_stats = knowledge_graph.get_statistics()
    
    print("📊 FINAL STATISTICS:")
    print(f"   Vector DB chunks: {vector_stats.get('total_chunks', 0)}")
    print(f"   Vector DB papers: {vector_stats.get('unique_papers', 0)}")
    print(f"   Knowledge Graph nodes: {graph_stats.get('total_nodes', 0)}")
    print(f"   Knowledge Graph papers: {graph_stats.get('paper_nodes', 0)}")
    print()

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "reindex":
        reindex_all_papers()
    else:
        diagnose()
        
        print("\nTo reindex all papers, run:")
        print("  python diagnose_rag.py reindex")