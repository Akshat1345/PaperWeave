# debug_rag.py - Debug RAG System
"""
Run this script to diagnose RAG issues.

Usage:
    python debug_rag.py
"""

import sys
from modules.database import db
from modules.vector_db import vector_db
from modules.knowledge_graph import knowledge_graph
from config import config

def print_section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")

def check_database():
    """Check database status."""
    print_section("DATABASE CHECK")
    
    try:
        stats = db.get_database_stats()
        print(f"✅ Total papers: {stats.get('total_papers', 0)}")
        print(f"✅ Total sections: {stats.get('total_sections', 0)}")
        print(f"✅ Papers with contributions: {stats.get('papers_with_contributions', 0)}")
        
        # Get recent jobs
        jobs = db.get_recent_jobs(limit=5)
        print(f"\n📋 Recent jobs:")
        for job in jobs:
            print(f"  Job {job['id']}: {job['topic']} - {job['status']} ({job['num_papers']} papers)")
            
            # Get papers for this job
            papers = db.get_papers_by_job(job['id'])
            print(f"    Papers in DB: {len(papers)}")
            for p in papers[:3]:
                print(f"      - Paper {p['id']}: {p['title'][:50]}...")
        
        return True
    except Exception as e:
        print(f"❌ Database error: {e}")
        return False

def check_vector_db():
    """Check vector database status."""
    print_section("VECTOR DATABASE CHECK")
    
    try:
        stats = vector_db.get_statistics()
        print(f"✅ Total chunks: {stats.get('total_chunks', 0)}")
        print(f"✅ Unique papers: {stats.get('unique_papers', 0)}")
        print(f"✅ Unique jobs: {stats.get('unique_jobs', 0)}")
        print(f"✅ Avg chunks/paper: {stats.get('avg_chunks_per_paper', 0):.1f}")
        
        # Debug collection
        print("\n🔍 Collection sample:")
        vector_db.debug_collection()
        
        return stats.get('total_chunks', 0) > 0
    except Exception as e:
        print(f"❌ Vector DB error: {e}")
        return False

def check_knowledge_graph():
    """Check knowledge graph status."""
    print_section("KNOWLEDGE GRAPH CHECK")
    
    try:
        stats = knowledge_graph.get_statistics()
        print(f"✅ Total nodes: {stats.get('total_nodes', 0)}")
        print(f"✅ Total edges: {stats.get('total_edges', 0)}")
        print(f"✅ Paper nodes: {stats.get('paper_nodes', 0)}")
        print(f"✅ Author nodes: {stats.get('author_nodes', 0)}")
        print(f"✅ Concept nodes: {stats.get('concept_nodes', 0)}")
        
        return True
    except Exception as e:
        print(f"❌ Knowledge graph error: {e}")
        return False

def test_search(job_id=None):
    """Test vector search."""
    print_section("SEARCH TEST")
    
    if not job_id:
        jobs = db.get_recent_jobs(limit=1)
        if not jobs:
            print("❌ No jobs found")
            return False
        job_id = jobs[0]['id']
    
    print(f"Testing with job_id: {job_id}")
    
    # Get papers for this job
    papers = db.get_papers_by_job(job_id)
    paper_ids = [p['id'] for p in papers]
    
    print(f"Papers in job: {paper_ids}")
    
    # Test queries
    test_queries = [
        "methodology",
        "approach",
        "method",
        "neural network",
        "transformer"
    ]
    
    for query in test_queries:
        print(f"\n🔍 Query: '{query}'")
        
        # Search WITHOUT filter
        results_no_filter = vector_db.search(query, top_k=5)
        print(f"  Without filter: {len(results_no_filter)} results")
        
        # Search WITH filter
        results_with_filter = vector_db.search(query, top_k=5, filter_paper_ids=paper_ids)
        print(f"  With filter: {len(results_with_filter)} results")
        
        if results_with_filter:
            top = results_with_filter[0]
            print(f"  Top result: {top['relevance_score']:.3f} - {top['metadata'].get('title', 'Unknown')[:50]}")
        else:
            print(f"  ⚠️ No results with filter!")
    
    return True

def test_rag_query(job_id=None):
    """Test full RAG query."""
    print_section("RAG QUERY TEST")
    
    from modules.rag_engine import rag_engine
    
    if not job_id:
        jobs = db.get_recent_jobs(limit=1)
        if not jobs:
            print("❌ No jobs found")
            return False
        job_id = jobs[0]['id']
    
    test_questions = [
        "What methodology is used?",
        "Summarize the approach",
        "What are the main contributions?"
    ]
    
    for question in test_questions:
        print(f"\n❓ Question: {question}")
        
        try:
            result = rag_engine.query(question, job_id=job_id)
            
            print(f"  Confidence: {result.get('confidence', 'unknown')}")
            print(f"  Sources: {len(result.get('sources', []))}")
            print(f"  Answer: {result.get('answer', 'No answer')[:200]}...")
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
    
    return True

def fix_recommendations():
    """Provide fix recommendations."""
    print_section("RECOMMENDATIONS")
    
    stats = vector_db.get_statistics()
    
    if stats.get('total_chunks', 0) == 0:
        print("⚠️ CRITICAL: Vector DB is empty!")
        print("\n🔧 SOLUTION:")
        print("  1. Go to http://localhost:5000")
        print("  2. Process some papers (3-5 papers recommended)")
        print("  3. After processing completes, click 'Reindex Papers'")
        print("  4. Wait for success message")
        print("  5. Try asking questions again")
        return
    
    if stats.get('unique_papers', 0) < 3:
        print("⚠️ WARNING: Very few papers indexed")
        print("\n🔧 RECOMMENDATION: Process at least 3-5 papers for better results")
        return
    
    # Check if threshold is too strict
    print(f"Current similarity threshold: {config.RAG_SIMILARITY_THRESHOLD}")
    if config.RAG_SIMILARITY_THRESHOLD > 0.4:
        print("⚠️ Threshold might be too strict")
        print("\n🔧 TRY: Lower threshold in config.py:")
        print("  RAG_SIMILARITY_THRESHOLD: float = 0.25")
    
    print("\n✅ System looks healthy!")
    print("\n💡 Tips for better results:")
    print("  - Use specific questions related to your topic")
    print("  - Try simpler queries first: 'What is the methodology?'")
    print("  - Make sure papers are related to your question")

def main():
    """Main debug function."""
    print("\n" + "="*60)
    print("  🔍 RAG SYSTEM DIAGNOSTIC TOOL")
    print("="*60)
    
    # Check all components
    db_ok = check_database()
    vector_ok = check_vector_db()
    graph_ok = check_knowledge_graph()
    
    print_section("OVERALL STATUS")
    
    if db_ok:
        print("✅ Database: OK")
    else:
        print("❌ Database: FAILED")
    
    if vector_ok:
        print("✅ Vector DB: OK")
    else:
        print("❌ Vector DB: EMPTY or FAILED")
    
    if graph_ok:
        print("✅ Knowledge Graph: OK")
    else:
        print("❌ Knowledge Graph: FAILED")
    
    # Run tests if possible
    if db_ok and vector_ok:
        print("\n" + "="*60)
        response = input("Run search tests? (y/n): ").strip().lower()
        if response == 'y':
            test_search()
        
        response = input("\nRun RAG query tests? (y/n): ").strip().lower()
        if response == 'y':
            test_rag_query()
    
    # Provide recommendations
    fix_recommendations()
    
    print("\n" + "="*60)
    print("  Diagnostic complete!")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()