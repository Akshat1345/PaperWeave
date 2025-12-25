#!/usr/bin/env python
"""Comprehensive RAG diagnostic to identify indexing and retrieval issues."""

import sqlite3
import json
from modules.database import db
from modules.vector_db import vector_db
from modules.knowledge_graph import knowledge_graph
from modules.hybrid_rag import hybrid_rag_engine
from modules.utils import logger

def diagnose_rag_system():
    """Run comprehensive diagnostics on the RAG system."""
    
    print("\n" + "="*80)
    print("🔍 RAG SYSTEM COMPREHENSIVE DIAGNOSTIC")
    print("="*80)
    
    # 1. Database Status
    print("\n1️⃣  DATABASE STATUS")
    print("-" * 80)
    with db.get_connection() as conn:
        cursor = conn.cursor()
        
        # Get all jobs
        cursor.execute('SELECT id, topic, created_at FROM processing_jobs ORDER BY id DESC LIMIT 5')
        jobs = cursor.fetchall()
        print(f"   Recent jobs: {len(jobs)}")
        for job in jobs:
            print(f"   - Job {job['id']}: {job['topic']}")
        
        # Get papers per job
        if jobs:
            latest_job_id = jobs[0]['id']
            cursor.execute('SELECT COUNT(*) as count FROM papers WHERE job_id = ?', (latest_job_id,))
            paper_count = cursor.fetchone()['count']
            print(f"\n   ✓ Latest job ({latest_job_id}): {paper_count} papers")
            
            # Get sections per job
            cursor.execute('''
                SELECT COUNT(*) as count FROM paper_sections ps
                JOIN papers p ON ps.paper_id = p.id
                WHERE p.job_id = ?
            ''', (latest_job_id,))
            section_count = cursor.fetchone()['count']
            print(f"   ✓ Paper sections: {section_count}")
            
            # Get surveys per job
            cursor.execute('''
                SELECT COUNT(DISTINCT p.id) as count FROM papers p
                LEFT JOIN paper_surveys ps ON p.id = ps.paper_id
                WHERE p.job_id = ? AND ps.id IS NOT NULL
            ''', (latest_job_id,))
            survey_count = cursor.fetchone()['count']
            print(f"   ✓ Papers with surveys: {survey_count}")
    
    # 2. Vector Database Status
    print("\n2️⃣  VECTOR DATABASE STATUS")
    print("-" * 80)
    try:
        vec_stats = vector_db.get_statistics()
        print(f"   ✓ Total chunks: {vec_stats.get('total_chunks', 0)}")
        print(f"   ✓ Unique papers: {vec_stats.get('unique_papers', 0)}")
        
        if vec_stats.get('total_chunks', 0) == 0:
            print("   ⚠️  WARNING: Vector DB is empty! Papers may not be indexed.")
        else:
            print("   ✅ Vector DB populated")
    except Exception as e:
        print(f"   ❌ ERROR: {e}")
    
    # 3. Knowledge Graph Status
    print("\n3️⃣  KNOWLEDGE GRAPH STATUS")
    print("-" * 80)
    try:
        kg_nodes = knowledge_graph.graph.number_of_nodes()
        kg_edges = knowledge_graph.graph.number_of_edges()
        print(f"   ✓ Graph nodes: {kg_nodes}")
        print(f"   ✓ Graph edges: {kg_edges}")
        
        if kg_nodes == 0:
            print("   ⚠️  WARNING: Knowledge graph is empty!")
        else:
            print("   ✅ Knowledge graph populated")
    except Exception as e:
        print(f"   ❌ ERROR: {e}")
    
    # 4. BM25 Index Status
    print("\n4️⃣  BM25 INDEX STATUS")
    print("-" * 80)
    try:
        bm25_docs = len(hybrid_rag_engine.bm25.documents)
        bm25_terms = len(hybrid_rag_engine.bm25.inverted_index)
        print(f"   ✓ BM25 documents: {bm25_docs}")
        print(f"   ✓ BM25 unique terms: {bm25_terms}")
        
        if bm25_docs == 0:
            print("   ⚠️  WARNING: BM25 index is empty!")
        else:
            print("   ✅ BM25 index populated")
    except Exception as e:
        print(f"   ❌ ERROR: {e}")
    
    # 5. Test Retrieval Pipeline
    print("\n5️⃣  RETRIEVAL PIPELINE TEST")
    print("-" * 80)
    test_query = "analyse all papers and tell me the research gap"
    print(f"   Testing query: '{test_query}'")
    
    try:
        # Test BM25 retrieval
        bm25_results = hybrid_rag_engine._retrieve_bm25(test_query, top_k=20, job_id=None)
        print(f"   ✓ BM25 retrieval: {len(bm25_results)} results")
        if bm25_results:
            print(f"     - Top result: {bm25_results[0]['metadata']['title'][:60]}")
        else:
            print("   ⚠️  BM25 returned no results")
        
        # Test Semantic retrieval
        semantic_results = hybrid_rag_engine._retrieve_semantic(test_query, top_k=20, job_id=None)
        print(f"   ✓ Semantic retrieval: {len(semantic_results)} results")
        if semantic_results:
            print(f"     - Top result: {semantic_results[0]['metadata']['title'][:60]}")
        else:
            print("   ⚠️  Semantic search returned no results")
        
        # Check if either has results
        if not bm25_results and not semantic_results:
            print("\n   🔴 CRITICAL: Both BM25 and Semantic search returned 0 results!")
            print("   This means papers are likely not indexed in the system.")
        
    except Exception as e:
        print(f"   ❌ ERROR during retrieval: {e}")
        import traceback
        traceback.print_exc()
    
    # 6. Full RAG Query Test
    print("\n6️⃣  FULL RAG QUERY TEST")
    print("-" * 80)
    try:
        result = hybrid_rag_engine.query(test_query, job_id=None)
        print(f"   ✓ Answer: {result['answer'][:150]}...")
        print(f"   ✓ Confidence: {result['confidence']}")
        print(f"   ✓ Sources: {len(result.get('sources', []))}")
        
        if result['confidence'] == 'low' or len(result.get('sources', [])) == 0:
            print("   🔴 RAG returned LOW confidence or no sources")
        else:
            print("   ✅ RAG working correctly")
    except Exception as e:
        print(f"   ❌ ERROR during RAG query: {e}")
        import traceback
        traceback.print_exc()
    
    # 7. Recommendations
    print("\n7️⃣  RECOMMENDATIONS")
    print("-" * 80)
    print("   If RAG is failing:")
    print("   - Check if papers were successfully processed (Database section)")
    print("   - Verify Vector DB has chunks indexed (Vector DB section)")
    print("   - Ensure Knowledge Graph was built (KG section)")
    print("   - Check BM25 index is populated (BM25 section)")
    print("   - If all above are ✅ but RAG still fails, check Ollama is running")
    
    print("\n" + "="*80 + "\n")

if __name__ == '__main__':
    diagnose_rag_system()
