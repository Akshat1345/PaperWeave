#!/usr/bin/env python
"""Test RAG with different job_id scenarios to replicate frontend behavior."""

from modules.hybrid_rag import hybrid_rag_engine
from modules.database import db
from modules.utils import logger
import json

def test_rag_with_job_filtering():
    """Test RAG with job_id to simulate frontend behavior."""
    
    print("\n" + "="*80)
    print("🧪 RAG JOB_ID FILTERING TEST")
    print("="*80)
    
    # Get latest job
    with db.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT id, topic FROM processing_jobs ORDER BY id DESC LIMIT 1')
        latest_job = cursor.fetchone()
        
        if not latest_job:
            print("No jobs found!")
            return
        
        job_id = latest_job['id']
        topic = latest_job['topic']
        print(f"\nLatest Job: ID={job_id}, Topic='{topic}'")
    
    test_query = "analyse all papers and tell me the research gap"
    
    print("\n" + "-"*80)
    print(f"Test Query: '{test_query}'")
    print("-"*80)
    
    # Test 1: No job_id (searches all papers)
    print("\n1️⃣  WITHOUT job_id filter (searches ALL papers):")
    try:
        result = hybrid_rag_engine.query(test_query, job_id=None)
        print(f"   ✓ Confidence: {result['confidence']}")
        print(f"   ✓ Sources: {len(result.get('sources', []))}")
        print(f"   ✓ Answer: {result['answer'][:100]}...")
    except Exception as e:
        print(f"   ❌ ERROR: {e}")
    
    # Test 2: With job_id (should only search papers from this job)
    print(f"\n2️⃣  WITH job_id={job_id} filter (searches ONLY this job's papers):")
    try:
        result = hybrid_rag_engine.query(test_query, job_id=job_id)
        print(f"   ✓ Confidence: {result['confidence']}")
        print(f"   ✓ Sources: {len(result.get('sources', []))}")
        print(f"   ✓ Answer: {result['answer'][:100]}...")
        
        # Check if sources belong to this job
        if result.get('sources'):
            job_papers = {}
            for source in result['sources']:
                paper_id = source.get('paper_id')
                if paper_id not in job_papers:
                    job_papers[paper_id] = 0
                job_papers[paper_id] += 1
            
            # Verify these papers belong to the job
            with db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT id, job_id FROM papers WHERE id IN ({})'.format(
                    ','.join(map(str, job_papers.keys()))
                ))
                papers = cursor.fetchall()
                
                jobs_in_results = set(p['job_id'] for p in papers)
                if len(jobs_in_results) > 1 or (len(jobs_in_results) == 1 and list(jobs_in_results)[0] != job_id):
                    print(f"   ⚠️  WARNING: Results from multiple jobs: {jobs_in_results}")
                else:
                    print(f"   ✅ All sources from job {job_id}")
        
    except Exception as e:
        print(f"   ❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 3: Check if job has papers indexed
    print(f"\n3️⃣  JOB {job_id} INDEXING STATUS:")
    try:
        with db.get_connection() as conn:
            cursor = conn.cursor()
            
            # Papers in job
            cursor.execute('SELECT COUNT(*) as count FROM papers WHERE job_id = ?', (job_id,))
            paper_count = cursor.fetchone()['count']
            print(f"   ✓ Papers in job: {paper_count}")
            
            # Sections in job
            cursor.execute('''
                SELECT COUNT(*) as count FROM paper_sections ps
                JOIN papers p ON ps.paper_id = p.id
                WHERE p.job_id = ?
            ''', (job_id,))
            section_count = cursor.fetchone()['count']
            print(f"   ✓ Sections in job: {section_count}")
            
            # Check if any of these sections are in BM25
            if section_count > 0:
                cursor.execute('''
                    SELECT DISTINCT ps.id FROM paper_sections ps
                    JOIN papers p ON ps.paper_id = p.id
                    WHERE p.job_id = ?
                    LIMIT 5
                ''', (job_id,))
                section_ids = [row['id'] for row in cursor.fetchall()]
                
                bm25_indexed = sum(1 for sid in section_ids if sid in hybrid_rag_engine.bm25.document_metadata)
                print(f"   ✓ Sections in BM25 index: {bm25_indexed}/{len(section_ids)}")
    except Exception as e:
        print(f"   ❌ ERROR: {e}")
    
    print("\n" + "="*80 + "\n")

if __name__ == '__main__':
    test_rag_with_job_filtering()
