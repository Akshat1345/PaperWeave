#!/usr/bin/env python3
# force_reindex.py - Force complete reindexing with verbose output
"""
Force reindex all papers with detailed logging.
"""

import os
import json
import sys

print("🔄 FORCE REINDEXING - Loading modules...")

from modules.database import db
from modules.vector_db import vector_db
from modules.knowledge_graph import knowledge_graph

print("\n" + "="*70)
print("🔄 FORCE COMPLETE REINDEXING")
print("="*70 + "\n")

# Get all papers
print("1️⃣ Fetching papers from database...")
jobs = db.get_recent_jobs(limit=100)
all_papers = []

for job in jobs:
    papers = db.get_papers_by_job(job['id'])
    all_papers.extend(papers)

print(f"   Found {len(all_papers)} total papers\n")

if len(all_papers) == 0:
    print("❌ No papers in database!")
    sys.exit(1)

# Check compiled files
print("2️⃣ Checking compiled files...")
papers_with_data = []
for paper in all_papers:
    if paper.get('compiled_json_path') and os.path.exists(paper['compiled_json_path']):
        papers_with_data.append(paper)
        print(f"   ✅ {paper['title'][:50]}")
    else:
        print(f"   ⚠️  No data: {paper['title'][:50]}")

print(f"\n   Papers with compiled data: {len(papers_with_data)}\n")

if len(papers_with_data) == 0:
    print("❌ No papers have compiled data!")
    sys.exit(1)

# Start indexing
print("3️⃣ Starting indexing process...\n")

indexed_count = 0
total_chunks = 0

for i, paper in enumerate(papers_with_data, 1):
    print(f"[{i}/{len(papers_with_data)}] Processing: {paper['title'][:60]}")
    print(f"   Paper ID: {paper['id']}")
    print(f"   ArXiv ID: {paper['arxiv_id']}")
    print(f"   Compiled: {paper['compiled_json_path']}")
    
    try:
        # Load data
        print(f"   📖 Loading compiled data...")
        with open(paper['compiled_json_path'], 'r', encoding='utf-8') as f:
            paper_data = json.load(f)
        
        # Check what's in the data
        print(f"   📊 Data contains:")
        print(f"      - Sections: {len(paper_data.get('sections_text', {}))}")
        print(f"      - Abstract: {'✅' if paper_data.get('metadata', {}).get('abstract') else '❌'}")
        print(f"      - Contributions: {'✅' if paper_data.get('contributions') else '❌'}")
        
        # Index in Vector DB
        print(f"   🔍 Indexing in Vector Database...")
        try:
            chunks = vector_db.index_paper(paper['id'], paper_data)
            print(f"   ✅ Indexed {chunks} chunks")
            total_chunks += chunks
        except Exception as ve:
            print(f"   ❌ Vector DB error: {ve}")
            continue
        
        # Add to Knowledge Graph
        print(f"   🕸️  Adding to Knowledge Graph...")
        try:
            knowledge_graph.add_paper(paper['id'], paper_data)
            print(f"   ✅ Added to graph")
        except Exception as ge:
            print(f"   ❌ Graph error: {ge}")
        
        # Link citations
        if paper_data.get('references'):
            print(f"   🔗 Linking {len(paper_data['references'])} references...")
            try:
                links = knowledge_graph.link_citations(paper['id'], paper_data['references'])
                if links > 0:
                    print(f"   ✅ Created {links} citation links")
            except Exception as ce:
                print(f"   ❌ Citation error: {ce}")
        
        indexed_count += 1
        print(f"   ✅ COMPLETE\n")
        
    except Exception as e:
        print(f"   ❌ ERROR: {e}\n")
        import traceback
        traceback.print_exc()
        continue

# Save graph
print("\n4️⃣ Saving knowledge graph...")
try:
    knowledge_graph.save_graph()
    print("   ✅ Saved\n")
except Exception as e:
    print(f"   ❌ Error: {e}\n")

# Final stats
print("="*70)
print("INDEXING COMPLETE")
print("="*70)
print(f"✅ Successfully indexed: {indexed_count}/{len(papers_with_data)}")
print(f"📊 Total chunks created: {total_chunks}")

# Verify
print("\n" + "="*70)
print("VERIFICATION")
print("="*70)

vector_stats = vector_db.get_statistics()
graph_stats = knowledge_graph.get_statistics()

print(f"\nVector Database:")
print(f"  Total chunks: {vector_stats.get('total_chunks', 0)}")
print(f"  Unique papers: {vector_stats.get('unique_papers', 0)}")
print(f"  Avg chunks/paper: {vector_stats.get('avg_chunks_per_paper', 0):.1f}")

print(f"\nKnowledge Graph:")
print(f"  Total nodes: {graph_stats.get('total_nodes', 0)}")
print(f"  Paper nodes: {graph_stats.get('paper_nodes', 0)}")
print(f"  Author nodes: {graph_stats.get('author_nodes', 0)}")
print(f"  Concept nodes: {graph_stats.get('concept_nodes', 0)}")

if vector_stats.get('total_chunks', 0) > 0:
    print(f"\n✅ SUCCESS! Vector DB is populated")
    print(f"\nNow you can:")
    print(f"  1. Start Flask: python app.py")
    print(f"  2. Go to http://localhost:5000")
    print(f"  3. Ask RAG questions!")
else:
    print(f"\n❌ FAILED! Vector DB still empty")
    print(f"   Check errors above")

print()