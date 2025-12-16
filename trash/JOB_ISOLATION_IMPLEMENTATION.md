# Job Isolation Implementation

## Overview

This document describes the complete job isolation system implemented to ensure that each research topic/query processes only papers from its specific job, preventing cross-contamination between different topics in the database.

## Problem Statement

Previously, when processing a new topic (e.g., 3 papers on "quantum computing"), the RAG system would query ALL papers in the database (e.g., 20 papers from various previous jobs). This caused:

- Mixed results from unrelated topics
- Incorrect context in literature surveys
- No separation between different research queries
- Confusion for users expecting topic-specific results

## Solution Architecture

### 1. Database Foundation

The SQLite database already had a `job_id` column in the `papers` table, which uniquely identifies which processing job each paper belongs to. This serves as the foundation for isolation.

### 2. Vector Database (ChromaDB) Isolation

**File:** `modules/vector_db.py`

**Changes:**

- **Indexing (index_paper method):**

  - Added database query to fetch `job_id` for each paper
  - Added `'job_id': job_id` to metadata for ALL chunks:
    - Abstract chunks
    - Contributions chunks
    - Section chunks
  - This ensures every document in ChromaDB knows which job it belongs to

- **Search (search method):**
  - Added `filter_job_id` parameter (optional)
  - When provided, builds ChromaDB where clause: `where["job_id"] = filter_job_id`
  - Filters results to only return documents from the specified job

```python
# Example metadata now includes job_id
{
    'paper_id': 123,
    'job_id': 5,  # NEW: Enables job filtering
    'arxiv_id': '2301.00942v1',
    'title': 'Paper Title',
    'section_type': 'abstract',
    ...
}
```

### 3. BM25 Retriever Isolation

**File:** `modules/hybrid_rag.py` (BM25Retriever class)

**Changes:**

- **Index Loading (\_load_index method):**

  - Modified SQL query to fetch `job_id` from database
  - Added `'job_id': job_id` to document_metadata for all indexed documents
  - Ensures BM25 inverted index tracks job ownership

- **Search (search method):**
  - Added `job_id` parameter (optional)
  - After BM25 scoring, filters results to only include documents with matching job_id
  - Post-scoring filter: `if job_id is not None and metadata.get('job_id') != job_id: continue`

```python
# BM25 now filters after scoring
for idx, score in top_results:
    metadata = self.document_metadata[idx]
    if job_id is not None and metadata.get('job_id') != job_id:
        continue  # Skip documents from other jobs
    results.append(...)
```

### 4. Hybrid RAG Engine Integration

**File:** `modules/hybrid_rag.py` (HybridRAGEngine class)

**Changes:**

- **Query Method Signature:**

  - Updated `query(question: str, job_id: Optional[int] = None)` to require job_id parameter

- **Retrieval Methods:**

  - `_retrieve_bm25()`: Added job_id parameter, passes to BM25 search
  - `_retrieve_semantic()`: Added job_id parameter, passes to vector_db.search
  - Both retrievers now filter by job_id during retrieval

- **Context Enrichment:**
  - `_enrich_with_context()`: Added job_id parameter
  - Passes job_id to knowledge graph when finding related papers
  - Ensures related papers are also from the same job

### 5. Knowledge Graph Isolation

**File:** `modules/knowledge_graph.py`

**Changes:**

- **Paper Node Storage (add_paper method):**

  - Added database query to fetch job_id for each paper
  - Added `job_id` attribute to all paper nodes in the graph
  - Enables filtering when traversing graph relationships

- **Related Papers (find_related_papers method):**
  - Added `job_id` parameter (optional)
  - Auto-detects job_id from source paper if not provided
  - Filters ALL relationship types by job_id:
    - Papers that cite this paper
    - Papers cited by this paper
    - Papers by same authors
    - Papers with shared concepts
  - Each traversal checks: `if job_id is not None and node.get('job_id') != job_id: continue`

```python
# Knowledge graph now filters related papers by job
for paper in graph.successors(author):
    if job_id is not None and graph.nodes[paper].get('job_id') != job_id:
        continue  # Skip papers from other jobs
    related.append(paper)
```

### 6. API Endpoint Integration

**File:** `app.py`

**Changes:**

- **RAG Query Endpoint:**
  - Updated `/rag/query` to require `job_id` in request
  - Added validation: returns 400 error if job_id missing
  - Passes job_id to `hybrid_rag_engine.query()`

```python
@app.route('/rag/query', methods=['POST'])
def rag_query():
    job_id = data.get('job_id')
    if not job_id:
        return jsonify({'error': 'job_id is required'}), 400

    result = hybrid_rag_engine.query(question, job_id=job_id)
```

- **Frontend Integration (templates/index.html):**
  - Updated RAG query JavaScript to include `job_id` from `currentJobId` variable
  - Request body now includes: `{question: "...", job_id: 5}`

### 7. Automatic Indexing During Processing

**File:** `app.py` (process_papers_background function)

**Changes:**

- After successful compilation of each paper:
  - **Vector DB Indexing:** Automatically calls `vector_db.index_paper(paper_id, result)`
  - **Knowledge Graph:** Automatically calls:
    - `knowledge_graph.add_paper(paper_id, result)`
    - `knowledge_graph.link_citations(paper_id, result['references'])`
  - This ensures new papers are immediately available for RAG queries
  - No manual indexing required - happens automatically during processing

```python
# After saving paper sections, contributions, references:
try:
    # Index in vector DB
    chunks = vector_db.index_paper(paper_id, result)

    # Add to knowledge graph
    knowledge_graph.add_paper(paper_id, result)
    knowledge_graph.link_citations(paper_id, result['references'])
except Exception as e:
    logger.error(f"Error indexing: {e}")
```

## Data Flow

### When Processing New Papers (e.g., Job 5 with 3 papers):

1. **User submits:** Topic "quantum computing", 3 papers
2. **Backend creates:** Job ID = 5 in database
3. **Scraper downloads:** 3 papers, saved to DB with job_id=5
4. **Compiler processes:** Each paper extracts text, sections, contributions
5. **Automatic indexing:**
   - Vector DB: Stores all chunks with `{'job_id': 5, ...}`
   - BM25: Indexes documents with `{'job_id': 5, ...}`
   - Knowledge Graph: Adds nodes with `job_id=5`
6. **Ready for queries:** Papers from job 5 are now queryable

### When Querying (RAG):

1. **User asks:** "What are the key findings?" (with job_id=5 from frontend)
2. **BM25 retrieves:** Top-k documents, filters to only job_id=5 papers
3. **Semantic retrieves:** Top-k documents, ChromaDB filters to only job_id=5
4. **Fusion:** Combines both retrieval sets (all from job_id=5)
5. **Knowledge Graph:** Finds related papers, filters to only job_id=5
6. **Answer generation:** LLM receives context ONLY from the 3 papers in job 5
7. **Result:** Answer based solely on quantum computing papers, no contamination

## Benefits

1. **Complete Isolation:** Each job/topic processes independently
2. **No Cross-Contamination:** Papers from different topics never mix
3. **Accurate Results:** Answers only reference papers from the current query
4. **Scalable:** Can have hundreds of jobs in database, queries remain fast
5. **Automatic:** Indexing happens during processing, no manual steps
6. **Consistent:** All retrieval methods (BM25, semantic, graph) respect job boundaries

## Testing & Verification

To verify job isolation works:

1. Process 3 papers on Topic A (creates job_id=1)
2. Process 3 papers on Topic B (creates job_id=2)
3. Query job 1: Should only reference Topic A papers
4. Query job 2: Should only reference Topic B papers
5. Check vector DB: `vector_db.search(query, filter_job_id=1)` should return different results than `filter_job_id=2`
6. Check BM25: Similar filtering should work
7. Check knowledge graph: Related papers should be from same job

## Migration Notes

- **Existing Papers:** Old papers without job_id in metadata can still be accessed
  - Query without job_id parameter returns all papers (backward compatible)
  - Query with job_id only returns papers from that job
- **Reindexing:** If needed, use `force_reindex.py` to rebuild indexes with job_id metadata
- **Background Surveys:** `regenerate_surveys.py` should also filter by job_id when generating literature surveys (needs separate update if not already implemented)

## Files Modified

1. `modules/vector_db.py` - Added job_id to indexing and search
2. `modules/hybrid_rag.py` - Added job_id to BM25, semantic, and enrichment
3. `modules/knowledge_graph.py` - Added job_id to nodes and filtering
4. `app.py` - Added job_id to RAG endpoint and automatic indexing
5. `templates/index.html` - Added job_id to frontend RAG queries

## Summary

The complete job isolation system ensures that **"whenever I enter a topic and number of papers, it should only give output wrt to those papers and not all the papers in the database"**. Every retrieval component (BM25, semantic search, knowledge graph) now respects job boundaries, and automatic indexing ensures new papers are immediately available without cross-contamination.
