# Technical Architecture - AI Research Assistant

## System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     AI Research Assistant                        │
│                                                                   │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │   Frontend   │───▶│   Flask API  │◀──▶│  Database    │      │
│  │  (HTML/JS)   │    │  (app.py)    │    │ (SQLite)     │      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
│                             ▲                                     │
│        ┌────────────────────┼────────────────────┐              │
│        │                    │                    │              │
│   ┌────▼─────┐         ┌───▼────┐         ┌─────▼────┐         │
│   │ Scraper  │         │Compiler│         │ Hybrid   │         │
│   │ (arXiv)  │         │ (LLM)  │         │ RAG      │         │
│   └──────────┘         └────────┘         └──────────┘         │
│        ▲                    ▲                    ▲              │
│   ┌────┴──────┐         ┌───┴──────┐      ┌────┴───────┐      │
│   │ PDFs      │         │JSON      │      │ BM25 +     │      │
│   │Downloads  │         │Compiled  │      │ Semantic   │      │
│   └───────────┘         └──────────┘      └────────────┘      │
│                                                                  │
│   ┌──────────────┐    ┌──────────────┐                        │
│   │ Vector DB    │    │  Knowledge   │                        │
│   │ (ChromaDB)   │    │  Graph       │                        │
│   └──────────────┘    │  (NetworkX)  │                        │
│                       └──────────────┘                        │
│                                                                 │
│   ┌──────────────────────────────────────┐                   │
│   │  Survey Generator (IEEE Format)      │                   │
│   │  - Related Work                      │                   │
│   │  - Methodology                       │                   │
│   │  - Contributions                     │                   │
│   │  - Gaps & Future Work                │                   │
│   └──────────────────────────────────────┘                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Data Flow Architecture

### 1. Paper Processing Pipeline

```
arXiv Search
    ↓
┌───────────────────────┐
│ Scraper Module        │ Downloads PDF files
│ (scraper.py)          │
└───────────────────────┘
    ↓ [PDFs stored]
┌───────────────────────┐
│ Compiler Agent        │ Extracts content
│ (compiler.py)         │ using Ollama LLM
│ - Text extraction     │
│ - Section splitting   │
│ - Reference parsing   │
└───────────────────────┘
    ↓ [JSON compiled]
┌───────────────────────────────────────────┐
│ Database Storage                          │
│ - Metadata (title, authors, abstract)     │
│ - Sections (introduction, method, etc)    │
│ - Contributions (problem, innovation)     │
│ - References (citations)                  │
└───────────────────────────────────────────┘
    ↓
┌───────────────────────────────────────────┐
│ Vector Indexing                           │
│ - Chunk paper sections                    │
│ - Generate embeddings                     │
│ - Store in ChromaDB                       │
│ - Create knowledge graph                  │
└───────────────────────────────────────────┘
    ↓
┌───────────────────────────────────────────┐
│ Survey Generation                         │
│ - LLM generates 5-section surveys         │
│ - Store in database                       │
└───────────────────────────────────────────┘
```

### 2. Query Processing Pipeline

```
User Question
    ↓
┌─────────────────────────────────────────┐
│ Query Preprocessing (hybrid_rag.py)     │
│ - Extract keywords                      │
│ - Identify technical terms              │
│ - Expand query variants                 │
└─────────────────────────────────────────┘
    ↓
    ├─────────────────────┬─────────────────────┐
    ▼                     ▼                     ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│ BM25 Search  │  │ Semantic     │  │ Knowledge    │
│ (keyword)    │  │ Search       │  │ Graph        │
│              │  │ (embeddings) │  │ (relations)  │
│ Top 20       │  │ Top 20       │  │ Top 5        │
└──────────────┘  └──────────────┘  └──────────────┘
    │                   │                     │
    └───────────────────┼─────────────────────┘
                        ▼
            ┌──────────────────────────┐
            │ Reciprocal Rank Fusion   │
            │ (RRF - weighted combine) │
            │ Formula:                 │
            │ score = Σ(1/(k+rank))    │
            └──────────────────────────┘
                        ▼
            ┌──────────────────────────┐
            │ Cross-Encoder Reranking  │
            │ (LLM ranks top 15)       │
            │ Human-quality ranking    │
            └──────────────────────────┘
                        ▼
            ┌──────────────────────────┐
            │ Deduplication            │
            │ Remove redundant chunks  │
            │ Group by paper           │
            └──────────────────────────┘
                        ▼
            ┌──────────────────────────┐
            │ Context Building         │
            │ Format for LLM           │
            │ Add citations            │
            └──────────────────────────┘
                        ▼
            ┌──────────────────────────┐
            │ Answer Generation        │
            │ LLM creates response     │
            │ Add confidence score     │
            └──────────────────────────┘
                        ▼
            Answer + Sources + Confidence
```

---

## Hybrid RAG Deep Dive

### BM25 Algorithm

**Why BM25?**
- Probabilistic model used by search engines
- Excellent for keyword matching
- Handles term frequency normalization
- Document length normalization

**Formula:**
```
BM25(D, Q) = Σ(IDF(qi) * (f(qi, D) * (k1 + 1)) / (f(qi, D) + k1 * (1 - b + b * |D| / avgdl)))

Where:
- IDF = Inverse Document Frequency
- f(qi, D) = frequency of term qi in document D
- |D| = document length
- avgdl = average document length
- k1 = 1.5 (term saturation)
- b = 0.75 (length normalization)
```

### Semantic Search

**Why Semantic Search?**
- Captures meaning, not just keywords
- Finds similar concepts
- Handles synonyms
- Understands context

**Process:**
1. Query → Embedding (SentenceTransformer)
2. Document sections → Embeddings (pre-computed)
3. Similarity calculation (cosine distance)
4. Rank by similarity score

### Reciprocal Rank Fusion (RRF)

**Why RRF?**
- Combines strengths of both methods
- No parameter tuning needed
- Proven effective in fusion

**Formula:**
```
RRF_score = Σ(1 / (k + rank))

Where k = 60 (constant)
```

**Example:**
```
Query: "transformer neural networks"

BM25 Results:
1. Paper A (BM25 rank: 1)
2. Paper B (BM25 rank: 3)
3. Paper C (BM25 rank: 5)

Semantic Results:
1. Paper C (semantic rank: 1)
2. Paper A (semantic rank: 2)
3. Paper D (semantic rank: 4)

RRF Fusion:
Paper A: 1/(60+1) + 1/(60+2) = 0.0330
Paper C: 1/(60+5) + 1/(60+1) = 0.0313
Paper B: 1/(60+3) = 0.0161
Paper D: 1/(60+4) = 0.0160

Final Ranking: A, C, B, D
```

### Cross-Encoder Reranking

**Why Reranking?**
- LLM can judge relevance better
- Handles complex queries
- Considers full context

**Process:**
1. Take top 15 from RRF
2. Format for LLM evaluation
3. LLM ranks by relevance
4. Return reranked list

---

## Database Schema

### Tables

```sql
-- Processing Jobs
processing_jobs
├── id (PK)
├── topic
├── num_papers
├── status (pending, processing, completed, failed)
├── progress (0-100)
├── current_step
├── created_at
├── completed_at
└── error_message

-- Papers
papers
├── id (PK)
├── job_id (FK)
├── arxiv_id (UNIQUE)
├── title
├── authors (JSON array)
├── abstract
├── published_date
├── categories
├── pdf_url
├── pdf_path
├── citation_count
├── metadata_json
├── compiled_json_path
└── processing_status

-- Sections (for RAG)
paper_sections
├── id (PK)
├── paper_id (FK)
├── section_name
├── content (text)
├── summary
└── word_count

-- Contributions
paper_contributions
├── id (PK)
├── paper_id (FK)
├── main_problem
├── key_innovation
├── methodology
├── major_results
├── limitations
└── research_gaps

-- References
paper_references
├── id (PK)
├── paper_id (FK)
├── reference_index
├── authors
├── title
├── year
└── venue

-- Surveys (NEW)
paper_surveys
├── id (PK)
├── paper_id (FK, UNIQUE)
├── related_work (text)
├── methodology_survey (text)
├── contributions_summary (text)
├── research_gaps (text)
├── context_analysis (text)
├── full_survey_json
└── generated_at
```

### Indexes for Performance
```sql
idx_papers_arxiv_id        -- Fast paper lookup
idx_papers_job_id          -- Fast job filtering
idx_sections_paper_id      -- Fast section retrieval
idx_contributions_paper_id -- Fast contribution lookup
idx_surveys_paper_id       -- Fast survey access
```

---

## Vector Database (ChromaDB)

### Collection Structure

```
collection: research_papers
│
├── Documents
│   └── text chunks (500-1000 words)
│
├── Embeddings
│   └── 384-dim vectors (SentenceTransformer)
│
└── Metadata
    ├── paper_id
    ├── arxiv_id
    ├── title
    ├── section_type (abstract, intro, method, results, etc)
    ├── chunk_index
    └── priority (high/normal)
```

### Chunking Strategy

```
Paper Text
    ↓
[Split into sections]
    - Abstract (high priority)
    - Key contributions (high priority)
    - Other sections (normal priority)
    ↓
[Chunk with overlap]
- Chunk size: 600 words
- Overlap: 100 words
- Min chunk size: 50 words
    ↓
[Generate embeddings]
    ↓
[Store in ChromaDB]
    └─ Total chunks per paper: ~10-20
```

### Query Process

```
Query Text
    ↓
[Generate embedding]
    ↓
[Similarity search]
    ├─ Cosine similarity calculation
    └─ Distance to threshold check
    ↓
[Return top K results]
    ├─ Default K = 15
    └─ Minimum similarity = 0.35
```

---

## Knowledge Graph Structure

### Node Types

```
Nodes:
├── Paper nodes (paper_123)
│   └── Attributes: title, arxiv_id, year, abstract
│
├── Author nodes (author_name)
│   └── Attributes: name
│
└── Concept nodes (concept_machine_learning)
    └── Attributes: name

Edges:
├── authored (author → paper)
├── cites (paper → paper)
├── discusses (paper → concept)
├── co_authored (author → author)
└── related_concept (concept → concept)
```

### Usage

```
1. Paper Relationships
   paper_1 --cites--> paper_2

2. Topic Clustering
   Find papers discussing same concepts
   
3. Author Networks
   Find collaborating authors
   
4. Related Papers
   Papers with overlapping citations
```

---

## Survey Generation Pipeline

### Prompt Templates

#### 1. Related Work & Context
```
Input:
- Title, abstract
- Intro/background sections
- Key problem
- Innovations

Output: 
- Historical context
- Problem space
- Positioning vs prior work
- Novelty highlights
```

#### 2. Methodology Survey
```
Input:
- Method/approach sections
- Algorithms
- Techniques

Output:
- Core approach explanation
- Algorithm breakdown
- Novelty vs traditional
- Advantages
```

#### 3. Contributions Summary
```
Input:
- Main contributions
- Results achieved
- Innovation details

Output:
- Problem statement
- Key innovations (bullets)
- Major results
- Uniqueness statement
```

#### 4. Research Gaps
```
Input:
- Limitations mentioned
- Future work section
- Methodology gaps

Output:
- Explicit limitations
- Implied gaps
- Future directions
- Open problems
```

#### 5. Context Analysis
```
Input:
- Title, authors
- Citation count
- Abstract

Output:
- Field positioning
- Impact/influence
- Work classification
- Audience relevance
```

---

## Performance Characteristics

### Processing Time
```
Scraping 10 papers:      2-5 min
Compiling 10 papers:     10-15 min
Vector indexing:         3-5 min
Survey generation:       5-10 min
Total for 10 papers:     20-35 min
```

### Storage Requirements
```
Per paper (average):
- PDF:                   ~2 MB
- Compiled JSON:         ~0.5 MB
- Embeddings:            ~50 KB
- Survey:                ~20 KB
- Total per paper:       ~2.6 MB

For 100 papers:
- Total storage:         ~260 MB
- Database:              ~5 MB
```

### Query Response
```
BM25 search:             <100 ms
Semantic search:         <500 ms
RRF fusion:              <50 ms
Cross-encoder rerank:    2-3 sec
Total query time:        3-4 sec
```

---

## Error Handling Strategy

### Critical Errors
```
1. PDF Download Fails
   → Retry up to 3 times
   → Mark as failed
   → Continue with next paper

2. Compilation Error
   → Fall back to basic extraction
   → Skip LLM processing
   → Save partial data

3. Vector Indexing Error
   → Store without embeddings
   → Still queryable by basic search
   → Log for review

4. Survey Generation Error
   → Save blank template
   → User can regenerate later
   → No blocking
```

### Recovery Mechanisms
```
1. Partial failures
   → Process continues
   → Partial results saved
   → Manual review available

2. Database corruption
   → Auto-repair with backup
   → Fallback to latest snapshot
   → Validation on startup

3. OOM (Out of Memory)
   → Batch processing
   → Reduce chunk size
   → Process fewer papers at once
```

---

## Configuration Parameters

### RAG Tuning
```python
RAG_TOP_K_RESULTS = 15              # More = broader answers
RAG_SIMILARITY_THRESHOLD = 0.35     # Lower = more results
RAG_INITIAL_RETRIEVAL = 30          # Higher = better quality
RAG_ENABLE_QUERY_EXPANSION = True   # Multiple queries
RAG_ENABLE_RERANKING = True         # Cross-encoder reranking
RAG_MAX_CONTEXT_LENGTH = 4000       # Max context words
RAG_TEMPERATURE = 0.2               # Lower = more focused
```

### Chunking Tuning
```python
CHUNK_SIZE = 600                    # Larger = more context
CHUNK_OVERLAP = 100                 # Larger = more connections
MIN_CHUNK_SIZE = 50                 # Minimum viable chunk
```

### BM25 Parameters
```python
k1 = 1.5                            # Term saturation
b = 0.75                            # Length normalization
```

---

## Testing Strategy

### Unit Tests
```
test_bm25_indexing.py
├── Test tokenization
├── Test IDF calculation
└── Test BM25 ranking

test_vector_db.py
├── Test embedding generation
├── Test similarity search
└── Test metadata filtering

test_rag_query.py
├── Test query preprocessing
├── Test RRF fusion
└── Test answer generation
```

### Integration Tests
```
test_full_pipeline.py
├── Scrape → Compile → Index
├── Query → Answer
└── Survey generation
```

### Performance Tests
```
test_performance.py
├── Query latency
├── Memory usage
├── Disk I/O
└── Vector search speed
```

---

## Security Considerations

### Data Privacy
```
✓ All data stored locally
✓ No external API calls (except arXiv)
✓ No telemetry or tracking
✓ Database encryption optional
```

### Input Validation
```
✓ Query length limits
✓ File type validation
✓ PDF safety checks
✓ SQL injection prevention
```

### Resource Limits
```
✓ Max papers per job: 50
✓ Max query length: 500 chars
✓ Max file size: 100 MB
✓ Max concurrent jobs: 1
```

---

## Future Enhancements

### Planned Features
```
1. Multi-user support
2. Async query processing
3. Real-time indexing
4. Graph visualization UI
5. Paper recommendation
6. Cross-paper analysis
7. Batch survey generation
8. PDF annotation
```

### Optimization Ideas
```
1. GPU acceleration for embeddings
2. Caching for frequent queries
3. Incremental indexing
4. Parallel processing
5. Distributed vector DB
6. Query result caching
```

---

**Document Version**: 2.0
**Last Updated**: December 2025
**Status**: Complete
