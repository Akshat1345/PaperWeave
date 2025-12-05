# Quick Reference Card

## 🚀 Start System

### macOS/Linux
```bash
chmod +x start.sh
./start.sh
```

### Windows
```bash
start.bat
```

### Manual Start
```bash
# Terminal 1: Start Ollama
ollama serve

# Terminal 2: Start Flask
python app.py

# Browser: http://localhost:5000
```

---

## 📖 Workflow

### 1. Search Papers
```
Topic: "Deep Learning in Medical Imaging"
Number: 10
Click: "Start Processing"
```

### 2. Wait for Completion
```
Processing:
  10% Searching arXiv...
  30% Downloading papers...
  60% Compiling with LLM...
  95% Generating surveys...
  100% Done!
```

### 3. View Results
```
URL: http://localhost:5000/results?job_id=1
Tabs:
  - Overview: Statistics
  - Papers: All papers + surveys
  - RAG Q&A: Ask questions
```

### 4. Ask Questions
```
Examples:
  "What methodologies are used?"
  "What challenges exist?"
  "Compare approaches"
  "Future research directions?"
```

---

## 🔧 Configuration

### Edit `config.py`

#### RAG Tuning
```python
RAG_TOP_K_RESULTS = 15              # More results
RAG_SIMILARITY_THRESHOLD = 0.35     # Lower = more results
RAG_TEMPERATURE = 0.2               # Lower = focused answers
```

#### Performance
```python
CHUNK_SIZE = 600                    # Smaller = faster
CHUNK_OVERLAP = 100
RAG_INITIAL_RETRIEVAL = 30
```

---

## 🛠️ Troubleshooting

### RAG Not Working
```bash
# Check vector DB status
curl http://localhost:5000/rag/index_status

# Reindex papers
curl -X POST http://localhost:5000/rag/reindex

# Check logs
tail -f research_assistant.log
```

### Ollama Issues
```bash
# Check if running
ollama ps

# Pull model if needed
ollama pull llama3.2

# Check models
ollama list
```

### Database Issues
```bash
# Reset database (WARNING: Deletes all data)
rm research_assistant.db
python -c "from modules.database import db"

# Check database
sqlite3 research_assistant.db ".tables"
```

---

## 📊 API Quick Reference

### Search Papers
```
POST /start_processing
{
  "topic": "Deep Learning",
  "num_papers": 10
}
```

### Ask Question
```
POST /rag/query
{
  "question": "What are the main challenges?"
}
```

### Generate Surveys
```
POST /surveys/generate
{
  "job_id": 1
}
```

### Get Results
```
GET /results/comprehensive?job_id=1
```

### Check Status
```
GET /rag/index_status
GET /status
```

---

## 📁 Important Files

```
app.py                     Main application
config.py                  Configuration
modules/
  hybrid_rag.py           RAG engine
  survey_generator.py     Survey generation
  database.py             Database
  vector_db.py            Embeddings
  knowledge_graph.py      Paper relationships
templates/
  index.html              Main page
  results.html            Results page
research_assistant.db     Data (SQLite)
```

---

## 💾 Data Locations

```
Downloaded PDFs:           data/pdfs/
Compiled JSON:            processed/compiled/
Vector embeddings:        processed/chroma_db/
Images extracted:         processed/images/
Knowledge graph:          processed/knowledge_graph.pkl
Database:                 research_assistant.db
Logs:                     research_assistant.log
Cache:                    processed/cache/
```

---

## ⚡ Performance Tips

### Faster Processing
```python
# In config.py
CHUNK_SIZE = 400              # Smaller
PAGE_LIMIT = 15               # Fewer pages
RAG_TOP_K_RESULTS = 10        # Fewer results
```

### Better Results
```python
# In config.py
CHUNK_SIZE = 800              # Larger
CHUNK_OVERLAP = 150           # More overlap
RAG_TOP_K_RESULTS = 25        # More results
RAG_TEMPERATURE = 0.1         # More focused
```

### Memory Optimization
```python
# Process fewer papers
# Reduce chunk size
# Use batch mode
# Clear cache periodically
```

---

## 🎓 Query Examples

### Methodology Questions
```
"What are the main deep learning architectures used?"
"How do CNNs differ from RNNs?"
"Describe the training procedures mentioned"
```

### Gap Analysis
```
"What limitations are mentioned?"
"What future work is suggested?"
"What open problems remain?"
"What challenges are not solved?"
```

### Comparison
```
"Compare supervised vs unsupervised learning"
"What are the differences between methods?"
"Which approaches are most effective?"
```

### General
```
"Summarize the key findings"
"What datasets are used?"
"What evaluation metrics are reported?"
"What are the main contributions?"
```

---

## 🔄 Hybrid RAG Components

### BM25
- Keyword-based retrieval
- Good for exact terms
- Fast
- No neural network needed

### Semantic Search
- Meaning-based retrieval
- Finds conceptual matches
- Slower but smarter
- Uses embeddings

### Reciprocal Rank Fusion (RRF)
- Combines BM25 + Semantic
- Optimal fusion
- Formula: 1/(k+rank)

### Cross-Encoder Reranking
- LLM-based reranking
- Human-quality ranking
- Final accuracy boost

---

## 📝 Literature Survey Sections

Each paper gets 5 auto-generated sections:

1. **Related Work & Context**
   - Historical background
   - Problem space
   - Positioning

2. **Methodology Survey**
   - Technical approach
   - Algorithms
   - Innovations

3. **Key Contributions**
   - Main contributions
   - Key results
   - Uniqueness

4. **Research Gaps & Future Work**
   - Limitations
   - Open problems
   - Future directions

5. **Context Analysis**
   - Field positioning
   - Impact/influence
   - Audience relevance

---

## 🎯 Common Workflows

### Create Literature Review (30 min)
```
1. Search for topic (5 min)
2. Process 10-20 papers (20 min)
3. Review auto-generated surveys (5 min)
4. Download results
```

### Find Research Gaps (10 min)
```
1. Process papers
2. Query: "What gaps exist?"
3. Review results
4. Export findings
```

### Compare Methodologies (15 min)
```
1. Process papers
2. Query: "Compare approaches"
3. Review comparison
4. Export analysis
```

### Write Related Work (20 min)
```
1. Process papers
2. Get surveys
3. Query for context
4. Compile into paper
```

---

## 🔑 Key Shortcuts

### Skip Steps (for testing)
```bash
# Just search, no compilation
# Just compile, no surveys
# Just index, no queries
# Edit config.py to disable steps
```

### Manual Commands
```bash
# Generate surveys for job 1
curl -X POST http://localhost:5000/surveys/generate -H "Content-Type: application/json" -d '{"job_id": 1}'

# Reindex papers
curl -X POST http://localhost:5000/rag/reindex

# Get comprehensive results
curl http://localhost:5000/results/comprehensive?job_id=1
```

---

## 📊 Monitoring

### Check Processing
```
http://localhost:5000/status
```

### Check RAG Status
```
http://localhost:5000/rag/index_status
```

### View Statistics
```
http://localhost:5000/stats
```

### Job History
```
http://localhost:5000/jobs/history?limit=50
```

---

## 💡 Pro Tips

1. **Start small** - Process 3-5 papers first
2. **Monitor logs** - `tail -f research_assistant.log`
3. **Save results** - Download ZIP after each job
4. **Tweak config** - Small changes big impact
5. **Reindex often** - Fresh embeddings = better queries
6. **Use specific queries** - Better answers
7. **Check confidence** - High > Medium > Low
8. **Explore surveys** - Tons of info there

---

## 🚨 Emergency Procedures

### System Won't Start
```
1. Check Ollama: ollama serve
2. Check Python: python --version
3. Check logs: research_assistant.log
4. Reinstall deps: pip install -r requirements.txt
5. Reset DB: rm research_assistant.db
```

### Query Returns Nothing
```
1. Reindex: /rag/reindex
2. Check vector DB: /rag/index_status
3. Verify papers compiled
4. Try simpler query
5. Lower similarity_threshold
```

### Out of Memory
```
1. Reduce CHUNK_SIZE to 400
2. Process fewer papers (3-5)
3. Increase system RAM
4. Restart app
5. Clear cache
```

### Surveys Not Generating
```
1. Check Ollama: ollama ps
2. Check logs: research_assistant.log
3. Verify papers are compiled
4. Try manually: /surveys/generate
5. Check model: ollama list
```

---

## 📚 Resources

- **SETUP_GUIDE.md** - Complete installation guide
- **TECHNICAL_DOCS.md** - Architecture & algorithms
- **IMPLEMENTATION_SUMMARY.md** - What's been built
- **config.py** - All configuration options
- **research_assistant.log** - Debug information

---

## 🎓 Learning Resources

### Understanding RAG
- Read: TECHNICAL_DOCS.md - Hybrid RAG Deep Dive
- Watch YouTube: "RAG systems explained"
- Try: Different RAG queries

### Understanding BM25
- Formula in TECHNICAL_DOCS.md
- Adjust k1 and b parameters
- See difference in results

### Tuning Performance
- Modify config.py
- Run benchmark queries
- Monitor latency
- Adjust until satisfied

---

## ✅ Checklist Before Running

- [ ] Python 3.10+ installed
- [ ] Ollama installed and running
- [ ] Model downloaded (`ollama list`)
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] NLTK data downloaded
- [ ] spaCy model downloaded
- [ ] Disk space available (~1GB minimum)
- [ ] RAM available (4GB+ recommended)

---

**Last Updated**: December 2025
**Version**: 2.0
**Status**: Production Ready ✅

Questions? Check TECHNICAL_DOCS.md or SETUP_GUIDE.md
