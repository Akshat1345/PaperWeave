# app.py - IMPROVED Flask Application
from flask import Flask, render_template, request, jsonify, send_file
import os
import json
import threading
import zipfile
from datetime import datetime
from config import config
from modules.scraper import ArxivScraper
from modules.compiler import CompilationAgent
from modules.database import db
from modules.vector_db import vector_db
from modules.knowledge_graph import knowledge_graph
from modules.hybrid_rag import hybrid_rag_engine
from modules.survey_generator import survey_generator
from modules.utils import logger, format_duration

app = Flask(__name__)

# Global variable for current job tracking
current_job = {
    'is_processing': False,
    'job_id': None,
    'start_time': None
}

# ========== ERROR HANDLERS ==========

@app.errorhandler(Exception)
def handle_error(error):
    """Global error handler."""
    logger.error(f"Unhandled exception: {error}", exc_info=True)
    return jsonify({
        'error': str(error),
        'type': type(error).__name__
    }), 500

@app.errorhandler(404)
def not_found(error):
    """404 handler."""
    return jsonify({'error': 'Resource not found'}), 404

# ========== ROUTES ==========

@app.route('/')
def index():
    """Main page."""
    return render_template('index.html')

@app.route('/results')
def results():
    """Results page with surveys and RAG."""
    return render_template('results.html')

@app.route('/start_processing', methods=['POST'])
def start_processing():
    """Start new processing job."""
    global current_job
    
    if current_job['is_processing']:
        return jsonify({
            'error': 'Processing already in progress',
            'current_job_id': current_job['job_id']
        }), 400
    
    try:
        data = request.json
        topic = data.get('topic', '').strip()
        num_papers = int(data.get('num_papers', 5))
        
        # Validation
        if not topic:
            return jsonify({'error': 'Topic is required'}), 400
        
        if not (1 <= num_papers <= 20):
            return jsonify({'error': 'Number of papers must be between 1 and 20'}), 400
        
        # Check Ollama connection
        compiler = CompilationAgent(config.DATA_DIR, config.PROCESSED_DIR)
        if not compiler.check_ollama_connection():
            return jsonify({
                'error': 'Ollama service not running',
                'hint': 'Run "ollama serve" in a separate terminal and ensure the model is downloaded'
            }), 503
        
        # Create job in database
        job_id = db.create_job(topic, num_papers)
        
        # Update global state
        current_job = {
            'is_processing': True,
            'job_id': job_id,
            'start_time': datetime.now()
        }
        
        logger.info(f"🚀 Starting job {job_id}: {topic} ({num_papers} papers)")
        
        # Start processing in background thread
        thread = threading.Thread(
            target=process_papers_background,
            args=(job_id, topic, num_papers)
        )
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'message': 'Processing started successfully',
            'job_id': job_id
        })
    
    except ValueError as e:
        return jsonify({'error': f'Invalid input: {str(e)}'}), 400
    except Exception as e:
        logger.error(f"Error starting processing: {e}", exc_info=True)
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/status')
def get_status():
    """Get current processing status."""
    if not current_job['is_processing']:
        # No active job - check if there's a completed recent job
        recent_jobs = db.get_recent_jobs(limit=1)
        if recent_jobs:
            latest_job = recent_jobs[0]
            return jsonify({
                'is_processing': False,
                'status': latest_job.get('status', 'unknown'),
                'progress': latest_job.get('progress', 0),
                'current_step': latest_job.get('current_step', 'Completed'),
                'message': 'Processing completed'
            })
        
        return jsonify({
            'is_processing': False,
            'message': 'No active processing job'
        })
    
    job_id = current_job['job_id']
    job_data = db.get_job(job_id)
    
    if not job_data:
        return jsonify({'error': 'Job not found'}), 404
    
    # Calculate elapsed time
    if current_job['start_time']:
        elapsed = (datetime.now() - current_job['start_time']).total_seconds()
        job_data['elapsed_time'] = format_duration(elapsed)
    
    # Get papers for this job
    papers = db.get_papers_by_job(job_id)
    job_data['papers_downloaded'] = len(papers)
    job_data['papers_completed'] = sum(1 for p in papers if p['processing_status'] == 'completed')
    
    # Add is_processing flag
    job_data['is_processing'] = True
    
    return jsonify(job_data)

@app.route('/results')
def get_results():
    """Get results of current or latest job."""
    job_id = current_job.get('job_id')
    
    if not job_id:
        # Get most recent job
        recent_jobs = db.get_recent_jobs(limit=1)
        if not recent_jobs:
            return jsonify({'error': 'No results available'}), 404
        job_id = recent_jobs[0]['id']
    
    job_data = db.get_job(job_id)
    if not job_data:
        return jsonify({'error': 'Job not found'}), 404
    
    # Get all papers for this job
    papers = db.get_papers_by_job(job_id)
    
    # Load compiled data for each paper
    results = []
    for paper in papers:
        if paper['compiled_json_path'] and os.path.exists(paper['compiled_json_path']):
            try:
                with open(paper['compiled_json_path'], 'r', encoding='utf-8') as f:
                    compiled_data = json.load(f)
                    results.append(compiled_data)
            except Exception as e:
                logger.error(f"Error loading compiled data: {e}")
                results.append({
                    'metadata': json.loads(paper['metadata_json']),
                    'error': 'Could not load compilation data',
                    'status': 'error'
                })
        else:
            results.append({
                'metadata': json.loads(paper['metadata_json']) if paper['metadata_json'] else {},
                'status': paper['processing_status']
            })
    
    return jsonify({
        'job': job_data,
        'results': results,
        'summary': {
            'total_papers': len(papers),
            'completed_papers': sum(1 for p in papers if p['processing_status'] == 'completed'),
            'topic': job_data['topic']
        }
    })

@app.route('/jobs/history')
def get_job_history():
    """Get processing job history."""
    limit = request.args.get('limit', 50, type=int)
    jobs = db.get_recent_jobs(limit)
    return jsonify({'jobs': jobs})

@app.route('/jobs/<int:job_id>')
def get_job_details(job_id):
    """Get details of a specific job."""
    job_data = db.get_job(job_id)
    if not job_data:
        return jsonify({'error': 'Job not found'}), 404
    
    papers = db.get_papers_by_job(job_id)
    
    return jsonify({
        'job': job_data,
        'papers': [
            {
                'arxiv_id': p['arxiv_id'],
                'title': p['title'],
                'authors': json.loads(p['authors']) if p['authors'] else [],
                'citation_count': p['citation_count'],
                'status': p['processing_status']
            }
            for p in papers
        ]
    })

@app.route('/download_results')
def download_results():
    """Download results as comprehensive ZIP file with PDFs, JSONs, and surveys."""
    job_id = request.args.get('job_id', type=int)
    
    if not job_id:
        job_id = current_job.get('job_id')
    
    if not job_id:
        return jsonify({'error': 'No job specified'}), 400
    
    job_data = db.get_job(job_id)
    if not job_data:
        return jsonify({'error': 'Job not found'}), 404
    
    papers = db.get_papers_by_job(job_id)
    surveys = db.get_surveys_by_job(job_id)
    
    if not papers:
        return jsonify({'error': 'No papers found for this job'}), 404
    
    # Create ZIP file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    zip_filename = f"research_results_job{job_id}_{timestamp}.zip"
    zip_path = os.path.join(config.PROCESSED_DIR, zip_filename)
    
    try:
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Add comprehensive summary JSON
            summary = {
                'job': job_data,
                'total_papers': len(papers),
                'papers_with_surveys': len(surveys),
                'generated_at': timestamp,
                'topic': job_data.get('topic', 'Unknown')
            }
            
            summary_file = os.path.join(config.PROCESSED_DIR, 'summary.json')
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2)
            zipf.write(summary_file, 'summary.json')
            os.remove(summary_file)
            
            # Add compiled JSONs
            for paper in papers:
                if paper['compiled_json_path'] and os.path.exists(paper['compiled_json_path']):
                    zipf.write(
                        paper['compiled_json_path'],
                        f"compiled/{os.path.basename(paper['compiled_json_path'])}"
                    )
            
            # Add PDFs
            for paper in papers:
                if paper['pdf_path'] and os.path.exists(paper['pdf_path']):
                    zipf.write(
                        paper['pdf_path'],
                        f"pdfs/{os.path.basename(paper['pdf_path'])}"
                    )
            
            # Add literature surveys as separate files
            for survey in surveys:
                paper = next((p for p in papers if p['id'] == survey['paper_id']), None)
                if paper:
                    survey_content = f"""
# Literature Survey: {paper['title']}
ArXiv ID: {paper['arxiv_id']}
Generated: {survey.get('generated_at', 'N/A')}

## Related Work & Context
{survey.get('related_work', 'N/A')}

## Methodology Survey
{survey.get('methodology_survey', 'N/A')}

## Key Contributions
{survey.get('contributions_summary', 'N/A')}

## Research Gaps & Future Work
{survey.get('research_gaps', 'N/A')}

## Context Analysis
{survey.get('context_analysis', 'N/A')}
"""
                    survey_filename = f"surveys/survey_{paper['arxiv_id'].replace('/', '_')}.md"
                    zipf.writestr(survey_filename, survey_content)
            
            # Add overall summary
            overall_summary = _generate_overall_summary(papers, surveys, job_data)
            zipf.writestr('OVERALL_SUMMARY.md', overall_summary)
        
        return send_file(zip_path, as_attachment=True, download_name=zip_filename)
    
    except Exception as e:
        logger.error(f"Error creating ZIP: {e}", exc_info=True)
        return jsonify({'error': f'Error creating download: {str(e)}'}), 500

def _generate_overall_summary(papers, surveys, job_data):
    """Generate an overall summary document."""
    content = f"""# Research Analysis Summary
Topic: {job_data.get('topic', 'Unknown')}
Total Papers: {len(papers)}
Surveys Generated: {len(surveys)}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Papers Analyzed
"""
    for i, paper in enumerate(papers, 1):
        content += f"\n{i}. **{paper['title']}**\n"
        content += f"   - ArXiv ID: {paper['arxiv_id']}\n"
        content += f"   - Citations: {paper['citation_count']}\n"
        content += f"   - Status: {paper['processing_status']}\n"
    
    content += "\n\n## Survey Status\n"
    content += f"- Papers with complete surveys: {len(surveys)}\n"
    content += f"- Papers pending survey: {len(papers) - len(surveys)}\n"
    
    return content

@app.route('/stats')
def get_stats():
    """Get database statistics."""
    stats = db.get_database_stats()
    
    # Add RAG statistics
    vector_stats = vector_db.get_statistics()
    graph_stats = knowledge_graph.get_statistics()
    
    stats['vector_db'] = vector_stats
    stats['knowledge_graph'] = graph_stats
    
    return jsonify(stats)

@app.route('/admin')
def admin_panel():
    """Admin panel for system maintenance."""
    return render_template('admin.html')

@app.route('/knowledge_graph')
def view_knowledge_graph():
    """View the knowledge graph visualization."""
    try:
        graph = knowledge_graph.graph
        
        if graph.number_of_nodes() == 0:
            return render_template('error.html', 
                                 error='Knowledge graph is empty. Please reindex the database.'), 404
        
        # Prepare graph data for web visualization
        nodes = []
        edges = []
        
        for node in graph.nodes():
            node_data = graph.nodes[node]
            node_type = 'paper' if node.startswith('paper_') else (
                        'author' if node.startswith('author_') else 'concept')
            
            nodes.append({
                'id': node,
                'label': node_data.get('title', node_data.get('name', node))[:50],
                'type': node_type,
                'full_label': node_data.get('title', node_data.get('name', node)),
                'citations': node_data.get('citation_count', 0) if node_type == 'paper' else 0
            })
        
        for source, target, key in graph.edges(keys=True):
            edge_data = graph.edges[source, target, key]
            edges.append({
                'source': source,
                'target': target,
                'type': edge_data.get('type', 'unknown')
            })
        
        return render_template('knowledge_graph.html', 
                             nodes=nodes, 
                             edges=edges,
                             stats=knowledge_graph.get_statistics())
        
    except Exception as e:
        logger.error(f"Error viewing knowledge graph: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

# ========== RAG ENDPOINTS ==========

@app.route('/rag/query', methods=['POST'])
def rag_query():
    """
    Answer a question using Hybrid RAG (BM25 + Semantic).
    
    Request body:
    {
        "question": "What are the main challenges?",
        "paper_id": 1  // optional
    }
    """
    try:
        data = request.json
        question = data.get('question', '').strip()
        paper_id = data.get('paper_id')
        
        logger.info(f"📝 Hybrid RAG Query received: '{question}'")
        
        if not question:
            return jsonify({'error': 'Question is required'}), 400
        
        # Check if vector DB has data
        stats = vector_db.get_statistics()
        if stats.get('total_chunks', 0) == 0:
            logger.warning("⚠️  Vector DB is empty!")
            return jsonify({
                'answer': 'No papers have been indexed yet. Please click "Reindex All Papers" button first.',
                'sources': [],
                'confidence': 'low',
                'error': 'vector_db_empty'
            })
        
        logger.info(f"📊 Vector DB has {stats.get('total_chunks')} chunks from {stats.get('unique_papers')} papers")
        
        # Query Hybrid RAG engine
        result = hybrid_rag_engine.query(question, specific_paper_id=paper_id)
        
        logger.info(f"✅ Hybrid RAG query completed: {len(result.get('sources', []))} sources")
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"❌ RAG query error: {e}", exc_info=True)
        return jsonify({
            'error': str(e),
            'answer': f'Error processing your question: {str(e)}',
            'sources': [],
            'confidence': 'error'
        }), 500

@app.route('/rag/summary')
def get_research_summary():
    """Get overall research summary across all papers."""
    try:
        # For now, return basic summary from knowledge graph
        stats = {
            'summary': 'Research summary feature uses knowledge graph analysis',
            'statistics': knowledge_graph.get_research_overview()
        }
        return jsonify(stats)
    except Exception as e:
        logger.error(f"Summary generation error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/rag/related/<int:paper_id>')
def get_related_papers(paper_id):
    """Get papers related to a specific paper."""
    try:
        related = knowledge_graph.find_related_papers(paper_id, max_results=10)
        return jsonify({'paper_id': paper_id, 'related_papers': related})
    except Exception as e:
        logger.error(f"Error finding related papers: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/rag/index_status')
def get_index_status():
    """Get indexing status."""
    try:
        vector_stats = vector_db.get_statistics()
        graph_stats = knowledge_graph.get_statistics()
        db_stats = db.get_database_stats()
        
        return jsonify({
            'vector_db': vector_stats,
            'knowledge_graph': graph_stats,
            'database': {
                'total_papers': db_stats.get('total_papers', 0)
            },
            'indexed_percentage': (vector_stats.get('unique_papers', 0) / max(db_stats.get('total_papers', 1), 1)) * 100
        })
    except Exception as e:
        logger.error(f"Error getting index status: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/rag/reindex', methods=['POST'])
def reindex_all():
    """Reindex all papers (admin function)."""
    try:
        logger.info("🔄 Starting manual reindex...")
        
        # Get all papers from database
        jobs = db.get_recent_jobs(limit=100)
        all_papers = []
        
        for job in jobs:
            job_papers = db.get_papers_by_job(job['id'])
            all_papers.extend(job_papers)
        
        indexed_count = 0
        total_chunks = 0
        errors = []
        
        for paper in all_papers:
            if not paper.get('compiled_json_path') or not os.path.exists(paper['compiled_json_path']):
                continue
            
            try:
                with open(paper['compiled_json_path'], 'r', encoding='utf-8') as f:
                    paper_data = json.load(f)
                
                # Index in vector DB
                chunks = vector_db.index_paper(paper['id'], paper_data)
                total_chunks += chunks
                
                # Add to knowledge graph
                knowledge_graph.add_paper(paper['id'], paper_data)
                
                # Link citations
                if paper_data.get('references'):
                    knowledge_graph.link_citations(paper['id'], paper_data['references'])
                
                indexed_count += 1
                logger.info(f"✅ Indexed paper {paper['id']}: {chunks} chunks")
                
            except Exception as e:
                error_msg = f"Paper {paper['id']}: {str(e)}"
                errors.append(error_msg)
                logger.error(f"❌ {error_msg}")
        
        # Save knowledge graph
        knowledge_graph.save_graph()
        
        # Get final stats
        vector_stats = vector_db.get_statistics()
        graph_stats = knowledge_graph.get_statistics()
        
        return jsonify({
            'success': True,
            'message': f'Reindexed {indexed_count} papers',
            'indexed_count': indexed_count,
            'total_papers': len(all_papers),
            'total_chunks': total_chunks,
            'errors': errors,
            'stats': {
                'vector_db': vector_stats,
                'knowledge_graph': graph_stats
            }
        })
        
    except Exception as e:
        logger.error(f"Reindexing error: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500

# ========== LITERATURE SURVEY ENDPOINTS ==========

@app.route('/surveys/generate', methods=['POST'])
def generate_surveys():
    """Generate literature surveys for all papers in a job."""
    try:
        data = request.json
        job_id = data.get('job_id')
        
        if not job_id:
            return jsonify({'error': 'job_id is required'}), 400
        
        logger.info(f"📚 Generating surveys for job {job_id}...")
        
        result = survey_generator.compile_job_surveys(job_id)
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Survey generation error: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/surveys/<int:paper_id>')
def get_paper_survey(paper_id):
    """Get literature survey for a specific paper."""
    try:
        survey = db.get_paper_survey(paper_id)
        
        if not survey:
            return jsonify({'error': 'Survey not found', 'paper_id': paper_id}), 404
        
        return jsonify({
            'paper_id': paper_id,
            'survey': survey
        })
        
    except Exception as e:
        logger.error(f"Error retrieving survey: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/surveys/job/<int:job_id>')
def get_job_surveys(job_id):
    """Get all surveys for a job."""
    try:
        surveys = db.get_surveys_by_job(job_id)
        
        return jsonify({
            'job_id': job_id,
            'total_surveys': len(surveys),
            'surveys': surveys
        })
        
    except Exception as e:
        logger.error(f"Error retrieving surveys: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/surveys/overall')
def get_overall_survey():
    """Generate an overall literature survey synthesizing all papers in a job."""
    try:
        job_id = request.args.get('job_id', type=int)
        
        if not job_id:
            return jsonify({'error': 'job_id parameter required'}), 400
        
        job_data = db.get_job(job_id)
        if not job_data:
            return jsonify({'error': 'Job not found'}), 404
        
        papers = db.get_papers_by_job(job_id)
        surveys = db.get_surveys_by_job(job_id)
        
        if not papers:
            return jsonify({'error': 'No papers found for this job'}), 404
        
        # Generate overall survey using Ollama
        overall_survey = _generate_comprehensive_literature_survey(
            papers, surveys, job_data
        )
        
        return jsonify(overall_survey)
        
    except Exception as e:
        logger.error(f"Error generating overall survey: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

def _generate_comprehensive_literature_survey(papers, surveys, job_data):
    """Generate comprehensive literature survey across all papers."""
    try:
        import ollama
        
        logger.info(f"Generating overall survey for {len(papers)} papers")
        
        # Collect key information from all papers
        papers_summary = []
        for paper in papers[:10]:  # Limit to 10 to avoid token overflow
            paper_info = {
                'title': paper.get('title', 'Unknown'),
                'abstract': paper.get('abstract', '')[:250] if paper.get('abstract') else 'No abstract',
                'arxiv_id': paper.get('arxiv_id', 'Unknown')
            }
            
            # Add survey if available
            survey = next((s for s in surveys if s.get('paper_id') == paper.get('id')), None)
            if survey:
                paper_info['contributions'] = survey.get('contributions_summary', '')[:150]
                paper_info['gaps'] = survey.get('research_gaps', '')[:150]
            
            papers_summary.append(paper_info)
        
        # Build comprehensive prompt
        papers_text = "\n\n".join([
            f"Paper {i+1}: {p['title']}\n"
            f"Abstract: {p['abstract']}\n"
            f"Contributions: {p.get('contributions', 'N/A')}\n"
            f"Gaps: {p.get('gaps', 'N/A')}"
            for i, p in enumerate(papers_summary)
        ])
        
        prompt = f"""Analyze these {len(papers_summary)} research papers on "{job_data.get('topic', 'Unknown')}" and write a comprehensive literature survey.

PAPERS:
{papers_text}

Write 5 concise paragraphs (2-3 sentences each):

1. DOMAIN: Define the research area and its importance
2. METHODS: Main approaches and techniques used
3. FINDINGS: Key results and consensus across papers
4. CHALLENGES: Common limitations and obstacles
5. FUTURE: Research gaps and future directions

Use formal academic style. Reference papers as [Paper 1], [Paper 2], etc.

Format each paragraph with a clear heading."""

        logger.info("Calling Ollama to generate survey...")
        response = ollama.chat(
            model=config.OLLAMA_MODEL,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.3, "num_predict": 1500}
        )
        
        content = response['message']['content'].strip()
        logger.info(f"Generated survey: {len(content)} characters")
        
        # Parse sections using simpler regex patterns
        import re
        
        sections = {
            'domain_scope': '',
            'methodologies': '',
            'key_findings': '',
            'challenges': '',
            'future_directions': ''
        }
        
        # Try to extract sections by numbered headers
        parts = re.split(r'\n\s*[12345]\.\s*', content)
        
        if len(parts) >= 5:
            sections['domain_scope'] = parts[1].strip() if len(parts) > 1 else ''
            sections['methodologies'] = parts[2].strip() if len(parts) > 2 else ''
            sections['key_findings'] = parts[3].strip() if len(parts) > 3 else ''
            sections['challenges'] = parts[4].strip() if len(parts) > 4 else ''
            sections['future_directions'] = parts[5].strip() if len(parts) > 5 else ''
        else:
            # Fallback: just split content into 5 equal parts
            lines = content.split('\n')
            chunk_size = len(lines) // 5
            if chunk_size > 0:
                sections['domain_scope'] = '\n'.join(lines[:chunk_size]).strip()
                sections['methodologies'] = '\n'.join(lines[chunk_size:chunk_size*2]).strip()
                sections['key_findings'] = '\n'.join(lines[chunk_size*2:chunk_size*3]).strip()
                sections['challenges'] = '\n'.join(lines[chunk_size*3:chunk_size*4]).strip()
                sections['future_directions'] = '\n'.join(lines[chunk_size*4:]).strip()
            else:
                # Last resort: put everything in domain_scope
                sections['domain_scope'] = content
        
        # Convert newlines to HTML breaks for display
        for key in sections:
            if sections[key]:
                sections[key] = sections[key].replace('\n', '<br>')
        
        sections['stats'] = {
            'total_papers': len(papers),
            'papers_with_surveys': len(surveys),
            'topic': job_data.get('topic', 'Unknown')
        }
        
        logger.info("Survey generation complete")
        return sections
        
    except Exception as e:
        logger.error(f"Error in comprehensive survey generation: {e}", exc_info=True)
        return {
            'error': str(e),
            'domain_scope': f'Error generating survey: {str(e)}',
            'methodologies': 'N/A',
            'key_findings': 'N/A',
            'challenges': 'N/A',
            'future_directions': 'N/A',
            'stats': {'total_papers': len(papers), 'papers_with_surveys': len(surveys)}
        }

@app.route('/results/comprehensive')
def get_comprehensive_results():
    """Get comprehensive results including surveys and all metadata."""
    try:
        job_id = request.args.get('job_id', type=int)
        
        if not job_id:
            recent_jobs = db.get_recent_jobs(limit=1)
            if not recent_jobs:
                return jsonify({'error': 'No results available'}), 404
            job_id = recent_jobs[0]['id']
        
        job_data = db.get_job(job_id)
        if not job_data:
            return jsonify({'error': 'Job not found'}), 404
        
        papers = db.get_papers_by_job(job_id)
        surveys = db.get_surveys_by_job(job_id)
        
        # Build comprehensive results
        results = []
        for paper in papers:
            paper_result = {
                'paper': {
                    'id': paper['id'],
                    'arxiv_id': paper['arxiv_id'],
                    'title': paper['title'],
                    'authors': json.loads(paper['authors']) if paper['authors'] else [],
                    'abstract': paper['abstract'],
                    'citation_count': paper['citation_count'],
                    'published_date': paper['published_date'],
                    'status': paper['processing_status']
                }
            }
            
            # Add compiled data
            if paper['compiled_json_path'] and os.path.exists(paper['compiled_json_path']):
                try:
                    with open(paper['compiled_json_path'], 'r', encoding='utf-8') as f:
                        compiled = json.load(f)
                        paper_result['compiled_data'] = compiled
                except:
                    pass
            
            # Add survey if available
            paper_survey = next((s for s in surveys if s['paper_id'] == paper['id']), None)
            if paper_survey:
                paper_result['survey'] = {
                    'literature_survey': paper_survey.get('literature_survey'),
                    'related_work': paper_survey.get('related_work'),
                    'methodology_survey': paper_survey.get('methodology_survey'),
                    'contributions_summary': paper_survey.get('contributions_summary'),
                    'research_gaps': paper_survey.get('research_gaps'),
                    'context_analysis': paper_survey.get('context_analysis'),
                    'reference_count': paper_survey.get('reference_count', 0)
                }
            
            results.append(paper_result)
        
        return jsonify({
            'job': job_data,
            'summary': {
                'total_papers': len(papers),
                'papers_with_surveys': len(surveys),
                'completed_papers': sum(1 for p in papers if p['processing_status'] == 'completed'),
                'topic': job_data['topic']
            },
            'results': results
        })
        
    except Exception as e:
        logger.error(f"Error getting comprehensive results: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

# ========== BACKGROUND PROCESSING ==========

def process_papers_background(job_id: int, topic: str, num_papers: int):
    """Background processing function."""
    global current_job
    
    try:
        start_time = datetime.now()
        
        # Update job status
        db.update_job_status(job_id, 'processing', 10, 'Searching arXiv...')
        
        # Step 1: Scraping
        logger.info("Step 1: Scraping papers...")
        scraper = ArxivScraper(config.DATA_DIR)
        papers_metadata = scraper.search_and_download(topic, num_papers)
        
        if not papers_metadata:
            db.update_job_status(job_id, 'failed', 100, 'No papers found', 
                                error='No papers matched the search query')
            current_job['is_processing'] = False
            return
        
        db.update_job_status(job_id, 'processing', 30, 
                           f'Downloaded {len(papers_metadata)} papers')
        
        # Save papers to database with error handling
        paper_ids = {}
        for paper_meta in papers_metadata:
            try:
                paper_id = db.save_paper(job_id, paper_meta)
                paper_ids[paper_meta['arxiv_id']] = paper_id
                logger.debug(f"Saved paper {paper_meta['arxiv_id']} to database (ID: {paper_id})")
            except Exception as e:
                logger.error(f"Error saving paper to database: {e}")
                # Continue with other papers
                continue
        
        # Step 2: Compilation
        logger.info("Step 2: Compiling papers...")
        compiler = CompilationAgent(config.DATA_DIR, config.PROCESSED_DIR)
        
        for i, paper_meta in enumerate(papers_metadata):
            arxiv_id = paper_meta['arxiv_id']
            
            # Skip if not saved to database
            if arxiv_id not in paper_ids:
                logger.warning(f"Skipping {arxiv_id} - not in database")
                continue
                
            paper_id = paper_ids[arxiv_id]
            
            progress = 30 + (i / len(papers_metadata)) * 65
            db.update_job_status(
                job_id, 'processing', int(progress),
                f'Processing paper {i+1}/{len(papers_metadata)}: {paper_meta["title"][:50]}'
            )
            
            try:
                result = compiler.process_paper(paper_meta)
                
                if result and result.get('status') == 'completed':
                    # Update paper in database
                    db.update_paper_compilation(paper_id, result.get('json_file'), 'completed')
                    
                    # Save sections
                    if result.get('sections_text'):
                        try:
                            db.save_paper_sections(paper_id, result['sections_text'])
                        except Exception as e:
                            logger.error(f"Error saving sections: {e}")
                    
                    # Save summaries
                    if result.get('sections_summary'):
                        for section_name, summary in result['sections_summary'].items():
                            try:
                                db.update_section_summary(paper_id, section_name, summary)
                            except Exception as e:
                                logger.error(f"Error saving summary for {section_name}: {e}")
                    
                    # Save contributions
                    if result.get('contributions'):
                        try:
                            db.save_paper_contributions(paper_id, result['contributions'])
                        except Exception as e:
                            logger.error(f"Error saving contributions: {e}")
                    
                    # Save references
                    if result.get('references'):
                        try:
                            db.save_paper_references(paper_id, result['references'])
                        except Exception as e:
                            logger.error(f"Error saving references: {e}")
                else:
                    db.update_paper_compilation(paper_id, None, 'failed')
                    
            except Exception as e:
                logger.error(f"Error processing paper {paper_meta['title']}: {e}", exc_info=True)
                try:
                    db.update_paper_compilation(paper_id, None, 'failed')
                except:
                    pass
        
        # Mark job as completed
        end_time = datetime.now()
        processing_time = format_duration((end_time - start_time).total_seconds())
        
        db.update_job_status(job_id, 'completed', 95, 
                           f'Generating literature surveys... {len(papers_metadata)} papers')
        
        # Step 3: Generate literature surveys
        logger.info("Step 3: Generating literature surveys...")
        try:
            survey_result = survey_generator.compile_job_surveys(job_id)
            logger.info(f"✅ Generated {survey_result.get('total_surveys', 0)} surveys")
        except Exception as e:
            logger.error(f"Error generating surveys: {e}", exc_info=True)
        
        # Save knowledge graph to disk
        try:
            knowledge_graph.save_graph()
            logger.info("Knowledge graph saved successfully")
        except Exception as e:
            logger.error(f"Error saving knowledge graph: {e}")
        
        db.update_job_status(job_id, 'completed', 100, 
                           f'Completed! Processed {len(papers_metadata)} papers in {processing_time}')
        
        logger.info(f"✅ Job {job_id} completed successfully in {processing_time}")
        
    except Exception as e:
        logger.error(f"Background processing error for job {job_id}: {e}", exc_info=True)
        try:
            db.update_job_status(job_id, 'failed', 0, 'Processing failed', error=str(e))
        except:
            pass
    
    finally:
        current_job['is_processing'] = False

# ========== STARTUP ==========

if __name__ == '__main__':
    logger.info("=" * 60)
    logger.info("🚀 AI Research Assistant Starting...")
    logger.info("=" * 60)
    logger.info(f"Data directory: {config.DATA_DIR}")
    logger.info(f"Processed directory: {config.PROCESSED_DIR}")
    logger.info(f"Database: {config.DATABASE_PATH}")
    logger.info(f"Ollama model: {config.OLLAMA_MODEL}")
    logger.info("=" * 60)
    
    app.run(debug=True, threaded=True, host='0.0.0.0', port=5000)