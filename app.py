# app.py - IMPROVED Flask Application
from flask import Flask, render_template, request, jsonify, send_file
import os
import json
import threading
import time
import zipfile
from datetime import datetime
from config import config
from modules.scraper import ArxivScraper
from modules.compiler import CompilationAgent
from modules.database import db
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
                'status': latest_job['status'],
                'progress': latest_job['progress'],
                'current_step': latest_job['current_step'],
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
    """Download results as ZIP file."""
    job_id = request.args.get('job_id', type=int)
    
    if not job_id:
        job_id = current_job.get('job_id')
    
    if not job_id:
        return jsonify({'error': 'No job specified'}), 400
    
    job_data = db.get_job(job_id)
    if not job_data:
        return jsonify({'error': 'Job not found'}), 404
    
    papers = db.get_papers_by_job(job_id)
    
    if not papers:
        return jsonify({'error': 'No papers found for this job'}), 404
    
    # Create ZIP file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    zip_filename = f"research_results_job{job_id}_{timestamp}.zip"
    zip_path = os.path.join(config.PROCESSED_DIR, zip_filename)
    
    try:
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Add summary JSON
            summary = {
                'job': job_data,
                'papers': len(papers),
                'generated_at': timestamp
            }
            
            summary_file = os.path.join(config.PROCESSED_DIR, 'summary.json')
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
            zipf.write(summary_file, 'summary.json')
            os.remove(summary_file)
            
            # Add compiled JSONs
            for paper in papers:
                if paper['compiled_json_path'] and os.path.exists(paper['compiled_json_path']):
                    zipf.write(
                        paper['compiled_json_path'],
                        os.path.basename(paper['compiled_json_path'])
                    )
            
            # Add PDFs (optional)
            for paper in papers:
                if paper['pdf_path'] and os.path.exists(paper['pdf_path']):
                    zipf.write(
                        paper['pdf_path'],
                        f"pdfs/{os.path.basename(paper['pdf_path'])}"
                    )
        
        return send_file(zip_path, as_attachment=True, download_name=zip_filename)
    
    except Exception as e:
        logger.error(f"Error creating ZIP: {e}", exc_info=True)
        return jsonify({'error': f'Error creating download: {str(e)}'}), 500

@app.route('/stats')
def get_stats():
    """Get database statistics."""
    stats = db.get_database_stats()
    return jsonify(stats)

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