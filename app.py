# app.py - Main Flask Application
from flask import Flask, render_template, request, jsonify, send_file
import os
import json
import threading
import time
from datetime import datetime
import zipfile
import ollama
from modules.scraper import ArxivScraper
from modules.compiler import CompilationAgent
from modules.utils import ensure_directories

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'data'
PROCESSED_FOLDER = 'processed'
ensure_directories([UPLOAD_FOLDER, PROCESSED_FOLDER])

# Global variables for tracking processing
processing_status = {
    'is_processing': False,
    'current_step': '',
    'progress': 0,
    'total_papers': 0,
    'processed_papers': 0,
    'current_paper': '',
    'results': []
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start_processing', methods=['POST'])
def start_processing():
    global processing_status
    
    if processing_status['is_processing']:
        return jsonify({'error': 'Processing already in progress'}), 400
    
    data = request.json
    topic = data.get('topic', '').strip()
    num_papers = int(data.get('num_papers', 5))
    
    if not topic:
        return jsonify({'error': 'Topic is required'}), 400
    
    if not (1 <= num_papers <= 20):
        return jsonify({'error': 'Number of papers must be between 1 and 20'}), 400
    
    # Reset status
    processing_status = {
        'is_processing': True,
        'current_step': 'Initializing...',
        'progress': 0,
        'total_papers': num_papers,
        'processed_papers': 0,
        'current_paper': '',
        'results': [],
        'topic': topic,
        'start_time': datetime.now().isoformat()
    }
    
    # Start processing in background thread
    thread = threading.Thread(target=process_papers_background, args=(topic, num_papers))
    thread.daemon = True
    thread.start()
    
    return jsonify({'message': 'Processing started successfully'})

@app.route('/status')
def get_status():
    return jsonify(processing_status)

@app.route('/results')
def get_results():
    if not processing_status['results']:
        return jsonify({'error': 'No results available'}), 404
    
    return jsonify({
        'results': processing_status['results'],
        'summary': {
            'total_papers': processing_status['total_papers'],
            'processed_papers': processing_status['processed_papers'],
            'topic': processing_status.get('topic', ''),
            'processing_time': processing_status.get('processing_time', '')
        }
    })

@app.route('/download_results')
def download_results():
    if not processing_status['results']:
        return jsonify({'error': 'No results to download'}), 404
    
    # Create zip file with all results
    zip_filename = f"research_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
    zip_path = os.path.join(PROCESSED_FOLDER, zip_filename)
    
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        # Add JSON results
        results_json = os.path.join(PROCESSED_FOLDER, 'results_summary.json')
        with open(results_json, 'w') as f:
            json.dump({
                'results': processing_status['results'],
                'summary': {
                    'total_papers': processing_status['total_papers'],
                    'topic': processing_status.get('topic', ''),
                    'processing_time': processing_status.get('processing_time', '')
                }
            }, f, indent=2)
        zipf.write(results_json, 'results_summary.json')
        
        # Add individual paper JSONs
        for result in processing_status['results']:
            if result.get('json_file') and os.path.exists(result['json_file']):
                zipf.write(result['json_file'], os.path.basename(result['json_file']))
    
    return send_file(zip_path, as_attachment=True)

def process_papers_background(topic, num_papers):
    global processing_status
    
    try:
        start_time = datetime.now()
        
        # Step 1: Scraping
        processing_status['current_step'] = 'Searching and downloading papers...'
        processing_status['progress'] = 10
        
        scraper = ArxivScraper(UPLOAD_FOLDER)
        papers_metadata = scraper.search_and_download(topic, num_papers)
        
        if not papers_metadata:
            processing_status['is_processing'] = False
            processing_status['current_step'] = 'Error: No papers found'
            return
        
        processing_status['progress'] = 30
        processing_status['current_step'] = f'Downloaded {len(papers_metadata)} papers'
        
        # Step 2: Compilation
        compiler = CompilationAgent(UPLOAD_FOLDER, PROCESSED_FOLDER)
        results = []
        
        for i, paper in enumerate(papers_metadata):
            processing_status['current_step'] = f'Processing paper {i+1}/{len(papers_metadata)}'
            processing_status['current_paper'] = paper.get('title', 'Unknown')
            processing_status['processed_papers'] = i
            processing_status['progress'] = 30 + (i / len(papers_metadata)) * 60
            
            try:
                result = compiler.process_paper(paper)
                if result:
                    results.append(result)
            except Exception as e:
                print(f"Error processing paper {paper.get('title', '')}: {e}")
                results.append({
                    'title': paper.get('title', 'Unknown'),
                    'error': str(e),
                    'status': 'failed'
                })
        
        # Final results
        processing_status['results'] = results
        processing_status['processed_papers'] = len(results)
        processing_status['progress'] = 100
        processing_status['current_step'] = 'Completed!'
        processing_status['is_processing'] = False
        
        end_time = datetime.now()
        processing_time = str(end_time - start_time).split('.')[0]
        processing_status['processing_time'] = processing_time
        
    except Exception as e:
        processing_status['is_processing'] = False
        processing_status['current_step'] = f'Error: {str(e)}'
        print(f"Background processing error: {e}")

if __name__ == '__main__':
    app.run(debug=True, threaded=True)