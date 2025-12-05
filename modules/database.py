# modules/database.py - SQLite Database Management
import sqlite3
import json
from contextlib import contextmanager
from datetime import datetime
from typing import List, Dict, Optional, Any
from config import config

class DatabaseManager:
    """Manages all database operations for the research assistant."""
    
    def __init__(self, db_path: str = None):
        self.db_path = db_path or config.DATABASE_PATH
        self.init_database()
    
    @contextmanager
    def get_connection(self):
        """Context manager for database connections."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Return rows as dictionaries
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()
    
    def init_database(self):
        """Initialize database schema."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Processing Jobs Table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS processing_jobs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    topic TEXT NOT NULL,
                    num_papers INTEGER NOT NULL,
                    status TEXT DEFAULT 'pending',
                    progress INTEGER DEFAULT 0,
                    current_step TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    started_at TIMESTAMP,
                    completed_at TIMESTAMP,
                    error_message TEXT,
                    results_summary TEXT
                )
            ''')
            
            # Papers Table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS papers (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id INTEGER NOT NULL,
                    arxiv_id TEXT UNIQUE NOT NULL,
                    title TEXT NOT NULL,
                    authors TEXT,
                    abstract TEXT,
                    published_date TEXT,
                    categories TEXT,
                    pdf_url TEXT,
                    pdf_path TEXT,
                    citation_count INTEGER DEFAULT 0,
                    influential_citation_count INTEGER DEFAULT 0,
                    metadata_json TEXT,
                    compiled_json_path TEXT,
                    processing_status TEXT DEFAULT 'pending',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (job_id) REFERENCES processing_jobs(id)
                )
            ''')
            
            # Paper Sections Table (for RAG indexing later)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS paper_sections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    paper_id INTEGER NOT NULL,
                    section_name TEXT NOT NULL,
                    content TEXT NOT NULL,
                    summary TEXT,
                    word_count INTEGER,
                    FOREIGN KEY (paper_id) REFERENCES papers(id)
                )
            ''')
            
            # Key Contributions Table (for Knowledge Graph)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS paper_contributions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    paper_id INTEGER NOT NULL,
                    main_problem TEXT,
                    key_innovation TEXT,
                    methodology TEXT,
                    major_results TEXT,
                    limitations TEXT,
                    research_gaps TEXT,
                    extracted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (paper_id) REFERENCES papers(id)
                )
            ''')
            
            # References Table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS paper_references (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    paper_id INTEGER NOT NULL,
                    reference_index TEXT,
                    authors TEXT,
                    title TEXT,
                    year TEXT,
                    venue TEXT,
                    FOREIGN KEY (paper_id) REFERENCES papers(id)
                )
            ''')
            
            # Literature Survey Table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS paper_surveys (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    paper_id INTEGER NOT NULL,
                    related_work TEXT,
                    methodology_survey TEXT,
                    contributions_summary TEXT,
                    research_gaps TEXT,
                    context_analysis TEXT,
                    full_survey_json TEXT,
                    generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (paper_id) REFERENCES papers(id),
                    UNIQUE(paper_id)
                )
            ''')
            
            # Create indexes for better query performance
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_papers_arxiv_id ON papers(arxiv_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_papers_job_id ON papers(job_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_sections_paper_id ON paper_sections(paper_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_contributions_paper_id ON paper_contributions(paper_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_surveys_paper_id ON paper_surveys(paper_id)')
            
            conn.commit()
    
    # ========== JOB MANAGEMENT ==========
    
    def create_job(self, topic: str, num_papers: int) -> int:
        """Create a new processing job."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO processing_jobs (topic, num_papers, status, started_at)
                VALUES (?, ?, 'processing', ?)
            ''', (topic, num_papers, datetime.now()))
            return cursor.lastrowid
    
    def update_job_status(self, job_id: int, status: str, progress: int = None, 
                         current_step: str = None, error: str = None):
        """Update job status and progress."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            updates = ['status = ?']
            params = [status]
            
            if progress is not None:
                updates.append('progress = ?')
                params.append(progress)
            
            if current_step:
                updates.append('current_step = ?')
                params.append(current_step)
            
            if error:
                updates.append('error_message = ?')
                params.append(error)
            
            if status == 'completed':
                updates.append('completed_at = ?')
                params.append(datetime.now())
            
            params.append(job_id)
            
            query = f"UPDATE processing_jobs SET {', '.join(updates)} WHERE id = ?"
            cursor.execute(query, params)
    
    def get_job(self, job_id: int) -> Optional[Dict]:
        """Get job details."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM processing_jobs WHERE id = ?', (job_id,))
            row = cursor.fetchone()
            return dict(row) if row else None
    
    def get_recent_jobs(self, limit: int = 50) -> List[Dict]:
        """Get recent processing jobs."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT id, topic, num_papers, status, progress, created_at, completed_at
                FROM processing_jobs
                ORDER BY created_at DESC
                LIMIT ?
            ''', (limit,))
            return [dict(row) for row in cursor.fetchall()]
    
    # ========== PAPER MANAGEMENT ==========
    
    def save_paper(self, job_id: int, paper_metadata: Dict) -> int:
        """Save paper metadata to database."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Ensure all values are proper types for SQLite
            authors_json = json.dumps(paper_metadata.get('authors', []))
            categories_json = json.dumps(paper_metadata.get('categories', []))
            metadata_json = json.dumps(paper_metadata, default=str)  # Handle any non-serializable types
            
            cursor.execute('''
                INSERT OR REPLACE INTO papers (
                    job_id, arxiv_id, title, authors, abstract, published_date,
                    categories, pdf_url, pdf_path, citation_count, 
                    influential_citation_count, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                int(job_id),
                str(paper_metadata.get('arxiv_id', '')),
                str(paper_metadata.get('title', '')),
                authors_json,
                str(paper_metadata.get('abstract', '')),
                str(paper_metadata.get('published', '')),
                categories_json,
                str(paper_metadata.get('pdf_url', '')),
                str(paper_metadata.get('pdf_file', '')),
                int(paper_metadata.get('citation_count', 0)),
                int(paper_metadata.get('influential_citation_count', 0)),
                metadata_json
            ))
            
            return cursor.lastrowid
    
    def update_paper_compilation(self, paper_id: int, compiled_path: str, status: str = 'completed'):
        """Update paper with compilation results."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE papers 
                SET compiled_json_path = ?, processing_status = ?
                WHERE id = ?
            ''', (compiled_path, status, paper_id))
    
    def get_papers_by_job(self, job_id: int) -> List[Dict]:
        """Get all papers for a specific job."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM papers WHERE job_id = ?', (job_id,))
            return [dict(row) for row in cursor.fetchall()]
    
    def get_paper_by_arxiv_id(self, arxiv_id: str) -> Optional[Dict]:
        """Check if paper already exists."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM papers WHERE arxiv_id = ?', (arxiv_id,))
            row = cursor.fetchone()
            return dict(row) if row else None
    
    # ========== SECTIONS MANAGEMENT ==========
    
    def save_paper_sections(self, paper_id: int, sections_data: Dict):
        """Save extracted sections for a paper."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            for section_name, content in sections_data.items():
                if isinstance(content, str) and content.strip():
                    cursor.execute('''
                        INSERT INTO paper_sections (paper_id, section_name, content, word_count)
                        VALUES (?, ?, ?, ?)
                    ''', (paper_id, section_name, content, len(content.split())))
    
    def update_section_summary(self, paper_id: int, section_name: str, summary: str):
        """Update summary for a specific section."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE paper_sections 
                SET summary = ?
                WHERE paper_id = ? AND section_name = ?
            ''', (summary, paper_id, section_name))
    
    # ========== CONTRIBUTIONS MANAGEMENT ==========
    
    def save_paper_contributions(self, paper_id: int, contributions: Dict):
        """Save extracted key contributions."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO paper_contributions (
                    paper_id, main_problem, key_innovation, methodology,
                    major_results, limitations, research_gaps
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                paper_id,
                contributions.get('main_problem', ''),
                contributions.get('key_innovation', ''),
                contributions.get('core_methodology', ''),
                contributions.get('major_results', ''),
                contributions.get('limitations', ''),
                contributions.get('research_gaps', '')
            ))
    
    # ========== REFERENCES MANAGEMENT ==========
    
    def save_paper_references(self, paper_id: int, references: List[Dict]):
        """Save extracted references."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            for ref in references:
                cursor.execute('''
                    INSERT INTO paper_references (
                        paper_id, reference_index, authors, title, year, venue
                    ) VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    paper_id,
                    ref.get('id', ''),
                    ref.get('authors', ''),
                    ref.get('title', ''),
                    ref.get('year', ''),
                    ref.get('venue', '')
                ))
    
    # ========== STATISTICS & ANALYTICS ==========
    
    def get_database_stats(self) -> Dict:
        """Get overall database statistics."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            stats = {}
            
            cursor.execute('SELECT COUNT(*) as count FROM processing_jobs')
            stats['total_jobs'] = cursor.fetchone()['count']
            
            cursor.execute('SELECT COUNT(*) as count FROM papers')
            stats['total_papers'] = cursor.fetchone()['count']
            
            cursor.execute('SELECT COUNT(*) as count FROM paper_sections')
            stats['total_sections'] = cursor.fetchone()['count']
            
            cursor.execute('SELECT COUNT(*) as count FROM paper_contributions')
            stats['papers_with_contributions'] = cursor.fetchone()['count']
            
            cursor.execute('''
                SELECT AVG(citation_count) as avg_citations 
                FROM papers 
                WHERE citation_count > 0
            ''')
            stats['avg_citations'] = cursor.fetchone()['avg_citations'] or 0
            
            return stats
    
    # ========== SURVEY MANAGEMENT ==========
    
    def save_paper_survey(self, paper_id: int, survey: Dict):
        """Save generated literature survey for a paper."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            survey_sections = survey.get('survey_sections', {})
            full_survey_json = json.dumps(survey, default=str)
            
            cursor.execute('''
                INSERT OR REPLACE INTO paper_surveys (
                    paper_id, related_work, methodology_survey, 
                    contributions_summary, research_gaps, context_analysis, full_survey_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                paper_id,
                survey_sections.get('related_work', {}).get('content', ''),
                survey_sections.get('methodology_survey', {}).get('content', ''),
                survey_sections.get('contributions_summary', {}).get('content', ''),
                survey_sections.get('research_gaps', {}).get('content', ''),
                survey_sections.get('context_analysis', {}).get('content', ''),
                full_survey_json
            ))
    
    def get_paper_survey(self, paper_id: int) -> Optional[Dict]:
        """Retrieve survey for a specific paper."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM paper_surveys WHERE paper_id = ?', (paper_id,))
            row = cursor.fetchone()
            
            if row:
                survey_dict = dict(row)
                if survey_dict.get('full_survey_json'):
                    survey_dict['survey_data'] = json.loads(survey_dict['full_survey_json'])
                return survey_dict
            return None
    
    def get_surveys_by_job(self, job_id: int) -> List[Dict]:
        """Get all surveys for papers in a job."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT ps.* FROM paper_surveys ps
                JOIN papers p ON ps.paper_id = p.id
                WHERE p.job_id = ?
            ''', (job_id,))
            return [dict(row) for row in cursor.fetchall()]

# Global database instance
db = DatabaseManager()