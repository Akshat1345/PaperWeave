# modules/scraper.py - IMPROVED ArXiv Scraper
import requests
import os
import time
import xml.etree.ElementTree as ET
from urllib.parse import quote_plus
import feedparser
from typing import List, Dict, Optional
from config import config
from modules.utils import (
    logger, get_organized_pdf_path, is_valid_arxiv_id, 
    is_valid_pdf, format_file_size, ProgressTracker
)

class ArxivScraper:
    """
    Enhanced ArXiv scraper with:
    - Advanced query building
    - Citation metrics from Semantic Scholar
    - Better deduplication
    - Organized storage
    """
    
    def __init__(self, output_folder: str):
        self.output_folder = output_folder
        self.pdfs_folder = os.path.join(output_folder, 'pdfs')
        os.makedirs(self.pdfs_folder, exist_ok=True)
        
        self.base_url = config.ARXIV_BASE_URL
        
        # Request session with proper headers
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'AI Research Assistant/2.0 (Educational Project)',
            'Accept': 'application/atom+xml'
        })
        
        logger.info("ArxivScraper initialized")
    
    def build_query(self, query: str, filters: Optional[Dict] = None) -> str:
        """
        Build advanced arXiv query with filters.
        
        Args:
            query: Base search query
            filters: Optional filters (author, category, year, etc.)
        
        Returns:
            URL-encoded query string
        
        Examples:
            - Simple: "machine learning"
            - With author: query="neural networks", filters={"author": "Goodfellow"}
            - With category: query="transformers", filters={"category": "cs.AI"}
            - With year: query="GPT", filters={"year": 2023}
        """
        search_parts = [f'all:{query}']
        
        if filters:
            if filters.get('author'):
                search_parts.append(f'au:{filters["author"]}')
            
            if filters.get('category'):
                search_parts.append(f'cat:{filters["category"]}')
            
            if filters.get('year'):
                year = filters['year']
                search_parts.append(f'submittedDate:[{year}0101 TO {year}1231]')
            
            if filters.get('title_keywords'):
                search_parts.append(f'ti:{filters["title_keywords"]}')
        
        combined_query = ' AND '.join(search_parts)
        return quote_plus(combined_query)
    
    def search_and_download(self, query: str, max_results: int = 5, 
                          filters: Optional[Dict] = None, 
                          fetch_citations: bool = True) -> List[Dict]:
        """
        Search arXiv and download papers with enhanced metadata.
        
        Args:
            query: Search query
            max_results: Maximum number of papers
            filters: Optional search filters
            fetch_citations: Whether to fetch citation counts
        
        Returns:
            List of paper metadata dictionaries
        """
        logger.info(f"🔍 Searching arXiv for: '{query}' (max_results={max_results})")
        
        if filters:
            logger.info(f"   Filters: {filters}")
        
        try:
            # Try feedparser method first
            papers_metadata = self.search_with_feedparser(query, max_results, filters)
            
            if not papers_metadata:
                logger.warning("📡 Feedparser failed, trying alternative method...")
                papers_metadata = self.search_with_requests(query, max_results, filters)
            
            if not papers_metadata:
                logger.error("❌ No papers found with either method")
                return []
            
            # Deduplicate
            papers_metadata = self.deduplicate_papers(papers_metadata)
            
            # Enrich with citation data
            if fetch_citations and config.ENABLE_CITATION_FETCH:
                papers_metadata = self.enrich_with_citations(papers_metadata)
            
            logger.info(f"✅ Successfully retrieved {len(papers_metadata)} papers")
            return papers_metadata
            
        except Exception as e:
            logger.error(f"❌ Error in arXiv search: {e}", exc_info=True)
            return []
    
    def search_with_feedparser(self, query: str, max_results: int, 
                              filters: Optional[Dict] = None) -> List[Dict]:
        """Search using feedparser (more reliable)."""
        try:
            search_query = self.build_query(query, filters)
            url = (f"{self.base_url}?search_query={search_query}&start=0"
                   f"&max_results={max_results}&sortBy=relevance&sortOrder=descending")
            
            logger.debug(f"📡 Fetching from: {url}")
            
            feed = feedparser.parse(url)
            
            if not hasattr(feed, 'entries') or not feed.entries:
                logger.warning("No entries found in feed")
                return []
            
            papers_metadata = []
            progress = ProgressTracker(len(feed.entries[:max_results]), "Downloading papers")
            
            for i, entry in enumerate(feed.entries[:max_results]):
                logger.info(f"📄 Processing paper {i+1}/{min(len(feed.entries), max_results)}: {entry.title[:60]}...")
                
                # Extract metadata
                metadata = {
                    'title': entry.title.replace('\n', ' ').strip(),
                    'authors': [author.name for author in getattr(entry, 'authors', [])],
                    'abstract': getattr(entry, 'summary', '').replace('\n', ' ').strip(),
                    'published': getattr(entry, 'published', ''),
                    'arxiv_id': entry.id.split('/')[-1] if hasattr(entry, 'id') else f'unknown_{i}',
                    'categories': [tag.term for tag in getattr(entry, 'tags', [])],
                    'pdf_url': None,
                    'source': 'arxiv'
                }
                
                # Validate arXiv ID
                if not is_valid_arxiv_id(metadata['arxiv_id']):
                    logger.warning(f"⚠️  Invalid arXiv ID: {metadata['arxiv_id']}")
                
                # Find PDF link
                for link in getattr(entry, 'links', []):
                    if link.type == 'application/pdf':
                        metadata['pdf_url'] = link.href
                        break
                
                # Fallback PDF URL
                if not metadata['pdf_url']:
                    paper_id = entry.id.split('/')[-1]
                    metadata['pdf_url'] = f"https://arxiv.org/pdf/{paper_id}.pdf"
                
                # Download PDF with organized storage
                if metadata['pdf_url']:
                    pdf_path = get_organized_pdf_path(query, metadata['arxiv_id'])
                    
                    if self.download_pdf(metadata['pdf_url'], pdf_path):
                        metadata['pdf_file'] = pdf_path
                        metadata['pdf_filename'] = os.path.basename(pdf_path)
                        metadata['pdf_size'] = format_file_size(os.path.getsize(pdf_path))
                        papers_metadata.append(metadata)
                        logger.info(f"✅ Downloaded: {os.path.basename(pdf_path)} ({metadata['pdf_size']})")
                    else:
                        logger.error(f"❌ Failed to download: {metadata['title'][:50]}...")
                
                progress.update()
                time.sleep(config.ARXIV_RATE_LIMIT_DELAY)
            
            progress.complete()
            return papers_metadata
            
        except Exception as e:
            logger.error(f"❌ Feedparser method failed: {e}", exc_info=True)
            return []
    
    def search_with_requests(self, query: str, max_results: int, 
                            filters: Optional[Dict] = None) -> List[Dict]:
        """Fallback: Direct XML parsing."""
        try:
            search_query = self.build_query(query, filters)
            url = (f"{self.base_url}?search_query={search_query}&start=0"
                   f"&max_results={max_results}&sortBy=relevance&sortOrder=descending")
            
            logger.debug(f"📡 Direct request to: {url}")
            
            response = self.session.get(url, timeout=30, allow_redirects=True)
            response.raise_for_status()
            
            root = ET.fromstring(response.content)
            ns = {'atom': 'http://www.w3.org/2005/Atom', 
                  'arxiv': 'http://arxiv.org/schemas/atom'}
            
            entries = root.findall('atom:entry', ns)
            
            if not entries:
                logger.warning("No entries found in XML response")
                return []
            
            papers_metadata = []
            
            for i, entry in enumerate(entries[:max_results]):
                try:
                    title_elem = entry.find('atom:title', ns)
                    title = title_elem.text.replace('\n', ' ').strip() if title_elem is not None else f"Unknown Title {i}"
                    
                    logger.info(f"📄 Processing paper {i+1}: {title[:50]}...")
                    
                    metadata = {
                        'title': title,
                        'authors': [],
                        'abstract': '',
                        'published': '',
                        'arxiv_id': f'paper_{i}',
                        'categories': [],
                        'pdf_url': None,
                        'source': 'arxiv'
                    }
                    
                    # Extract authors
                    authors = entry.findall('atom:author', ns)
                    for author in authors:
                        name_elem = author.find('atom:name', ns)
                        if name_elem is not None:
                            metadata['authors'].append(name_elem.text.strip())
                    
                    # Extract abstract
                    summary_elem = entry.find('atom:summary', ns)
                    if summary_elem is not None:
                        metadata['abstract'] = summary_elem.text.replace('\n', ' ').strip()
                    
                    # Extract arXiv ID
                    id_elem = entry.find('atom:id', ns)
                    if id_elem is not None:
                        metadata['arxiv_id'] = id_elem.text.split('/')[-1]
                    
                    # Extract categories
                    categories = entry.findall('atom:category', ns)
                    for cat in categories:
                        term = cat.get('term')
                        if term:
                            metadata['categories'].append(term)
                    
                    # Find PDF link
                    links = entry.findall('atom:link', ns)
                    for link in links:
                        if link.get('type') == 'application/pdf':
                            metadata['pdf_url'] = link.get('href')
                            break
                    
                    if not metadata['pdf_url']:
                        metadata['pdf_url'] = f"https://arxiv.org/pdf/{metadata['arxiv_id']}.pdf"
                    
                    # Download PDF
                    pdf_path = get_organized_pdf_path(query, metadata['arxiv_id'])
                    
                    if self.download_pdf(metadata['pdf_url'], pdf_path):
                        metadata['pdf_file'] = pdf_path
                        metadata['pdf_filename'] = os.path.basename(pdf_path)
                        papers_metadata.append(metadata)
                        logger.info(f"✅ Downloaded: {os.path.basename(pdf_path)}")
                    
                    time.sleep(config.ARXIV_RATE_LIMIT_DELAY)
                    
                except Exception as e:
                    logger.error(f"❌ Error processing entry {i}: {e}")
                    continue
            
            return papers_metadata
            
        except Exception as e:
            logger.error(f"❌ Direct requests method failed: {e}", exc_info=True)
            return []
    
    def download_pdf(self, url: str, filepath: str, max_retries: int = None) -> bool:
        """Download PDF with retry logic and validation."""
        max_retries = max_retries or config.PDF_MAX_RETRIES
        
        # Skip if file already exists and is valid
        if os.path.exists(filepath) and is_valid_pdf(filepath):
            logger.info(f"📦 PDF already exists: {os.path.basename(filepath)}")
            return True
        
        for attempt in range(max_retries):
            try:
                logger.debug(f"📥 Download attempt {attempt + 1}/{max_retries}")
                
                response = self.session.get(
                    url, 
                    timeout=config.PDF_DOWNLOAD_TIMEOUT, 
                    stream=True, 
                    allow_redirects=True
                )
                response.raise_for_status()
                
                # Check content type
                content_type = response.headers.get('content-type', '').lower()
                if 'pdf' not in content_type and 'octet-stream' not in content_type:
                    logger.warning(f"⚠️  Unexpected content type: {content_type}")
                
                # Write file
                os.makedirs(os.path.dirname(filepath), exist_ok=True)
                with open(filepath, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                
                # Validate downloaded file
                if not is_valid_pdf(filepath):
                    logger.error(f"❌ Downloaded file is not a valid PDF")
                    if os.path.exists(filepath):
                        os.remove(filepath)
                    continue
                
                file_size = os.path.getsize(filepath)
                logger.debug(f"✅ Download successful ({format_file_size(file_size)})")
                return True
                
            except requests.exceptions.Timeout:
                logger.warning(f"⏰ Download timeout (attempt {attempt + 1})")
            except requests.exceptions.RequestException as e:
                logger.warning(f"🌐 Network error (attempt {attempt + 1}): {e}")
            except Exception as e:
                logger.error(f"❌ Download error (attempt {attempt + 1}): {e}")
            
            # Clean up partial file
            if os.path.exists(filepath):
                try:
                    os.remove(filepath)
                except:
                    pass
            
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                logger.info(f"⏳ Waiting {wait_time}s before retry...")
                time.sleep(wait_time)
        
        logger.error(f"❌ Failed to download after {max_retries} attempts")
        return False
    
    def deduplicate_papers(self, papers_metadata: List[Dict]) -> List[Dict]:
        """
        Remove duplicate papers based on arXiv ID.
        
        Args:
            papers_metadata: List of paper metadata
        
        Returns:
            Deduplicated list
        """
        seen_ids = set()
        unique_papers = []
        duplicates_removed = 0
        
        for paper in papers_metadata:
            arxiv_id = paper.get('arxiv_id')
            if arxiv_id not in seen_ids:
                seen_ids.add(arxiv_id)
                unique_papers.append(paper)
            else:
                duplicates_removed += 1
                logger.debug(f"⚠️  Skipping duplicate: {paper['title'][:50]}...")
        
        if duplicates_removed > 0:
            logger.info(f"🧹 Removed {duplicates_removed} duplicate papers")
        
        return unique_papers
    
    def enrich_with_citations(self, papers_metadata: List[Dict]) -> List[Dict]:
        """
        Fetch citation counts from Semantic Scholar.
        
        Args:
            papers_metadata: List of paper metadata
        
        Returns:
            Enriched paper metadata with citation counts
        """
        logger.info("📊 Fetching citation metrics from Semantic Scholar...")
        
        for paper in papers_metadata:
            try:
                arxiv_id = paper.get('arxiv_id')
                if not arxiv_id or not is_valid_arxiv_id(arxiv_id):
                    continue
                
                url = f"{config.SEMANTIC_SCHOLAR_API}/paper/arXiv:{arxiv_id}"
                params = {'fields': 'citationCount,influentialCitationCount,year'}
                
                response = requests.get(url, params=params, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    paper['citation_count'] = data.get('citationCount', 0)
                    paper['influential_citation_count'] = data.get('influentialCitationCount', 0)
                    logger.debug(f"   {paper['title'][:40]}: {paper['citation_count']} citations")
                elif response.status_code == 404:
                    logger.debug(f"   Paper not found in Semantic Scholar: {arxiv_id}")
                    paper['citation_count'] = 0
                    paper['influential_citation_count'] = 0
                else:
                    logger.debug(f"   Could not fetch citations for {arxiv_id}: {response.status_code}")
                    paper['citation_count'] = 0
                    paper['influential_citation_count'] = 0
                
                time.sleep(0.5)  # Rate limiting
                
            except Exception as e:
                logger.debug(f"   Citation fetch error for {paper.get('title', 'Unknown')}: {e}")
                paper['citation_count'] = 0
                paper['influential_citation_count'] = 0
        
        return papers_metadata