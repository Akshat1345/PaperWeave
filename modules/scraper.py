# modules/scraper.py - FIXED VERSION
import requests
import os
import time
import xml.etree.ElementTree as ET
from urllib.parse import quote_plus, urlparse
import feedparser

class ArxivScraper:
    def __init__(self, output_folder):
        self.output_folder = output_folder
        self.pdfs_folder = os.path.join(output_folder, 'pdfs')
        os.makedirs(self.pdfs_folder, exist_ok=True)
        
        # Use the correct arXiv API base URL
        self.base_url = "http://export.arxiv.org/api/query"
        
        # Request session with proper headers
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'AI Research Assistant/1.0 (Educational Project)',
            'Accept': 'application/atom+xml'
        })
    
    def search_and_download(self, query, max_results=5):
        """Search arXiv and download papers using direct API calls."""
        print(f"🔍 Searching arXiv for: '{query}'")
        
        try:
            # Method 1: Try with feedparser (more reliable)
            papers_metadata = self.search_with_feedparser(query, max_results)
            
            if not papers_metadata:
                # Method 2: Fallback to direct requests
                print("📡 Trying alternative method...")
                papers_metadata = self.search_with_requests(query, max_results)
            
            if not papers_metadata:
                print("❌ No papers found with either method")
                return []
            
            print(f"✅ Found {len(papers_metadata)} papers")
            return papers_metadata
            
        except Exception as e:
            print(f"❌ Error in arXiv search: {e}")
            return []
    
    def search_with_feedparser(self, query, max_results):
        """Use feedparser library for more reliable parsing."""
        try:
            # Construct search URL
            search_query = quote_plus(query)
            url = f"{self.base_url}?search_query=all:{search_query}&start=0&max_results={max_results}&sortBy=relevance&sortOrder=descending"
            
            print(f"📡 Fetching from: {url}")
            
            # Parse feed
            feed = feedparser.parse(url)
            
            if not hasattr(feed, 'entries') or not feed.entries:
                print("❌ No entries found in feed")
                return []
            
            papers_metadata = []
            
            for i, entry in enumerate(feed.entries[:max_results]):
                print(f"📄 Processing paper {i+1}/{len(feed.entries[:max_results])}: {entry.title}")
                
                # Extract metadata
                metadata = {
                    'title': entry.title.replace('\n', ' ').strip(),
                    'authors': [author.name for author in getattr(entry, 'authors', [])],
                    'abstract': getattr(entry, 'summary', '').replace('\n', ' ').strip(),
                    'published': getattr(entry, 'published', ''),
                    'arxiv_id': entry.id.split('/')[-1] if hasattr(entry, 'id') else f'unknown_{i}',
                    'categories': [tag.term for tag in getattr(entry, 'tags', [])],
                    'pdf_url': None
                }
                
                # Find PDF link
                for link in getattr(entry, 'links', []):
                    if link.type == 'application/pdf':
                        metadata['pdf_url'] = link.href
                        break
                
                # Fallback PDF URL construction
                if not metadata['pdf_url'] and hasattr(entry, 'id'):
                    paper_id = entry.id.split('/')[-1]
                    metadata['pdf_url'] = f"https://arxiv.org/pdf/{paper_id}.pdf"
                
                if metadata['pdf_url']:
                    # Download PDF
                    pdf_filename = f"{metadata['arxiv_id'].replace('/', '_').replace(':', '_')}.pdf"
                    pdf_path = os.path.join(self.pdfs_folder, pdf_filename)
                    
                    if self.download_pdf(metadata['pdf_url'], pdf_path):
                        metadata['pdf_file'] = pdf_path
                        metadata['pdf_filename'] = pdf_filename
                        papers_metadata.append(metadata)
                        print(f"✅ Downloaded: {pdf_filename}")
                    else:
                        print(f"❌ Failed to download: {metadata['title'][:50]}...")
                
                # Rate limiting
                time.sleep(1)
            
            return papers_metadata
            
        except Exception as e:
            print(f"❌ Feedparser method failed: {e}")
            return []
    
    def search_with_requests(self, query, max_results):
        """Alternative method using direct requests."""
        try:
            # Construct search URL
            search_query = quote_plus(query)
            url = f"{self.base_url}?search_query=all:{search_query}&start=0&max_results={max_results}&sortBy=relevance&sortOrder=descending"
            
            print(f"📡 Direct request to: {url}")
            
            # Make request with proper handling
            response = self.session.get(url, timeout=30, allow_redirects=True)
            response.raise_for_status()
            
            # Parse XML
            root = ET.fromstring(response.content)
            
            # Define namespace
            ns = {'atom': 'http://www.w3.org/2005/Atom', 
                  'arxiv': 'http://arxiv.org/schemas/atom'}
            
            entries = root.findall('atom:entry', ns)
            
            if not entries:
                print("❌ No entries found in XML response")
                return []
            
            papers_metadata = []
            
            for i, entry in enumerate(entries[:max_results]):
                try:
                    title_elem = entry.find('atom:title', ns)
                    title = title_elem.text.replace('\n', ' ').strip() if title_elem is not None else f"Unknown Title {i}"
                    
                    print(f"📄 Processing paper {i+1}: {title[:50]}...")
                    
                    # Extract metadata
                    metadata = {
                        'title': title,
                        'authors': [],
                        'abstract': '',
                        'published': '',
                        'arxiv_id': f'paper_{i}',
                        'categories': [],
                        'pdf_url': None
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
                    
                    # Fallback PDF URL
                    if not metadata['pdf_url']:
                        metadata['pdf_url'] = f"https://arxiv.org/pdf/{metadata['arxiv_id']}.pdf"
                    
                    # Download PDF
                    if metadata['pdf_url']:
                        pdf_filename = f"{metadata['arxiv_id'].replace('/', '_').replace(':', '_')}.pdf"
                        pdf_path = os.path.join(self.pdfs_folder, pdf_filename)
                        
                        if self.download_pdf(metadata['pdf_url'], pdf_path):
                            metadata['pdf_file'] = pdf_path
                            metadata['pdf_filename'] = pdf_filename
                            papers_metadata.append(metadata)
                            print(f"✅ Downloaded: {pdf_filename}")
                        else:
                            print(f"❌ Failed to download: {title[:50]}...")
                    
                    # Rate limiting
                    time.sleep(1.5)
                    
                except Exception as e:
                    print(f"❌ Error processing entry {i}: {e}")
                    continue
            
            return papers_metadata
            
        except Exception as e:
            print(f"❌ Direct requests method failed: {e}")
            return []
    
    def download_pdf(self, url, filepath, max_retries=3):
        """Download PDF with improved error handling."""
        for attempt in range(max_retries):
            try:
                print(f"📥 Downloading attempt {attempt + 1}: {os.path.basename(filepath)}")
                
                # Use session for consistent headers
                response = self.session.get(url, timeout=45, stream=True, allow_redirects=True)
                response.raise_for_status()
                
                # Check if it's actually a PDF
                content_type = response.headers.get('content-type', '').lower()
                if 'pdf' not in content_type and 'octet-stream' not in content_type:
                    print(f"⚠️  Warning: Unexpected content type: {content_type}")
                
                # Write file
                with open(filepath, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                
                # Verify file size and basic PDF structure
                file_size = os.path.getsize(filepath)
                if file_size < 1000:  # Less than 1KB
                    print(f"❌ File too small ({file_size} bytes)")
                    os.remove(filepath)
                    continue
                
                # Basic PDF validation
                with open(filepath, 'rb') as f:
                    header = f.read(8)
                    if not header.startswith(b'%PDF'):
                        print(f"❌ Not a valid PDF file")
                        os.remove(filepath)
                        continue
                
                print(f"✅ Successfully downloaded ({file_size:,} bytes)")
                return True
                
            except requests.exceptions.Timeout:
                print(f"⏰ Download timeout (attempt {attempt + 1})")
            except requests.exceptions.RequestException as e:
                print(f"🌐 Network error (attempt {attempt + 1}): {e}")
            except Exception as e:
                print(f"❌ Download error (attempt {attempt + 1}): {e}")
            
            # Clean up partial file
            if os.path.exists(filepath):
                try:
                    os.remove(filepath)
                except:
                    pass
            
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff
                print(f"⏳ Waiting {wait_time}s before retry...")
                time.sleep(wait_time)
        
        print(f"❌ Failed to download after {max_retries} attempts")
        return False