# modules/compiler.py - IMPROVED Compilation Agent
import os
import fitz  # PyMuPDF
import pdfplumber
import json
import ollama
import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from config import config
from modules.utils import (
    logger, clean_text, get_file_hash, get_cache_path, 
    cache_exists, ProgressTracker
)

class CompilationAgent:
    """
    Enhanced compilation agent with:
    - Reference extraction
    - Equation extraction
    - Key contribution analysis
    - Caption extraction
    - Smart caching
    - Section-specific summarization
    """
    
    def __init__(self, input_folder: str, output_folder: str):
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.compiled_folder = os.path.join(output_folder, 'compiled')
        self.images_folder = os.path.join(output_folder, 'images')
        self.cache_folder = config.CACHE_DIR
        
        os.makedirs(self.compiled_folder, exist_ok=True)
        os.makedirs(self.images_folder, exist_ok=True)
        os.makedirs(self.cache_folder, exist_ok=True)
        
        # Configuration from config
        self.MODEL_NAME = config.OLLAMA_MODEL
        self.PAGE_LIMIT = config.PAGE_LIMIT
        self.WORD_LIMIT = config.WORD_LIMIT
        
        self.SECTION_KEYWORDS = [
            "abstract", "introduction", "background", "related work",
            "method", "methodology", "approach", "model",
            "experiment", "results", "discussion",
            "conclusion", "future work", "acknowledgement", "references"
        ]
        
        logger.info("CompilationAgent initialized")
    
    def check_ollama_connection(self) -> bool:
        """Check if Ollama service is running."""
        try:
            response = ollama.chat(
                model=self.MODEL_NAME,
                messages=[{"role": "user", "content": "test"}],
                options={"num_predict": 1}
            )
            logger.info(f"✅ Ollama connection successful (model: {self.MODEL_NAME})")
            return True
        except Exception as e:
            logger.error(f"❌ Ollama connection failed: {e}")
            logger.error("💡 Make sure to run 'ollama serve' in a separate terminal")
            return False
    
    def process_paper(self, paper_metadata: Dict) -> Optional[Dict]:
        """
        Process a single paper through the complete compilation pipeline.
        
        Args:
            paper_metadata: Paper metadata from scraper
        
        Returns:
            Compilation results or None on error
        """
        pdf_path = paper_metadata.get('pdf_file')
        if not pdf_path or not os.path.exists(pdf_path):
            logger.error(f"PDF not found: {pdf_path}")
            return None
        
        pdf_basename = os.path.splitext(os.path.basename(pdf_path))[0]
        arxiv_id = paper_metadata.get('arxiv_id', pdf_basename)
        
        logger.info(f"📄 Processing: {paper_metadata.get('title', 'Unknown')[:60]}")
        
        # Check cache first
        if config.ENABLE_CACHING and cache_exists(arxiv_id, 'compilation'):
            logger.info(f"📦 Using cached compilation for {arxiv_id}")
            cache_path = get_cache_path(arxiv_id, 'compilation')
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    cached_result = json.load(f)
                    cached_result['from_cache'] = True
                    return cached_result
            except Exception as e:
                logger.warning(f"Cache read error: {e}, reprocessing...")
        
        try:
            # Check if PDF is too large
            is_large = self.is_large_pdf(pdf_path)
            
            if is_large:
                logger.warning(f"⚠️  Large PDF detected (>{self.PAGE_LIMIT} pages or >{self.WORD_LIMIT} words)")
            
            # Extract all content
            logger.info("   Extracting sections...")
            sections_text = self.extract_sections_text(pdf_path)
            
            logger.info("   Extracting tables...")
            tables = self.extract_tables(pdf_path)
            
            logger.info("   Extracting images...")
            images = self.extract_images(pdf_path, pdf_basename)
            
            # NEW: Extract equations
            equations = []
            if config.ENABLE_EQUATIONS:
                logger.info("   Extracting equations...")
                equations = self.extract_equations(pdf_path)
            
            # NEW: Extract captions
            captions = {}
            if config.ENABLE_CAPTIONS:
                logger.info("   Extracting captions...")
                captions = self.extract_captions(pdf_path)
            
            # NEW: Extract references
            references = []
            if config.ENABLE_REFERENCES:
                logger.info("   Extracting references...")
                references = self.extract_references(sections_text)
            
            # Generate summaries
            sections_summary = {}
            if not is_large and sections_text:
                logger.info("   Generating summaries...")
                for section_name, text in sections_text.items():
                    if text.strip() and len(text.split()) > 30:
                        logger.info(f"      Summarizing: {section_name}")
                        sections_summary[section_name] = self.summarize_section(text, section_name)
                    else:
                        sections_summary[section_name] = text
            else:
                sections_summary = {sec: "[Skipped - Large PDF]" for sec in sections_text.keys()}
            
            # NEW: Extract key contributions
            contributions = {}
            if sections_summary:
                logger.info("   Extracting key contributions...")
                contributions = self.extract_key_contributions(sections_summary)
            
            # Prepare output data
            result = {
                'metadata': paper_metadata,
                'processing_info': {
                    'processed_at': datetime.now().isoformat(),
                    'is_large_pdf': is_large,
                    'sections_found': len(sections_text),
                    'tables_found': len(tables),
                    'images_found': len(images),
                    'equations_found': len(equations),
                    'references_found': len(references),
                    'model_used': self.MODEL_NAME
                },
                'sections_text': sections_text,
                'sections_summary': sections_summary,
                'tables': tables,
                'images': images,
                'equations': equations,
                'captions': captions,
                'references': references,
                'contributions': contributions,
                'status': 'completed',
                'from_cache': False
            }
            
            # Save to JSON
            json_filename = f"{pdf_basename}_compiled.json"
            json_path = os.path.join(self.compiled_folder, json_filename)
            
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            
            result['json_file'] = json_path
            
            # Save to cache
            if config.ENABLE_CACHING:
                cache_path = get_cache_path(arxiv_id, 'compilation')
                with open(cache_path, 'w', encoding='utf-8') as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)
            
            logger.info(f"✅ Compilation complete: {json_filename}")
            return result
            
        except Exception as e:
            logger.error(f"❌ Error processing {paper_metadata.get('title', 'Unknown')}: {e}", exc_info=True)
            return {
                'metadata': paper_metadata,
                'error': str(e),
                'status': 'failed',
                'processed_at': datetime.now().isoformat()
            }
    
    def is_large_pdf(self, pdf_path: str) -> bool:
        """Check if PDF exceeds processing limits."""
        try:
            total_words = 0
            with pdfplumber.open(pdf_path) as pdf:
                if len(pdf.pages) > self.PAGE_LIMIT:
                    return True
                for page in pdf.pages:
                    text = page.extract_text() or ""
                    total_words += len(text.split())
                    if total_words > self.WORD_LIMIT:
                        return True
            return False
        except:
            return True
    
    # === CONTINUED IN PART 2 ===
    # modules/compiler.py - Part 2: Extraction Methods
# (Continuation of CompilationAgent class)

    def extract_sections_text(self, pdf_path: str) -> Dict[str, str]:
        """Extract sections using improved algorithm."""
        sections = {}
        current_section = "Abstract"
        text_accum = []

        try:
            doc = fitz.open(pdf_path)
        except Exception as e:
            logger.error(f"Error opening PDF: {e}")
            return {}

        # Enhanced section patterns
        section_patterns = {
            'abstract': r'\b(abstract|summary)\b',
            'introduction': r'\b(introduction|intro)\b',
            'background': r'\b(background|related\s+work|literature\s+review|prior\s+work)\b',
            'methodology': r'\b(method|methodology|approach|framework|model|technique)\b',
            'experiments': r'\b(experiment|evaluation|results|findings|analysis)\b',
            'discussion': r'\b(discussion|interpretation|implications)\b',
            'conclusion': r'\b(conclusion|conclusions|summary|future\s+work)\b',
            'references': r'\b(references|bibliography|citations)\b',
            'acknowledgement': r'\b(acknowledgement|acknowledgments|thanks)\b'
        }

        compiled_patterns = {k: re.compile(v, re.IGNORECASE) for k, v in section_patterns.items()}
        numbered_section_pattern = re.compile(r'^\s*(\d+\.?\d*\.?\d*)\s+([A-Z][^.!?]*)', re.MULTILINE)
        
        all_text = ""
        font_info = []
        
        # Collect text with font information
        for page_num, page in enumerate(doc):
            blocks = page.get_text("dict")["blocks"]
            for block in blocks:
                if "lines" not in block:
                    continue
                for line in block["lines"]:
                    line_text = ""
                    line_fonts = []
                    for span in line["spans"]:
                        text = span.get("text", "").strip()
                        if text:
                            line_text += text + " "
                            line_fonts.append({
                                'size': span.get('size', 0),
                                'flags': span.get('flags', 0),
                                'font': span.get('font', '')
                            })
                    
                    if line_text.strip():
                        all_text += line_text + "\n"
                        if line_fonts:
                            avg_size = sum(f['size'] for f in line_fonts) / len(line_fonts)
                            has_bold = any(f['flags'] & 2**4 for f in line_fonts)
                            font_info.append({
                                'text': line_text.strip(),
                                'size': avg_size,
                                'bold': has_bold,
                                'page': page_num
                            })

        # Calculate header threshold
        sizes = [info['size'] for info in font_info if info['size'] > 0]
        if sizes:
            median_size = sorted(sizes)[len(sizes)//2]
            header_threshold = median_size + 1.0
        else:
            header_threshold = 12.0

        # Process lines
        lines = all_text.split('\n')
        i = 0
        
        while i < len(lines):
            line = lines[i].strip()
            if not line:
                i += 1
                continue
            
            # Skip references content
            if current_section.lower() == 'references':
                i += 1
                continue
            
            # Check for numbered sections
            numbered_match = numbered_section_pattern.match(line)
            if numbered_match:
                if text_accum:
                    sections[current_section] = clean_text(" ".join(text_accum))
                    text_accum = []
                current_section = f"{numbered_match.group(1)} {numbered_match.group(2).strip()}"
                i += 1
                continue
            
            # Check for keyword-based sections
            section_found = False
            for section_key, pattern in compiled_patterns.items():
                if pattern.search(line) and self.is_potential_header(line, font_info, i, header_threshold):
                    if len(line.split()) <= 8 and len(line) < 100:
                        if text_accum:
                            sections[current_section] = clean_text(" ".join(text_accum))
                            text_accum = []
                        current_section = section_key.title()
                        section_found = True
                        break
            
            if section_found:
                i += 1
                continue
            
            # Check formatting-based headers
            if self.is_potential_header(line, font_info, i, header_threshold):
                words = line.split()
                if (2 <= len(words) <= 8 and 
                    (line.istitle() or line.isupper()) and 
                    not line.endswith('.')):
                    
                    if text_accum:
                        sections[current_section] = clean_text(" ".join(text_accum))
                        text_accum = []
                    current_section = line.title()
                    i += 1
                    continue
            
            # Regular content
            if not self.is_junk_line(line):
                text_accum.append(line)
            
            i += 1
        
        # Final section
        if text_accum:
            sections[current_section] = clean_text(" ".join(text_accum))
        
        doc.close()
        return self.merge_small_sections(sections)
    
    def is_potential_header(self, line: str, font_info: List, line_index: int, threshold: float) -> bool:
        """Check if line could be a section header."""
        if line_index >= len(font_info):
            return False
        info = font_info[line_index]
        return (info['size'] >= threshold or info['bold']) and len(line.split()) <= 8
    
    def is_junk_line(self, line: str) -> bool:
        """Filter out junk lines."""
        line = line.strip()
        if len(line) < 3 or re.match(r'^\d+$', line):
            return True
        if re.match(r'^\[?\d+\]?$', line):
            return True
        if re.match(r'^(https?://|doi:|www\.)', line.lower()):
            return True
        return False
    
    def merge_small_sections(self, sections: Dict, min_words: int = 30) -> Dict:
        """Merge small sections with previous section."""
        merged = {}
        prev_key = None
        
        for key, content in sections.items():
            word_count = len(content.split())
            if word_count < min_words and prev_key and prev_key in merged:
                merged[prev_key] += " " + content
            else:
                merged[key] = content
                prev_key = key
        return merged
    
    def extract_tables(self, pdf_path: str) -> List[Dict]:
        """Extract tables from PDF."""
        tables = []
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for i, page in enumerate(pdf.pages):
                    page_tables = page.extract_tables()
                    for j, table in enumerate(page_tables):
                        if table and len(table) > 0:
                            tables.append({
                                "page": i + 1,
                                "table_index": j,
                                "rows": len(table),
                                "cols": len(table[0]) if table else 0,
                                "content": table
                            })
        except Exception as e:
            logger.error(f"Error extracting tables: {e}")
        return tables
    
    def extract_images(self, pdf_path: str, pdf_basename: str) -> List[str]:
        """Extract images from PDF."""
        images = []
        try:
            doc = fitz.open(pdf_path)
            for page_index, page in enumerate(doc):
                page_images = page.get_images(full=True)
                for img_index, img in enumerate(page_images):
                    try:
                        xref = img[0]
                        pix = fitz.Pixmap(doc, xref)
                        
                        img_filename = f"{pdf_basename}_page{page_index+1}_img{img_index}.png"
                        img_path = os.path.join(self.images_folder, img_filename)
                        
                        if pix.n < 5:
                            pix.save(img_path)
                        else:
                            pix_rgb = fitz.Pixmap(fitz.csRGB, pix)
                            pix_rgb.save(img_path)
                            pix_rgb = None
                        
                        pix = None
                        images.append(img_path)
                    except Exception as e:
                        logger.debug(f"Error extracting image: {e}")
            doc.close()
        except Exception as e:
            logger.error(f"Error in image extraction: {e}")
        return images
    
    def extract_equations(self, pdf_path: str) -> List[Dict]:
        """
        Extract LaTeX-style equations from PDF.
        """
        equations = []
        try:
            doc = fitz.open(pdf_path)
            for page_num, page in enumerate(doc):
                text = page.get_text()
                
                # LaTeX equation patterns
                latex_patterns = [
                    (r'\\begin\{equation\}(.*?)\\end\{equation\}', 'equation'),
                    (r'\\begin\{align\}(.*?)\\end\{align\}', 'align'),
                    (r'\$\$(.*?)\$\$', 'display'),
                    (r'\\\[(.*?)\\\]', 'display'),
                    (r'\$(.*?)\$', 'inline')
                ]
                
                for pattern, eq_type in latex_patterns:
                    matches = re.findall(pattern, text, re.DOTALL)
                    for match in matches:
                        cleaned = match.strip()
                        if len(cleaned) > 5:  # Skip very short matches
                            equations.append({
                                'page': page_num + 1,
                                'content': cleaned,
                                'type': eq_type
                            })
            
            doc.close()
            
            # Deduplicate equations
            seen = set()
            unique_equations = []
            for eq in equations:
                if eq['content'] not in seen:
                    seen.add(eq['content'])
                    unique_equations.append(eq)
            
            return unique_equations[:20]  # Limit to 20 equations
            
        except Exception as e:
            logger.error(f"Equation extraction error: {e}")
            return []
    
    def extract_captions(self, pdf_path: str) -> Dict[str, List[Dict]]:
        """Extract figure and table captions."""
        captions = {'figures': [], 'tables': []}
        
        try:
            doc = fitz.open(pdf_path)
            for page_num, page in enumerate(doc):
                text = page.get_text()
                
                # Figure captions
                fig_pattern = r'(?:Figure|Fig\.?)\s+(\d+)[:\.]?\s*([^\n]+)'
                for match in re.finditer(fig_pattern, text, re.IGNORECASE):
                    captions['figures'].append({
                        'number': match.group(1),
                        'caption': match.group(2).strip()[:200],
                        'page': page_num + 1
                    })
                
                # Table captions
                table_pattern = r'Table\s+(\d+)[:\.]?\s*([^\n]+)'
                for match in re.finditer(table_pattern, text, re.IGNORECASE):
                    captions['tables'].append({
                        'number': match.group(1),
                        'caption': match.group(2).strip()[:200],
                        'page': page_num + 1
                    })
            
            doc.close()
        except Exception as e:
            logger.error(f"Caption extraction error: {e}")
        
        return captions
    
    def extract_references(self, sections_text: Dict) -> List[Dict]:
        """Parse references section."""
        references_section = (sections_text.get('References', '') or 
                            sections_text.get('Bibliography', '') or 
                            sections_text.get('references', ''))
        
        if not references_section:
            return []
        
        references = []
        
        # Pattern: [1] Authors. (Year). Title.
        pattern = r'\[(\d+)\]\s*([^.]+?)\.?\s*\((\d{4})\)\.?\s*([^.]+)\.?'
        
        for match in re.finditer(pattern, references_section):
            references.append({
                'id': match.group(1),
                'authors': match.group(2).strip(),
                'year': match.group(3),
                'title': match.group(4).strip()
            })
        
        # Alternative pattern: Author et al., "Title", Year
        if len(references) < 3:
            alt_pattern = r'([A-Z][a-z]+(?:\s+et\s+al\.)?),?\s*["""]([^"""]+)["""],?\s*(\d{4})'
            for match in re.finditer(alt_pattern, references_section):
                references.append({
                    'id': str(len(references) + 1),
                    'authors': match.group(1).strip(),
                    'year': match.group(3),
                    'title': match.group(2).strip()
                })
        
        return references[:50]  # Limit to 50 references
    
    # === CONTINUED IN PART 3 (Summarization) ===
    # modules/compiler.py - Part 3: Summarization & Key Contributions
# (Continuation of CompilationAgent class)

    def summarize_section(self, text: str, section_name: str = None) -> str:
        """
        Context-aware section summarization.
        
        Args:
            text: Section text to summarize
            section_name: Name of section (for context-specific prompts)
        
        Returns:
            Summary text
        """
        text = text.strip()
        if not text or len(text.split()) < 30:
            return text

        # Section-specific prompts
        section_prompts = {
            'Abstract': """Condense this abstract to 2-3 sentences highlighting:
1. The main problem addressed
2. The proposed solution/method
3. Key results/contributions""",
            
            'Introduction': """Summarize the introduction focusing on:
1. The research problem and motivation
2. Main research questions or objectives
3. Key innovations or proposed approach""",
            
            'Methodology': """Summarize the methodology focusing on:
1. Core algorithmic approach
2. Key innovations or modifications
3. Critical implementation details""",
            
            'Results': """Extract key findings:
1. Main quantitative results (metrics, scores)
2. Comparisons with baselines
3. Statistical significance of improvements""",
            
            'Experiments': """Summarize experimental setup and results:
1. Experimental design and datasets used
2. Key performance metrics
3. Main findings and comparisons""",
            
            'Discussion': """Summarize the discussion:
1. Interpretation of results
2. Comparison with related work
3. Implications and significance""",
            
            'Conclusion': """Identify:
1. Main takeaways and contributions
2. Stated limitations
3. Future work directions""",
            
            'Background': """Summarize background and related work:
1. Key concepts and definitions
2. Relevant prior research
3. Gaps in existing literature"""
        }
        
        # Select appropriate prompt
        if config.ENABLE_SECTION_SPECIFIC_PROMPTS and section_name:
            prompt_template = section_prompts.get(section_name, 
                "Summarize the following text clearly and concisely, capturing main ideas:")
        else:
            prompt_template = "Summarize the following text clearly and concisely, capturing main ideas:"
        
        summaries = []
        for idx, chunk in enumerate(self.chunk_text(text, max_words=config.CHUNK_SIZE_WORDS)):
            logger.debug(f"   Summarizing chunk {idx+1} ({len(chunk.split())} words)...")
            
            prompt = f"""{prompt_template}

Provide only the summary, without any introductory phrases.

---
{chunk}
---

Summary:"""
            
            try:
                resp = ollama.chat(
                    model=self.MODEL_NAME,
                    messages=[{"role": "user", "content": prompt}],
                    options={"temperature": 0.3}  # Lower temperature for factual summaries
                )
                summaries.append(resp['message']['content'].strip())
            except Exception as e:
                logger.error(f"Summarization error: {e}")
                summaries.append("[Summarization failed]")

        # If multiple chunks, combine summaries
        if len(summaries) > 1:
            final_prompt = f"""Combine the following partial summaries into a coherent, concise overall summary:

{chr(10).join(f'{i+1}. {s}' for i, s in enumerate(summaries))}

Provide only the final summary:"""
            
            try:
                resp = ollama.chat(
                    model=self.MODEL_NAME,
                    messages=[{"role": "user", "content": final_prompt}],
                    options={"temperature": 0.3}
                )
                return resp['message']['content'].strip()
            except:
                return " ".join(summaries)

        return summaries[0] if summaries else "[Empty]"
    
    def extract_key_contributions(self, sections_summary: Dict) -> Dict:
        """
        Extract key contributions using LLM analysis.
        
        This is CRITICAL for knowledge graph construction.
        
        Args:
            sections_summary: Dictionary of section summaries
        
        Returns:
            Dictionary with structured contribution information
        """
        # Combine relevant sections
        relevant_sections = ['abstract', 'introduction', 'conclusion', 
                           'methodology', 'results']
        
        combined_text = ""
        for section_name, summary in sections_summary.items():
            section_lower = section_name.lower()
            if any(rel in section_lower for rel in relevant_sections):
                combined_text += f"\n{section_name}:\n{summary}\n"
        
        if not combined_text.strip():
            return {
                "main_problem": "Not available",
                "key_innovation": "Not available",
                "core_methodology": "Not available",
                "major_results": "Not available",
                "limitations": "Not available",
                "research_gaps": "Not available"
            }
        
        prompt = f"""Analyze this research paper and extract structured information.

Paper content:
{combined_text}

Extract the following information and respond ONLY with a valid JSON object (no additional text):

{{
  "main_problem": "What specific problem does this paper address?",
  "key_innovation": "What is novel or innovative about their approach?",
  "core_methodology": "What are the main techniques/algorithms used?",
  "major_results": "What are the key quantitative or qualitative results?",
  "limitations": "What limitations do the authors acknowledge?",
  "research_gaps": "What future work or research gaps do they identify?"
}}

JSON response:"""
        
        try:
            resp = ollama.chat(
                model=self.MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.2},
                format='json'  # Request JSON format
            )
            
            response_text = resp['message']['content'].strip()
            
            # Clean up response (remove markdown code blocks if present)
            if response_text.startswith('```'):
                response_text = re.sub(r'```json\n?|\n?```', '', response_text).strip()
            
            contributions = json.loads(response_text)
            
            # Validate structure
            required_keys = ["main_problem", "key_innovation", "core_methodology", 
                           "major_results", "limitations", "research_gaps"]
            
            for key in required_keys:
                if key not in contributions:
                    contributions[key] = "Not available"
            
            logger.info("   ✅ Key contributions extracted successfully")
            return contributions
            
        except json.JSONDecodeError as e:
            logger.error(f"   ❌ JSON parsing error: {e}")
            logger.debug(f"   Response was: {response_text[:200]}")
            return {
                "main_problem": "Extraction failed",
                "key_innovation": "Extraction failed",
                "core_methodology": "Extraction failed",
                "major_results": "Extraction failed",
                "limitations": "Extraction failed",
                "research_gaps": "Extraction failed"
            }
        except Exception as e:
            logger.error(f"   ❌ Contribution extraction error: {e}")
            return {
                "main_problem": f"Error: {str(e)}",
                "key_innovation": "Error",
                "core_methodology": "Error",
                "major_results": "Error",
                "limitations": "Error",
                "research_gaps": "Error"
            }
    
    def chunk_text(self, text: str, max_words: int = 500) -> List[str]:
        """
        Split text into chunks for processing.
        
        Args:
            text: Text to chunk
            max_words: Maximum words per chunk
        
        Yields:
            Text chunks
        """
        words = text.split()
        for i in range(0, len(words), max_words):
            yield " ".join(words[i:i+max_words])