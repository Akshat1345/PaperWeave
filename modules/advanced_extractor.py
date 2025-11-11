# modules/advanced_extractor.py - FINAL VERSION
import os
import fitz
import re
import ollama
from collections import defaultdict
import json

class AdvancedSectionExtractor:
    """
    Multi-pass section extraction with LLM validation.
    This is the most accurate approach without training a custom model.
    """
    
    def __init__(self, model_name="llama3.2:latest"):
        self.model_name = model_name
        
        # Common academic section patterns
        self.standard_sections = {
            'abstract': ['abstract', 'summary', 'executive summary'],
            'introduction': ['introduction', 'background and introduction', 'intro'],
            'related_work': ['related work', 'literature review', 'prior work', 'background', 'previous work'],
            'methodology': ['method', 'methods', 'methodology', 'approach', 'proposed method', 'our approach'],
            'experiments': ['experiment', 'experiments', 'experimental setup', 'evaluation', 'experimental results'],
            'results': ['results', 'results and discussion', 'findings', 'experimental results'],
            'discussion': ['discussion', 'analysis', 'interpretation'],
            'conclusion': ['conclusion', 'conclusions', 'concluding remarks', 'summary and conclusion'],
            'future_work': ['future work', 'future directions', 'future research'],
            'acknowledgments': ['acknowledgment', 'acknowledgments', 'acknowledgements'],
            'references': ['references', 'bibliography', 'citations']
        }
    
    def extract_sections_with_validation(self, pdf_path):
        """
        Three-pass extraction with LLM validation.
        """
        print("🔍 Starting advanced section extraction...")
        
        # Pass 1: Get all candidate sections
        candidates = self._extract_candidate_sections(pdf_path)
        
        if not candidates:
            print("❌ No candidate sections found")
            return {}
        
        print(f"✅ Found {len(candidates)} candidate sections")
        
        # Pass 2: LLM validation of section boundaries
        validated_sections = self._validate_with_llm(candidates)
        
        print(f"✅ Validated {len(validated_sections)} accurate sections")
        
        # Pass 3: Extract full content for validated sections
        final_sections = self._extract_full_content(pdf_path, validated_sections)
        
        return final_sections
    
    def _extract_candidate_sections(self, pdf_path):
        """Extract ALL possible section candidates."""
        try:
            doc = fitz.open(pdf_path)
        except Exception as e:
            print(f"Error opening PDF: {e}")
            return []
        
        candidates = []
        
        # Collect all text with formatting info
        for page_num, page in enumerate(doc):
            blocks = page.get_text("dict")["blocks"]
            
            for block_idx, block in enumerate(blocks):
                if "lines" not in block:
                    continue
                
                for line in block["lines"]:
                    line_text = ""
                    spans = line.get("spans", [])
                    
                    if not spans:
                        continue
                    
                    font_sizes = []
                    is_bold = False
                    
                    for span in spans:
                        text = span.get("text", "").strip()
                        if text:
                            line_text += text + " "
                            font_sizes.append(span.get('size', 0))
                            if span.get('flags', 0) & 2**4:
                                is_bold = True
                    
                    line_text = line_text.strip()
                    if not line_text or len(line_text) < 2:
                        continue
                    
                    avg_font = sum(font_sizes) / len(font_sizes) if font_sizes else 0
                    
                    # Check if potential header
                    if self._is_potential_section_header(line_text, avg_font, is_bold, page_num):
                        candidates.append({
                            'text': line_text,
                            'page': page_num + 1,
                            'font_size': avg_font,
                            'is_bold': is_bold,
                            'block_idx': block_idx
                        })
        
        doc.close()
        return candidates
    
    def _is_potential_section_header(self, text, font_size, is_bold, page_num):
        """Initial filtering - be permissive."""
        if len(text) > 150 or len(text.split()) > 15:
            return False
        
        # Skip pure numbers, citations, URLs
        if re.match(r'^\d+$', text) or re.match(r'^\[?\d+\]?$', text):
            return False
        if re.match(r'^(https?://|doi:|www\.)', text.lower()):
            return False
        
        words = text.split()
        
        # Strategy 1: Numbered sections
        if re.match(r'^\d+\.?\d*\.?\d*\s+', text):
            return True
        
        # Strategy 2: All caps short text
        if text.isupper() and 2 <= len(words) <= 8:
            return True
        
        # Strategy 3: Known keywords
        if text.istitle() or text[0].isupper():
            text_lower = text.lower()
            for variants in self.standard_sections.values():
                if any(variant in text_lower for variant in variants):
                    return True
        
        # Strategy 4: Bold + reasonable size
        if is_bold and len(words) <= 8:
            return True
        
        # Strategy 5: Roman numerals
        if re.match(r'^[IVX]+\.?\s+', text):
            return True
        
        return False
    
    def _validate_with_llm(self, candidates):
        """Use LLM to validate section headers."""
        if not candidates:
            return []
        
        validated = []
        batch_size = 10
        
        for i in range(0, len(candidates), batch_size):
            batch = candidates[i:i + batch_size]
            prompt = self._create_validation_prompt(batch)
            
            try:
                print(f"🤖 Validating batch {i//batch_size + 1}...")
                response = ollama.chat(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}]
                )
                
                result = response['message']['content']
                validated_indices = self._parse_validation_response(result)
                
                for idx in validated_indices:
                    if 0 <= idx < len(batch):
                        validated.append(batch[idx])
                
            except Exception as e:
                print(f"⚠️  Validation error: {e}")
                # Fallback: accept exact matches
                for candidate in batch:
                    if self._strict_pattern_match(candidate['text']):
                        validated.append(candidate)
        
        return validated
    
    def _create_validation_prompt(self, candidates):
        """Create validation prompt."""
        candidate_list = "\n".join([
            f"{i}. \"{c['text']}\" (Page {c['page']}, Bold: {c['is_bold']})"
            for i, c in enumerate(candidates)
        ])
        
        prompt = f"""You are analyzing potential section headers from an academic research paper.
Determine which candidates are REAL section headers.

REAL section headers:
- Main divisions (Introduction, Methods, Results, Conclusion)
- Subsection titles
- Numbered sections (1., 2.1, etc.)

NOT section headers:
- Figure/Table captions
- Author names or affiliations
- Regular bold text
- References to sections in body text

Candidates:
{candidate_list}

Respond ONLY with comma-separated numbers of real headers.
Example: 0,2,5,7
If none are real headers, respond: NONE

Your response:"""
        
        return prompt
    
    def _parse_validation_response(self, response):
        """Parse LLM response."""
        response = response.strip().upper()
        if response == "NONE":
            return []
        numbers = re.findall(r'\d+', response)
        return [int(n) for n in numbers]
    
    def _strict_pattern_match(self, text):
        """Fallback: strict matching."""
        text_lower = text.lower().strip()
        
        for variants in self.standard_sections.values():
            if text_lower in variants:
                return True
        
        if re.match(r'^\d+\.?\d*\.?\d*\s+[A-Z]', text):
            return True
        
        return False
    
    def _extract_full_content(self, pdf_path, validated_sections):
        """Extract full content for validated sections."""
        if not validated_sections:
            return {}
        
        # Sort by page and position
        validated_sections.sort(key=lambda x: (x['page'], x['block_idx']))
        
        try:
            doc = fitz.open(pdf_path)
        except:
            return {}
        
        sections = {}
        
        # Extract all text
        full_text = ""
        for page in doc:
            full_text += page.get_text()
        
        doc.close()
        
        # For each section, find content until next section
        for i, section in enumerate(validated_sections):
            section_name = section['text']
            
            # Find section start in text
            section_start = full_text.find(section_name)
            if section_start == -1:
                continue
            
            # Find next section or end
            if i < len(validated_sections) - 1:
                next_section_name = validated_sections[i + 1]['text']
                section_end = full_text.find(next_section_name, section_start + len(section_name))
                if section_end == -1:
                    section_end = len(full_text)
            else:
                section_end = len(full_text)
            
            # Extract content
            content = full_text[section_start + len(section_name):section_end]
            content = self._clean_section_content(content)
            
            if content and len(content.split()) > 10:  # At least 10 words
                sections[section_name] = content
        
        return sections
    
    def _clean_section_content(self, text):
        """Clean extracted content."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove page numbers
        text = re.sub(r'\b\d+\b(?=\s*$)', '', text)
        
        # Remove citations
        text = re.sub(r'\[\d+(?:,\s*\d+)*\]', '', text)
        
        # Remove URLs
        text = re.sub(r'https?://\S+', '', text)
        
        return text.strip()


def extract_sections_accurately(pdf_path, model_name="llama3.2:latest"):
    """
    Drop-in replacement for extract_sections_text.
    """
    extractor = AdvancedSectionExtractor(model_name)
    return extractor.extract_sections_with_validation(pdf_path)