# # modules/compiler.py - FINAL VERSION
# import os
# import fitz  # PyMuPDF
# import pdfplumber
# import json
# import ollama
# import re
# from datetime import datetime

# class CompilationAgent:
#     def __init__(self, input_folder, output_folder):
#         self.input_folder = input_folder
#         self.output_folder = output_folder
#         self.compiled_folder = os.path.join(output_folder, 'compiled')
#         self.images_folder = os.path.join(output_folder, 'images')
        
#         os.makedirs(self.compiled_folder, exist_ok=True)
#         os.makedirs(self.images_folder, exist_ok=True)
        
#         # Configuration
#         self.MODEL_NAME = "llama3.2:latest"
#         self.PAGE_LIMIT = 20
#         self.WORD_LIMIT = 15000
    
#     def process_paper(self, paper_metadata):
#         """Process a single paper through the compilation pipeline."""
#         pdf_path = paper_metadata.get('pdf_file')
#         if not pdf_path or not os.path.exists(pdf_path):
#             return None
        
#         pdf_basename = os.path.splitext(os.path.basename(pdf_path))[0]
#         print(f"\nProcessing: {paper_metadata.get('title', 'Unknown')}")
        
#         try:
#             # Check if PDF is too large
#             is_large = self.is_large_pdf(pdf_path)
            
#             # Extract content using advanced method
#             sections_text = self.extract_sections_text(pdf_path)
#             tables = self.extract_tables(pdf_path)
#             images = self.extract_images(pdf_path, pdf_basename)
            
#             # Generate summaries (skip for large PDFs)
#             sections_summary = {}
#             if not is_large and sections_text:
#                 for section_name, text in sections_text.items():
#                     if text.strip() and len(text.split()) >= 30:
#                         print(f"  -> Summarizing: {section_name}")
#                         sections_summary[section_name] = self.summarize_section(text)
#                     else:
#                         sections_summary[section_name] = text  # Too short, keep as-is
#             else:
#                 sections_summary = {sec: "[Skipped - Large PDF]" for sec in sections_text.keys()}
            
#             # Prepare output data
#             result = {
#                 'metadata': paper_metadata,
#                 'processing_info': {
#                     'processed_at': datetime.now().isoformat(),
#                     'is_large_pdf': is_large,
#                     'sections_found': len(sections_text),
#                     'tables_found': len(tables),
#                     'images_found': len(images)
#                 },
#                 'sections_text': sections_text,
#                 'sections_summary': sections_summary,
#                 'tables': tables,
#                 'images': images,
#                 'status': 'completed'
#             }
            
#             # Save to JSON
#             json_filename = f"{pdf_basename}_compiled.json"
#             json_path = os.path.join(self.compiled_folder, json_filename)
            
#             with open(json_path, 'w', encoding='utf-8') as f:
#                 json.dump(result, f, ensure_ascii=False, indent=2)
            
#             result['json_file'] = json_path
#             print(f"✅ Compiled: {json_filename}")
            
#             return result
            
#         except Exception as e:
#             print(f"❌ Error processing {paper_metadata.get('title', 'Unknown')}: {e}")
#             return {
#                 'metadata': paper_metadata,
#                 'error': str(e),
#                 'status': 'failed',
#                 'processed_at': datetime.now().isoformat()
#             }
    
#     def is_large_pdf(self, pdf_path):
#         """Check if PDF is too large for processing."""
#         try:
#             total_words = 0
#             with pdfplumber.open(pdf_path) as pdf:
#                 if len(pdf.pages) > self.PAGE_LIMIT:
#                     return True
#                 for page in pdf.pages:
#                     text = page.extract_text() or ""
#                     total_words += len(text.split())
#                     if total_words > self.WORD_LIMIT:
#                         return True
#             return False
#         except:
#             return True
    
#     def extract_sections_text(self, pdf_path):
#         """
#         Extract sections using advanced LLM-validated approach.
#         Falls back to basic extraction if advanced method fails.
#         """
#         try:
#             # Try advanced extraction first
#             from modules.advanced_extractor import extract_sections_accurately
            
#             print("🔍 Using advanced LLM-validated section extraction...")
#             sections = extract_sections_accurately(pdf_path, self.MODEL_NAME)
            
#             if sections and len(sections) > 0:
#                 print(f"✅ Successfully extracted {len(sections)} sections")
#                 return sections
#             else:
#                 print("⚠️  Advanced extraction returned no sections, using fallback...")
#                 return self.extract_sections_fallback(pdf_path)
                
#         except Exception as e:
#             print(f"⚠️  Advanced extraction failed: {e}")
#             print("📝 Using fallback extraction method...")
#             return self.extract_sections_fallback(pdf_path)
    
#     def extract_sections_fallback(self, pdf_path):
#         """
#         Fallback section extraction using basic rules.
#         Simple but reliable method.
#         """
#         sections = {}
#         current_section = "Abstract"
#         text_accum = []

#         try:
#             doc = fitz.open(pdf_path)
#         except Exception as e:
#             print(f"Error opening PDF: {e}")
#             return {}

#         # Simple patterns
#         section_patterns = {
#             'abstract': re.compile(r'\b(abstract|summary)\b', re.IGNORECASE),
#             'introduction': re.compile(r'\b(introduction|intro)\b', re.IGNORECASE),
#             'methodology': re.compile(r'\b(method|methodology|approach)\b', re.IGNORECASE),
#             'experiments': re.compile(r'\b(experiment|evaluation|results)\b', re.IGNORECASE),
#             'conclusion': re.compile(r'\b(conclusion|conclusions)\b', re.IGNORECASE),
#             'references': re.compile(r'\b(references|bibliography)\b', re.IGNORECASE),
#         }
        
#         numbered_pattern = re.compile(r'^\s*(\d+\.?\d*)\s+([A-Z][^.!?]{2,50})\s*$')
        
#         # Extract all text
#         all_text = ""
#         for page in doc:
#             all_text += page.get_text() + "\n"
        
#         doc.close()
        
#         # Process line by line
#         lines = all_text.split('\n')
        
#         for line in lines:
#             line = line.strip()
#             if not line:
#                 continue
            
#             # Skip references content
#             if current_section.lower() == 'references':
#                 continue
            
#             # Check for numbered sections
#             numbered_match = numbered_pattern.match(line)
#             if numbered_match:
#                 if text_accum:
#                     sections[current_section] = ' '.join(text_accum)
#                     text_accum = []
#                 current_section = f"{numbered_match.group(1)} {numbered_match.group(2)}"
#                 continue
            
#             # Check for keyword sections
#             section_found = False
#             for section_key, pattern in section_patterns.items():
#                 if pattern.search(line) and len(line.split()) <= 5:
#                     if text_accum:
#                         sections[current_section] = ' '.join(text_accum)
#                         text_accum = []
#                     current_section = section_key.title()
#                     section_found = True
#                     break
            
#             if section_found:
#                 continue
            
#             # Add to current section
#             if len(line) > 10 and not re.match(r'^\d+$', line):
#                 text_accum.append(line)
        
#         # Add final section
#         if text_accum:
#             sections[current_section] = ' '.join(text_accum)
        
#         # Clean sections
#         cleaned_sections = {}
#         for name, content in sections.items():
#             cleaned = re.sub(r'\s+', ' ', content).strip()
#             if len(cleaned.split()) >= 10:
#                 cleaned_sections[name] = cleaned
        
#         return cleaned_sections
    
#     def extract_tables(self, pdf_path):
#         """Extract tables from PDF."""
#         tables = []
#         try:
#             with pdfplumber.open(pdf_path) as pdf:
#                 for i, page in enumerate(pdf.pages):
#                     page_tables = page.extract_tables()
#                     for j, table in enumerate(page_tables):
#                         if table:
#                             tables.append({
#                                 "page": i + 1,
#                                 "table_index": j,
#                                 "content": table
#                             })
#         except Exception as e:
#             print(f"Error extracting tables: {e}")
#         return tables
    
#     def extract_images(self, pdf_path, pdf_basename):
#         """Extract images from PDF."""
#         images = []
#         try:
#             doc = fitz.open(pdf_path)
#             for page_index, page in enumerate(doc):
#                 page_images = page.get_images(full=True)
#                 for img_index, img in enumerate(page_images):
#                     try:
#                         xref = img[0]
#                         pix = fitz.Pixmap(doc, xref)
                        
#                         img_filename = f"{pdf_basename}_page{page_index+1}_img{img_index}.png"
#                         img_path = os.path.join(self.images_folder, img_filename)
                        
#                         if pix.n < 5:
#                             pix.save(img_path)
#                         else:
#                             pix_rgb = fitz.Pixmap(fitz.csRGB, pix)
#                             pix_rgb.save(img_path)
#                             pix_rgb = None
                        
#                         pix = None
#                         images.append(img_path)
#                     except Exception as e:
#                         print(f"Error extracting image: {e}")
#             doc.close()
#         except Exception as e:
#             print(f"Error in image extraction: {e}")
#         return images
    
#     def summarize_section(self, text):
#         """Summarize a section with Ollama."""
#         text = text.strip()
#         if not text or len(text.split()) < 30:
#             return text

#         summaries = []
#         for idx, chunk in enumerate(self.chunk_text(text)):
#             print(f"   Summarizing chunk {idx+1} ({len(chunk.split())} words)...")
#             prompt = f"""Summarize the following text clearly and concisely.
# Provide only the summary, without any introductory phrases.
# Capture the main ideas and key insights without unnecessary detail.
# ---
# {chunk}"""
#             try:
#                 resp = ollama.chat(model=self.MODEL_NAME, messages=[{"role": "user", "content": prompt}])
#                 summaries.append(resp['message']['content'].strip())
#             except Exception as e:
#                 print("Summarization error:", e)
#                 summaries.append("[Error]")

#         if len(summaries) > 1:
#             final_prompt = "Combine the following summaries into a coherent, concise overall summary:\n" + "\n".join(summaries)
#             try:
#                 resp = ollama.chat(model=self.MODEL_NAME, messages=[{"role": "user", "content": final_prompt}])
#                 return resp['message']['content'].strip()
#             except:
#                 return " ".join(summaries)

#         return summaries[0] if summaries else "[Empty]"
    
#     def chunk_text(self, text, max_words=500):
#         """Split text into chunks."""
#         words = text.split()
#         for i in range(0, len(words), max_words):
#             yield " ".join(words[i:i+max_words])

