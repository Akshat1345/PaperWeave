# modules/survey_generator.py - Literature Survey Generation
import json
import os
from typing import Dict, List, Optional
from config import config
from modules.utils import logger
from modules.database import db
import ollama

class LiteratureSurveyGenerator:
    """
    Generates comprehensive literature surveys for each paper.
    Surveys follow IEEE conference paper format:
    - Related Work & Context
    - Methodologies Survey
    - Key Contributions Summary
    - Research Gaps & Future Work
    """
    
    def __init__(self):
        self.model = config.OLLAMA_MODEL
        logger.info("Literature Survey Generator initialized")
    
    def generate_survey_for_paper(self, paper_id: int, paper_data: Dict, job_id: Optional[int] = None) -> Dict:
        """
        Generate a comprehensive literature survey for a specific paper.
        
        Args:
            paper_id: Database paper ID
            paper_data: Complete compiled paper data
            job_id: Optional job_id to filter references by job context
        
        Returns:
            Survey data with multiple sections
        """
        try:
            metadata = paper_data.get('metadata', {})
            sections_text = paper_data.get('sections_text', {})
            contributions = paper_data.get('contributions', {})
            references = paper_data.get('references', [])
            
            logger.info(f"Generating survey for paper {paper_id}: {metadata.get('title', '')[:60]}")
            
            # Extract key information
            abstract = metadata.get('abstract', '')
            title = metadata.get('title', '')
            
            # Step 1: Generate Related Work Analysis
            related_work = self._generate_related_work(
                title, abstract, sections_text, contributions, references
            )
            
            # Step 2: Generate Methodology Survey
            methodology_survey = self._generate_methodology_survey(
                sections_text, contributions
            )
            
            # Step 3: Generate Contributions Summary
            contributions_summary = self._generate_contributions_summary(contributions)
            
            # Step 4: Generate Research Gaps Analysis
            research_gaps = self._generate_research_gaps(
                title, abstract, contributions, references
            )
            
            # Step 5: Generate Overall Context
            context_analysis = self._generate_context_analysis(
                title, metadata, contributions
            )
            
            # Step 6: Generate Academic Literature Survey with job-filtered citations
            literature_survey = self._generate_literature_survey(
                title, abstract, sections_text, references, job_id=job_id
            )
            
            survey = {
                'paper_id': paper_id,
                'arxiv_id': metadata.get('arxiv_id', ''),
                'title': title,
                'abstract': abstract,
                'survey_sections': {
                    'literature_survey': literature_survey,  # NEW: Academic-style survey
                    'related_work': related_work,
                    'methodology_survey': methodology_survey,
                    'contributions_summary': contributions_summary,
                    'research_gaps': research_gaps,
                    'context_analysis': context_analysis
                },
                'reference_count': len(references),
                'generated': True
            }
            
            logger.info(f"✅ Survey generated for paper {paper_id}")
            return survey
            
        except Exception as e:
            logger.error(f"Error generating survey: {e}", exc_info=True)
            return {
                'paper_id': paper_id,
                'error': str(e),
                'generated': False
            }
    
    def _generate_related_work(self, title: str, abstract: str, 
                               sections_text: Dict, contributions: Dict, 
                               references: List) -> Dict:
        """Generate Related Work & Context section."""
        try:
            # Collect intro/background content
            related_sections = {k: v for k, v in sections_text.items() 
                              if any(kw in k.lower() for kw in ['intro', 'background', 'related', 'survey'])}
            
            context_text = "\n".join(related_sections.values())[:2000]
            
            prompt = f"""Analyze the following research paper and generate a comprehensive "Related Work & Context" section for a literature survey.

PAPER TITLE: {title}

ABSTRACT: {abstract}

BACKGROUND/INTRO CONTENT:
{context_text}

KEY PROBLEM: {contributions.get('main_problem', '')}
KEY INNOVATION: {contributions.get('key_innovation', '')}

Generate a "Related Work & Context" section that:
1. Places the paper in its research domain
2. Identifies the research problem space
3. Notes how this work relates to prior research areas
4. Highlights the novelty compared to existing work
5. Provides historical context if applicable

FORMAT AS ACADEMIC PROSE (2-4 paragraphs):"""

            response = ollama.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.3, "num_predict": 500}
            )
            
            return {
                'content': response['message']['content'].strip(),
                'reference_count': len(references),
                'section_type': 'related_work'
            }
        except Exception as e:
            logger.error(f"Error generating related work: {e}")
            return {
                'content': f"Error: {str(e)}",
                'error': str(e),
                'section_type': 'related_work'
            }
    
    def _generate_methodology_survey(self, sections_text: Dict, contributions: Dict) -> Dict:
        """Generate Methodology Survey section."""
        try:
            method_sections = {k: v for k, v in sections_text.items() 
                             if any(kw in k.lower() for kw in ['method', 'approach', 'model', 'algorithm'])}
            
            method_text = "\n".join(method_sections.values())[:2000]
            
            prompt = f"""Based on the following methodology content from a research paper, generate a comprehensive "Methodology Survey" section.

METHODOLOGY CONTENT:
{method_text}

CORE METHODOLOGY: {contributions.get('core_methodology', '')}

Generate a "Methodology Survey" that:
1. Explains the core technical approach
2. Breaks down key algorithms or techniques used
3. Identifies novel methodological contributions
4. Compares with traditional approaches in the field
5. Highlights advantages of this methodology

FORMAT AS ACADEMIC PROSE (2-4 paragraphs):"""

            response = ollama.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.3, "num_predict": 500}
            )
            
            return {
                'content': response['message']['content'].strip(),
                'section_type': 'methodology'
            }
        except Exception as e:
            logger.error(f"Error generating methodology survey: {e}")
            return {
                'content': f"Error: {str(e)}",
                'error': str(e),
                'section_type': 'methodology'
            }
    
    def _generate_contributions_summary(self, contributions: Dict) -> Dict:
        """Generate Contributions Summary section."""
        try:
            contrib_prompt = f"""Summarize the key contributions of a research paper for a literature survey.

MAIN PROBLEM: {contributions.get('main_problem', '')}
KEY INNOVATION: {contributions.get('key_innovation', '')}
CORE METHODOLOGY: {contributions.get('core_methodology', '')}
MAJOR RESULTS: {contributions.get('major_results', '')}

Generate a "Key Contributions" section (1-2 paragraphs) that:
1. Clearly states the main problem addressed
2. Highlights the key innovation or insight
3. Summarizes major results achieved
4. Emphasizes what makes this work unique in the field

FORMAT AS BULLET POINTS FOLLOWED BY SUMMARY:"""

            response = ollama.chat(
                model=self.model,
                messages=[{"role": "user", "content": contrib_prompt}],
                options={"temperature": 0.3, "num_predict": 400}
            )
            
            return {
                'content': response['message']['content'].strip(),
                'section_type': 'contributions'
            }
        except Exception as e:
            logger.error(f"Error generating contributions summary: {e}")
            return {
                'content': f"Error: {str(e)}",
                'error': str(e),
                'section_type': 'contributions'
            }
    
    def _generate_research_gaps(self, title: str, abstract: str, 
                               contributions: Dict, references: List) -> Dict:
        """Generate Research Gaps & Future Work section."""
        try:
            gaps_prompt = f"""Analyze a research paper and identify research gaps and future work opportunities.

PAPER TITLE: {title}
ABSTRACT: {abstract}
LIMITATIONS: {contributions.get('limitations', '')}
RESEARCH GAPS: {contributions.get('research_gaps', '')}

Generate a "Research Gaps & Future Work" section (2-3 paragraphs) that:
1. Identifies explicit limitations mentioned in the paper
2. Infers implicit research gaps from the work
3. Suggests natural extensions and future research directions
4. Identifies open problems in the domain
5. Proposes potential improvements to the methodology

FORMAT AS ACADEMIC PROSE:"""

            response = ollama.chat(
                model=self.model,
                messages=[{"role": "user", "content": gaps_prompt}],
                options={"temperature": 0.4, "num_predict": 500}
            )
            
            return {
                'content': response['message']['content'].strip(),
                'section_type': 'research_gaps'
            }
        except Exception as e:
            logger.error(f"Error generating research gaps: {e}")
            return {
                'content': f"Error: {str(e)}",
                'error': str(e),
                'section_type': 'research_gaps'
            }
    
    def _generate_context_analysis(self, title: str, metadata: Dict, 
                                   contributions: Dict) -> Dict:
        """Generate Overall Context Analysis."""
        try:
            context_prompt = f"""Create a concise contextual analysis of a research paper.

TITLE: {title}
AUTHORS: {metadata.get('authors', [])}
PUBLISHED: {metadata.get('published', '')}
CITATION COUNT: {metadata.get('citation_count', 0)}

KEY PROBLEM: {contributions.get('main_problem', '')}
MAJOR RESULTS: {contributions.get('major_results', '')}

Generate a "Context Analysis" (1-2 paragraphs) that:
1. Places the work in its historical and research context
2. Discusses its impact and influence in the field
3. Notes if it's a foundational, incremental, or paradigm-shifting work
4. Identifies which researchers/groups would benefit from this work
5. Discusses potential applications

FORMAT AS ACADEMIC PROSE:"""

            response = ollama.chat(
                model=self.model,
                messages=[{"role": "user", "content": context_prompt}],
                options={"temperature": 0.3, "num_predict": 400}
            )
            
            return {
                'content': response['message']['content'].strip(),
                'section_type': 'context',
                'citation_influence': metadata.get('citation_count', 0)
            }
        except Exception as e:
            logger.error(f"Error generating context analysis: {e}")
            return {
                'content': f"Error: {str(e)}",
                'error': str(e),
                'section_type': 'context'
            }
    
    def _generate_literature_survey(self, title: str, abstract: str, 
                                    sections_text: Dict, references: List,
                                    job_id: Optional[int] = None) -> Dict:
        """
        Generate academic-style Literature Survey section.
        This creates a dense, citation-heavy narrative review of related work,
        similar to the Literature Survey sections in IEEE conference papers.
        Only uses citations from papers in the same job.
        """
        try:
            # Extract introduction and related work sections
            relevant_sections = {k: v for k, v in sections_text.items() 
                               if any(kw in k.lower() for kw in 
                                     ['intro', 'related', 'background', 'survey', 'literature', 'previous'])}
            
            survey_content = "\n".join(relevant_sections.values())[:3000]
            
            # Extract reference titles/context if available
            # Only use references from this paper (since we're generating for individual papers)
            ref_context = ""
            ref_mapping = {}  # Map real citation indices to display indices
            if references:
                ref_list = []
                # Limit to references available in this paper
                max_refs = min(len(references), 20)
                for i, ref in enumerate(references[:max_refs], 1):
                    ref_mapping[i] = i  # Direct mapping for this paper's refs
                    if isinstance(ref, dict):
                        ref_title = ref.get('title', 'Unknown')
                        ref_authors = ref.get('authors', [])
                        author_str = ""
                        if ref_authors:
                            if len(ref_authors) == 1:
                                author_str = ref_authors[0].split()[-1]  # Last name
                            elif len(ref_authors) == 2:
                                author_str = f"{ref_authors[0].split()[-1]} and {ref_authors[1].split()[-1]}"
                            else:
                                author_str = f"{ref_authors[0].split()[-1]} et al."
                        ref_list.append(f"[{i}] {author_str}: {ref_title[:80]}" if author_str else f"[{i}] {ref_title[:80]}")
                    elif isinstance(ref, str):
                        ref_list.append(f"[{i}] {ref[:100]}")
                ref_context = "\n".join(ref_list)
            
            # Create prompt that tells LLM to ONLY use citations that exist
            prompt = f"""You are writing the LITERATURE SURVEY section for an IEEE conference research paper.

PAPER TITLE: {title}

ABSTRACT: {abstract}

RELATED WORK CONTENT FROM PAPER:
{survey_content}

REFERENCES AVAILABLE IN THIS PAPER (use ONLY these):
{ref_context if ref_context else "No external references available - focus on the paper's own contributions"}

CRITICAL INSTRUCTIONS:
- ONLY use citation numbers from the references listed above
- If there are {len(references)} references, ONLY use citations [1] through [{len(references)}]
- DO NOT create fake citations like [7] if only 3 references exist
- If no external references available, write about the paper's own work and methodologies
- Each citation used MUST correspond to a reference in the list above

Generate a comprehensive LITERATURE SURVEY section following this style:
- Write in dense, flowing academic prose (like IEEE papers)
- Discuss multiple related works in each sentence using ONLY available citations
- Cover the progression of research in this domain
- Mention specific techniques, algorithms, models, and approaches
- Include author names when introducing key work if available
- Discuss results, accuracies, and performance metrics when mentioned
- Connect different works showing how the field evolved
- Write 2-3 substantial paragraphs with multiple citations per sentence
- Use transitions like "In recent years", "Several studies", "Furthermore", "Additionally"
- End with current challenges or limitations in existing approaches

IMPORTANT:
- Use ONLY citation numbers [1] through [{len(references)}] if references exist
- Make it read like a real literature survey from an IEEE paper
- Be specific about techniques and results
- Keep the narrative flowing naturally from one work to another
- If citations are limited, focus on synthesizing the paper's own contributions

Write ONLY the literature survey content (no headers or section titles):"""

            response = ollama.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.4, "num_predict": 800}
            )
            
            content = response['message']['content'].strip()
            
            # Clean up any headers if Ollama added them
            content = content.replace('LITERATURE SURVEY', '').replace('Literature Survey', '').strip()
            
            return {
                'content': content,
                'section_type': 'literature_survey',
                'reference_count': len(references),
                'max_citation': len(references),  # Track max citation used
                'style': 'academic_ieee'
            }
            
        except Exception as e:
            logger.error(f"Error generating literature survey: {e}")
            return {
                'content': f"Error: {str(e)}",
                'error': str(e),
                'section_type': 'literature_survey'
            }
    
    def compile_job_surveys(self, job_id: int) -> Dict:
        """
        Generate surveys for all papers in a job and create a master survey.
        """
        try:
            papers = db.get_papers_by_job(job_id)
            logger.info(f"Generating surveys for {len(papers)} papers in job {job_id}")
            
            all_surveys = []
            
            for paper in papers:
                if not paper.get('compiled_json_path') or not os.path.exists(paper['compiled_json_path']):
                    continue
                
                try:
                    with open(paper['compiled_json_path'], 'r', encoding='utf-8') as f:
                        paper_data = json.load(f)
                    
                    # Pass job_id to generate survey with job context
                    survey = self.generate_survey_for_paper(paper['id'], paper_data, job_id=job_id)
                    all_surveys.append(survey)
                    
                    # Save survey to database
                    db.save_paper_survey(paper['id'], survey)
                    
                except Exception as e:
                    logger.error(f"Error processing paper {paper['id']}: {e}")
                    continue
            
            logger.info(f"✅ Generated {len(all_surveys)} surveys")
            
            return {
                'success': True,
                'job_id': job_id,
                'total_surveys': len(all_surveys),
                'surveys': all_surveys
            }
            
        except Exception as e:
            logger.error(f"Error compiling surveys: {e}", exc_info=True)
            return {
                'success': False,
                'error': str(e)
            }

# Global instance
survey_generator = LiteratureSurveyGenerator()
