#!/usr/bin/env python3
"""
Regenerate Literature Surveys for Existing Papers
This script will regenerate surveys with the new "Literature Survey" section
for all papers that already have surveys.
"""

import os
import json
from modules.database import db
from modules.survey_generator import survey_generator
from modules.utils import logger

def regenerate_surveys():
    """Regenerate all paper surveys to include literature survey section."""
    
    print("\n" + "="*70)
    print("📚 REGENERATING LITERATURE SURVEYS")
    print("="*70 + "\n")
    
    # Get all papers with existing surveys
    with db.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT p.id, p.title, p.arxiv_id, p.compiled_json_path
            FROM papers p
            JOIN paper_surveys ps ON p.id = ps.paper_id
            ORDER BY p.id
        """)
        papers = [dict(row) for row in cursor.fetchall()]
    
    print(f"Found {len(papers)} papers with existing surveys\n")
    
    if not papers:
        print("No papers found. Process some papers first.")
        return
    
    regenerated = 0
    errors = []
    
    for i, paper in enumerate(papers, 1):
        paper_id = paper['id']
        title = paper['title'][:60]
        compiled_path = paper['compiled_json_path']
        
        print(f"[{i}/{len(papers)}] Processing: {title}...")
        
        if not compiled_path or not os.path.exists(compiled_path):
            print(f"  ⚠️  Skipped - compiled data not found")
            continue
        
        try:
            # Load compiled paper data
            with open(compiled_path, 'r', encoding='utf-8') as f:
                paper_data = json.load(f)
            
            # Regenerate survey with new literature survey section
            survey = survey_generator.generate_survey_for_paper(paper_id, paper_data)
            
            if survey.get('generated'):
                # Save to database
                db.save_paper_survey(paper_id, survey)
                regenerated += 1
                
                # Check if literature survey was generated
                lit_survey = survey.get('survey_sections', {}).get('literature_survey', {})
                if lit_survey.get('content'):
                    print(f"  ✅ Survey regenerated with Literature Survey ({len(lit_survey['content'])} chars)")
                else:
                    print(f"  ⚠️  Survey regenerated but Literature Survey is empty")
            else:
                print(f"  ❌ Failed: {survey.get('error', 'Unknown error')}")
                errors.append(f"Paper {paper_id}: {survey.get('error')}")
                
        except Exception as e:
            print(f"  ❌ Error: {e}")
            errors.append(f"Paper {paper_id}: {str(e)}")
    
    print("\n" + "="*70)
    print("📊 REGENERATION SUMMARY")
    print("="*70)
    print(f"Total papers: {len(papers)}")
    print(f"Successfully regenerated: {regenerated}")
    print(f"Errors: {len(errors)}")
    
    if errors:
        print("\n❌ Errors encountered:")
        for error in errors[:5]:  # Show first 5 errors
            print(f"  - {error}")
        if len(errors) > 5:
            print(f"  ... and {len(errors) - 5} more")
    
    print("\n✅ Done! View your surveys at: http://localhost:5000/results")
    print("="*70 + "\n")

if __name__ == "__main__":
    regenerate_surveys()
