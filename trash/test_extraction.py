# test_extraction.py - FINAL VERSION
import os
import sys
from modules.advanced_extractor import extract_sections_accurately
import json
from pprint import pprint

def test_single_paper(pdf_path):
    """Test section extraction on a single paper."""
    print(f"\n{'='*80}")
    print(f"Testing: {os.path.basename(pdf_path)}")
    print(f"{'='*80}\n")
    
    if not os.path.exists(pdf_path):
        print(f"❌ File not found: {pdf_path}")
        return None
    
    try:
        # Extract sections
        sections = extract_sections_accurately(pdf_path)
        
        if not sections:
            print("❌ No sections extracted!")
            return None
        
        # Display results
        print(f"\n✅ Successfully extracted {len(sections)} sections:\n")
        
        for i, (section_name, content) in enumerate(sections.items(), 1):
            word_count = len(content.split())
            preview = content[:150] + "..." if len(content) > 150 else content
            
            print(f"{i}. Section: '{section_name}'")
            print(f"   Words: {word_count}")
            print(f"   Preview: {preview}")
            print()
        
        return sections
        
    except Exception as e:
        print(f"❌ Error during extraction: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_all_papers_in_folder(folder_path):
    """Test all PDFs in a folder."""
    if not os.path.exists(folder_path):
        print(f"❌ Folder not found: {folder_path}")
        return
    
    pdf_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.pdf')]
    
    if not pdf_files:
        print(f"❌ No PDF files found in {folder_path}")
        return
    
    print(f"\n🔍 Found {len(pdf_files)} PDF files to test\n")
    
    results = {}
    
    for pdf_file in pdf_files:
        pdf_path = os.path.join(folder_path, pdf_file)
        sections = test_single_paper(pdf_path)
        
        if sections:
            results[pdf_file] = {
                'success': True,
                'num_sections': len(sections),
                'section_names': list(sections.keys())
            }
        else:
            results[pdf_file] = {
                'success': False,
                'num_sections': 0,
                'section_names': []
            }
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}\n")
    
    successful = sum(1 for r in results.values() if r['success'])
    print(f"✅ Successful: {successful}/{len(pdf_files)}")
    print(f"❌ Failed: {len(pdf_files) - successful}/{len(pdf_files)}")
    
    # Save results
    with open('test_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📁 Detailed results saved to: test_results.json")

def compare_old_vs_new(pdf_path):
    """Compare old extraction vs new extraction."""
    from modules.compiler import CompilationAgent
    
    print(f"\n{'='*80}")
    print(f"COMPARISON TEST: {os.path.basename(pdf_path)}")
    print(f"{'='*80}\n")
    
    if not os.path.exists(pdf_path):
        print(f"❌ File not found: {pdf_path}")
        return
    
    # Test new method
    print("🆕 Testing NEW (LLM-validated) extraction...")
    new_sections = extract_sections_accurately(pdf_path)
    
    # Test old method
    print("\n🔄 Testing OLD (rule-based) extraction...")
    compiler = CompilationAgent('data', 'processed')
    old_sections = compiler.extract_sections_fallback(pdf_path)
    
    # Compare
    print(f"\n{'='*80}")
    print("COMPARISON RESULTS")
    print(f"{'='*80}\n")
    
    print(f"OLD method found: {len(old_sections)} sections")
    print(f"NEW method found: {len(new_sections)} sections")
    
    print("\n📊 OLD sections:")
    for name in old_sections.keys():
        print(f"  - {name}")
    
    print("\n📊 NEW sections:")
    for name in new_sections.keys():
        print(f"  - {name}")
    
    # Find differences
    old_only = set(old_sections.keys()) - set(new_sections.keys())
    new_only = set(new_sections.keys()) - set(old_sections.keys())
    
    if old_only:
        print(f"\n⚠️  Sections ONLY in OLD method:")
        for name in old_only:
            print(f"  - {name}")
    
    if new_only:
        print(f"\n✨ Sections ONLY in NEW method:")
        for name in new_only:
            print(f"  - {name}")
    
    if not old_only and not new_only:
        print("\n✅ Both methods found the same sections!")

def interactive_test():
    """Interactive testing mode."""
    print("""
╔════════════════════════════════════════════════════════════════╗
║         SECTION EXTRACTION TEST SUITE                          ║
╚════════════════════════════════════════════════════════════════╝

Choose test mode:
1. Test single PDF
2. Test all PDFs in folder (data/pdfs)
3. Compare old vs new method
4. Exit

""")
    
    choice = input("Enter choice (1-4): ").strip()
    
    if choice == '1':
        pdf_path = input("Enter PDF path: ").strip()
        test_single_paper(pdf_path)
        
    elif choice == '2':
        folder = input("Enter folder path (default: data/pdfs): ").strip()
        if not folder:
            folder = 'data/pdfs'
        test_all_papers_in_folder(folder)
        
    elif choice == '3':
        pdf_path = input("Enter PDF path: ").strip()
        compare_old_vs_new(pdf_path)
        
    elif choice == '4':
        print("👋 Goodbye!")
        sys.exit(0)
    
    else:
        print("❌ Invalid choice!")

if __name__ == "__main__":
    # Check if Ollama is running
    try:
        import ollama
        ollama.chat(model="llama3.2:latest", messages=[{"role": "user", "content": "test"}])
        print("✅ Ollama is running\n")
    except Exception as e:
        print("❌ Ollama is not running!")
        print("💡 Please run 'ollama serve' in another terminal\n")
        sys.exit(1)
    
    # If PDF path provided as argument
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
        test_single_paper(pdf_path)
    else:
        # Interactive mode
        interactive_test()