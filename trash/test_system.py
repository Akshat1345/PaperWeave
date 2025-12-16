# test_system.py - Comprehensive System Testing Script
"""
Run this script to test all components of the research assistant.

Usage:
    python test_system.py
"""

import sys
import os
import requests
import time
from datetime import datetime

# Colors for terminal output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    END = '\033[0m'

def print_test(name):
    print(f"\n{Colors.BLUE}[TEST]{Colors.END} {name}")

def print_success(msg):
    print(f"{Colors.GREEN}✅ {msg}{Colors.END}")

def print_error(msg):
    print(f"{Colors.RED}❌ {msg}{Colors.END}")

def print_warning(msg):
    print(f"{Colors.YELLOW}⚠️  {msg}{Colors.END}")

def test_imports():
    """Test if all required modules can be imported."""
    print_test("Testing Python imports...")
    
    modules = [
        ('flask', 'Flask'),
        ('requests', 'requests'),
        ('feedparser', 'feedparser'),
        ('fitz', 'PyMuPDF'),
        ('pdfplumber', 'pdfplumber'),
        ('ollama', 'ollama'),
    ]
    
    all_ok = True
    for module_name, display_name in modules:
        try:
            __import__(module_name)
            print_success(f"{display_name} imported successfully")
        except ImportError as e:
            print_error(f"Failed to import {display_name}: {e}")
            all_ok = False
    
    return all_ok

def test_directories():
    """Test if required directories exist or can be created."""
    print_test("Testing directory structure...")
    
    dirs = ['data', 'processed', 'processed/cache', 'processed/compiled', 'modules', 'templates']
    
    all_ok = True
    for dir_path in dirs:
        if os.path.exists(dir_path):
            print_success(f"Directory exists: {dir_path}")
        else:
            try:
                os.makedirs(dir_path, exist_ok=True)
                print_success(f"Created directory: {dir_path}")
            except Exception as e:
                print_error(f"Failed to create {dir_path}: {e}")
                all_ok = False
    
    return all_ok

def test_files():
    """Test if required files exist."""
    print_test("Testing required files...")
    
    required_files = [
        'config.py',
        'modules/__init__.py',
        'modules/scraper.py',
        'modules/compiler.py',
        'modules/database.py',
        'modules/utils.py',
        'app.py'
    ]
    
    all_ok = True
    for file_path in required_files:
        if os.path.exists(file_path):
            print_success(f"File exists: {file_path}")
        else:
            print_error(f"Missing file: {file_path}")
            all_ok = False
    
    return all_ok

def test_config():
    """Test if config can be loaded."""
    print_test("Testing configuration...")
    
    try:
        from config import config
        print_success(f"Config loaded successfully")
        print(f"   - Data dir: {config.DATA_DIR}")
        print(f"   - Ollama model: {config.OLLAMA_MODEL}")
        print(f"   - Page limit: {config.PAGE_LIMIT}")
        return True
    except Exception as e:
        print_error(f"Failed to load config: {e}")
        return False

def test_database():
    """Test database initialization."""
    print_test("Testing database...")
    
    try:
        from modules.database import db
        
        # Get stats
        stats = db.get_database_stats()
        print_success("Database initialized successfully")
        print(f"   - Total jobs: {stats['total_jobs']}")
        print(f"   - Total papers: {stats['total_papers']}")
        return True
    except Exception as e:
        print_error(f"Database error: {e}")
        return False

def test_ollama():
    """Test Ollama connection."""
    print_test("Testing Ollama connection...")
    
    try:
        import ollama
        from config import config
        
        response = ollama.chat(
            model=config.OLLAMA_MODEL,
            messages=[{"role": "user", "content": "Say 'OK' if you can read this."}],
            options={"num_predict": 10}
        )
        
        print_success(f"Ollama connected (model: {config.OLLAMA_MODEL})")
        print(f"   - Response: {response['message']['content'][:50]}")
        return True
    except Exception as e:
        print_error(f"Ollama connection failed: {e}")
        print_warning("Make sure 'ollama serve' is running")
        return False

def test_flask_app():
    """Test if Flask app starts."""
    print_test("Testing Flask application (checking files only)...")
    
    try:
        # Don't actually run the app, just check it can be imported
        import app
        print_success("Flask app can be imported")
        return True
    except Exception as e:
        print_error(f"Flask app error: {e}")
        return False

def test_scraper():
    """Test scraper module."""
    print_test("Testing scraper module...")
    
    try:
        from modules.scraper import ArxivScraper
        from config import config
        
        scraper = ArxivScraper(config.DATA_DIR)
        print_success("Scraper initialized successfully")
        
        # Test query building
        query = scraper.build_query("machine learning", {"year": 2023})
        print_success(f"Query builder working: {query[:50]}...")
        
        return True
    except Exception as e:
        print_error(f"Scraper error: {e}")
        return False

def test_compiler():
    """Test compiler module."""
    print_test("Testing compiler module...")
    
    try:
        from modules.compiler import CompilationAgent
        from config import config
        
        compiler = CompilationAgent(config.DATA_DIR, config.PROCESSED_DIR)
        print_success("Compiler initialized successfully")
        
        # Test Ollama connection through compiler
        if compiler.check_ollama_connection():
            print_success("Compiler can connect to Ollama")
        else:
            print_warning("Compiler could not connect to Ollama")
        
        return True
    except Exception as e:
        print_error(f"Compiler error: {e}")
        return False

def test_api_endpoint():
    """Test if Flask API is running (optional)."""
    print_test("Testing Flask API endpoints (if running)...")
    
    try:
        response = requests.get('http://localhost:5000/status', timeout=2)
        if response.status_code in [200, 404]:  # Either is fine
            print_success("Flask app is running and responding")
            return True
        else:
            print_warning(f"Flask app returned status: {response.status_code}")
            return False
    except requests.exceptions.RequestException:
        print_warning("Flask app is not running (this is OK if you haven't started it)")
        print("   To start: python app.py")
        return None  # Neither pass nor fail

def run_all_tests():
    """Run all tests and report results."""
    print("\n" + "=" * 60)
    print("🧪 AI RESEARCH ASSISTANT - SYSTEM TEST")
    print("=" * 60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    tests = [
        ("Python Imports", test_imports),
        ("Directory Structure", test_directories),
        ("Required Files", test_files),
        ("Configuration", test_config),
        ("Database", test_database),
        ("Ollama Connection", test_ollama),
        ("Flask App", test_flask_app),
        ("Scraper Module", test_scraper),
        ("Compiler Module", test_compiler),
        ("API Endpoints", test_api_endpoint)
    ]
    
    results = []
    for test_name, test_func in tests:
        result = test_func()
        results.append((test_name, result))
        time.sleep(0.5)  # Brief pause between tests
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, r in results if r is True)
    failed = sum(1 for _, r in results if r is False)
    skipped = sum(1 for _, r in results if r is None)
    total = len(results)
    
    for test_name, result in results:
        if result is True:
            print(f"{Colors.GREEN}✅ PASS{Colors.END} - {test_name}")
        elif result is False:
            print(f"{Colors.RED}❌ FAIL{Colors.END} - {test_name}")
        else:
            print(f"{Colors.YELLOW}⏭️  SKIP{Colors.END} - {test_name}")
    
    print("\n" + "-" * 60)
    print(f"Total: {total} tests")
    print(f"{Colors.GREEN}Passed: {passed}{Colors.END}")
    print(f"{Colors.RED}Failed: {failed}{Colors.END}")
    print(f"{Colors.YELLOW}Skipped: {skipped}{Colors.END}")
    print("-" * 60)
    
    if failed == 0:
        print(f"\n{Colors.GREEN}🎉 All critical tests passed!{Colors.END}")
        print("You can now run: python app.py")
        return True
    else:
        print(f"\n{Colors.RED}⚠️  {failed} test(s) failed. Please fix issues before proceeding.{Colors.END}")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)