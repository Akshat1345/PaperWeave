# modules/utils.py - FINAL VERSION
import os

def ensure_directories(dirs):
    """Create directories if they don't exist."""
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
        # Create subdirectories
        os.makedirs(os.path.join(dir_path, 'pdfs'), exist_ok=True)
        os.makedirs(os.path.join(dir_path, 'compiled'), exist_ok=True)
        os.makedirs(os.path.join(dir_path, 'images'), exist_ok=True)