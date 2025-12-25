"""
Production Configuration Module
Handles environment variables, security, and production settings
"""
import os
from datetime import timedelta

class ProductionConfig:
    """Production configuration with security and performance optimizations."""
    
    # ==================== FLASK CORE ====================
    SECRET_KEY = os.environ.get('SECRET_KEY') or os.urandom(32)
    DEBUG = False
    TESTING = False
    
    # ==================== SESSION CONFIGURATION ====================
    SESSION_COOKIE_SECURE = True
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Lax'
    PERMANENT_SESSION_LIFETIME = timedelta(hours=24)
    
    # ==================== SECURITY HEADERS ====================
    SECURITY_HEADERS = {
        'Strict-Transport-Security': 'max-age=31536000; includeSubDomains',
        'X-Content-Type-Options': 'nosniff',
        'X-Frame-Options': 'SAMEORIGIN',
        'X-XSS-Protection': '1; mode=block',
        'Content-Security-Policy': "default-src 'self'; script-src 'self' 'unsafe-inline' 'unsafe-eval'; style-src 'self' 'unsafe-inline';"
    }
    
    # ==================== CORS CONFIGURATION ====================
    CORS_ENABLED = os.environ.get('ENABLE_CORS', 'False').lower() == 'true'
    CORS_ORIGINS = os.environ.get('ALLOWED_ORIGINS', 'http://localhost:3000').split(',')
    
    # ==================== DATABASE ====================
    SQLALCHEMY_DATABASE_URI = f"sqlite:///{os.environ.get('DATABASE_PATH', 'research_assistant.db')}"
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_POOL_SIZE = int(os.environ.get('DATABASE_POOL_SIZE', 10))
    SQLALCHEMY_POOL_TIMEOUT = int(os.environ.get('DATABASE_TIMEOUT', 30))
    
    # ==================== FILE UPLOAD ====================
    MAX_CONTENT_LENGTH = int(os.environ.get('MAX_UPLOAD_SIZE', 100)) * 1024 * 1024  # MB to bytes
    UPLOAD_FOLDER = os.environ.get('DATA_DIR', 'data')
    ALLOWED_EXTENSIONS = {'pdf'}
    
    # ==================== RATE LIMITING ====================
    RATELIMIT_ENABLED = os.environ.get('RATELIMIT_ENABLED', 'True').lower() == 'true'
    RATELIMIT_DEFAULT = os.environ.get('RATELIMIT_DEFAULT', '100 per hour')
    RATELIMIT_STORAGE_URL = os.environ.get('RATELIMIT_STORAGE_URL', 'memory://')
    
    # ==================== CACHING ====================
    CACHE_TYPE = os.environ.get('CACHE_TYPE', 'simple')
    CACHE_DEFAULT_TIMEOUT = int(os.environ.get('CACHE_DEFAULT_TIMEOUT', 300))
    
    # ==================== LOGGING ====================
    LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')
    LOG_FILE = os.environ.get('LOG_FILE', 'logs/research_assistant.log')
    LOG_MAX_BYTES = int(os.environ.get('LOG_MAX_BYTES', 10485760))
    LOG_BACKUP_COUNT = int(os.environ.get('LOG_BACKUP_COUNT', 5))
    
    # ==================== PERFORMANCE ====================
    MAX_CONCURRENT_JOBS = int(os.environ.get('MAX_CONCURRENT_JOBS', 2))
    JOB_TIMEOUT = int(os.environ.get('JOB_TIMEOUT', 3600))
    
    # ==================== FEATURE FLAGS ====================
    ENABLE_METRICS = os.environ.get('ENABLE_METRICS', 'False').lower() == 'true'
    ENABLE_PERFORMANCE_LOGGING = os.environ.get('ENABLE_PERFORMANCE_LOGGING', 'False').lower() == 'true'


class DevelopmentConfig:
    """Development configuration."""
    
    DEBUG = True
    TESTING = False
    SECRET_KEY = 'dev-secret-key-change-in-production'
    
    # Less restrictive for development
    SESSION_COOKIE_SECURE = False
    CORS_ENABLED = True
    CORS_ORIGINS = ['http://localhost:3000', 'http://localhost:5000']
    
    RATELIMIT_ENABLED = False


class TestingConfig:
    """Testing configuration."""
    
    TESTING = True
    DEBUG = True
    SECRET_KEY = 'test-secret-key'
    
    # Use in-memory database for tests
    SQLALCHEMY_DATABASE_URI = 'sqlite:///:memory:'
    
    RATELIMIT_ENABLED = False


# Configuration dictionary
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}


def get_config():
    """Get configuration based on environment."""
    env = os.environ.get('FLASK_ENV', 'development')
    return config.get(env, config['default'])
