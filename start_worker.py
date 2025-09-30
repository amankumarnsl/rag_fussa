#!/usr/bin/env python3
"""
Celery worker startup script for RAG FUSSA API
"""
import os
import sys
from celery import Celery

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.celery_app import celery_app

if __name__ == "__main__":
    # Start Celery worker
    celery_app.worker_main([
        'worker',
        '--loglevel=info',
        '--concurrency=4',
        '--queues=default,document_processing,embeddings',
        '--hostname=worker@%h'
    ])
