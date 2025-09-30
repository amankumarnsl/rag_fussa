"""
Celery configuration for RAG FUSSA API
"""
import os
from celery import Celery
from kombu import Queue

# Redis configuration
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# Create Celery instance
celery_app = Celery(
    "rag_fussa",
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=["src.tasks.document_processing", "src.tasks.embedding_tasks"]
)

# Celery configuration
celery_app.conf.update(
    # Task routing
    task_routes={
        "src.tasks.document_processing.*": {"queue": "document_processing"},
        "src.tasks.embedding_tasks.*": {"queue": "embeddings"},
    },
    
    # Queue configuration
    task_default_queue="default",
    task_queues=(
        Queue("default", routing_key="default"),
        Queue("document_processing", routing_key="document_processing"),
        Queue("embeddings", routing_key="embeddings"),
    ),
    
    # Task execution settings
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    
    # Task time limits
    task_soft_time_limit=300,  # 5 minutes soft limit
    task_time_limit=600,       # 10 minutes hard limit
    
    # Result backend settings
    result_expires=3600,       # 1 hour
    result_backend_max_retries=10,
    
    # Worker settings
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=50,
    
    # Retry settings
    task_acks_late=True,
    worker_disable_rate_limits=False,
    
    # Monitoring
    worker_send_task_events=True,
    task_send_sent_event=True,
    
    # Task result settings
    task_ignore_result=False,
    task_store_eager_result=True,
)

# Task priority settings
celery_app.conf.task_annotations = {
    "*": {
        "rate_limit": "10/s",
        "time_limit": 600,
        "soft_time_limit": 300,
    },
    "src.tasks.embedding_tasks.generate_embeddings": {
        "rate_limit": "5/s",  # Slower for OpenAI API limits
        "time_limit": 900,
        "soft_time_limit": 600,
    },
    "src.tasks.document_processing.process_document": {
        "rate_limit": "2/s",  # Very slow for large documents
        "time_limit": 1800,
        "soft_time_limit": 1200,
    }
}

if __name__ == "__main__":
    celery_app.start()
