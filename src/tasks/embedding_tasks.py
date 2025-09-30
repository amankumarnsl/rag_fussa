"""
Embedding generation tasks for RAG FUSSA API
"""
import time
from typing import List
from celery import current_task
from ..celery_app import celery_app
from ..utils.logging_config import get_logger, log_api_call, log_api_response, log_error
from ..utils.error_handling import handle_openai_error, openai_retry

logger = get_logger("embedding_tasks")

@celery_app.task(bind=True, name="src.tasks.embedding_tasks.generate_embeddings")
def generate_embeddings_async(self, texts: List[str]) -> List[List[float]]:
    """
    Generate embeddings for a list of texts using OpenAI API with progress tracking
    """
    start_time = time.time()
    task_id = self.request.id
    
    try:
        logger.info("Starting embedding generation", 
                   task_id=task_id, 
                   text_count=len(texts))
        
        # Update task status
        self.update_state(
            state="PROGRESS",
            meta={"step": "generating", "progress": 0, "message": f"Generating embeddings for {len(texts)} texts"}
        )
        
        # Process in batches to handle large numbers of texts
        batch_size = 20  # OpenAI batch limit
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (len(texts) + batch_size - 1) // batch_size
            
            # Update progress
            progress = int((i / len(texts)) * 100)
            self.update_state(
                state="PROGRESS",
                meta={
                    "step": "generating",
                    "progress": progress,
                    "message": f"Processing batch {batch_num}/{total_batches} ({len(batch)} texts)"
                }
            )
            
            # Generate embeddings for this batch
            batch_embeddings = generate_batch_embeddings(batch)
            all_embeddings.extend(batch_embeddings)
            
            logger.info("Batch embeddings generated", 
                       batch_num=batch_num, 
                       total_batches=total_batches,
                       batch_size=len(batch))
        
        # Calculate duration
        duration_ms = (time.time() - start_time) * 1000
        
        logger.info("Embedding generation completed", 
                   task_id=task_id,
                   total_texts=len(texts),
                   total_embeddings=len(all_embeddings),
                   duration_ms=duration_ms)
        
        # Final status update
        self.update_state(
            state="SUCCESS",
            meta={"step": "completed", "progress": 100, "message": f"Generated {len(all_embeddings)} embeddings"}
        )
        
        return all_embeddings
        
    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        error = handle_openai_error(e)
        
        log_error(e, {
            "operation": "generate_embeddings_async",
            "task_id": task_id,
            "text_count": len(texts),
            "duration_ms": duration_ms
        })
        
        # Update task state with error
        self.update_state(
            state="FAILURE",
            meta={"step": "error", "progress": 0, "message": f"Embedding generation failed: {error.message}"}
        )
        
        raise

@celery_app.task(name="src.tasks.embedding_tasks.generate_batch_embeddings")
def generate_batch_embeddings(texts: List[str]) -> List[List[float]]:
    """Generate embeddings for a batch of texts"""
    try:
        import openai
        from ..config.config import OPENAI_API_KEY
        
        openai.api_key = OPENAI_API_KEY
        
        log_api_call("openai", "embeddings", "POST", 
                    model="text-embedding-3-small", 
                    text_count=len(texts))
        
        @openai_retry
        def _get_embeddings():
            return openai.embeddings.create(
                input=texts,
                model="text-embedding-3-small"
            )
        
        response = _get_embeddings()
        embeddings = [item.embedding for item in response.data]
        
        log_api_response("openai", "embeddings", 200, 0, 
                        model="text-embedding-3-small", 
                        embedding_count=len(embeddings))
        
        return embeddings
        
    except Exception as e:
        error = handle_openai_error(e)
        log_error(e, {"operation": "generate_batch_embeddings", "text_count": len(texts)})
        raise error

@celery_app.task(name="src.tasks.embedding_tasks.generate_single_embedding")
def generate_single_embedding(text: str) -> List[float]:
    """Generate embedding for a single text (for queries)"""
    try:
        import openai
        from ..config.config import OPENAI_API_KEY
        
        openai.api_key = OPENAI_API_KEY
        
        log_api_call("openai", "embeddings", "POST", 
                    model="text-embedding-3-small", 
                    text_count=1)
        
        @openai_retry
        def _get_embedding():
            return openai.embeddings.create(
                input=[text],
                model="text-embedding-3-small"
            )
        
        response = _get_embedding()
        embedding = response.data[0].embedding
        
        log_api_response("openai", "embeddings", 200, 0, 
                        model="text-embedding-3-small", 
                        embedding_count=1)
        
        return embedding
        
    except Exception as e:
        error = handle_openai_error(e)
        log_error(e, {"operation": "generate_single_embedding", "text": text[:100]})
        raise error
