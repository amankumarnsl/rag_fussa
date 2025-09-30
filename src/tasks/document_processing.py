"""
Document processing tasks for RAG FUSSA API
"""
import os
import time
from typing import Dict, Any, Optional
from celery import current_task
from ..celery_app import celery_app
from ..utils.logging_config import get_logger, log_processing_step, log_error, log_performance_metric
from ..utils.error_handling import handle_openai_error, handle_pinecone_error, handle_s3_error
from ..processors.pdf_processor import process_pdf
from ..processors.video_processor import process_video
from ..processors.image_processor import process_image
from ..utils.text_pipeline import process_text_file_to_chunks

logger = get_logger("document_processing")

@celery_app.task(bind=True, name="src.tasks.document_processing.process_document")
def process_document_task(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main document processing task - handles the complete pipeline
    """
    start_time = time.time()
    task_id = self.request.id
    
    try:
        logger.info("Starting document processing task", 
                   task_id=task_id, 
                   uuid=request_data.get("uuid"),
                   url=str(request_data.get("url")))
        
        # Update task status
        self.update_state(
            state="PROGRESS",
            meta={"step": "initializing", "progress": 0, "message": "Starting document processing"}
        )
        
        # Step 1: Download file from S3
        self.update_state(
            state="PROGRESS", 
            meta={"step": "downloading", "progress": 10, "message": "Downloading file from S3"}
        )
        
        file_content, filename, file_type = download_file_from_s3(request_data["url"])
        
        # Step 2: Extract text content
        self.update_state(
            state="PROGRESS",
            meta={"step": "extracting", "progress": 30, "message": f"Extracting text from {file_type}"}
        )
        
        text_filepath = extract_text_content(file_content, filename, file_type)
        
        # Step 3: Process text into chunks
        self.update_state(
            state="PROGRESS",
            meta={"step": "chunking", "progress": 50, "message": "Processing text into chunks"}
        )
        
        processed_chunks = process_text_file_to_chunks(
            text_filepath=text_filepath,
            filename=filename,
            file_type=file_type,
            chunk_strategy="semantic"
        )
        
        # Step 4: Generate embeddings
        self.update_state(
            state="PROGRESS",
            meta={"step": "embeddings", "progress": 70, "message": "Generating embeddings"}
        )
        
        embeddings = generate_embeddings_async([chunk["content"] for chunk in processed_chunks])
        
        # Step 5: Upload to Pinecone
        self.update_state(
            state="PROGRESS",
            meta={"step": "uploading", "progress": 90, "message": "Uploading vectors to Pinecone"}
        )
        
        upload_vectors_to_pinecone_async(processed_chunks, embeddings, filename, file_type, request_data)
        
        # Step 6: Update backend status
        self.update_state(
            state="PROGRESS",
            meta={"step": "completing", "progress": 95, "message": "Updating backend status"}
        )
        
        update_backend_status_async(request_data["uuid"], "COMPLETED")
        
        # Calculate duration and log performance
        duration_ms = (time.time() - start_time) * 1000
        log_performance_metric("document_processing_duration", duration_ms, "ms",
                              task_id=task_id, file_type=file_type, chunks_created=len(processed_chunks))
        
        result = {
            "success": True,
            "message": f"Successfully processed {file_type} file: {filename}",
            "task_id": task_id,
            "uuid": request_data["uuid"],
            "filename": filename,
            "file_type": file_type,
            "chunks_created": len(processed_chunks),
            "duration_ms": duration_ms
        }
        
        logger.info("Document processing completed successfully", **result)
        
        # Final status update
        self.update_state(
            state="SUCCESS",
            meta={"step": "completed", "progress": 100, "message": "Document processing completed"}
        )
        
        return result
        
    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        error_msg = f"Document processing failed: {str(e)}"
        
        log_error(e, {
            "operation": "process_document_task",
            "task_id": task_id,
            "uuid": request_data.get("uuid"),
            "duration_ms": duration_ms
        })
        
        # Update backend with failure status
        try:
            update_backend_status_async(request_data["uuid"], "FAILED", error_msg)
        except:
            pass  # Don't fail the task if backend update fails
        
        # Update task state with error
        self.update_state(
            state="FAILURE",
            meta={"step": "error", "progress": 0, "message": error_msg, "error": str(e)}
        )
        
        raise

@celery_app.task(name="src.tasks.document_processing.download_file_from_s3")
def download_file_from_s3(s3_url: str) -> tuple[bytes, str, str]:
    """Download file from S3"""
    try:
        import boto3
        from urllib.parse import urlparse, unquote
        from ..config.config import AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION
        
        # Parse S3 URL
        parsed_url = urlparse(str(s3_url))
        bucket_name = parsed_url.netloc.split('.')[0]
        object_key = parsed_url.path.lstrip('/')
        object_key = unquote(object_key)
        
        # Extract filename and file type
        filename = os.path.basename(object_key)
        file_extension = filename.split('.')[-1].lower()
        file_type = get_file_type_from_extension(file_extension)
        
        # Download from S3
        s3_client = boto3.client(
            's3',
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
            region_name=AWS_REGION
        )
        
        response = s3_client.get_object(Bucket=bucket_name, Key=object_key)
        file_content = response['Body'].read()
        
        logger.info("File downloaded from S3", 
                   bucket=bucket_name, 
                   key=object_key, 
                   size_bytes=len(file_content),
                   filename=filename,
                   file_type=file_type)
        
        return file_content, filename, file_type
        
    except Exception as e:
        error = handle_s3_error(e)
        logger.error("Failed to download file from S3", error=str(error), s3_url=s3_url)
        raise

@celery_app.task(name="src.tasks.document_processing.extract_text_content")
def extract_text_content(file_content: bytes, filename: str, file_type: str) -> str:
    """Extract text content from file"""
    try:
        log_processing_step("text_extraction", file_type, filename)
        
        if file_type == 'pdf':
            text_filepath = process_pdf(file_content, filename)
        elif file_type == 'video':
            text_filepath = process_video(file_content, filename)
        elif file_type == 'image':
            text_filepath = process_image(file_content, filename)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
        
        logger.info("Text extraction completed", 
                   file_type=file_type, 
                   filename=filename, 
                   text_filepath=text_filepath)
        
        return text_filepath
        
    except Exception as e:
        logger.error("Text extraction failed", 
                    error=str(e), 
                    file_type=file_type, 
                    filename=filename)
        raise

@celery_app.task(name="src.tasks.document_processing.upload_vectors_to_pinecone")
def upload_vectors_to_pinecone_async(chunks: list, embeddings: list, filename: str, file_type: str, request_data: Dict[str, Any]):
    """Upload vectors to Pinecone"""
    try:
        from pinecone import Pinecone
        from ..config.config import PINECONE_API_KEY, get_index_for_file_type
        
        pc = Pinecone(api_key=PINECONE_API_KEY)
        index = get_index_for_file_type(file_type)
        
        # Prepare vectors
        vectors = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            vector_id = f"{filename}_{i}"
            vectors.append({
                "id": vector_id,
                "values": embedding,
                "metadata": {
                    "name": request_data["name"],
                    "uuid": request_data["uuid"],
                    "url": str(request_data["url"]),
                    "type": request_data["type"],
                    "trainingStatus": request_data["trainingStatus"],
                    "filename": filename,
                    "chunk_index": i,
                    "content": chunk["content"],
                    "file_type": file_type,
                    **chunk["metadata"]
                }
            })
        
        # Upload to Pinecone
        index.upsert(vectors=vectors, namespace=filename)
        
        logger.info("Vectors uploaded to Pinecone", 
                   namespace=filename, 
                   vector_count=len(vectors),
                   file_type=file_type)
        
    except Exception as e:
        error = handle_pinecone_error(e)
        logger.error("Failed to upload vectors to Pinecone", 
                    error=str(error), 
                    filename=filename,
                    vector_count=len(chunks))
        raise

@celery_app.task(name="src.tasks.document_processing.update_backend_status")
def update_backend_status_async(uuid: str, status: str, failure_reason: str = None):
    """Update backend status"""
    try:
        import requests
        import os
        
        backend_base_url = os.getenv("BACKEND_BASE_URL", "192.168.68.72")
        backend_port = os.getenv("BACKEND_PORT", "4504")
        backend_endpoint = os.getenv("BACKEND_ENDPOINT_PATH", "/upload/internal/update-document-entry")
        
        backend_url = f"http://{backend_base_url}:{backend_port}{backend_endpoint}"
        
        payload = {
            "search": {"uuid": uuid},
            "update": {"$set": {"trainingStatus": status}}
        }
        
        if status == "FAILED" and failure_reason:
            payload["update"]["$set"]["trainingFailedReason"] = failure_reason
        
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        
        response = requests.put(backend_url, json=payload, headers=headers, timeout=10)
        
        if response.status_code == 200:
            logger.info("Backend status updated successfully", uuid=uuid, status=status)
        else:
            logger.warning("Backend update failed", 
                          uuid=uuid, 
                          status=status, 
                          status_code=response.status_code)
        
    except Exception as e:
        logger.error("Failed to update backend status", 
                    error=str(e), 
                    uuid=uuid, 
                    status=status)

def get_file_type_from_extension(file_extension: str) -> str:
    """Determine file type from extension"""
    file_extension = file_extension.lower()
    
    if file_extension == 'pdf':
        return 'pdf'
    elif file_extension in ['mp4', 'avi', 'mov', 'wmv', 'flv', 'webm']:
        return 'video'
    elif file_extension in ['jpg', 'jpeg', 'png', 'gif', 'bmp', 'tiff', 'webp']:
        return 'image'
    else:
        return 'unknown'
