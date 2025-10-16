import os
import io
import json
import boto3
import openai
import requests
import time
import asyncio
from urllib.parse import urlparse, unquote
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import PyPDF2
from typing import List, Dict, Any

from .config.config import *
from .config.schemas import *
from .processors.pdf_processor import process_pdf, get_pdf_info, extract_pdf_text
from .processors.video_processor import process_video, is_video_file, extract_video_audio_transcript
from .processors.image_processor import process_image, is_image_file, extract_text_from_image
from .utils.text_pipeline import process_text_file_to_chunks
from .utils.smart_chunking import process_extracted_text
from .utils.health_checks import check_all_dependencies, get_overall_health_status, HealthStatus
from .utils.error_handling import (
    ServiceError, OpenAIServiceError, PineconeServiceError, S3ServiceError, BackendServiceError,
    handle_openai_error, handle_pinecone_error, handle_s3_error, handle_backend_error,
    safe_api_call, safe_async_api_call, create_fallback_response, openai_retry, pinecone_retry, s3_retry, backend_retry
)
from .utils.cpu_config import run_cpu_task, debug_print_cpu_info, get_cpu_info

# Simple debug toggle - set to True for console prints, False for silence
DEBUG_PRINT = os.getenv("DEBUG_PRINT", "false").lower() == "true"

# API Method Toggle - set to True for responses.create(), False for chat.completions.create()
USE_RESPONSE_API = os.getenv("USE_RESPONSE_API", "false").lower() == "true"

# Debug file logging setup
debug_file = None
if DEBUG_PRINT:
    import datetime
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    
    # Generate filename with datetime (newest first sorting)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    debug_filename = f"logs/{timestamp}_debug.log"
    
    # Open debug file for writing
    debug_file = open(debug_filename, "w", encoding="utf-8")
    print(f"📝 Debug file: {debug_filename}")

# Simple debug print function with flexible variable printing
def debug_print(message, **kwargs):
    if DEBUG_PRINT:
        # Print to console
        print(message)
        # Handle n number of variables
        for key, value in kwargs.items():
            print(f"  {key}: {value}")
        
        # Also write to debug file
        if debug_file:
            debug_file.write(f"{message}\n")
            debug_file.flush()  # Ensure immediate write
            for key, value in kwargs.items():
                debug_file.write(f"  {key}: {value}\n")
                debug_file.flush()

# Startup message
if DEBUG_PRINT:
    print("🚀 Starting RAG FUSSA API")
    debug_print_cpu_info()

# Mock logger to replace all logger calls with debug_print
class MockLogger:
    def info(self, message, **kwargs):
        debug_print(message, **kwargs)
    
    def warning(self, message, **kwargs):
        debug_print(f"WARNING: {message}", **kwargs)
    
    def error(self, message, **kwargs):
        debug_print(f"ERROR: {message}", **kwargs)

# Replace all logger calls with mock logger
logger = MockLogger()

app = FastAPI(
    title="RAG FUSSA API", 
    version="1.0.0",
    description="Production-ready RAG API with health checks and structured logging"
)

# Configure CORS for production
allowed_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:8080").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    
    # Extract user info from headers or query params
    user_id = request.headers.get("x-user-id") or request.query_params.get("user_id")
    conversation_id = request.headers.get("x-conversation-id") or request.query_params.get("conversation_id")
    
    # Start request logging
    request_id = f"req_{int(time.time() * 1000)}"
    debug_print("Request started", request_id=request_id, user_id=user_id, conversation_id=conversation_id)
    
    try:
        response = await call_next(request)
        duration_ms = (time.time() - start_time) * 1000
        debug_print("Request completed", request_id=request_id, status_code=response.status_code, duration_ms=duration_ms)
        return response
    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        debug_print("Request error", request_id=request_id, error=str(e), method=request.method, url=str(request.url), duration_ms=duration_ms)
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error", "request_id": request_id}
)

# Initialize clients
s3_client = boto3.client(
    's3',
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=AWS_REGION
)

from pinecone import Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
pdf_index = pc.Index(PINECONE_PDF_INDEX)
video_index = pc.Index(PINECONE_VIDEO_INDEX)
image_index = pc.Index(PINECONE_IMAGE_INDEX)

openai.api_key = OPENAI_API_KEY

# Global configuration
DEFAULT_TOP_K = 3  # Default number of chunks to retrieve

# Note: Conversation storage moved to request payload
# user_conversations and user_openai_conversations removed


def extract_filename_from_s3_url(s3_url):
    """Extract filename from S3 URL."""
    parsed_url = urlparse(str(s3_url))
    object_key = parsed_url.path.lstrip('/')
    # URL decode the object key to handle special characters (Arabic, spaces, etc.)
    object_key = unquote(object_key)
    filename = os.path.basename(object_key)
    return filename


async def get_file_type(file_extension):
    """Determine file type from extension."""
    file_extension = file_extension.lower()
    
    if file_extension == 'pdf':
        return 'pdf'
    elif await is_video_file(file_extension):
        return 'video'
    elif await is_image_file(file_extension):
        return 'image'
    else:
        return 'unknown'


def get_index_for_file_type(file_type):
    """Get appropriate Pinecone index for file type."""
    if file_type == 'pdf':
        return pdf_index
    elif file_type == 'video':
        return video_index
    elif file_type == 'image':
        return image_index
    else:
        raise ValueError(f"Unsupported file type: {file_type}")




async def get_embeddings(texts):
    """Get embeddings from OpenAI using text-embedding-3-small with retry logic."""
    try:
        debug_print("OpenAI API call", service="openai", endpoint="embeddings", method="POST", model="text-embedding-3-small", text_count=len(texts))
        
        @openai_retry
        async def _get_embeddings():
            return await asyncio.to_thread(
                lambda: openai.embeddings.create(
        input=texts,
        model="text-embedding-3-small"
    )
            )
        
        response = await _get_embeddings()
        embeddings = [item.embedding for item in response.data]
        
        debug_print("OpenAI API response", service="openai", endpoint="embeddings", status=200, model="text-embedding-3-small", embedding_count=len(embeddings))
        return embeddings
        
    except Exception as e:
        error = handle_openai_error(e)
        debug_print("Get embeddings error", operation="get_embeddings", error=str(e), text_count=len(texts))
        raise error


async def download_document_from_s3(s3_url: str) -> bytes:
    """Download document content from S3 URL"""
    try:
        debug_print("Downloading document from S3", s3_url=s3_url)
        
        # Download document content using requests with asyncio.to_thread
        response = await asyncio.to_thread(requests.get, s3_url, timeout=300)  # 5 minute timeout
        response.raise_for_status()
        
        debug_print("Document downloaded successfully", s3_url=s3_url, size_bytes=len(response.content))
        return response.content
        
    except Exception as e:
        debug_print("S3 download failed", s3_url=s3_url, error=str(e))
        raise Exception(f"Failed to download document from S3: {str(e)}")


async def store_chunks_in_pinecone(chunks: List[Dict], embeddings: List[List[float]], filename: str, file_type: str):
    """Store chunks with embeddings in Pinecone"""
    try:
        debug_print("Storing chunks in Pinecone", filename=filename, file_type=file_type, chunk_count=len(chunks))
        
        # Get appropriate index for file type
        index = get_index_for_file_type(file_type)
        
        # Prepare vectors for Pinecone
        vectors = []
        for i, chunk in enumerate(chunks):
            vector_data = {
                "id": f"{filename}_{i}",
                "values": embeddings[i],
                "metadata": {
                    **chunk["metadata"],
                    "filename": filename,
                    "file_type": file_type,
                    "chunk_index": i,
                    "s3_url": chunk["metadata"].get("s3_url", ""),
                    "content": chunk["content"][:1000]  # Store first 1000 chars for search
                }
            }
            vectors.append(vector_data)
        
        # Store in Pinecone with namespace
        await asyncio.to_thread(index.upsert, vectors=vectors, namespace=filename)
        
        debug_print("Chunks stored successfully in Pinecone", filename=filename, vector_count=len(vectors))
        
    except Exception as e:
        debug_print("Pinecone storage failed", filename=filename, error=str(e))
        raise Exception(f"Failed to store chunks in Pinecone: {str(e)}")


async def update_document_status(uuid: str, status: str, failure_reason: str = None):
    """Update document status in backend"""
    try:
        # Construct backend URL from environment variables
        backend_base_url = os.getenv("BACKEND_BASE_URL", "192.168.68.72")
        backend_port = os.getenv("BACKEND_PORT", "4504")
        backend_endpoint = os.getenv("BACKEND_ENDPOINT_PATH", "/upload/internal/update-document-entry")
        
        backend_url = f"http://{backend_base_url}:{backend_port}{backend_endpoint}"
        debug_print("Updating backend status", backend_url=backend_url, uuid=uuid, status=status)
        
        payload = {
            "search": {
                "uuid": uuid
            },
            "update": {
                "$set": {
                    "trainingStatus": status
                }
            }
        }
        
        # Add failure reason only if status is FAILED and reason is provided
        if status == "FAILED" and failure_reason:
            payload["update"]["$set"]["trainingFailedReason"] = failure_reason
        
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        
        # Use improved error handling with retry logic
        try:
            response = await asyncio.to_thread(
                safe_api_call, "backend", requests.put, backend_url, json=payload, headers=headers, timeout=10
            )
            
            if response.status_code == 200:
                logger.info("Backend status updated successfully", uuid=uuid, status=status)
            else:
                logger.warning("Backend update failed", uuid=uuid, status=status, status_code=response.status_code, response_text=response.text)
                
        except Exception as backend_error:
            # Handle backend errors gracefully
            error = handle_backend_error(backend_error)
            logger.error("Backend update failed", uuid=uuid, status=status, error=str(error))
            # Don't raise exception - backend update failure shouldn't stop document processing
            
    except Exception as e:
        debug_print("Update document status error", operation="update_document_status", error=str(e), uuid=uuid, status=status)
        # Don't raise exception - backend update failure shouldn't stop document processing


@app.post("/ai-service/internal/process-document-data")
async def train(request: TrainRequest):
    """Start document processing asynchronously"""
    request_id = f"req_{int(time.time() * 1000)}"
    
    try:
        logger.info("Starting async document training", 
                   uuid=request.uuid, 
                   name=request.name, 
                   url=str(request.url),
                   type=request.type,
                   request_id=request_id)
        
        # Step 1: Update backend with PROCESSING status
        await update_document_status(request.uuid, "PROCESSING")
        
        # Step 2: Start async Celery task
        request_data = {
                    "name": request.name,
                    "uuid": request.uuid,
                    "url": str(request.url),
                    "type": request.type,
            "trainingStatus": request.trainingStatus
        }
        
        # Step 3: Download document from S3
        debug_print("Downloading document from S3", uuid=request.uuid, s3_url=str(request.url))
        document_content = await download_document_from_s3(str(request.url))
        
        # Step 4: Extract filename and determine file type
        filename = extract_filename_from_s3_url(request.url)
        file_extension = filename.split('.')[-1].lower()
        file_type = await get_file_type(file_extension)
        
        debug_print("Document type detected", uuid=request.uuid, filename=filename, file_type=file_type)
        
        # Step 5: Process document based on type
        text_content = ""
        if file_type == "pdf":
            debug_print("Processing PDF document", uuid=request.uuid, filename=filename)
            text_content = await extract_pdf_text(document_content)
        elif file_type == "video":
            debug_print("Processing video document", uuid=request.uuid, filename=filename)
            text_content = await asyncio.to_thread(extract_video_audio_transcript, document_content, filename)
        elif file_type == "image":
            debug_print("Processing image document", uuid=request.uuid, filename=filename)
            text_content = await extract_text_from_image(document_content, filename)
        else:
            raise Exception(f"Unsupported file type: {file_type}")
        
        debug_print("Text extraction completed", uuid=request.uuid, text_length=len(text_content))
        
        # Step 6: Process text into chunks
        debug_print("Processing text into chunks", uuid=request.uuid, filename=filename)
        chunks = await process_extracted_text(text_content, filename, file_type, "semantic")
        
        if not chunks:
            raise Exception("No chunks generated from document")
        
        debug_print("Chunks generated", uuid=request.uuid, chunk_count=len(chunks))
        
        # Step 7: Generate embeddings for chunks
        debug_print("Generating embeddings", uuid=request.uuid, chunk_count=len(chunks))
        chunk_texts = [chunk["content"] for chunk in chunks]
        embeddings = await get_embeddings(chunk_texts)
        
        debug_print("Embeddings generated", uuid=request.uuid, embedding_count=len(embeddings))
        
        # Step 8: Store chunks in Pinecone
        debug_print("Storing chunks in Pinecone", uuid=request.uuid, filename=filename)
        await store_chunks_in_pinecone(chunks, embeddings, filename, file_type)
        
        # Step 9: Update backend with COMPLETED status
        debug_print("Document processing completed", uuid=request.uuid, filename=filename, chunk_count=len(chunks))
        await update_document_status(request.uuid, "COMPLETED")
        
        return {
            "success": True,
            "message": f"Document processing completed: {len(chunks)} chunks stored",
            "uuid": request.uuid,
            "status": "COMPLETED",
                    "filename": filename,
                    "file_type": file_type,
            "chunks_processed": len(chunks)
        }
        
    except Exception as e:
        debug_print("Start document training error", operation="start_document_training", error=str(e), uuid=request.uuid, request_id=request_id)
        
        # Update backend with FAILED status
        await update_document_status(request.uuid, "FAILED", str(e))
        
        raise HTTPException(status_code=500, detail=f"Failed to start document processing: {str(e)}")

@app.get("/ai-service/internal/task-status/{task_id}")
async def get_task_status(task_id: str):
    """Get the status of a processing task"""
    try:
        # No longer using Celery - return default completed status
        # task = celery_app.AsyncResult(task_id)
        
        # Since we're no longer using Celery, return a default completed status
        response = {
            "task_id": task_id,
            "status": "SUCCESS",
            "progress": 100,
            "message": "Task completed successfully (direct processing)"
        }
        
        return response
        
    except Exception as e:
        debug_print("Get task status error", operation="get_task_status", error=str(e), task_id=task_id)
        raise HTTPException(status_code=500, detail=f"Failed to get task status: {str(e)}")


@app.post("/ai-service/internal/unprocess-document-data", response_model=UntrainResponse)
async def untrain(request: UntrainRequest):
    try:
        debug_print("Starting document removal", uuid=request.uuid, url=str(request.url))
        
        # Step 1: Update backend with PROCESSING status
        await update_document_status(request.uuid, "PROCESSING")
        
        # Extract filename from S3 URL (URL decoding handled in extract_filename_from_s3_url)
        filename = extract_filename_from_s3_url(request.url)
        file_extension = filename.split('.')[-1].lower()
        file_type = await get_file_type(file_extension)
        
        if file_type == 'unknown':
            error_msg = f"Unsupported file type: {file_extension}"
            debug_print("Unsupported file type", uuid=request.uuid, file_extension=file_extension)
            await update_document_status(request.uuid, "FAILED", error_msg)
            raise HTTPException(status_code=400, detail=error_msg)
        
        # Get appropriate index for file type
        index = get_index_for_file_type(file_type)
        
        # Delete the entire namespace (all chunks for this file)
        try:
            # Get stats to check if namespace exists
            stats = await asyncio.to_thread(index.describe_index_stats)
            namespaces = stats.get('namespaces', {})
            
            if filename in namespaces:
                # Delete the namespace
                await asyncio.to_thread(index.delete, delete_all=True, namespace=filename)
                chunks_removed = namespaces[filename].get('vector_count', 0)
                
                debug_print("Document removal successful", uuid=request.uuid, filename=filename, chunks_removed=chunks_removed)
                
                # Update backend with PENDING status
                await update_document_status(request.uuid, "PENDING")
                
                return UntrainResponse(
                    success=True,
                    message=f"Successfully removed {file_type} file: {filename}",
                    s3_url=str(request.url),
                    file_type=file_type,
                    chunks_removed=chunks_removed
                )
            else:
                debug_print("File not found in Pinecone", uuid=request.uuid, filename=filename)
                
                # Update backend with COMPLETED status (file not found is still a successful operation)
                await update_document_status(request.uuid, "COMPLETED")
                
                return UntrainResponse(
                    success=False,
                    message=f"File not found: {filename}",
                    s3_url=str(request.url),
                    file_type=file_type,
                    chunks_removed=0
                )
        except Exception as e:
            # If namespace deletion fails, try alternative approach
            # Query and delete by metadata
            query_response = await asyncio.to_thread(
                index.query,
                vector=[0] * 1536,  # Dummy vector
                filter={"s3_url": str(request.url)},
                top_k=10000,
                include_metadata=False,
                namespace=filename
            )
            
            vector_ids = [match.id for match in query_response.matches]
            
            if vector_ids:
                await asyncio.to_thread(index.delete, ids=vector_ids, namespace=filename)
                
                debug_print("Document removal successful (fallback method)", uuid=request.uuid, filename=filename, chunks_removed=len(vector_ids))
                
                # Update backend with PENDING status
                await update_document_status(request.uuid, "PENDING")
                
                return UntrainResponse(
                    success=True,
                    message=f"Successfully removed {file_type} file: {filename}",
                    s3_url=str(request.url),
                    file_type=file_type,
                    chunks_removed=len(vector_ids)
                )
            else:
                debug_print("File not found in Pinecone (fallback method)", uuid=request.uuid, filename=filename)
                
                # Update backend with COMPLETED status (file not found is still a successful operation)
                await update_document_status(request.uuid, "COMPLETED")
                
                return UntrainResponse(
                    success=False,
                    message=f"File not found: {filename}",
                    s3_url=str(request.url),
                    file_type=file_type,
                    chunks_removed=0
                )
        
    except Exception as e:
        debug_print("Document removal failed", uuid=request.uuid, error=str(e))
        
        # Update backend with FAILED status
        await update_document_status(request.uuid, "FAILED", str(e))
        
        raise HTTPException(status_code=500, detail=f"Untrain failed: {str(e)}")


@app.post("/retrain")
async def retrain():
    return {"message": "Retrain endpoint - to be implemented"}


async def fetch_rag_internal(query: str, top_k: int = 5) -> Dict[str, Any]:
    """Internal function to fetch RAG results (used by both fetch_rag and ask-query-rag)"""
    try:
        logger.info("fetch_rag_internal: Starting RAG fetch", query=query, top_k=top_k)
        
        # Generate query embedding directly
        embedding_start = time.time()
        logger.info("fetch_rag_internal: Generating query embedding")
        query_embedding = await get_embeddings([query])
        embedding_time = (time.time() - embedding_start) * 1000
        print(f"⏱️ TIMING: Embedding Generation: {embedding_time:.2f}ms")
        logger.info("fetch_rag_internal: Query embedding generated")
        
        all_results = []
        
        # Search across all indexes
        search_start = time.time()
        logger.info("fetch_rag_internal: Starting search across indexes")
        indexes = [
            ("pdf", pdf_index),
            ("video", video_index), 
            ("image", image_index)
        ]
        
        for file_type, index in indexes:
            try:
                logger.info("fetch_rag_internal: Searching index", file_type=file_type)
                # Get all namespaces in this index
                stats = await asyncio.to_thread(index.describe_index_stats)
                namespaces = stats.get('namespaces', {})
                logger.info("fetch_rag_internal: Got namespaces", file_type=file_type, namespace_count=len(namespaces))
                
                if not namespaces:
                    # If no namespaces, search without namespace
                    search_response = await asyncio.to_thread(
                        index.query,
                        vector=query_embedding,
                        top_k=top_k,
                        include_metadata=True
                    )
                    
                    for match in search_response.matches:
                        result = {
                            "content": match.metadata.get("content", ""),
                            "score": float(match.score),
                            "file_type": file_type,
                            "filename": match.metadata.get("filename", "unknown"),
                            "s3_url": match.metadata.get("s3_url", ""),
                            "chunk_index": match.metadata.get("chunk_index", 0),
                            "metadata": match.metadata
                        }
                        all_results.append(result)
                else:
                    # Search each namespace separately
                    for namespace_name in namespaces.keys():
                        search_response = await asyncio.to_thread(
                            index.query,
                            vector=query_embedding,
                            top_k=top_k,
                            include_metadata=True,
                            namespace=namespace_name
                        )
                        
                        for match in search_response.matches:
                            result = {
                                "content": match.metadata.get("content", ""),
                                "score": float(match.score),
                                "file_type": file_type,
                                "filename": match.metadata.get("filename", namespace_name),
                                "s3_url": match.metadata.get("s3_url", ""),
                                "chunk_index": match.metadata.get("chunk_index", 0),
                                "namespace": namespace_name,
                                "metadata": match.metadata
                            }
                            all_results.append(result)
                            
            except Exception as e:
                # Continue with other indexes if one fails
                debug_print("Pinecone search error", operation="pinecone_search", error=str(e), index_type=file_type)
                continue
        
        # Sort all results by score (highest first)
        all_results.sort(key=lambda x: x["score"], reverse=True)
        
        # Limit to requested top_k
        final_results = all_results[:top_k]
        
        search_time = (time.time() - search_start) * 1000
        print(f"⏱️ TIMING: Pinecone Search: {search_time:.2f}ms")
        
        return {
            "success": True,
            "results": final_results,
            "total_retrieved": len(final_results)
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "results": [],
            "total_retrieved": 0
        }


@app.post("/fetch_rag", response_model=RAGQueryResponse)
async def fetch_rag(request: RAGQueryRequest):
    """Retrieve relevant content from RAG system without AI processing"""
    try:
        result = await fetch_rag_internal(request.query, request.top_k)
        
        if result["success"]:
            return RAGQueryResponse(
                success=True,
                message=f"Found {result['total_retrieved']} results across all indexes",
                results=result["results"]
            )
        else:
            raise HTTPException(status_code=500, detail=result["error"])
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG query failed: {str(e)}")


def filter_conversation_history(conversation_history: List[Dict]) -> List[Dict]:
    """Filter conversation history to keep all questions but only the last answer for optimization"""
    if not conversation_history:
        return []
    
    filtered_history = []
    
    for i, msg in enumerate(conversation_history):
        # Check if this is the last message
        is_last = (i == len(conversation_history) - 1)
        
        if is_last:
            # Keep both question and answer for the last message
            filtered_history.append({
                "question": msg.get("question", ""),
                "answer": msg.get("answer", "")
            })
        else:
            # Keep only question for older messages
            filtered_history.append({
                "question": msg.get("question", "")
            })
    
    logger.info("Conversation history filtered", 
                original_count=len(conversation_history), 
                filtered_count=len(filtered_history),
                answers_kept=1 if conversation_history else 0)
    
    return filtered_history


async def analyze_query_and_classify(request_data: dict) -> Dict[str, str]:
    """Combined function: Rephrase query + Classify type in ONE API call using gpt-4.1-nano"""
    try:
        import openai
        
        # Get conversation history from request and filter it
        conversation_history = request_data.get("conversationHistory", [])
        filtered_history = filter_conversation_history(conversation_history)
        query = request_data.get("question", "")
        
        logger.info("Starting combined query analysis", query=query, history_count=len(filtered_history))
        
        # Quick check for obvious greetings - no need to rephrase
        query_lower = query.lower().strip()
        if query_lower in ['hello', 'hi', 'hey', 'good morning', 'good evening', 'thanks', 'thank you', 'bye', 'goodbye']:
            return {
                "rephrasedQuery": query,
                "classification": "GENERAL_CONVERSATION"
            }
        
        # Build conversation context
        context = ""
        if filtered_history:
            context_parts = []
            for msg in filtered_history:
                context_parts.append(f"User: {msg['question']}")
                if msg.get('answer'):  # Only last message will have answer
                    context_parts.append(f"Assistant: {msg['answer'][:200]}...")
            context = "\n".join(context_parts)
        
        # Combined prompt for both rephrasing and classification
        combined_prompt = f"""You have TWO tasks to perform on the user's query:

CONVERSATION CONTEXT:
{context if context else "No previous conversation"}

IMPORTANT NOTE: For optimization, only the most recent answer is included in conversation history. All previous questions were answered, but those answers are excluded to save tokens. This is an ongoing conversation with full context maintained.

CURRENT USER QUERY: {query}

TASK 1 - REPHRASE QUERY (if needed):
- If the query is a follow-up question referencing previous conversation, rephrase it to be standalone
- Include necessary context from conversation history
- If query is already clear and standalone, keep it as is
- Examples:
  * "What about that?" → "What about [specific topic from context]?"
  * "Can you explain the second point?" → "Can you explain [specific point from context]?"
  * "Tell me more" → "Tell me more about [topic from context]"

TASK 2 - CLASSIFY QUERY TYPE:
Determine if this is GENERAL_CONVERSATION or KNOWLEDGE_QUESTION

GENERAL_CONVERSATION:
- Greetings, pleasantries, thank you, goodbye
- Questions about our current conversation ("What did I ask earlier?", "What's my name?")
- Personal opinions, advice, general chat
- Relationship advice, life coaching, generic facts NOT in knowledge base

KNOWLEDGE_QUESTION:
- Questions that need specific information from uploaded documents
- Technical queries, factual questions requiring knowledge base
- Document-specific questions

Return your response in this EXACT JSON format (no other text):
{{
  "rephrasedQuery": "the rephrased query or original if no rephrasing needed",
  "classification": "GENERAL_CONVERSATION or KNOWLEDGE_QUESTION"
}}"""

        response = openai.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[
                {"role": "system", "content": "You are a query analyzer. Analyze and return results in JSON format."},
                {"role": "user", "content": combined_prompt}
            ],
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response.choices[0].message.content.strip())
        
        rephrased_query = result.get("rephrasedQuery", query)
        classification = result.get("classification", "KNOWLEDGE_QUESTION")
        
        print(f"🔍 DEBUG: Combined Analysis Result: {result}")
        print(f"🔍 DEBUG: Final Rephrased Query: {rephrased_query}")
        print(f"🔍 DEBUG: Final Classification: {classification}")
        
        logger.info("Combined query analysis completed", 
                   original_query=query, 
                   rephrased_query=rephrased_query, 
                   classification=classification)
        
        return {
            "rephrasedQuery": rephrased_query,
            "classification": classification
        }
        
    except Exception as e:
        debug_print("Combined query analysis error", operation="analyze_query_and_classify", error=str(e), query=query)
        # Fallback: return original query and default to knowledge question
        return {
            "rephrasedQuery": query,
            "classification": "KNOWLEDGE_QUESTION"
        }


async def generate_conversation_title(first_message: str) -> str:
    """Generate conversation title using OpenAI - handles all cases"""
    try:
        import openai
        
        prompt = f"""Generate a short, descriptive title (max 50 characters) for this conversation based on the user's first message.

USER'S FIRST MESSAGE: {first_message}

INSTRUCTIONS:
- If it's a greeting (hi, hello, hey), create a title like General Chat or New Conversation
- If it's a specific question, create a descriptive title
- If it's a help request, create a relevant title
- Always return a meaningful title, never empty or generic
- Keep it under 50 characters
- Make it clear and specific
- Do NOT include quotes around the title

Examples:
- "Hi" → General Chat
- "Hello" → New Conversation 
- "What is the constitution?" → Constitution Discussion
- "Can you help me?" → Help Request
- "Tell me about AI" → AI Discussion

Return ONLY the title without quotes, nothing else."""

        response = openai.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[
                {"role": "system", "content": "You are a conversation title generator. Return only the title."},
                {"role": "user", "content": prompt}
            ]
        )
        
        title = response.choices[0].message.content.strip()
        logger.info("Conversation title generated", title=title, first_message=first_message)
        return title if title else "New Conversation"
        
    except Exception as e:
        debug_print("Generate conversation title error", operation="generate_conversation_title", error=str(e), first_message=first_message)
        return "New Conversation"


# Note: save_conversation_locally removed - conversations now handled by backend
# Note: rephrase_followup_query and classify_query_type removed - now combined in analyze_query_and_classify


async def generate_general_conversation_answer(request_data: dict) -> Dict[str, str]:
    """Generate general conversation response using gpt-4.1-nano with filtered conversation history"""
    try:
        import openai
        
        # Get and filter conversation history
        conversation_history = request_data.get("conversationHistory", [])
        filtered_history = filter_conversation_history(conversation_history)
        
        # Get conversation ID from request (null for new chat)
        previous_response_id = request_data.get("conversationId")
        query = request_data.get("question", "")
        
        # Build conversation context
        context = ""
        if filtered_history:
            context_parts = []
            for msg in filtered_history:
                context_parts.append(f"User: {msg['question']}")
                if msg.get('answer'):  # Only last message will have answer
                    context_parts.append(f"Assistant: {msg['answer']}")
            context = "\n".join(context_parts)
        
        # Create conversational input
        input_message = f"""CONVERSATION CONTEXT:
{context if context else "No previous conversation"}

IMPORTANT NOTE: For optimization, only the most recent answer is included in conversation history. All previous questions were answered, but those answers are excluded to save tokens. This is an ongoing conversation with full context maintained.

CURRENT USER QUERY: {query}

You are handling general conversation. Decide between three behaviors and output ONLY the final message text (no labels):

LANGUAGE REQUIREMENT:
- IMPORTANT: Respond in the SAME LANGUAGE as the user's question
- If the user asked in Hindi, respond in Hindi
- If the user asked in Hinglish, respond in Hinglish
- If the user asked in English, respond in English
- Maintain the same language style and tone throughout the conversation
- Handle mixed languages naturally (code-switching)

1) GREETING_REPLY
- Trigger if the message is a greeting/pleasantry/small talk (hello, hi, hey, good morning, thanks, how are you, nice to meet you, what's up, good night) or a very short friendly check-in.
- Respond warmly and briefly, 1-2 sentences max.

2) CONVERSATION_CONTEXT
- Trigger for questions about our current conversation, user's personal information shared in this chat, or past exchanges in this session.
- Examples: "What's my name?", "What did I say earlier?", "What did you tell me about X?", "What was my first question?", "What did I ask you about my project?"
- Answer these questions naturally using your conversation memory and context from this chat session.
- Be helpful and reference what was discussed earlier.

3) POLITE_REFUSAL
- Trigger ONLY for general knowledge questions NOT related to our conversation (e.g., relationship advice, life coaching, opinions, generic facts, news, health/financial/legal advice, etc.).
- Do NOT refuse questions about our current conversation or user's personal information shared in this chat.
- Politely say you can't answer general questions and that you only answer using the user's uploaded knowledge base. Suggest adding relevant documents if they want help.
- Keep to 1-2 sentences, kind and clear.

Examples:
- User: "hello" → "Hi there! Great to see you. How can I help today?"
- User: "What's my name?" → "Your name is [name from conversation memory]"
- User: "What did I say earlier?" → "You mentioned [specific content from conversation]"
- User: "What did you tell me about my project?" → "I told you that [specific advice from conversation]"
- User: "hi can you help me to manage my break up with my gf?" → "I'm sorry, but I can't provide general personal advice. I only answer using information from your uploaded knowledge base. You can add relevant documents and ask again."
- User: "who is prime minister of india" → "Sorry, I can't answer general questions. I only respond using your uploaded knowledge base."
"""
        
        print(f"🔧 Using API Method: {'responses.create()' if USE_RESPONSE_API else 'chat.completions.create()'}")
        
        if USE_RESPONSE_API:
            # ===== METHOD 1: responses.create() API =====
            try:
                if previous_response_id:
                    # Continue existing conversation
                    logger.info("Continuing general conversation with responses.create()")
                    response = openai.responses.create(
                        model="gpt-4.1-nano",
                        input=input_message,
                        previous_response_id=previous_response_id,
                        max_output_tokens=300,
                        temperature=0.2
                    )
                else:
                    # Start new conversation
                    logger.info("Starting new general conversation with responses.create()")
                    response = openai.responses.create(
                        model="gpt-4.1-nano",
                        input=input_message,
                        max_output_tokens=300,
                        temperature=0.2
                    )
                
                ai_answer = response.output_text.strip()
                new_conversation_id = response.id
                
                # Extract token usage
                input_tokens = response.usage.input_tokens if hasattr(response, 'usage') else 'N/A'
                output_tokens = response.usage.output_tokens if hasattr(response, 'usage') else 'N/A'
                total_tokens = response.usage.total_tokens if hasattr(response, 'usage') else 'N/A'
                
                print(f"🎫 TOKENS: Input={input_tokens} | Output={output_tokens} | Total={total_tokens}")
                logger.info("Generated general conversation response using responses.create()")
                return {"answer": ai_answer, "conversationId": new_conversation_id}
                
            except Exception as e:
                print(f"❌ responses.create() ERROR: {str(e)}")
                raise e
        
        else:
            # ===== METHOD 2: chat.completions.create() API =====
            try:
                # Convert input_message to chat format
                messages = [
                    {"role": "system", "content": "You are a helpful assistant for general conversation."},
                    {"role": "user", "content": input_message}
                ]
                
                response = openai.chat.completions.create(
                    model="gpt-4.1-nano",
                    messages=messages
                )
                
                ai_answer = response.choices[0].message.content.strip()
                new_conversation_id = f"chat_{int(time.time() * 1000)}"  # Generate simple ID
                
                # Extract token usage
                input_tokens = response.usage.prompt_tokens
                output_tokens = response.usage.completion_tokens
                total_tokens = response.usage.total_tokens
                
                print(f"🎫 TOKENS: Input={input_tokens} | Output={output_tokens} | Total={total_tokens}")
                logger.info("Generated general conversation response using chat.completions.create()")
                return {"answer": ai_answer, "conversationId": new_conversation_id}
                
            except Exception as e:
                print(f"❌ chat.completions.create() ERROR: {str(e)}")
                raise e
        
    except Exception as e:
        debug_print("Generate general conversation answer error", operation="generate_general_conversation_answer", error=str(e))
        return {"answer": "Hello! I'm here to help you with any questions you might have. How can I assist you today?", "conversationId": request_data.get("conversationId")}


@app.post("/ai-service/internal/ask-question", response_model=AskQueryRAGResponse)
async def ask_query_rag(request: AskQueryRAGRequest):
    """Ask a question to the RAG system with conversational context using conversation_id"""
    start_time = time.time()
    request_id = f"req_{int(time.time() * 1000)}"
    
    try:
        debug_print("Processing AI query", 
                   conversation_id=request.conversationId, 
                   question=request.question,
                   type=request.type,
                   request_id=request_id)
        
        # Check if this is the first message in the conversation
        is_first_message = request.conversationId is None or request.conversationId == ""
        conversation_title = None
        
        print(f"🔍 DEBUG: Is First Message: {is_first_message}")
        print(f"🔍 DEBUG: Conversation ID from request: {request.conversationId}")
        
        if is_first_message:
            print(f"🔍 DEBUG: Generating conversation title for: {request.question}")
            logger.info("First message detected", conversation_id=request.conversationId)
            conversation_title = await generate_conversation_title(request.question)
            print(f"🔍 DEBUG: Generated Title: {conversation_title}")
        else:
            print(f"🔍 DEBUG: Not a first message, skipping title generation")
        
        # Handle DOCUMENT type messages (placeholder for now)
        if request.type == "DOCUMENT":
            document_count = len(request.documents) if request.documents else 0
            logger.info("Document type message received", conversation_id=request.conversationId, document_count=document_count)
            # TODO: Process documents and add to context
            # For now, just process the message normally
        
        # Step 1: Combined query analysis (rephrase + classify) in ONE API call
        request_data = {
            "question": request.question,
            "conversationId": request.conversationId,
            "conversationHistory": request.conversationHistory
        }
        
        # Step 1: Combined query analysis timing
        analysis_start = time.time()
        debug_print("Starting combined query analysis", conversation_id=request.conversationId)
        analysis_result = await analyze_query_and_classify(request_data)
        rephrased_query = analysis_result["rephrasedQuery"]
        query_type = analysis_result["classification"]
        analysis_time = (time.time() - analysis_start) * 1000
        
        print(f"⏱️ TIMING: Combined Analysis: {analysis_time:.2f}ms")
        print(f"🔍 DEBUG: Original Query: {request.question}")
        print(f"🔍 DEBUG: Rephrased Query: {rephrased_query}")
        print(f"🔍 DEBUG: Classification: {query_type}")
        print(f"🔍 DEBUG: Conversation History Length: {len(request.conversationHistory)}")
        
        debug_print("Combined analysis completed", 
                   original_query=request.question, 
                   rephrased_query=rephrased_query, 
                   query_type=query_type)
        
        # Step 2: RAG retrieval timing
        rag_start = time.time()
        rag_result = await fetch_rag_internal(rephrased_query, DEFAULT_TOP_K)
        rag_time = (time.time() - rag_start) * 1000
        
        if not rag_result["success"]:
            raise HTTPException(status_code=500, detail=f"Content retrieval failed: {rag_result['error']}")
        
        retrieved_content = rag_result["results"]
        total_retrieved = rag_result["total_retrieved"]
        
        print(f"⏱️ TIMING: RAG Retrieval: {rag_time:.2f}ms")
        print(f"🔍 DEBUG: RAG Success: {rag_result['success']}")
        print(f"🔍 DEBUG: Total Retrieved: {total_retrieved}")
        print(f"🔍 DEBUG: Retrieved Content Preview: {retrieved_content[0]['content'][:100] if retrieved_content else 'No content'}")
        
        debug_print("Content retrieved", conversation_id=request.conversationId, total_retrieved=total_retrieved)
        
        if query_type == "GENERAL_CONVERSATION":
            # Handle general conversation without RAG
            debug_print("General conversation detected", conversation_id=request.conversationId)
            try:
                # Step 3a: General conversation timing
                general_start = time.time()
                print(f"🔍 DEBUG: Generating GENERAL conversation answer")
                logger.info("Starting general conversation answer generation", conversation_id=request.conversationId)
                result = await generate_general_conversation_answer(request_data)
                general_time = (time.time() - general_start) * 1000
                logger.info("General conversation answer generated", conversation_id=request.conversationId)
                ai_answer = result["answer"]
                new_conversation_id = result["conversationId"]
                print(f"⏱️ TIMING: General Answer Generation: {general_time:.2f}ms")
                print(f"🔍 DEBUG: General Answer Generated: {ai_answer[:100]}...")
                print(f"🔍 DEBUG: New Conversation ID: {new_conversation_id}")
            except Exception as e:
                print(f"🔍 DEBUG: ERROR in General Conversation: {str(e)}")
                logger.error("General conversation answer generation failed", conversation_id=request.conversationId, error=str(e))
                raise
            
            # Conversation saved by backend - no local storage needed
            
            # Return response without retrieved content for general chat
            return AskQueryRAGResponse(
                success=True,
                message="General conversation response",
                conversationId=new_conversation_id or "new_conversation",
                answer=ai_answer,
                retrieved_content=[],
                total_retrieved=0,
                conversationTitle=conversation_title
            )
        else:
            # Handle knowledge question with RAG using rephrased query
            logger.info("Knowledge question detected", conversation_id=request.conversationId)
            try:
                # Step 3b: Knowledge answer timing
                knowledge_start = time.time()
                print(f"🔍 DEBUG: Generating KNOWLEDGE answer with {len(retrieved_content)} content pieces")
                logger.info("Starting knowledge question answer generation", conversation_id=request.conversationId)
                result = await generate_conversational_ai_answer(request_data, retrieved_content)
                knowledge_time = (time.time() - knowledge_start) * 1000
                logger.info("Knowledge question answer generated", conversation_id=request.conversationId)
                ai_answer = result["answer"]
                new_conversation_id = result["conversationId"]
                print(f"⏱️ TIMING: Knowledge Answer Generation: {knowledge_time:.2f}ms")
                print(f"🔍 DEBUG: Knowledge Answer Generated: {ai_answer[:100]}...")
                print(f"🔍 DEBUG: New Conversation ID: {new_conversation_id}")
            except Exception as e:
                print(f"🔍 DEBUG: ERROR in Knowledge Generation: {str(e)}")
                logger.error("Knowledge question answer generation failed", conversation_id=request.conversationId, error=str(e))
                raise
        
        # Step 4: Conversation saved by backend - no local storage needed
        
        # Log performance metrics
        duration_ms = (time.time() - start_time) * 1000
        
        print(f"⏱️ TIMING: TOTAL REQUEST TIME: {duration_ms:.2f}ms")
        print(f"⏱️ TIMING BREAKDOWN:")
        print(f"   - Combined Analysis: {analysis_time:.2f}ms")
        print(f"   - RAG Retrieval: {rag_time:.2f}ms")
        if query_type == "GENERAL_CONVERSATION":
            print(f"   - General Answer: {general_time:.2f}ms")
        else:
            print(f"   - Knowledge Answer: {knowledge_time:.2f}ms")
        
        debug_print("Performance metric", metric="ask_query_duration", value=f"{duration_ms}ms", conversation_id=request.conversationId, query_type=query_type, total_retrieved=total_retrieved)
        
        logger.info("AI query completed successfully", 
                   conversation_id=request.conversationId,
                   query_type=query_type,
                   total_retrieved=total_retrieved,
                   duration_ms=duration_ms)
        
        # Step 5: Return structured response
        return AskQueryRAGResponse(
            success=True,
            message="AI answer generated successfully",
            conversationId=new_conversation_id or "new_conversation",
            answer=ai_answer,
            retrieved_content=retrieved_content,
            total_retrieved=total_retrieved,
            conversationTitle=conversation_title
        )
        
    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        
        debug_print("Ask query RAG error", operation="ask_query_rag", error=str(e), conversation_id=request.conversationId, question=request.question, duration_ms=duration_ms, request_id=request_id)
        
        # Handle cases where retrieval works but AI fails
        try:
            rag_result = await fetch_rag_internal(request.question, DEFAULT_TOP_K)
            retrieved_content = rag_result.get("results", [])
            
            # Error handling - conversation saved by backend
            error_answer = "Sorry, I couldn't generate an answer due to a technical issue, but I found some relevant content below."
            
            logger.warning("AI processing failed, returning retrieved content", 
                          conversation_id=request.conversationId,
                          retrieved_count=len(retrieved_content),
                          error=str(e))
            
            return AskQueryRAGResponse(
                success=False,
                message=f"AI processing failed, but retrieved content available: {str(e)}",
                conversationId=request.conversationId or "new_conversation",
                answer=error_answer,
                retrieved_content=retrieved_content,
                total_retrieved=len(retrieved_content),
                conversationTitle=conversation_title
            )
        except Exception as fallback_error:
            debug_print("Ask query RAG fallback error", operation="ask_query_rag_fallback", error=str(fallback_error), conversation_id=request.conversationId, original_error=str(e))
            raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")


async def generate_conversational_ai_answer(request_data: dict, retrieved_content: List[Dict[str, Any]]) -> Dict[str, str]:
    """Generate knowledge-based AI answer using gpt-4.1-nano with filtered conversation history"""
    try:
        import openai
        
        # Get and filter conversation history
        conversation_history = request_data.get("conversationHistory", [])
        filtered_history = filter_conversation_history(conversation_history)
        query = request_data.get("question", "")
        
        # Handle case where no content was retrieved
        if not retrieved_content:
            return {"answer": """I'm sorry, but I don't have information about that topic in my current knowledge base. Please try asking about something else or upload relevant documents first.""", "conversationId": request_data.get("conversationId")}

        # Prepare context from retrieved content
        context_parts = []
        for i, content in enumerate(retrieved_content, 1):
            context_parts.append(content.get('content', ''))
        
        context = "\n\n---\n\n".join(context_parts)
        
        # Build conversation context
        conversation_context = ""
        if filtered_history:
            context_parts_conv = []
            for msg in filtered_history:
                context_parts_conv.append(f"User: {msg['question']}")
                if msg.get('answer'):  # Only last message will have answer
                    context_parts_conv.append(f"Assistant: {msg['answer'][:200]}...")
            conversation_context = "\n".join(context_parts_conv)
        
        # Create input with improved relevance detection
        input_message = f"""CONVERSATION HISTORY:
{conversation_context if conversation_context else "No previous conversation"}

IMPORTANT NOTE: For optimization, only the most recent answer is included in conversation history. All previous questions were answered, but those answers are excluded to save tokens. This is an ongoing conversation with full context maintained.

You are an expert assistant that answers questions using the provided context information. Follow these guidelines:

RELEVANCE GUIDELINES:
- Analyze the context to find any information that relates to the user's question
- Look for related concepts, synonyms, and connected topics
- If the context contains useful information about the topic, use it to answer
- If the context has partial information, provide what you can and mention what's missing
- Only decline if the context has absolutely no connection to the question

CONTENT RULES:
- Use only information from the provided context
- Never add information not in the context
- Never use your general knowledge to supplement
- If context partially addresses the question, answer what you can
- Be honest about limitations when information is incomplete

LANGUAGE REQUIREMENT:
- IMPORTANT: Respond in the SAME LANGUAGE as the user's question
- If the user asked in Hindi, respond in Hindi
- If the user asked in Hinglish, respond in Hinglish
- If the user asked in English, respond in English
- Maintain the same language style and tone throughout the conversation
- Handle mixed languages naturally (code-switching)

CONTEXT INFORMATION:
{context}

USER QUESTION: {query}

WRITING STYLE:
- Write naturally and conversationally
- Present information as your expertise
- Don't mention "context", "retrieved content", or "based on information provided"
- If context only covers part of the question, provide available information and note what's missing
- Use bullet points and clear structure when helpful
- Keep paragraphs concise and readable

Provide a clear, natural answer based on the available information."""

        # Get conversation ID from request (null for new chat)
        previous_response_id = request_data.get("conversationId")
        
        print(f"🔍 DEBUG: Previous Response ID: {previous_response_id}")
        print(f"🔧 Using API Method: {'responses.create()' if USE_RESPONSE_API else 'chat.completions.create()'}")
        
        if USE_RESPONSE_API:
            # ===== METHOD 1: responses.create() API =====
            try:
                if previous_response_id:
                    # Continue existing conversation
                    logger.info("Continuing conversation with responses.create()", previous_response_id=previous_response_id)
                    response = openai.responses.create(
                        model="gpt-4.1-nano",
                        input=input_message,
                        previous_response_id=previous_response_id,
                        max_output_tokens=1200,
                        temperature=0.3
                    )
                else:
                    # Start new conversation
                    logger.info("Starting new conversation with responses.create()")
                    response = openai.responses.create(
                        model="gpt-4.1-nano",
                        input=input_message,
                        max_output_tokens=1200,
                        temperature=0.3
                    )
                
                ai_answer = response.output_text.strip()
                new_conversation_id = response.id
                
                # Extract token usage
                input_tokens = response.usage.input_tokens if hasattr(response, 'usage') else 'N/A'
                output_tokens = response.usage.output_tokens if hasattr(response, 'usage') else 'N/A'
                total_tokens = response.usage.total_tokens if hasattr(response, 'usage') else 'N/A'
                
                print(f"🎫 TOKENS: Input={input_tokens} | Output={output_tokens} | Total={total_tokens}")
                logger.info("Generated response using responses.create()", response_id=new_conversation_id)
                return {"answer": ai_answer, "conversationId": new_conversation_id}
                
            except Exception as e:
                print(f"❌ responses.create() ERROR: {str(e)}")
                raise e
        
        else:
            # ===== METHOD 2: chat.completions.create() API =====
            try:
                # Convert input_message to chat format
                messages = [
                    {"role": "system", "content": "You are an expert assistant that answers questions using provided context."},
                    {"role": "user", "content": input_message}
                ]
                
                response = openai.chat.completions.create(
                    model="gpt-4.1-nano",
                    messages=messages
                )
                
                ai_answer = response.choices[0].message.content.strip()
                new_conversation_id = f"chat_{int(time.time() * 1000)}"  # Generate simple ID
                
                # Extract token usage
                input_tokens = response.usage.prompt_tokens
                output_tokens = response.usage.completion_tokens
                total_tokens = response.usage.total_tokens
                
                print(f"🎫 TOKENS: Input={input_tokens} | Output={output_tokens} | Total={total_tokens}")
                logger.info("Generated response using chat.completions.create()", conversation_id=new_conversation_id)
                return {"answer": ai_answer, "conversationId": new_conversation_id}
                
            except Exception as e:
                print(f"❌ chat.completions.create() ERROR: {str(e)}")
                raise e
        
    except Exception as e:
        debug_print("Generate conversational AI answer error", operation="generate_conversational_ai_answer", error=str(e))
        return {"answer": "I apologize, but I encountered an error while generating the answer. Please try your question again, or contact support if this error persists.", "conversationId": request_data.get("conversationId")}


# Health check endpoints
@app.get("/health")
async def health_check():
    """Comprehensive health check with all dependencies"""
    try:
        logger.info("Health check requested")
        
        # Check all dependencies
        dependency_results = await check_all_dependencies()
        overall_status = get_overall_health_status(dependency_results)
        
        # Prepare response
        health_data = {
            "status": overall_status.value,
            "timestamp": time.time(),
            "version": "1.0.0",
            "dependencies": {}
        }
        
        # Add dependency details
        for service, result in dependency_results.items():
            health_data["dependencies"][service] = {
                "status": result.status.value,
                "message": result.message,
                "details": result.details,
                "timestamp": result.timestamp
            }
        
        # Set HTTP status code based on overall health
        if overall_status == HealthStatus.HEALTHY:
            status_code = 200
        elif overall_status == HealthStatus.DEGRADED:
            status_code = 200  # Still operational but with issues
        else:
            status_code = 503  # Service unavailable
        
        logger.info(f"Health check completed", status=overall_status.value, status_code=status_code)
        return JSONResponse(status_code=status_code, content=health_data)
        
    except Exception as e:
        debug_print("Health check error", endpoint="health_check", error=str(e))
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": "Health check failed",
                "timestamp": time.time()
            }
        )

@app.get("/health/live")
async def liveness_check():
    """Simple liveness check for Kubernetes"""
    return {"status": "alive", "timestamp": time.time()}

@app.get("/health/ready")
async def readiness_check():
    """Readiness check for load balancers"""
    try:
        # Quick check of critical dependencies
        dependency_results = await check_all_dependencies()
        overall_status = get_overall_health_status(dependency_results)
        
        if overall_status in [HealthStatus.HEALTHY, HealthStatus.DEGRADED]:
            return {"status": "ready", "timestamp": time.time()}
        else:
            return JSONResponse(
                status_code=503,
                content={"status": "not_ready", "timestamp": time.time()}
            )
    except Exception as e:
        debug_print("Readiness check error", endpoint="readiness_check", error=str(e))
        return JSONResponse(
            status_code=503,
            content={"status": "not_ready", "error": str(e)}
        )

@app.get("/")
async def root():
    return {
        "message": "RAG FUSSA API", 
        "docs": "/docs",
        "health": "/health",
        "version": "1.0.0"
    }


if __name__ == "__main__":
    import uvicorn
    try:
        uvicorn.run("src.main:app", host="0.0.0.0", port=8000, reload=True)
    finally:
        # Close debug file on shutdown
        if debug_file:
            debug_file.close()
            print("📝 Debug file closed")
