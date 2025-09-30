import os
import io
import boto3
import openai
import requests
import time
from urllib.parse import urlparse, unquote
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import PyPDF2

from .config.config import *
from .config.schemas import *
from .processors.pdf_processor import process_pdf, get_pdf_info
from .processors.video_processor import process_video, is_video_file
from .processors.image_processor import process_image, is_image_file
from .utils.text_pipeline import process_text_file_to_chunks
from .utils.logging_config import (
    setup_logging, get_logger, log_request_start, log_request_end,
    log_api_call, log_api_response, log_processing_step, log_conversation_event,
    log_error, log_performance_metric, request_id_var, user_id_var, conversation_id_var
)
from .utils.health_checks import check_all_dependencies, get_overall_health_status, HealthStatus
from .utils.error_handling import (
    ServiceError, OpenAIServiceError, PineconeServiceError, S3ServiceError, BackendServiceError,
    handle_openai_error, handle_pinecone_error, handle_s3_error, handle_backend_error,
    safe_api_call, safe_async_api_call, create_fallback_response, openai_retry, pinecone_retry, s3_retry, backend_retry
)
from .celery_app import celery_app
from .tasks.document_processing import process_document_task
from .tasks.embedding_tasks import generate_single_embedding

# Initialize logging
setup_logging(
    log_level=os.getenv("LOG_LEVEL", "INFO"),
    log_file=os.getenv("LOG_FILE", "logs/rag_fussa.log")
)

logger = get_logger("main")
logger.info("Starting RAG FUSSA API")

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
    request_id = log_request_start(request, user_id, conversation_id)
    
    try:
        response = await call_next(request)
        duration_ms = (time.time() - start_time) * 1000
        log_request_end(request_id, response.status_code, duration_ms)
        return response
    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        log_error(e, {
            "request_id": request_id,
            "method": request.method,
            "url": str(request.url),
            "duration_ms": duration_ms
        })
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

# In-memory conversation storage for backup
user_conversations = {}

# OpenAI conversation tracking (Response API)
user_openai_conversations = {}  # {chat_id: previous_response_id}


def extract_filename_from_s3_url(s3_url):
    """Extract filename from S3 URL."""
    parsed_url = urlparse(str(s3_url))
    object_key = parsed_url.path.lstrip('/')
    # URL decode the object key to handle special characters (Arabic, spaces, etc.)
    object_key = unquote(object_key)
    filename = os.path.basename(object_key)
    return filename


def get_file_type(file_extension):
    """Determine file type from extension."""
    file_extension = file_extension.lower()
    
    if file_extension == 'pdf':
        return 'pdf'
    elif is_video_file(file_extension):
        return 'video'
    elif is_image_file(file_extension):
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




def get_embeddings(texts):
    """Get embeddings from OpenAI using text-embedding-3-small with retry logic."""
    try:
        log_api_call("openai", "embeddings", "POST", model="text-embedding-3-small", text_count=len(texts))
        
        @openai_retry
        def _get_embeddings():
            return openai.embeddings.create(
                input=texts,
                model="text-embedding-3-small"
            )
        
        response = _get_embeddings()
        embeddings = [item.embedding for item in response.data]
        
        log_api_response("openai", "embeddings", 200, 0, model="text-embedding-3-small", embedding_count=len(embeddings))
        return embeddings
        
    except Exception as e:
        error = handle_openai_error(e)
        log_error(e, {"operation": "get_embeddings", "text_count": len(texts)})
        raise error


async def update_document_status(uuid: str, status: str, failure_reason: str = None):
    """Update document status in backend"""
    try:
        # Construct backend URL from environment variables
        backend_base_url = os.getenv("BACKEND_BASE_URL", "192.168.68.72")
        backend_port = os.getenv("BACKEND_PORT", "4504")
        backend_endpoint = os.getenv("BACKEND_ENDPOINT_PATH", "/upload/internal/update-document-entry")
        
        backend_url = f"http://{backend_base_url}:{backend_port}{backend_endpoint}"
        logger.info("Updating backend status", backend_url=backend_url, uuid=uuid, status=status)
        
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
            response = safe_api_call("backend", requests.put, backend_url, json=payload, headers=headers, timeout=10)
            
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
        log_error(e, {"operation": "update_document_status", "uuid": uuid, "status": status})
        # Don't raise exception - backend update failure shouldn't stop document processing


@app.post("/ai-service/internal/process-document-data")
async def train(request: TrainRequest):
    """Start document processing asynchronously"""
    request_id = request_id_var.get() or "unknown"
    
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
        
        # Submit task to Celery
        task = process_document_task.delay(request_data)
        
        logger.info("Document processing task submitted", 
                   uuid=request.uuid,
                   task_id=task.id,
                   request_id=request_id)
        
        # Return immediate response with task ID
        return {
            "success": True,
            "message": "Document processing started",
            "task_id": task.id,
            "uuid": request.uuid,
            "status": "PROCESSING",
            "check_status_url": f"/ai-service/internal/task-status/{task.id}"
        }
        
    except Exception as e:
        log_error(e, {
            "operation": "start_document_training",
            "uuid": request.uuid,
            "request_id": request_id
        })
        
        # Update backend with FAILED status
        await update_document_status(request.uuid, "FAILED", str(e))
        
        raise HTTPException(status_code=500, detail=f"Failed to start document processing: {str(e)}")

@app.get("/ai-service/internal/task-status/{task_id}")
async def get_task_status(task_id: str):
    """Get the status of a processing task"""
    try:
        task = celery_app.AsyncResult(task_id)
        
        if task.state == "PENDING":
            response = {
                "task_id": task_id,
                "status": "PENDING",
                "progress": 0,
                "message": "Task is waiting to be processed"
            }
        elif task.state == "PROGRESS":
            response = {
                "task_id": task_id,
                "status": "PROGRESS",
                "progress": task.info.get("progress", 0),
                "step": task.info.get("step", "processing"),
                "message": task.info.get("message", "Processing...")
            }
        elif task.state == "SUCCESS":
            response = {
                "task_id": task_id,
                "status": "SUCCESS",
                "progress": 100,
                "result": task.result,
                "message": "Task completed successfully"
            }
        elif task.state == "FAILURE":
            response = {
                "task_id": task_id,
                "status": "FAILURE",
                "progress": 0,
                "error": str(task.info),
                "message": "Task failed"
            }
        else:
            response = {
                "task_id": task_id,
                "status": task.state,
                "message": f"Task state: {task.state}"
            }
        
        return response
        
    except Exception as e:
        log_error(e, {"operation": "get_task_status", "task_id": task_id})
        raise HTTPException(status_code=500, detail=f"Failed to get task status: {str(e)}")


@app.post("/unprocess-document-data", response_model=UntrainResponse)
async def untrain(request: UntrainRequest):
    try:
        # Extract filename from S3 URL (URL decoding handled in extract_filename_from_s3_url)
        filename = extract_filename_from_s3_url(request.url)
        file_extension = filename.split('.')[-1].lower()
        file_type = get_file_type(file_extension)
        
        if file_type == 'unknown':
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file type: {file_extension}"
            )
        
        # Get appropriate index for file type
        index = get_index_for_file_type(file_type)
        
        # Delete the entire namespace (all chunks for this file)
        try:
            # Get stats to check if namespace exists
            stats = index.describe_index_stats()
            namespaces = stats.get('namespaces', {})
            
            if filename in namespaces:
                # Delete the namespace
                index.delete(delete_all=True, namespace=filename)
                chunks_removed = namespaces[filename].get('vector_count', 0)
                
                return UntrainResponse(
                    success=True,
                    message=f"Successfully removed {file_type} file: {filename}",
                    s3_url=str(request.url),
                    file_type=file_type,
                    chunks_removed=chunks_removed
                )
            else:
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
            query_response = index.query(
                vector=[0] * 1536,  # Dummy vector
                filter={"s3_url": str(request.url)},
                top_k=10000,
                include_metadata=False,
                namespace=filename
            )
            
            vector_ids = [match.id for match in query_response.matches]
            
            if vector_ids:
                index.delete(ids=vector_ids, namespace=filename)
                
                return UntrainResponse(
                    success=True,
                    message=f"Successfully removed {file_type} file: {filename}",
                    s3_url=str(request.url),
                    file_type=file_type,
                    chunks_removed=len(vector_ids)
                )
            else:
                return UntrainResponse(
                    success=False,
                    message=f"File not found: {filename}",
                    s3_url=str(request.url),
                    file_type=file_type,
                    chunks_removed=0
                )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Untrain failed: {str(e)}")


@app.post("/retrain")
async def retrain():
    return {"message": "Retrain endpoint - to be implemented"}


async def fetch_rag_internal(query: str, top_k: int = 5) -> Dict[str, Any]:
    """Internal function to fetch RAG results (used by both fetch_rag and ask-query-rag)"""
    try:
        # Generate query embedding using async task
        query_embedding = generate_single_embedding.delay(query).get()
        
        all_results = []
        
        # Search across all indexes
        indexes = [
            ("pdf", pdf_index),
            ("video", video_index), 
            ("image", image_index)
        ]
        
        for file_type, index in indexes:
            try:
                # Get all namespaces in this index
                stats = index.describe_index_stats()
                namespaces = stats.get('namespaces', {})
                
                if not namespaces:
                    # If no namespaces, search without namespace
                    search_response = index.query(
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
                        search_response = index.query(
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
                log_error(e, {"operation": "pinecone_search", "index_type": file_type})
                continue
        
        # Sort all results by score (highest first)
        all_results.sort(key=lambda x: x["score"], reverse=True)
        
        # Limit to requested top_k
        final_results = all_results[:top_k]
        
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
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a conversation title generator. Return only the title."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=20,
            temperature=0.3
        )
        
        title = response.choices[0].message.content.strip()
        logger.info("Conversation title generated", title=title, first_message=first_message)
        return title if title else "New Conversation"
        
    except Exception as e:
        log_error(e, {"operation": "generate_conversation_title", "first_message": first_message})
        return "New Conversation"


def save_conversation_locally(chat_id: str, query: str, answer: str, title: str = None):
    """Save conversation to local storage for backup purposes"""
    import datetime
    
    if chat_id not in user_conversations:
        user_conversations[chat_id] = {
            "title": title,
            "created_at": datetime.datetime.now().isoformat(),
            "messages": []
        }
    
    conversation_entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "query": query,
        "answer": answer
    }
    
    user_conversations[chat_id]["messages"].append(conversation_entry)
    log_conversation_event("message_saved", chat_id, message_count=len(user_conversations[chat_id]['messages']))


async def rephrase_followup_query(chat_id: str, query: str) -> str:
    """Rephrase follow-up queries using conversation context"""
    try:
        import openai
        
        # Get conversation history for context
        conversation_data = user_conversations.get(chat_id, {})
        conversation_history = conversation_data.get("messages", []) if isinstance(conversation_data, dict) else conversation_data
        
        if not conversation_history:
            return query  # No history, return original query
        
        # Quick check for obvious non-follow-ups
        query_lower = query.lower().strip()
        if query_lower in ['hello', 'hi', 'hey', 'good morning', 'good evening', 'thanks', 'thank you', 'bye', 'goodbye']:
            return query  # Greetings, no rephrasing needed
        
        # Check if query is clearly standalone (longer than 10 words, contains specific topics)
        if len(query.split()) > 10 and any(word in query_lower for word in ['maharashtra', 'delhi', 'mumbai', 'friend', 'girlfriend', 'boyfriend', 'school', 'college', 'work', 'job']):
            return query  # Standalone questions, no rephrasing needed
        
        # Build conversation context
        context_parts = []
        for msg in conversation_history[-3:]:  # Last 3 exchanges
            context_parts.append(f"User: {msg['query']}")
            context_parts.append(f"Assistant: {msg['answer'][:200]}...")
        context = "\n".join(context_parts)
        
        rephrase_prompt = f"""You are a query rephraser. Analyze if the current query is a follow-up to previous conversation and rephrase it with full context.

CONVERSATION HISTORY:
{context}

CURRENT USER QUERY: {query}

REPHRASING RULES:
- ONLY rephrase if the query is clearly a follow-up to the previous conversation
- Follow-up indicators: "why not", "why", "how", "what about", "tell me more", "explain that", "so you're saying", "is it okay", "right?"
- If the query is standalone, personal, or about a different topic, return it as-is
- If the query is a greeting or personal statement, return it as-is

EXAMPLES:
- "why not" + context about constitution → "why not can i kill someone according to constitution"
- "tell me more" + context about rights → "tell me more about constitutional rights to life"
- "explain that" + context about Article 21 → "explain Article 21 of the constitution"
- "hello" → "hello" (no change needed)
- "i purposed her, is it okay" → "i purposed her, is it okay" (personal question, no change)
- "tell me about maharashtra" → "tell me about maharashtra" (standalone question, no change)
- "what's my name" → "what's my name" (personal question, no change)

IMPORTANT: Only expand if it's clearly a follow-up to the previous knowledge-based conversation. Don't force connections.

Return ONLY the rephrased query, nothing else."""

        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a query rephraser. Return only the rephrased query."},
                {"role": "user", "content": rephrase_prompt}
            ],
            max_tokens=100,
            temperature=0.1
        )
        
        rephrased_query = response.choices[0].message.content.strip()
        logger.info("Query rephrased", original_query=query, rephrased_query=rephrased_query, chat_id=chat_id)
        
        return rephrased_query
        
    except Exception as e:
        log_error(e, {"operation": "rephrase_followup_query", "chat_id": chat_id, "original_query": query})
        return query


async def classify_query_type(chat_id: str, query: str) -> str:
    """Classify if query is general conversation or knowledge question using GPT-4o-mini with full conversation context"""
    try:
        import openai
        
        # Get conversation history for context
        conversation_data = user_conversations.get(chat_id, {})
        conversation_history = conversation_data.get("messages", []) if isinstance(conversation_data, dict) else conversation_data
        
        # Build full conversation context
        context = ""
        if conversation_history:
            recent_messages = conversation_history[-3:]  # Last 3 exchanges for context
            context_parts = []
            for msg in recent_messages:
                context_parts.append(f"User: {msg['query']}")
                context_parts.append(f"Assistant: {msg['answer'][:150]}...")  # Truncate for brevity
            context = "\n".join(context_parts)
        
        classification_prompt = f"""You are a query classifier. Analyze the user's query and determine if it requires knowledge base search or is general conversation.

FULL CONVERSATION HISTORY:
{context if context else "No previous conversation"}

CURRENT USER QUERY: {query}

CLASSIFICATION RULES:
- GENERAL_CONVERSATION: 
  * Greetings (hello, hi, hey, good morning, thanks, how are you, nice to meet you)
  * Personal statements (my name is, I am, I live in)
  * Conversation about our chat (what's my name, what did I say, what did you tell me)
  * Follow-ups to general conversation (tell me more about yourself, what else can you do)
  * Confirmations (yes, no, okay, sure)
  * Goodbyes (bye, see you later)

- KNOWLEDGE_QUESTION:
  * Factual questions (what is, who is, explain, how does, when, where, why)
  * Questions about specific topics that require external knowledge
  * Follow-ups to knowledge questions (why not, tell me more about X, explain that concept)
  * Questions that could be answered from uploaded documents/videos

IMPORTANT CONTEXT ANALYSIS:
- If the query is a follow-up like "why not", "tell me more", "explain that" - check if the previous conversation was about a knowledge topic
- If previous conversation was about constitution, rights, laws, facts → classify as KNOWLEDGE_QUESTION
- If previous conversation was general chat → classify as GENERAL_CONVERSATION
- If no clear context, classify based on the query itself

Examples:
- "why not" after discussing constitution → KNOWLEDGE_QUESTION
- "why not" after saying hello → GENERAL_CONVERSATION  
- "tell me more" after discussing rights → KNOWLEDGE_QUESTION
- "tell me more" after general chat → GENERAL_CONVERSATION

Respond with ONLY one word: GENERAL_CONVERSATION or KNOWLEDGE_QUESTION"""

        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a query classifier. Respond with only 'GENERAL_CONVERSATION' or 'KNOWLEDGE_QUESTION'."},
                {"role": "user", "content": classification_prompt}
            ],
            max_tokens=10,
            temperature=0.1
        )
        
        classification = response.choices[0].message.content.strip()
        logger.info("Query classified", query=query, classification=classification, chat_id=chat_id)
        
        return classification if classification in ["GENERAL_CONVERSATION", "KNOWLEDGE_QUESTION"] else "KNOWLEDGE_QUESTION"
        
    except Exception as e:
        log_error(e, {"operation": "classify_query_type", "chat_id": chat_id, "query": query})
        return "KNOWLEDGE_QUESTION"  # Default to RAG if classification fails


async def generate_general_conversation_answer(chat_id: str, query: str) -> str:
    """Generate general conversation response with conversation context support using OpenAI responses.create()"""
    try:
        import openai
        
        # Check if chat has previous conversation
        previous_response_id = user_openai_conversations.get(chat_id)
        
        # Create conversational input
        input_message = f"""User says: {query}

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

        if previous_response_id:
            # Continue existing conversation
            logger.info("Continuing general conversation", chat_id=chat_id)
            response = openai.responses.create(
                model="gpt-4o",
                input=input_message,
                previous_response_id=previous_response_id,
                max_output_tokens=300,  # Shorter for general conversation
                temperature=0.2  # Lower temperature for more deterministic refusals
            )
        else:
            # Start new conversation
            logger.info("Starting new general conversation", chat_id=chat_id)
            response = openai.responses.create(
                model="gpt-4o",
                input=input_message,
                max_output_tokens=300,
                temperature=0.2
            )
        
        # Store response ID for future conversation continuity
        user_openai_conversations[chat_id] = response.id
        
        ai_answer = response.output_text.strip()
        
        logger.info("Generated general conversation response", chat_id=chat_id)
        return ai_answer
        
    except Exception as e:
        log_error(e, {"operation": "generate_general_conversation_answer", "chat_id": chat_id})
        return "Hello! I'm here to help you with any questions you might have. How can I assist you today?"


@app.post("/ai-service/internal/ask-question", response_model=AskQueryRAGResponse)
async def ask_query_rag(request: AskQueryRAGRequest):
    """Ask a question to the RAG system with conversational context using conversation_id"""
    start_time = time.time()
    request_id = request_id_var.get() or "unknown"
    
    try:
        logger.info("Processing AI query", 
                   conversation_id=request.conversationId, 
                   question=request.question,
                   type=request.type,
                   request_id=request_id)
        
        # Check if this is the first message in the conversation
        is_first_message = request.conversationId not in user_conversations
        conversation_title = None
        
        if is_first_message:
            logger.info("First message detected", conversation_id=request.conversationId)
            conversation_title = await generate_conversation_title(request.question)
        
        # Handle DOCUMENT type messages (placeholder for now)
        if request.type == "DOCUMENT":
            document_count = len(request.documents) if request.documents else 0
            logger.info("Document type message received", conversation_id=request.conversationId, document_count=document_count)
            # TODO: Process documents and add to context
            # For now, just process the message normally
        
        # Step 1: Rephrase query using conversation context
        rephrased_query = await rephrase_followup_query(request.conversationId, request.question)
        logger.info("Using rephrased query", original_query=request.question, rephrased_query=rephrased_query)
        
        # Step 2: Retrieve relevant content using rephrased query
        rag_result = await fetch_rag_internal(rephrased_query, DEFAULT_TOP_K)
        
        if not rag_result["success"]:
            raise HTTPException(status_code=500, detail=f"Content retrieval failed: {rag_result['error']}")
        
        retrieved_content = rag_result["results"]
        total_retrieved = rag_result["total_retrieved"]
        
        logger.info("Content retrieved", conversation_id=request.conversationId, total_retrieved=total_retrieved)
        
        # Step 3: Classify query type using GPT-4o-mini with full conversation context
        query_type = await classify_query_type(request.conversationId, request.question)
        
        if query_type == "GENERAL_CONVERSATION":
            # Handle general conversation without RAG
            logger.info("General conversation detected", conversation_id=request.conversationId)
            ai_answer = await generate_general_conversation_answer(request.conversationId, request.question)
            
            # Save conversation with title for first message
            save_conversation_locally(request.conversationId, request.question, ai_answer, conversation_title)
            
            # Return response without retrieved content for general chat
            return AskQueryRAGResponse(
                success=True,
                message="General conversation response",
                conversationId=request.conversationId,
                answer=ai_answer,
                retrieved_content=[],
                total_retrieved=0,
                conversationTitle=conversation_title
            )
        else:
            # Handle knowledge question with RAG using rephrased query
            logger.info("Knowledge question detected", conversation_id=request.conversationId)
            ai_answer = await generate_conversational_ai_answer(request.conversationId, rephrased_query, retrieved_content)
        
        # Step 4: Save conversation locally for backup
        save_conversation_locally(request.conversationId, request.question, ai_answer, conversation_title)
        
        # Log performance metrics
        duration_ms = (time.time() - start_time) * 1000
        log_performance_metric("ask_query_duration", duration_ms, "ms", 
                              conversation_id=request.conversationId, 
                              query_type=query_type,
                              total_retrieved=total_retrieved)
        
        logger.info("AI query completed successfully", 
                   conversation_id=request.conversationId,
                   query_type=query_type,
                   total_retrieved=total_retrieved,
                   duration_ms=duration_ms)
        
        # Step 5: Return structured response
        return AskQueryRAGResponse(
            success=True,
            message="AI answer generated successfully",
            conversationId=request.conversationId,
            answer=ai_answer,
            retrieved_content=retrieved_content,
            total_retrieved=total_retrieved,
            conversationTitle=conversation_title
        )
        
    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        
        log_error(e, {
            "operation": "ask_query_rag",
            "conversation_id": request.conversationId,
            "question": request.question,
            "duration_ms": duration_ms,
            "request_id": request_id
        })
        
        # Handle cases where retrieval works but AI fails
        try:
            rag_result = await fetch_rag_internal(request.question, DEFAULT_TOP_K)
            retrieved_content = rag_result.get("results", [])
            
            # Save failed attempt locally
            error_answer = "Sorry, I couldn't generate an answer due to a technical issue, but I found some relevant content below."
            save_conversation_locally(request.conversationId, request.question, error_answer, conversation_title)
            
            logger.warning("AI processing failed, returning retrieved content", 
                          conversation_id=request.conversationId,
                          retrieved_count=len(retrieved_content),
                          error=str(e))
            
            return AskQueryRAGResponse(
                success=False,
                message=f"AI processing failed, but retrieved content available: {str(e)}",
                conversationId=request.conversationId,
                answer=error_answer,
                retrieved_content=retrieved_content,
                total_retrieved=len(retrieved_content),
                conversationTitle=conversation_title
            )
        except Exception as fallback_error:
            log_error(fallback_error, {
                "operation": "ask_query_rag_fallback",
                "conversation_id": request.conversationId,
                "original_error": str(e)
            })
            raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")


async def generate_conversational_ai_answer(chat_id: str, query: str, retrieved_content: List[Dict[str, Any]]) -> str:
    """Generate conversational AI answer using OpenAI responses.create() with conversation tracking"""
    try:
        import openai
        
        # Handle case where no content was retrieved
        if not retrieved_content:
            return """I'm sorry, but I don't have information about that topic in my current knowledge base. Please try asking about something else or upload relevant documents first."""

        # Prepare context from retrieved content
        context_parts = []
        for i, content in enumerate(retrieved_content, 1):
            context_parts.append(content.get('content', ''))
        
        context = "\n\n---\n\n".join(context_parts)
        
        # Create input with improved relevance detection
        input_message = f"""You are an expert assistant that answers questions using the provided context information. Follow these guidelines:

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

        # Check if chat has previous conversation
        previous_response_id = user_openai_conversations.get(chat_id)
        
        if previous_response_id:
            # Continue existing conversation
            logger.info("Continuing conversation", chat_id=chat_id, previous_response_id=previous_response_id)
            response = openai.responses.create(
                model="gpt-4o",
                input=input_message,
                previous_response_id=previous_response_id,  # Continue conversation
                max_output_tokens=1200,
                temperature=0.3  # Balanced temperature for better relevance detection
            )
        else:
            # Start new conversation
            logger.info("Starting new conversation", chat_id=chat_id)
            response = openai.responses.create(
                model="gpt-4o",
                input=input_message,
                max_output_tokens=1200,
                temperature=0.3  # Balanced temperature for better relevance detection
            )
        
        # Store response ID for future conversation continuity
        user_openai_conversations[chat_id] = response.id
        
        ai_answer = response.output_text.strip()
        
        logger.info("Generated conversational response", chat_id=chat_id, response_id=response.id)
        return ai_answer
        
    except Exception as e:
        log_error(e, {"operation": "generate_conversational_ai_answer", "chat_id": chat_id})
        return "I apologize, but I encountered an error while generating the answer. Please try your question again, or contact support if this error persists."


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
        log_error(e, {"endpoint": "health_check"})
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
        log_error(e, {"endpoint": "readiness_check"})
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
    uvicorn.run("src.main:app", host="0.0.0.0", port=8000, reload=True)
