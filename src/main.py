import os
import io
import boto3
import openai
import requests
from urllib.parse import urlparse, unquote
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import PyPDF2

from .config.config import *
from .config.schemas import *
from .processors.pdf_processor import process_pdf, get_pdf_info
from .processors.video_processor import process_video, is_video_file
from .processors.image_processor import process_image, is_image_file
from .utils.text_pipeline import process_text_file_to_chunks

app = FastAPI(title="RAG FUSSA API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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
    """Get embeddings from OpenAI using text-embedding-3-small."""
    response = openai.embeddings.create(
        input=texts,
        model="text-embedding-3-small"
    )
    return [item.embedding for item in response.data]


async def update_document_status(uuid: str, status: str, failure_reason: str = None):
    """Update document status in backend"""
    try:
        # Construct backend URL from environment variables
        backend_base_url = os.getenv("BACKEND_BASE_URL", "192.168.68.72")
        backend_port = os.getenv("BACKEND_PORT", "4504")
        backend_endpoint = os.getenv("BACKEND_ENDPOINT_PATH", "/upload/internal/update-document-entry")
        
        backend_url = f"http://{backend_base_url}:{backend_port}{backend_endpoint}"
        print(f"🔗 Backend URL: {backend_url}")
        
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
        
        response = requests.put(backend_url, json=payload, headers=headers, timeout=10)
        
        if response.status_code == 200:
            print(f"✅ Backend updated: {uuid} -> {status}")
        else:
            print(f"⚠️ Backend update failed: {response.status_code} - {response.text}")
            
    except Exception as e:
        print(f"❌ Backend update error: {str(e)}")
        # Don't raise exception - backend update failure shouldn't stop document processing


@app.post("/ai-service/internal/process-document-data", response_model=TrainResponse)
async def train(request: TrainRequest):
    try:
        # Step 1: Update backend with PROCESSING status
        await update_document_status(request.uuid, "PROCESSING")
        
        # Parse S3 URL and extract filename
        parsed_url = urlparse(str(request.url))
        bucket_name = parsed_url.netloc.split('.')[0]
        object_key = parsed_url.path.lstrip('/')
        # URL decode the object key to handle special characters (Arabic, spaces, etc.)
        object_key = unquote(object_key)
        filename = extract_filename_from_s3_url(request.url)
        
        # Download file from S3
        response = s3_client.get_object(Bucket=bucket_name, Key=object_key)
        file_content = response['Body'].read()
        
        # Determine file type and process
        file_extension = object_key.split('.')[-1].lower()
        file_type = get_file_type(file_extension)
        
        if file_type == 'unknown':
            error_msg = f"Unsupported file type: {file_extension}"
            await update_document_status(request.uuid, "FAILED", error_msg)
            raise HTTPException(status_code=400, detail=error_msg)
        
        # Get appropriate index for file type
        index = get_index_for_file_type(file_type)
        
        # Step 2: Extract text and save to .txt file
        if file_type == 'pdf':
            text_filepath = process_pdf(file_content, filename)
        elif file_type == 'video':
            text_filepath = process_video(file_content, filename)
        elif file_type == 'image':
            text_filepath = process_image(file_content, filename)  # Will return text path too
        
        # Step 3: Common text processing pipeline (same for all file types)
        processed_chunks = process_text_file_to_chunks(
            text_filepath=text_filepath,
            filename=filename,
            file_type=file_type,
            chunk_strategy="semantic"
        )
        
        # Step 4: Generate embeddings for final chunks
        chunk_texts = [chunk["content"] for chunk in processed_chunks]
        embeddings = get_embeddings(chunk_texts)
        
        # Prepare vectors for Pinecone with namespace
        vectors = []
        for i, (chunk, embedding) in enumerate(zip(processed_chunks, embeddings)):
            vector_id = f"{filename}_{i}"
            vectors.append({
                "id": vector_id,
                "values": embedding,
                "metadata": {
                    "name": request.name,
                    "uuid": request.uuid,
                    "url": str(request.url),
                    "type": request.type,
                    "trainingStatus": request.trainingStatus,
                    "filename": filename,
                    "chunk_index": i,
                    "content": chunk["content"],
                    "file_type": file_type,
                    **chunk["metadata"]  # Include additional metadata from processors
                }
            })
        
        # Step 5: Upload to appropriate Pinecone index with filename as namespace
        index.upsert(vectors=vectors, namespace=filename)
        
        # Step 6: Update backend with COMPLETED status
        await update_document_status(request.uuid, "COMPLETED")
        
        return TrainResponse(
            success=True,
            message=f"Successfully trained {file_type} file: {filename}",
            name=request.name,
            uuid=request.uuid,
            url=str(request.url),
            type=request.type,
            trainingStatus="completed",
            file_type=file_type,
            chunks_created=len(processed_chunks)
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions (already handled above)
        raise
    except Exception as e:
        # Update backend with FAILED status for any other errors
        error_msg = f"Training failed: {str(e)}"
        await update_document_status(request.uuid, "FAILED", error_msg)
        raise HTTPException(status_code=500, detail=error_msg)


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
        # Generate query embedding
        query_embedding = get_embeddings([query])[0]
        
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
                print(f"Error searching {file_type} index: {str(e)}")
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
        print(f"📝 Generated conversation title: '{title}'")
        return title if title else "New Conversation"
        
    except Exception as e:
        print(f"❌ Title generation failed: {str(e)}")
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
    print(f"💾 Saved conversation for chat {chat_id} (total: {len(user_conversations[chat_id]['messages'])} messages)")


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
        print(f"🔄 Query rephrased: '{query}' → '{rephrased_query}'")
        print(f"📝 Rephrased query: {rephrased_query}")
        
        return rephrased_query
        
    except Exception as e:
        print(f"❌ Query rephrasing failed: {str(e)}, using original query")
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
        print(f"🧠 Query classification for '{query}': {classification}")
        
        return classification if classification in ["GENERAL_CONVERSATION", "KNOWLEDGE_QUESTION"] else "KNOWLEDGE_QUESTION"
        
    except Exception as e:
        print(f"❌ Query classification failed: {str(e)}, defaulting to KNOWLEDGE_QUESTION")
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
            print(f"💬 Continuing general conversation for chat {chat_id}")
            response = openai.responses.create(
                model="gpt-4o",
                input=input_message,
                previous_response_id=previous_response_id,
                max_output_tokens=300,  # Shorter for general conversation
                temperature=0.2  # Lower temperature for more deterministic refusals
            )
        else:
            # Start new conversation
            print(f"🆕 Starting new general conversation for chat {chat_id}")
            response = openai.responses.create(
                model="gpt-4o",
                input=input_message,
                max_output_tokens=300,
                temperature=0.2
            )
        
        # Store response ID for future conversation continuity
        user_openai_conversations[chat_id] = response.id
        
        ai_answer = response.output_text.strip()
        
        print(f"💬 Generated general conversation response for chat {chat_id}")
        return ai_answer
        
    except Exception as e:
        print(f"❌ General conversation failed: {str(e)}")
        return "Hello! I'm here to help you with any questions you might have. How can I assist you today?"


@app.post("/ai-service/internal/ask-question", response_model=AskQueryRAGResponse)
async def ask_query_rag(request: AskQueryRAGRequest):
    """Ask a question to the RAG system with conversational context using conversation_id"""
    try:
        print(f"🤖 Processing AI query for conversation {request.conversationId}: {request.question}")
        
        # Check if this is the first message in the conversation
        is_first_message = request.conversationId not in user_conversations
        conversation_title = None
        
        if is_first_message:
            print(f"🆕 First message detected for conversation {request.conversationId}")
            conversation_title = await generate_conversation_title(request.question)
        
        # Handle DOCUMENT type messages (placeholder for now)
        if request.type == "DOCUMENT":
            print(f"📄 Document type message received with {len(request.documents) if request.documents else 0} documents")
            # TODO: Process documents and add to context
            # For now, just process the message normally
        
        # Step 1: Rephrase query using conversation context
        rephrased_query = await rephrase_followup_query(request.conversationId, request.question)
        print(f"🔄 Using rephrased query: '{rephrased_query}'")
        
        # Step 2: Retrieve relevant content using rephrased query
        rag_result = await fetch_rag_internal(rephrased_query, DEFAULT_TOP_K)
        
        if not rag_result["success"]:
            raise HTTPException(status_code=500, detail=f"Content retrieval failed: {rag_result['error']}")
        
        retrieved_content = rag_result["results"]
        total_retrieved = rag_result["total_retrieved"]
        
        print(f"📚 Retrieved {total_retrieved} relevant chunks for conversation {request.conversationId}")
        
        # Step 3: Classify query type using GPT-4o-mini with full conversation context
        query_type = await classify_query_type(request.conversationId, request.question)
        
        if query_type == "GENERAL_CONVERSATION":
            # Handle general conversation without RAG
            print(f"💬 General conversation detected for conversation {request.conversationId}")
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
            print(f"🔍 Knowledge question detected for conversation {request.conversationId}")
            ai_answer = await generate_conversational_ai_answer(request.conversationId, rephrased_query, retrieved_content)
        
        # Step 4: Save conversation locally for backup
        save_conversation_locally(request.conversationId, request.question, ai_answer, conversation_title)
        
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
        # Handle cases where retrieval works but AI fails
        try:
            rag_result = await fetch_rag_internal(request.question, DEFAULT_TOP_K)
            retrieved_content = rag_result.get("results", [])
            
            # Save failed attempt locally
            error_answer = "Sorry, I couldn't generate an answer due to a technical issue, but I found some relevant content below."
            save_conversation_locally(request.conversationId, request.question, error_answer, conversation_title)
            
            return AskQueryRAGResponse(
                success=False,
                message=f"AI processing failed, but retrieved content available: {str(e)}",
                conversationId=request.conversationId,
                answer=error_answer,
                retrieved_content=retrieved_content,
                total_retrieved=len(retrieved_content),
                conversationTitle=conversation_title
            )
        except:
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
            print(f"💬 Continuing conversation for chat {chat_id} (previous: {previous_response_id})")
            response = openai.responses.create(
                model="gpt-4o",
                input=input_message,
                previous_response_id=previous_response_id,  # Continue conversation
                max_output_tokens=1200,
                temperature=0.3  # Balanced temperature for better relevance detection
            )
        else:
            # Start new conversation
            print(f"🆕 Starting new conversation for chat {chat_id}")
            response = openai.responses.create(
                model="gpt-4o",
                input=input_message,
                max_output_tokens=1200,
                temperature=0.3  # Balanced temperature for better relevance detection
            )
        
        # Store response ID for future conversation continuity
        user_openai_conversations[chat_id] = response.id
        
        ai_answer = response.output_text.strip()
        
        print(f"💬 Generated conversational response for chat {chat_id} (response_id: {response.id})")
        return ai_answer
        
    except Exception as e:
        print(f"❌ Conversational AI failed: {str(e)}")
        return "I apologize, but I encountered an error while generating the answer. Please try your question again, or contact support if this error persists."


@app.get("/")
async def root():
    return {"message": "RAG FUSSA API", "docs": "/docs"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.main:app", host="0.0.0.0", port=8000, reload=True)
