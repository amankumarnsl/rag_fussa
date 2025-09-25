import os
import io
import boto3
import openai
from urllib.parse import urlparse, unquote
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import PyPDF2

from config import *
from schemas import *
from pdf_processor import process_pdf, get_pdf_info
from video_processor import process_video, is_video_file
from image_processor import process_image, is_image_file
from text_pipeline import process_text_file_to_chunks

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


@app.post("/ai-service/internal/process-document-data", response_model=TrainResponse)
async def train(request: TrainRequest):
    try:
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
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file type: {file_extension}"
            )
        
        # Get appropriate index for file type
        index = get_index_for_file_type(file_type)
        
        # Step 1: Extract text and save to .txt file
        if file_type == 'pdf':
            text_filepath = process_pdf(file_content, filename)
        elif file_type == 'video':
            text_filepath = process_video(file_content, filename)
        elif file_type == 'image':
            text_filepath = process_image(file_content, filename)  # Will return text path too
        
        # Step 2: Common text processing pipeline (same for all file types)
        processed_chunks = process_text_file_to_chunks(
            text_filepath=text_filepath,
            filename=filename,
            file_type=file_type,
            chunk_strategy="semantic"
        )
        
        # Step 3: Generate embeddings for final chunks
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
        
        # Upload to appropriate Pinecone index with filename as namespace
        index.upsert(vectors=vectors, namespace=filename)
        
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
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")


@app.post("/untrain", response_model=UntrainResponse)
async def untrain(request: UntrainRequest):
    try:
        # Extract filename from S3 URL (URL decoding handled in extract_filename_from_s3_url)
        filename = extract_filename_from_s3_url(request.s3_url)
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
                    s3_url=str(request.s3_url),
                    file_type=file_type,
                    chunks_removed=chunks_removed
                )
            else:
                return UntrainResponse(
                    success=False,
                    message=f"File not found: {filename}",
                    s3_url=str(request.s3_url),
                    file_type=file_type,
                    chunks_removed=0
                )
        except Exception as e:
            # If namespace deletion fails, try alternative approach
            # Query and delete by metadata
            query_response = index.query(
                vector=[0] * 1536,  # Dummy vector
                filter={"s3_url": str(request.s3_url)},
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
                    s3_url=str(request.s3_url),
                    file_type=file_type,
                    chunks_removed=len(vector_ids)
                )
            else:
                return UntrainResponse(
                    success=False,
                    message=f"File not found: {filename}",
                    s3_url=str(request.s3_url),
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


def save_conversation_locally(chat_id: str, query: str, answer: str):
    """Save conversation to local storage for backup purposes"""
    import datetime
    
    if chat_id not in user_conversations:
        user_conversations[chat_id] = []
    
    conversation_entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "query": query,
        "answer": answer
    }
    
    user_conversations[chat_id].append(conversation_entry)
    print(f"💾 Saved conversation for chat {chat_id} (total: {len(user_conversations[chat_id])} messages)")


async def classify_query_type(chat_id: str, query: str) -> str:
    """Classify if query is general conversation or knowledge question using GPT-4o-mini"""
    try:
        import openai
        
        # Get conversation history for context
        conversation_history = user_conversations.get(chat_id, [])
        
        # Build conversation context
        context = ""
        if conversation_history:
            recent_messages = conversation_history[-3:]  # Last 3 exchanges for context
            context_parts = []
            for msg in recent_messages:
                context_parts.append(f"User: {msg['query']}")
                context_parts.append(f"Assistant: {msg['answer'][:100]}...")  # Truncate for brevity
            context = "\n".join(context_parts)
        
        classification_prompt = f"""You are a query classifier. Analyze the user's query and determine if it requires knowledge base search or is general conversation.

CONVERSATION HISTORY:
{context if context else "No previous conversation"}

CURRENT USER QUERY: {query}

CLASSIFICATION RULES:
- GENERAL_CONVERSATION: Greetings (hello, hi), pleasantries (how are you, thank you), personal statements (my name is), follow-ups that refer to previous conversation context (tell me more, explain that, what about it), confirmations (yes, no, okay), goodbyes
- KNOWLEDGE_QUESTION: Factual questions (what is, who is, explain, how does), requests for specific information, questions about topics that would require external knowledge

IMPORTANT: If the query is a follow-up like "tell me more" or "explain that", check if it refers to something from the conversation history. If yes, classify as GENERAL_CONVERSATION.

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
    """Generate general conversation response without RAG using OpenAI responses.create()"""
    try:
        import openai
        
        # Check if chat has previous conversation
        previous_response_id = user_openai_conversations.get(chat_id)
        
        # Create conversational input
        input_message = f"""User says: {query}

You are handling ONLY general conversation. Decide between two behaviors and output ONLY the final message text (no labels):

1) GREETING_REPLY
- Trigger if the message is a greeting/pleasantry/small talk (hello, hi, hey, good morning, thanks, how are you, nice to meet you, what's up, good night) or a very short friendly check-in.
- Respond warmly and briefly, 1-2 sentences max.

2) POLITE_REFUSAL
- Trigger for ANY other general/personal/open-ended question that is not tied to the user's uploaded knowledge (e.g., relationship advice, life coaching, opinions, generic facts, news, health/financial/legal advice, etc.).
- Do NOT answer the question. Politely say you can't answer general questions and that you only answer using the user's uploaded knowledge base. Suggest adding relevant documents if they want help.
- Keep to 1-2 sentences, kind and clear.

Examples:
- User: "hi can you help me to manage my break up with my gf?" → "I'm sorry, but I can't provide general personal advice. I only answer using information from your uploaded knowledge base. You can add relevant documents and ask again."
- User: "hello" → "Hi there! Great to see you. How can I help today?"
- User: "who is prime minister of india" → "Sorry, I can’t answer general questions. I only respond using your uploaded knowledge base."
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
        
        # Handle DOCUMENT type messages (placeholder for now)
        if request.type == "DOCUMENT":
            print(f"📄 Document type message received with {len(request.documents) if request.documents else 0} documents")
            # TODO: Process documents and add to context
            # For now, just process the message normally
        
        # Step 1: Retrieve relevant content using fetch_rag_internal
        rag_result = await fetch_rag_internal(request.question, DEFAULT_TOP_K)
        
        if not rag_result["success"]:
            raise HTTPException(status_code=500, detail=f"Content retrieval failed: {rag_result['error']}")
        
        retrieved_content = rag_result["results"]
        total_retrieved = rag_result["total_retrieved"]
        
        print(f"📚 Retrieved {total_retrieved} relevant chunks for conversation {request.conversationId}")
        
        # Step 2: Classify query type using GPT-4o-mini
        query_type = await classify_query_type(request.conversationId, request.question)
        
        if query_type == "GENERAL_CONVERSATION":
            # Handle general conversation without RAG
            print(f"💬 General conversation detected for conversation {request.conversationId}")
            ai_answer = await generate_general_conversation_answer(request.conversationId, request.question)
            
            # Return response without retrieved content for general chat
            return AskQueryRAGResponse(
                success=True,
                message="General conversation response",
                conversationId=request.conversationId,
                answer=ai_answer,
                retrieved_content=[],
                total_retrieved=0
            )
        else:
            # Handle knowledge question with RAG
            print(f"🔍 Knowledge question detected for conversation {request.conversationId}")
            ai_answer = await generate_conversational_ai_answer(request.conversationId, request.question, retrieved_content)
        
        # Step 3: Save conversation locally for backup
        save_conversation_locally(request.conversationId, request.question, ai_answer)
        
        # Step 4: Return structured response
        return AskQueryRAGResponse(
            success=True,
            message="AI answer generated successfully",
            conversationId=request.conversationId,
            answer=ai_answer,
            retrieved_content=retrieved_content,
            total_retrieved=total_retrieved
        )
        
    except Exception as e:
        # Handle cases where retrieval works but AI fails
        try:
            rag_result = await fetch_rag_internal(request.question, DEFAULT_TOP_K)
            retrieved_content = rag_result.get("results", [])
            
            # Save failed attempt locally
            error_answer = "Sorry, I couldn't generate an answer due to a technical issue, but I found some relevant content below."
            save_conversation_locally(request.conversationId, request.question, error_answer)
            
            return AskQueryRAGResponse(
                success=False,
                message=f"AI processing failed, but retrieved content available: {str(e)}",
                conversationId=request.conversationId,
                answer=error_answer,
                retrieved_content=retrieved_content,
                total_retrieved=len(retrieved_content)
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
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
