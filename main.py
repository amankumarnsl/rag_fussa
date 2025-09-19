import os
import io
import boto3
import openai
from urllib.parse import urlparse
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


def extract_filename_from_s3_url(s3_url):
    """Extract filename from S3 URL."""
    parsed_url = urlparse(str(s3_url))
    object_key = parsed_url.path.lstrip('/')
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


@app.post("/train", response_model=TrainResponse)
async def train(request: TrainRequest):
    try:
        # Parse S3 URL and extract filename
        parsed_url = urlparse(str(request.s3_url))
        bucket_name = parsed_url.netloc.split('.')[0]
        object_key = parsed_url.path.lstrip('/')
        filename = extract_filename_from_s3_url(request.s3_url)
        
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
                    "s3_url": str(request.s3_url),
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
            s3_url=str(request.s3_url),
            file_type=file_type,
            chunks_created=len(processed_chunks)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")


@app.post("/untrain", response_model=UntrainResponse)
async def untrain(request: UntrainRequest):
    try:
        # Extract filename from S3 URL
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


@app.post("/ask-query-rag", response_model=AskQueryRAGResponse)
async def ask_query_rag(request: AskQueryRAGRequest):
    """Ask a question to the RAG system and get an AI-generated answer based on retrieved content"""
    try:
        print(f"🤖 Processing AI query: {request.query}")
        
        # Step 1: Retrieve relevant content using fetch_rag_internal
        rag_result = await fetch_rag_internal(request.query, request.top_k)
        
        if not rag_result["success"]:
            raise HTTPException(status_code=500, detail=f"Content retrieval failed: {rag_result['error']}")
        
        retrieved_content = rag_result["results"]
        total_retrieved = rag_result["total_retrieved"]
        
        print(f"📚 Retrieved {total_retrieved} relevant chunks")
        
        # Step 2: Generate AI answer using OpenAI
        ai_answer = await generate_ai_answer(request.query, retrieved_content)
        
        # Step 3: Return structured response
        return AskQueryRAGResponse(
            success=True,
            message="AI answer generated successfully",
            query=request.query,
            answer=ai_answer,
            retrieved_content=retrieved_content,
            total_retrieved=total_retrieved
        )
        
    except Exception as e:
        # Handle cases where retrieval works but AI fails
        try:
            rag_result = await fetch_rag_internal(request.query, request.top_k)
            retrieved_content = rag_result.get("results", [])
            
            return AskQueryRAGResponse(
                success=False,
                message=f"AI processing failed, but retrieved content available: {str(e)}",
                query=request.query,
                answer="Sorry, I couldn't generate an answer due to a technical issue, but I found some relevant content below.",
                retrieved_content=retrieved_content,
                total_retrieved=len(retrieved_content)
            )
        except:
            raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")


async def generate_ai_answer(query: str, retrieved_content: List[Dict[str, Any]]) -> str:
    """Generate AI answer using OpenAI based on retrieved content"""
    try:
        import openai
        
        # Handle case where no content was retrieved
        if not retrieved_content:
            return """I apologize, but I couldn't find any relevant content in the knowledge base to answer your question. 

This could mean:
- The information you're looking for hasn't been uploaded to the system yet
- Your question might need to be phrased differently
- The content might be in a different format or context

Please try rephrasing your question or check if the relevant documents have been properly uploaded and processed."""

        # Prepare context from retrieved content
        context_parts = []
        for i, content in enumerate(retrieved_content, 1):
            source_info = f"Source {i} (from {content.get('filename', 'unknown')} - {content.get('file_type', 'unknown')})"
            context_parts.append(f"{source_info}:\n{content.get('content', '')}")
        
        context = "\n\n---\n\n".join(context_parts)
        
        # Create natural, user-friendly prompt for OpenAI with strict content relevance
        prompt = f"""You are a knowledgeable expert providing helpful information. Answer the user's question in a natural, conversational way.

CRITICAL RULE - CONTENT RELEVANCE:
- FIRST, carefully analyze if the provided context information actually contains relevant information to answer the user's question
- If the context does NOT contain relevant information to answer the question, you MUST respond with: "I'm sorry, but I don't have information about that topic in my current knowledge base. Please try asking about something else or upload relevant documents first."
- NEVER answer questions using general knowledge if the context doesn't contain relevant information
- ONLY answer if the context actually addresses the user's question

WRITING STYLE REQUIREMENTS (only if context is relevant):
- Write in a natural, friendly tone as if you're an expert explaining to a friend
- Use clear structure with bullet points, numbered lists, or sections when helpful
- Break up long paragraphs into digestible chunks
- Use bold formatting (**text**) for key points
- Don't mention "retrieved content", "sources", or "knowledge base"
- Write as if this is your own knowledge and expertise
- Make it engaging and easy to read

CONTENT GUIDELINES:
- Only use information from the provided context below
- If the context partially answers the question, provide what you can and mention what's missing
- Be accurate and don't add information not in the context
- If information seems incomplete or unclear, mention this naturally

USER QUESTION: {query}

CONTEXT INFORMATION:
{context}

Please provide a comprehensive, well-structured answer (or politely decline if context is not relevant):"""

        # Call OpenAI API
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful expert who ONLY answers questions if the provided context contains relevant information. If the context is not relevant to the question, you must politely decline. Never use general knowledge. Write naturally and conversationally when you do answer."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1200,
            temperature=0.3  # Lower temperature for strict adherence to context
        )
        
        ai_answer = response.choices[0].message.content.strip()
        
        return ai_answer  # Return clean answer without metadata
        
    except Exception as e:
        return f"""I apologize, but I encountered an error while generating the answer: {str(e)}

However, I was able to retrieve some potentially relevant content which you can review below in the retrieved_content section.

Please try your question again, or contact support if this error persists."""


@app.get("/")
async def root():
    return {"message": "RAG FUSSA API", "docs": "/docs"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
