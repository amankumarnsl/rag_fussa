import os
import io
import boto3
import pinecone
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

pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
index = pinecone.Index(PINECONE_INDEX_NAME)

openai.api_key = OPENAI_API_KEY


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




def get_embeddings(texts):
    """Get embeddings from OpenAI."""
    response = openai.embeddings.create(
        input=texts,
        model="text-embedding-ada-002"
    )
    return [item.embedding for item in response.data]


@app.post("/train", response_model=TrainResponse)
async def train(request: TrainRequest):
    try:
        # Parse S3 URL
        parsed_url = urlparse(str(request.s3_url))
        bucket_name = parsed_url.netloc.split('.')[0]
        object_key = parsed_url.path.lstrip('/')
        
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
        
        # Process file based on type
        if file_type == 'pdf':
            processed_chunks = process_pdf(file_content, request.file_name)
        elif file_type == 'video':
            processed_chunks = process_video(file_content, request.file_name)
        elif file_type == 'image':
            processed_chunks = process_image(file_content, request.file_name)
        
        # Extract content for embeddings
        chunk_texts = [chunk["content"] for chunk in processed_chunks]
        embeddings = get_embeddings(chunk_texts)
        
        # Prepare vectors for Pinecone
        vectors = []
        for i, (chunk, embedding) in enumerate(zip(processed_chunks, embeddings)):
            vector_id = f"{request.file_name}_{i}"
            vectors.append({
                "id": vector_id,
                "values": embedding,
                "metadata": {
                    "file_name": request.file_name,
                    "chunk_index": i,
                    "content": chunk["content"],
                    "file_type": file_type,
                    **chunk["metadata"]  # Include additional metadata from processors
                }
            })
        
        # Upload to Pinecone
        index.upsert(vectors=vectors)
        
        return TrainResponse(
            success=True,
            message=f"Successfully trained {file_type} file: {request.file_name}",
            file_name=request.file_name,
            chunks_created=len(processed_chunks)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")


@app.post("/untrain", response_model=UntrainResponse)
async def untrain(request: UntrainRequest):
    try:
        # Query to find all vectors for this file
        query_response = index.query(
            vector=[0] * 1536,  # Dummy vector
            filter={"file_name": request.file_name},
            top_k=10000,
            include_metadata=False
        )
        
        # Get vector IDs to delete
        vector_ids = [match.id for match in query_response.matches]
        
        if vector_ids:
            # Delete vectors from Pinecone
            index.delete(ids=vector_ids)
            
            return UntrainResponse(
                success=True,
                message=f"Successfully removed file: {request.file_name}",
                file_name=request.file_name,
                chunks_removed=len(vector_ids)
            )
        else:
            return UntrainResponse(
                success=False,
                message=f"File not found: {request.file_name}",
                file_name=request.file_name,
                chunks_removed=0
            )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Untrain failed: {str(e)}")


@app.post("/retrain")
async def retrain():
    return {"message": "Retrain endpoint - to be implemented"}


@app.post("/fetch_rag", response_model=RAGQueryResponse)
async def fetch_rag(request: RAGQueryRequest):
    return RAGQueryResponse(
        success=True,
        message="Fetch RAG endpoint - to be implemented",
        results=[]
    )


@app.get("/")
async def root():
    return {"message": "RAG FUSSA API", "docs": "/docs"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
