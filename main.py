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
    """Get embeddings from OpenAI."""
    response = openai.embeddings.create(
        input=texts,
        model="text-embedding-ada-002"
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
        
        # Process file based on type
        if file_type == 'pdf':
            processed_chunks = process_pdf(file_content, filename)
        elif file_type == 'video':
            processed_chunks = process_video(file_content, filename)
        elif file_type == 'image':
            processed_chunks = process_image(file_content, filename)
        
        # Extract content for embeddings
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
