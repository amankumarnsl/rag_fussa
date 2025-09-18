from pydantic import BaseModel, HttpUrl, Field
from typing import Optional, List, Dict, Any


class TrainRequest(BaseModel):
    s3_url: HttpUrl = Field(..., description="AWS S3 URL of the file to process")


class UntrainRequest(BaseModel):
    s3_url: HttpUrl = Field(..., description="AWS S3 URL of the file to remove from Pinecone")


class RAGQueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5


class TrainResponse(BaseModel):
    success: bool
    message: str
    s3_url: Optional[str] = None
    file_type: Optional[str] = None
    chunks_created: Optional[int] = None


class UntrainResponse(BaseModel):
    success: bool
    message: str
    s3_url: Optional[str] = None
    file_type: Optional[str] = None
    chunks_removed: Optional[int] = None


class RAGQueryResponse(BaseModel):
    success: bool
    message: str
    results: Optional[List[Dict[str, Any]]] = None
