from pydantic import BaseModel, HttpUrl, Field
from typing import Optional, List, Dict, Any


class TrainRequest(BaseModel):
    s3_url: HttpUrl
    file_name: str = Field(..., description="Name to save the file as in Pinecone")


class UntrainRequest(BaseModel):
    file_name: str = Field(..., description="Name of the file to remove from Pinecone")


class RAGQueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5


class TrainResponse(BaseModel):
    success: bool
    message: str
    file_name: Optional[str] = None
    chunks_created: Optional[int] = None


class UntrainResponse(BaseModel):
    success: bool
    message: str
    file_name: Optional[str] = None
    chunks_removed: Optional[int] = None


class RAGQueryResponse(BaseModel):
    success: bool
    message: str
    results: Optional[List[Dict[str, Any]]] = None
