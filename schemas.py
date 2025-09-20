from pydantic import BaseModel, HttpUrl, Field
from typing import Optional, List, Dict, Any


class TrainRequest(BaseModel):
    name: str = Field(..., description="Name of the document")
    uuid: str = Field(..., description="Unique identifier for the document")
    url: HttpUrl = Field(..., description="AWS S3 URL of the file to process")
    type: str = Field(..., description="Type of the document")
    trainingStatus: str = Field(..., description="Training status of the document")


class UntrainRequest(BaseModel):
    s3_url: HttpUrl = Field(..., description="AWS S3 URL of the file to remove from Pinecone")


class RAGQueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5


class AskQueryRAGRequest(BaseModel):
    query: str = Field(..., description="Question to ask the RAG system")
    chat_id: str = Field(..., description="Unique chat identifier for conversation tracking")


class TrainResponse(BaseModel):
    success: bool
    message: str
    name: Optional[str] = None
    uuid: Optional[str] = None
    url: Optional[str] = None
    type: Optional[str] = None
    trainingStatus: Optional[str] = None
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


class AskQueryRAGResponse(BaseModel):
    success: bool
    message: str
    query: str
    chat_id: str
    answer: str
    retrieved_content: Optional[List[Dict[str, Any]]] = None
    total_retrieved: Optional[int] = None
