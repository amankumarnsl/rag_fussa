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


class DocumentInfo(BaseModel):
    url: HttpUrl = Field(..., description="URL of the document")
    type: str = Field(..., description="Type of the document (PDF, IMAGE, AUDIO, etc.)")


class AskQueryRAGRequest(BaseModel):
    conversationId: str = Field(..., description="Unique conversation identifier")
    type: str = Field(..., description="Message type: TEXT or DOCUMENT")
    question: str = Field(..., description="Question to ask the RAG system")
    documents: Optional[List[DocumentInfo]] = Field(None, description="Additional documents (for DOCUMENT type)")


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
    conversationId: str
    answer: str
    retrieved_content: Optional[List[Dict[str, Any]]] = None
    total_retrieved: Optional[int] = None
