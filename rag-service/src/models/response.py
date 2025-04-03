# rag-service/src/models/response.py
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from datetime import datetime

class DocumentSource(BaseModel):
    """
    Model for document source information in the response.
    """
    source: str = Field(..., description="Document name")
    department: str = Field(..., description="Department (e.g., commercial, technical)")
    page: Optional[int] = Field(None, description="Page number")
    content: str = Field(..., description="Relevant excerpt")
    relevance: float = Field(..., description="Relevance score (0-1)")

class ResponseMetrics(BaseModel):
    """
    Model for response metrics information.
    """
    retrieval_time: float = Field(..., description="Time taken for retrieval (seconds)")
    generation_time: float = Field(..., description="Time taken for generation (seconds)")
    context_relevance: float = Field(..., description="Overall context relevance (0-1)")
    
class QueryResponse(BaseModel):
    """
    Response model for RAG query endpoint.
    """
    response: str = Field(..., description="Generated natural language response")
    sources: List[DocumentSource] = Field([], description="Documents used for the answer")
    metrics: Optional[ResponseMetrics] = Field(None, description="Optional debug information")
    conversation_id: Optional[str] = Field(None, description="ID for continuing the conversation")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat(), 
                          description="ISO format timestamp")
    language: Optional[str] = Field(None, description="Detected or specified language")