# rag-service/src/models/document.py
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from datetime import datetime

class DocumentInfo(BaseModel):
    """
    Model for document metadata information.
    """
    id: int = Field(..., description="Document ID")
    filename: str = Field(..., description="Document filename")
    department: str = Field(..., description="Department classification")
    added_date: str = Field(..., description="ISO format date when document was added")
    last_updated: str = Field(..., description="ISO format date when document was last updated")
    file_size: int = Field(..., description="File size in bytes")
    file_type: str = Field(..., description="File type extension (pdf, docx, txt)")
    page_count: Optional[int] = Field(None, description="Number of pages in document")
    embedding_count: Optional[int] = Field(None, description="Number of embeddings created")
    status: str = Field(..., description="Document status (added, updated)")
    tags: List[str] = Field([], description="List of document tags")

class DocumentUploadRequest(BaseModel):
    """
    Request model for document upload endpoint.
    """
    # File will be handled separately as multipart/form-data
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Document metadata")
    
    class Config:
        extra = "allow"  # Allow extra fields for flexibility

class DocumentUploadResponse(BaseModel):
    """
    Response model for document upload endpoint.
    """
    success: bool = Field(..., description="Whether the upload was successful")
    document_id: Optional[int] = Field(None, description="ID of the uploaded document")
    chunks_created: Optional[int] = Field(None, description="Number of chunks created")
    message: str = Field(..., description="Status message")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat(), 
                          description="ISO format timestamp")