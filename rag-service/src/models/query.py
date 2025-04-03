# rag-service/src/models/query.py
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Optional, Any

class QueryRequest(BaseModel):
    """
    Request model for RAG query endpoint.
    """
    query: str = Field(..., description="The user's natural language question")
    language: Optional[str] = Field(None, description="Optional language code (fr, ar, en)")
    department: Optional[str] = Field(None, description="Optional department filter (technical, commercial, etc.)")
    max_results: Optional[int] = Field(5, description="Maximum number of document chunks to return")
    include_metadata: Optional[bool] = Field(True, description="Whether to include document metadata")
    conversation_id: Optional[str] = Field(None, description="ID for maintaining conversation context")
    previous_messages: Optional[List[Dict[str, str]]] = Field(None, description="Optional conversation history")
    
    @validator('max_results')
    def validate_max_results(cls, v):
        if v is not None and (v < 1 or v > 20):
            raise ValueError('max_results must be between 1 and 20')
        return v
    
    @validator('previous_messages')
    def validate_previous_messages(cls, v):
        if v is not None:
            for msg in v:
                if 'role' not in msg or 'content' not in msg:
                    raise ValueError('Each message must have role and content fields')
                if msg['role'] not in ['user', 'assistant', 'system']:
                    raise ValueError('Message role must be user, assistant, or system')
        return v