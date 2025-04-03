# rag-service/src/api/endpoints.py
import os
import logging
import traceback
import json
import tempfile
import shutil
from typing import List, Dict, Optional, Any
from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import JSONResponse
import uuid
from datetime import datetime

from ..models.query import QueryRequest
from ..models.response import QueryResponse, DocumentSource, ResponseMetrics
from ..models.document import DocumentInfo, DocumentUploadRequest, DocumentUploadResponse
from ..core.document_processor import DocumentProcessor
from ..core.vector_store import VectorStore
from ..core.response_generator import ResponseGenerator
from ..core.document_database import DocumentDatabase
from ..core.file_watcher import FileWatcher

logger = logging.getLogger(__name__)

# Create a router for the RAG endpoints
router = APIRouter()

# Function to get dependencies from app state
def get_document_processor(request: Request) -> DocumentProcessor:
    return request.app.state.document_processor

def get_vector_store(request: Request) -> VectorStore:
    return request.app.state.vector_store

def get_response_generator(request: Request) -> ResponseGenerator:
    return request.app.state.response_generator

def get_document_database(request: Request) -> DocumentDatabase:
    return request.app.state.document_database

def get_file_watcher(request: Request) -> Optional[FileWatcher]:
    return getattr(request.app.state, "file_watcher", None)

def get_config(request: Request) -> Dict[str, Any]:
    return request.app.state.config


@router.post("/query", response_model=QueryResponse)
async def query_endpoint(
    query_request: QueryRequest,
    request: Request,
    document_processor: DocumentProcessor = Depends(get_document_processor),
    vector_store: VectorStore = Depends(get_vector_store),
    response_generator: ResponseGenerator = Depends(get_response_generator),
    config: Dict[str, Any] = Depends(get_config)
):
    """
    Process a query and return a response using RAG.
    """
    try:
        start_time = datetime.now()
        
        # Extract query parameters
        query = query_request.query
        language = query_request.language
        department = query_request.department
        max_results = query_request.max_results
        include_metadata = query_request.include_metadata
        conversation_id = query_request.conversation_id or str(uuid.uuid4())
        previous_messages = query_request.previous_messages or []
        
        # Convert previous messages to the format expected by the response generator
        conversation_history = []
        for msg in previous_messages:
            conversation_history.append({
                "role": msg["role"],
                "content": msg["content"]
            })
        
        # Determine which departments to search
        departments = None
        if department:
            departments = [department]
        
        # Retrieve relevant documents
        retrieval_start = datetime.now().timestamp()
        
        if departments:
            docs = vector_store.search_across_departments(query, departments, max_results)
        else:
            docs = vector_store.similarity_search(query, max_results)
        
        retrieval_time = datetime.now().timestamp() - retrieval_start
        
        # Evaluate context relevance
        max_relevance, has_relevant_context = vector_store.evaluate_context_relevance(
            query, docs, threshold=config.get("relevance_threshold", 0.7)
        )
        
        # Apply sentence window retrieval for more focused context
        windowed_docs = document_processor.enhanced_context_window(docs, query)
        
        # Prepare retrieval metrics
        metrics = {
            "retrieval_time": retrieval_time,
            "context_relevance": max_relevance,
            "has_relevant_context": has_relevant_context
        }
        
        # Generate response
        generation_start = datetime.now().timestamp()
        response_text, response_metrics = response_generator.generate_response(
            query, windowed_docs, metrics, conversation_history
        )
        generation_time = datetime.now().timestamp() - generation_start
        
        # Prepare the document sources for the response
        sources = []
        if include_metadata:
            for doc in docs:
                source = doc.metadata.get('source', 'unknown')
                department = doc.metadata.get('department', 'general')
                page = doc.metadata.get('page_number')
                relevance = doc.metadata.get('relevance_score', 0.0)
                
                sources.append(DocumentSource(
                    source=source,
                    department=department,
                    page=page,
                    content=doc.page_content[:500] + ("..." if len(doc.page_content) > 500 else ""),
                    relevance=relevance
                ))
        
        # Prepare final response
        response = QueryResponse(
            response=response_text,
            sources=sources,
            metrics=ResponseMetrics(
                retrieval_time=retrieval_time,
                generation_time=generation_time,
                context_relevance=max_relevance
            ),
            conversation_id=conversation_id,
            timestamp=datetime.now().isoformat(),
            language=response_metrics.get("language", language)
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}"
        )


@router.post("/documents", response_model=DocumentUploadResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    request: Request,
    document: UploadFile = File(...),
    metadata: str = Form("{}"),
    document_processor: DocumentProcessor = Depends(get_document_processor),
    vector_store: VectorStore = Depends(get_vector_store),
    document_database: DocumentDatabase = Depends(get_document_database),
    config: Dict[str, Any] = Depends(get_config)
):
    """
    Upload a document to the knowledge base.
    """
    try:
        # Parse metadata
        try:
            metadata_dict = json.loads(metadata)
        except json.JSONDecodeError:
            metadata_dict = {}
        
        # Get document parameters
        department = metadata_dict.get("department", "general")
        tags = metadata_dict.get("tags", [])
        description = metadata_dict.get("description", "")
        
        # Create department folder path
        data_folder = config.get("data_folder", "data")
        dept_folder = os.path.join(data_folder, department)
        os.makedirs(dept_folder, exist_ok=True)
        
        # Save the file
        file_path = os.path.join(dept_folder, document.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(document.file, buffer)
        
        # Process the document in the background
        def process_document():
            try:
                # Load and process the document
                docs = document_processor.load_document(file_path, department)
                
                if not docs:
                    logger.warning(f"No content extracted from {file_path}")
                    return
                
                # Add to document database
                doc_id = document_database.add_document(
                    file_path=file_path,
                    department=department,
                    tags=tags,
                    page_count=len(docs)
                )
                
                # Split into chunks and add to vector store
                chunks = document_processor.split_documents(docs)
                
                # Update vector stores
                vector_store.add_documents(chunks)
                vector_store.add_documents_to_department(department, chunks)
                
                logger.info(f"Successfully processed document {file_path} with {len(chunks)} chunks")
                
            except Exception as e:
                logger.error(f"Error processing document {file_path} in background: {e}")
                logger.error(traceback.format_exc())
        
        # Schedule background processing
        background_tasks.add_task(process_document)
        
        return DocumentUploadResponse(
            success=True,
            document_id=None,  # Will be set by background process
            chunks_created=None,  # Will be set by background process
            message=f"Document '{document.filename}' uploaded and scheduled for processing",
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Error uploading document: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Error uploading document: {str(e)}"
        )


@router.get("/documents", response_model=List[DocumentInfo])
async def list_documents(
    request: Request,
    department: Optional[str] = None,
    tag: Optional[str] = None,
    limit: int = 100,
    document_database: DocumentDatabase = Depends(get_document_database)
):
    """
    List documents in the knowledge base with optional filtering.
    """
    try:
        documents = document_database.list_documents(department, tag, limit)
        return documents
        
    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Error listing documents: {str(e)}"
        )


@router.get("/documents/{doc_id}", response_model=DocumentInfo)
async def get_document(
    doc_id: int,
    request: Request,
    document_database: DocumentDatabase = Depends(get_document_database)
):
    """
    Get document metadata by ID.
    """
    try:
        document = document_database.get_document(doc_id)
        
        if document is None:
            raise HTTPException(
                status_code=404,
                detail=f"Document with ID {doc_id} not found"
            )
            
        return document
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting document: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Error getting document: {str(e)}"
        )


@router.delete("/documents/{doc_id}")
async def delete_document(
    doc_id: int,
    request: Request,
    document_database: DocumentDatabase = Depends(get_document_database)
):
    """
    Delete a document from the knowledge base.
    """
    try:
        # Get document info before deleting
        document = document_database.get_document(doc_id)
        
        if document is None:
            raise HTTPException(
                status_code=404,
                detail=f"Document with ID {doc_id} not found"
            )
            
        # Delete document from database
        success = document_database.delete_document(doc_id)
        
        if not success:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to delete document with ID {doc_id}"
            )
            
        return {"success": True, "message": f"Document {doc_id} deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting document: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Error deleting document: {str(e)}"
        )


@router.get("/departments")
async def list_departments(
    request: Request,
    document_database: DocumentDatabase = Depends(get_document_database)
):
    """
    List departments with document counts.
    """
    try:
        departments = document_database.get_departments()
        return {"departments": [{"name": dept[0], "document_count": dept[1]} for dept in departments]}
        
    except Exception as e:
        logger.error(f"Error listing departments: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Error listing departments: {str(e)}"
        )


@router.post("/rebuild")
async def rebuild_indexes(
    request: Request,
    background_tasks: BackgroundTasks,
    vector_store: VectorStore = Depends(get_vector_store),
    document_processor: DocumentProcessor = Depends(get_document_processor),
    config: Dict[str, Any] = Depends(get_config)
):
    """
    Rebuild document indexes from scratch.
    """
    try:
        def rebuild_task():
            try:
                # Load all documents
                all_docs = document_processor.load_all_documents()
                
                if not all_docs:
                    logger.warning("No documents found to rebuild indexes")
                    return
                    
                # Split into chunks
                chunks = document_processor.split_documents(all_docs)
                
                # Rebuild main vector store
                vector_store.initialize_vectorstore(chunks, rebuild=True)
                
                # Rebuild department-specific vector stores
                departments = config.get("default_departments", ["general", "commercial", "technical", "safety", "regulatory"])
                for department in departments:
                    vector_store.initialize_department_vectorstore(department, chunks, rebuild=True)
                    
                logger.info(f"Successfully rebuilt indexes with {len(chunks)} chunks from {len(all_docs)} documents")
                
            except Exception as e:
                logger.error(f"Error in rebuild task: {e}")
                logger.error(traceback.format_exc())
        
        # Schedule background rebuild
        background_tasks.add_task(rebuild_task)
        
        return {
            "success": True,
            "message": "Index rebuild scheduled in the background"
        }
        
    except Exception as e:
        logger.error(f"Error scheduling rebuild: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Error scheduling rebuild: {str(e)}"
        )


@router.get("/health")
async def health_check(
    request: Request,
    vector_store: VectorStore = Depends(get_vector_store),
    document_database: DocumentDatabase = Depends(get_document_database)
):
    """
    Check the health of the RAG service.
    """
    try:
        # Get vector store stats
        index_stats = vector_store.get_index_stats()
        
        # Get document count
        document_count = document_database.get_document_count()
        
        # Check if file watcher is running
        file_watcher = get_file_watcher(request)
        file_watcher_status = "running" if file_watcher and file_watcher.is_running else "stopped"
        
        return {
            "status": "healthy",
            "document_count": document_count,
            "index_status": "ready",
            "index_stats": index_stats,
            "file_watcher": file_watcher_status,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in health check: {e}")
        logger.error(traceback.format_exc())
        return {
            "status": "degraded",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }