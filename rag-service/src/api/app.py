import os
import logging
from typing import Dict, Any

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
import yaml

from .endpoints import router
from ..utils.embeddings import get_embeddings_model
from ..utils.llm import get_llm_service
from ..core.document_processor import DocumentProcessor
from ..core.vector_store import VectorStore
from ..core.response_generator import ResponseGenerator
from ..core.document_database import DocumentDatabase
from ..core.file_watcher import FileWatcher

logger = logging.getLogger(__name__)

def create_app(config_path: str = "config/config.yaml"):
    """
    Create and configure the FastAPI application.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Configured FastAPI application
    """
    # Load configuration
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
    except Exception as e:
        logger.error(f"Error loading configuration from {config_path}: {e}")
        config = {}
    
    # Create FastAPI app
    app = FastAPI(
        title="ANP RAG Service",
        description="Retrieval Augmented Generation service for ANP port information",
        version="1.0.0"
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.get("cors", {}).get("origins", ["*"]),
        allow_credentials=config.get("cors", {}).get("allow_credentials", True),
        allow_methods=config.get("cors", {}).get("methods", ["*"]),
        allow_headers=config.get("cors", {}).get("headers", ["*"]),
    )
    
    # Add rate limiting middleware if needed
    if config.get("rate_limiting", {}).get("enabled", False):
        from .rate_limiter import RateLimiter
        app.add_middleware(
            RateLimiter,
            limit=config.get("rate_limiting", {}).get("limit", 100),
            window=config.get("rate_limiting", {}).get("window", 3600)
        )
    
    # Register routes
    app.include_router(router, prefix="/api")
    
    # Startup event
    @app.on_event("startup")
    async def startup_event():
        # Store config in app state
        app.state.config = config
        
        # Initialize embeddings model
        logger.info("Initializing embeddings model...")
        embeddings = get_embeddings_model(config.get("embeddings", {}))
        
        # Initialize LLM service
        logger.info("Initializing LLM service...")
        llm_service = get_llm_service(config.get("llm", {}))
        
        # Initialize document processor
        logger.info("Initializing document processor...")
        app.state.document_processor = DocumentProcessor(config.get("document_processor", {}))
        
        # Initialize document database
        logger.info("Initializing document database...")
        app.state.document_database = DocumentDatabase(config.get("document_database", {}))
        
        # Initialize vector store
        logger.info("Initializing vector store...")
        app.state.vector_store = VectorStore(embeddings, config.get("vector_store", {}))
        
        # Load existing vector stores
        logger.info("Loading vector stores...")
        try:
            app.state.vector_store.initialize_vectorstore()
            departments = config.get("document_processor", {}).get("default_departments", 
                                             ["general", "commercial", "technical", "safety", "regulatory"])
            app.state.vector_store.initialize_all_department_vectorstores()
            logger.info("Successfully initialized vector stores")
        except Exception as e:
            logger.error(f"Error initializing vector stores: {e}")
            logger.warning("Starting with empty vector stores - will need to add documents")
            # Create empty vector stores
            app.state.vector_store.initialize_vectorstore(rebuild=True)
            for dept in config.get("document_processor", {}).get("default_departments", ["general"]):
                app.state.vector_store.initialize_department_vectorstore(dept, rebuild=True)
        
        # Initialize response generator
        logger.info("Initializing response generator...")
        app.state.response_generator = ResponseGenerator(llm_service, config.get("response_generator", {}))
        
        # Initialize file watcher if enabled
        if config.get("file_watcher", {}).get("enabled", True):
            logger.info("Initializing file watcher...")
            app.state.file_watcher = FileWatcher(config.get("file_watcher", {}))
            
            # Register document processing callback
            def process_new_document(file_path: str, department: str):
                try:
                    logger.info(f"Processing new file from watcher: {file_path}")
                    docs = app.state.document_processor.load_document(file_path, department)
                    if docs:
                        # Add to document database
                        app.state.document_database.add_document(
                            file_path=file_path,
                            department=department,
                            page_count=len(docs)
                        )
                        
                        # Split and add to vector stores
                        chunks = app.state.document_processor.split_documents(docs)
                        app.state.vector_store.add_documents(chunks)
                        app.state.vector_store.add_documents_to_department(department, chunks)
                        logger.info(f"File watcher processed {file_path} with {len(chunks)} chunks")
                except Exception as e:
                    logger.error(f"Error processing file {file_path} from watcher: {e}")
            
            app.state.file_watcher.register_callback(process_new_document)
            app.state.file_watcher.setup()
            app.state.file_watcher.start()
            logger.info("File watcher started")
            
            # Check and process existing documents
            try:
                all_docs = app.state.document_processor.load_all_documents()
                if all_docs:
                    logger.info(f"Found {len(all_docs)} existing documents to process")
                    chunks = app.state.document_processor.split_documents(all_docs)
                    
                    # Force rebuild of main vector store with these documents to ensure it has content
                    app.state.vector_store.initialize_vectorstore(chunks=chunks, rebuild=True)
                    
                    # Process department-specific documents
                    for dept in app.state.config.get("document_processor", {}).get("default_departments", 
                                               ["general", "commercial", "technical", "safety", "regulatory"]):
                        # Filter docs for this department
                        dept_docs = [doc for doc in all_docs if doc.metadata.get("department") == dept]
                        if dept_docs:
                            dept_chunks = app.state.document_processor.split_documents(dept_docs)
                            app.state.vector_store.initialize_department_vectorstore(dept, chunks=dept_chunks, rebuild=True)
                            
                    logger.info(f"Successfully processed {len(all_docs)} documents into {len(chunks)} chunks")
                    
                    # Force a health check to verify vector stores have documents
                    index_stats = app.state.vector_store.get_index_stats()
                    logger.info(f"Vector store status: {index_stats}")
                    if index_stats['total_documents'] == 0:
                        logger.warning("Vector store appears empty after initialization. Forcing rebuild.")
                        # Try one more time with different approach
                        app.state.vector_store.initialize_vectorstore(chunks=chunks, rebuild=True)
                    
                else:
                    logger.warning("No documents found for processing")
            except Exception as e:
                logger.error(f"Error processing existing documents: {e}")
    
    # Shutdown event
    @app.on_event("shutdown")
    async def shutdown_event():
        # Stop file watcher if it's running
        if hasattr(app.state, "file_watcher") and app.state.file_watcher.is_running:
            app.state.file_watcher.stop()
            logger.info("File watcher stopped")
    
    return app