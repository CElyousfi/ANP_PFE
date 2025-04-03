# rag-service/main.py

import os
import sys
import logging
import argparse
import yaml
import traceback

# Set up correct import path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("rag-service.log")
    ]
)

logger = logging.getLogger(__name__)

def load_config(config_path):
    """Load configuration from a YAML file."""
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
            logger.info(f"Loaded configuration from {config_path}")
            return config
    except Exception as e:
        logger.warning(f"Error loading configuration from {config_path}: {e}")
        logger.warning("Using default configuration")
        return {}

def setup_data_directories(config):
    """Create necessary data directories based on configuration."""
    data_folder = config.get("document_processor", {}).get("data_folder", "data")
    default_departments = config.get("document_processor", {}).get("default_departments", 
                                             ["general", "commercial", "technical", "safety", "regulatory"])
    
    # Create main data folder
    os.makedirs(data_folder, exist_ok=True)
    logger.info(f"Created data folder: {data_folder}")
    
    # Create department subfolders
    for dept in default_departments:
        dept_path = os.path.join(data_folder, dept)
        os.makedirs(dept_path, exist_ok=True)
        logger.info(f"Created department folder: {dept_path}")
    
    # Create vector store directory
    faiss_index_path = config.get("vector_store", {}).get("faiss_index_path", "faiss_index")
    os.makedirs(faiss_index_path, exist_ok=True)
    logger.info(f"Created vector store directory: {faiss_index_path}")
    
    # Check for test documents
    has_documents = False
    for dept in default_departments:
        dept_path = os.path.join(data_folder, dept)
        if os.path.exists(dept_path):
            files = [f for f in os.listdir(dept_path) if os.path.isfile(os.path.join(dept_path, f))]
            if files:
                has_documents = True
                logger.info(f"Found {len(files)} documents in {dept} department")
    
    if not has_documents:
        logger.warning("No documents found in data directories")
        try:
            create_test_documents = input("Would you like to create test documents? (y/n): ").lower().strip() == 'y'
            if create_test_documents:
                script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "create_test_documents.py")
                if os.path.exists(script_path):
                    import subprocess
                    subprocess.run([sys.executable, script_path], check=True)
                    logger.info("Created test documents")
                else:
                    logger.error(f"Test document creation script not found at {script_path}")
        except:
            pass

def main():
    """Main entry point for the RAG service."""
    parser = argparse.ArgumentParser(description="ANP Retrieval Augmented Generation Service")
    parser.add_argument(
        "--config", type=str, default="config/config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--host", type=str, default=None,
        help="Host to bind the server to (overrides config)"
    )
    parser.add_argument(
        "--port", type=int, default=None,
        help="Port to bind the server to (overrides config)"
    )
    parser.add_argument(
        "--reload", action="store_true",
        help="Enable hot reloading for development"
    )
    parser.add_argument(
        "--log-level", type=str, default="info",
        choices=["debug", "info", "warning", "error", "critical"],
        help="Logging level"
    )
    parser.add_argument(
        "--setup-only", action="store_true",
        help="Only set up data directories, don't start server"
    )
    
    args = parser.parse_args()
    
    # Set log level
    log_level = getattr(logging, args.log_level.upper())
    logging.getLogger().setLevel(log_level)
    
    # Load configuration
    config_path = args.config
    config = load_config(config_path)
    
    # Set up data directories
    setup_data_directories(config)
    
    if args.setup_only:
        logger.info("Setup completed. Exiting as requested.")
        return
    
    # Get server settings
    host = args.host or config.get("server", {}).get("host", "0.0.0.0")
    port = args.port or config.get("server", {}).get("port", 8001)
    reload = args.reload or config.get("server", {}).get("reload", False)
    
    # Log startup information
    logger.info(f"Starting RAG service on {host}:{port}")
    logger.info(f"Environment: {'Development' if reload else 'Production'}")
    
    # Import here to avoid circular imports
    from src.api.app import create_app
    
    # Start the server with uvicorn directly
    import uvicorn
    uvicorn.run(
        create_app(config_path),
        host=host,
        port=port,
        log_level=args.log_level.lower()
    )

if __name__ == "__main__":
    main()