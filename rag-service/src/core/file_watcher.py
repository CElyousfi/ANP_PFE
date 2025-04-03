# rag-service/src/core/file_watcher.py
import os
import logging
import time
from typing import Callable, Set, Dict, Any
from threading import Timer

# Configure logging
logger = logging.getLogger(__name__)

# Check if watchdog is available
try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
    WATCHDOG_AVAILABLE = True
except ImportError:
    logger.warning("Watchdog package not installed. File monitoring disabled.")
    logger.warning("Install with: pip install watchdog")
    WATCHDOG_AVAILABLE = False

class FileWatcher:
    """
    Watches for file changes in a directory and triggers callbacks.
    This component can be disabled if the watchdog package is not available.
    """
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the file watcher with configuration settings.
        
        Args:
            config: Configuration dictionary containing:
                - data_folder: Base data folder to watch
                - file_processing_delay: Seconds to wait before processing (default: 5)
                - recursive_watch: Whether to watch subdirectories (default: True)
        """
        self.data_folder = config.get("data_folder", "data")
        self.file_processing_delay = config.get("file_processing_delay", 5)
        self.recursive_watch = config.get("recursive_watch", True)
        self.supported_extensions = config.get("supported_extensions", ['.pdf', '.docx', '.txt'])
        
        self.observer = None
        self.processing_queue: Set[str] = set()
        self.is_running = False
        self.callbacks = []
        
    def register_callback(self, callback: Callable[[str, str], None]):
        """
        Register a callback function to be called when a file changes.
        
        Args:
            callback: Function with signature (file_path, department) -> None
        """
        self.callbacks.append(callback)
        
    def setup(self) -> bool:
        """
        Set up the file watcher to monitor the data folder.
        
        Returns:
            True if setup was successful, False otherwise
        """
        if not WATCHDOG_AVAILABLE:
            logger.warning("File watching disabled because watchdog package is not installed")
            return False
            
        try:
            class NewFileHandler(FileSystemEventHandler):
                def __init__(self, file_watcher):
                    self.file_watcher = file_watcher
                    
                def on_created(self, event):
                    if event.is_directory:
                        return
                        
                    file_path = event.src_path
                    file_ext = os.path.splitext(file_path)[1].lower()
                    
                    if file_ext in self.file_watcher.supported_extensions:
                        logger.info(f"New file detected: {file_path}")
                        self.file_watcher.processing_queue.add(file_path)
                        Timer(self.file_watcher.file_processing_delay, 
                              self.process_file, 
                              args=[file_path]).start()
                
                def on_modified(self, event):
                    if event.is_directory:
                        return
                        
                    file_path = event.src_path
                    file_ext = os.path.splitext(file_path)[1].lower()
                    
                    # Only process modified files that aren't already in the queue
                    if file_ext in self.file_watcher.supported_extensions and file_path not in self.file_watcher.processing_queue:
                        logger.info(f"Modified file detected: {file_path}")
                        self.file_watcher.processing_queue.add(file_path)
                        Timer(self.file_watcher.file_processing_delay, 
                              self.process_file, 
                              args=[file_path]).start()
                        
                def process_file(self, file_path):
                    if file_path not in self.file_watcher.processing_queue:
                        return
                        
                    try:
                        logger.info(f"Processing file: {file_path}")
                        
                        # Determine department from path
                        rel_path = os.path.relpath(file_path, self.file_watcher.data_folder)
                        parts = rel_path.split(os.sep)
                        
                        # If file is in a department subdirectory, use that as the department
                        if len(parts) > 1:
                            department = parts[0]
                        else:
                            department = "general"
                            
                        # Call all registered callbacks
                        for callback in self.file_watcher.callbacks:
                            try:
                                callback(file_path, department)
                            except Exception as e:
                                logger.error(f"Error in file watcher callback for {file_path}: {e}")
                                
                    except Exception as e:
                        logger.error(f"Error processing file {file_path}: {e}")
                    finally:
                        self.file_watcher.processing_queue.discard(file_path)
            
            # Create and configure the observer
            self.observer = Observer()
            self.observer.schedule(
                NewFileHandler(self), 
                self.data_folder, 
                recursive=self.recursive_watch
            )
            
            logger.info(f"File watcher set up for directory: {self.data_folder}")
            return True
            
        except Exception as e:
            logger.error(f"Error setting up file watcher: {e}")
            return False
    
    def start(self) -> bool:
        """
        Start the file watcher.
        
        Returns:
            True if started successfully, False otherwise
        """
        if not self.observer:
            if not self.setup():
                logger.error("File watcher not set up and setup failed")
                return False
        
        try:
            self.observer.start()
            self.is_running = True
            logger.info("File watcher started")
            return True
        except Exception as e:
            logger.error(f"Error starting file watcher: {e}")
            return False
    
    def stop(self) -> bool:
        """
        Stop the file watcher.
        
        Returns:
            True if stopped successfully, False otherwise
        """
        if not self.observer:
            logger.warning("File watcher not set up, nothing to stop")
            return False
            
        if not self.is_running:
            logger.warning("File watcher not running, nothing to stop")
            return False
            
        try:
            self.observer.stop()
            self.observer.join()
            self.is_running = False
            logger.info("File watcher stopped")
            return True
        except Exception as e:
            logger.error(f"Error stopping file watcher: {e}")
            return False
    
    def scan_existing_files(self, process_callback: Callable[[str, str], None] = None) -> int:
        """
        Scan for existing files and process them.
        This is useful when starting the service to process files that already exist.
        
        Args:
            process_callback: Optional specific callback to use for processing
                              If None, all registered callbacks will be used
        
        Returns:
            Number of files found and processed
        """
        count = 0
        
        # Process files in the root data folder
        for file_name in os.listdir(self.data_folder):
            file_path = os.path.join(self.data_folder, file_name)
            
            if os.path.isfile(file_path) and os.path.splitext(file_path)[1].lower() in self.supported_extensions:
                logger.info(f"Found existing file: {file_path}")
                count += 1
                
                if process_callback:
                    process_callback(file_path, "general")
                else:
                    for callback in self.callbacks:
                        try:
                            callback(file_path, "general")
                        except Exception as e:
                            logger.error(f"Error in file watcher callback for {file_path}: {e}")
        
        # Process files in department folders
        for department in os.listdir(self.data_folder):
            dept_path = os.path.join(self.data_folder, department)
            
            if os.path.isdir(dept_path):
                for file_name in os.listdir(dept_path):
                    file_path = os.path.join(dept_path, file_name)
                    
                    if os.path.isfile(file_path) and os.path.splitext(file_path)[1].lower() in self.supported_extensions:
                        logger.info(f"Found existing file: {file_path}")
                        count += 1
                        
                        if process_callback:
                            process_callback(file_path, department)
                        else:
                            for callback in self.callbacks:
                                try:
                                    callback(file_path, department)
                                except Exception as e:
                                    logger.error(f"Error in file watcher callback for {file_path}: {e}")
        
        return count