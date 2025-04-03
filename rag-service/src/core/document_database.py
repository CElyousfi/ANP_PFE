# rag-service/src/core/document_database.py
import os
import sqlite3
import logging
from typing import List, Dict, Tuple, Optional, Any
from datetime import datetime

logger = logging.getLogger(__name__)

class DocumentDatabase:
    """
    Manages document metadata in a SQLite database.
    """
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the document database with configuration.
        
        Args:
            config: Configuration dictionary containing:
                - db_path: Path to the SQLite database
        """
        self.db_path = config.get("db_path", "documents.db")
        self._initialize_db()
        
    def _initialize_db(self) -> None:
        """
        Initialize the database schema if it doesn't exist.
        """
        os.makedirs(os.path.dirname(self.db_path) if os.path.dirname(self.db_path) else '.', exist_ok=True)
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Documents table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                department TEXT,
                added_date TEXT,
                last_updated TEXT,
                file_size INTEGER,
                file_type TEXT,
                page_count INTEGER,
                embedding_count INTEGER,
                status TEXT,
                UNIQUE(filename, department)
            )
            ''')
            
            # Document tags table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS document_tags (
                doc_id INTEGER,
                tag TEXT,
                FOREIGN KEY (doc_id) REFERENCES documents (id),
                PRIMARY KEY (doc_id, tag)
            )
            ''')
            
            # Departments table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS departments (
                name TEXT PRIMARY KEY,
                doc_count INTEGER DEFAULT 0,
                last_updated TEXT
            )
            ''')
            
            conn.commit()
            conn.close()
            logger.info(f"Document database initialized at {self.db_path}")
            
        except Exception as e:
            logger.error(f"Error setting up document database: {e}")
            
    def add_document(self, file_path: str, department: str = "general", 
                     tags: List[str] = None, page_count: int = None, 
                     embedding_count: int = None) -> int:
        """
        Add or update a document in the database.
        
        Args:
            file_path: Path to the document file
            department: Department classification
            tags: Optional list of tags for the document
            page_count: Optional count of pages in the document
            embedding_count: Optional count of embeddings created
            
        Returns:
            Document ID (positive integer) if successful, -1 if failed
        """
        try:
            # Get file details
            file_size = os.path.getsize(file_path)
            file_type = os.path.splitext(file_path)[1].lstrip('.').lower()
            filename = os.path.basename(file_path)
            current_time = datetime.now().isoformat()
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check if document already exists
            cursor.execute(
                "SELECT id FROM documents WHERE filename = ? AND department = ?", 
                (filename, department)
            )
            result = cursor.fetchone()
            
            if result:
                # Update existing document
                doc_id = result[0]
                cursor.execute('''
                    UPDATE documents SET 
                    last_updated = ?, 
                    file_size = ?, 
                    status = ?,
                    page_count = COALESCE(?, page_count),
                    embedding_count = COALESCE(?, embedding_count)
                    WHERE id = ?
                ''', (current_time, file_size, "updated", page_count, embedding_count, doc_id))
            else:
                # Insert new document
                cursor.execute('''
                    INSERT INTO documents 
                    (filename, department, added_date, last_updated, file_size, file_type, 
                     page_count, embedding_count, status)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (filename, department, current_time, current_time, file_size, 
                      file_type, page_count, embedding_count, "added"))
                doc_id = cursor.lastrowid
                
                # Update department statistics
                cursor.execute('''
                    INSERT INTO departments (name, doc_count, last_updated)
                    VALUES (?, 1, ?)
                    ON CONFLICT(name) DO UPDATE SET 
                    doc_count = doc_count + 1,
                    last_updated = ?
                ''', (department, current_time, current_time))
            
            # Update document tags
            if tags:
                cursor.execute("DELETE FROM document_tags WHERE doc_id = ?", (doc_id,))
                for tag in tags:
                    cursor.execute(
                        "INSERT INTO document_tags (doc_id, tag) VALUES (?, ?)",
                        (doc_id, tag)
                    )
            
            conn.commit()
            conn.close()
            return doc_id
            
        except Exception as e:
            logger.error(f"Error adding document to database: {e}")
            return -1
            
    def get_document(self, doc_id: int) -> Optional[Dict[str, Any]]:
        """
        Get document metadata by ID.
        
        Args:
            doc_id: Document ID
            
        Returns:
            Dictionary of document metadata or None if not found
        """
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute("SELECT * FROM documents WHERE id = ?", (doc_id,))
            result = cursor.fetchone()
            
            if not result:
                conn.close()
                return None
                
            document = dict(result)
            
            # Get document tags
            cursor.execute("SELECT tag FROM document_tags WHERE doc_id = ?", (doc_id,))
            tags = [row['tag'] for row in cursor.fetchall()]
            document['tags'] = tags
            
            conn.close()
            return document
            
        except Exception as e:
            logger.error(f"Error getting document: {e}")
            return None
            
    def list_documents(self, department: Optional[str] = None, 
                       tag: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """
        List documents with optional filtering.
        
        Args:
            department: Optional department filter
            tag: Optional tag filter
            limit: Maximum number of documents to return
            
        Returns:
            List of document metadata dictionaries
        """
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            query = "SELECT d.* FROM documents d"
            params = []
            
            if tag:
                query += " JOIN document_tags t ON d.id = t.doc_id WHERE t.tag = ?"
                params.append(tag)
                if department:
                    query += " AND d.department = ?"
                    params.append(department)
            elif department:
                query += " WHERE d.department = ?"
                params.append(department)
                
            query += " ORDER BY d.last_updated DESC LIMIT ?"
            params.append(limit)
            
            cursor.execute(query, params)
            results = [dict(row) for row in cursor.fetchall()]
            
            # Get tags for each document
            for doc in results:
                cursor.execute("SELECT tag FROM document_tags WHERE doc_id = ?", (doc['id'],))
                doc['tags'] = [row['tag'] for row in cursor.fetchall()]
                
            conn.close()
            return results
            
        except Exception as e:
            logger.error(f"Error listing documents: {e}")
            return []
            
    def get_departments(self) -> List[Tuple[str, int]]:
        """
        Get list of departments with document counts.
        
        Returns:
            List of (department_name, document_count) tuples
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("SELECT name, doc_count FROM departments ORDER BY name")
            departments = cursor.fetchall()
            
            conn.close()
            return departments
            
        except Exception as e:
            logger.error(f"Error getting departments: {e}")
            return []
    
    def get_document_count(self, department: Optional[str] = None) -> int:
        """
        Get total document count, optionally filtered by department.
        
        Args:
            department: Optional department filter
            
        Returns:
            Document count
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            if department:
                cursor.execute("SELECT COUNT(*) FROM documents WHERE department = ?", (department,))
            else:
                cursor.execute("SELECT COUNT(*) FROM documents")
                
            count = cursor.fetchone()[0]
            
            conn.close()
            return count
            
        except Exception as e:
            logger.error(f"Error getting document count: {e}")
            return 0
    
    def delete_document(self, doc_id: int) -> bool:
        """
        Delete a document from the database.
        
        Args:
            doc_id: Document ID
            
        Returns:
            True if successful, False otherwise
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get document department first for updating department count
            cursor.execute("SELECT department FROM documents WHERE id = ?", (doc_id,))
            result = cursor.fetchone()
            
            if not result:
                conn.close()
                return False
                
            department = result[0]
            
            # Delete document tags
            cursor.execute("DELETE FROM document_tags WHERE doc_id = ?", (doc_id,))
            
            # Delete document
            cursor.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
            
            # Update department count
            cursor.execute('''
                UPDATE departments 
                SET doc_count = MAX(0, doc_count - 1),
                    last_updated = ?
                WHERE name = ?
            ''', (datetime.now().isoformat(), department))
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            logger.error(f"Error deleting document: {e}")
            return False