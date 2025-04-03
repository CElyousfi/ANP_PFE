# rag-service/src/core/document_processor.py
import os
import logging
import traceback
from typing import List, Optional, Dict, Any
import glob
import re

from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

class DocumentProcessor:
   """
   Handles document loading, processing, and chunking.
   """
   def __init__(self, config: Dict[str, Any]):
       """
       Initialize the document processor with configuration.
       
       Args:
           config: Configuration dictionary containing:
               - chunk_size: Size of document chunks
               - chunk_overlap: Overlap between chunks
               - data_folder: Base folder for document storage
               - default_departments: List of default departments
       """
       self.chunk_size = config.get("chunk_size", 500)
       self.chunk_overlap = config.get("chunk_overlap", 200)
       self.data_folder = config.get("data_folder", "data")
       self.default_departments = config.get("default_departments", 
                                            ["general", "commercial", "technical", "safety", "regulatory"])
       
       self.text_splitter = RecursiveCharacterTextSplitter(
           chunk_size=self.chunk_size,
           chunk_overlap=self.chunk_overlap,
           separators=["\n\n", "\n", ". ", " ", ""]
       )
       
       # Ensure data folders exist
       os.makedirs(self.data_folder, exist_ok=True)
       for dept in self.default_departments:
           os.makedirs(os.path.join(self.data_folder, dept), exist_ok=True)

   def load_document(self, file_path: str, department: Optional[str] = None) -> List[Document]:
       """
       Load a single document from the specified path.
       
       Args:
           file_path: Path to the document
           department: Optional department classification
           
       Returns:
           List of Document objects
       """
       file_ext = os.path.splitext(file_path)[1].lower()
       docs = []
       
       try:
           logger.info(f"Processing file: {file_path}")
           
           # For plain text files (easier for testing)
           if file_ext == '.txt':
               try:
                   with open(file_path, 'r', encoding='utf-8') as f:
                       content = f.read()
                   
                   # Create a single document from the content
                   docs = [Document(
                       page_content=content,
                       metadata={
                           'source': os.path.basename(file_path),
                           'filetype': 'txt',
                           'department': department or 'general',
                           'page_number': 1
                       }
                   )]
                   
                   logger.info(f"Successfully loaded text file with {len(content)} characters")
               except UnicodeDecodeError:
                   # Try different encoding if UTF-8 fails
                   with open(file_path, 'r', encoding='latin-1') as f:
                       content = f.read()
                   
                   docs = [Document(
                       page_content=content,
                       metadata={
                           'source': os.path.basename(file_path),
                           'filetype': 'txt',
                           'department': department or 'general',
                           'page_number': 1
                       }
                   )]
                   
                   logger.info(f"Successfully loaded text file with latin-1 encoding, {len(content)} characters")
               
           # PDF files
           elif file_ext == '.pdf':
               try:
                   loader = PyPDFLoader(file_path)
                   docs = loader.load()
                   for doc in docs:
                       doc.metadata['source'] = os.path.basename(file_path)
                       doc.metadata['filetype'] = 'pdf'
                       doc.metadata['department'] = department or 'general'
                       page_num = doc.metadata.get('page', 0) + 1
                       doc.metadata['page_number'] = page_num
                   logger.info(f"Successfully loaded PDF with {len(docs)} pages")
               except Exception as pdf_error:
                   logger.error(f"Error loading PDF {file_path}: {pdf_error}")
                   # Create a fallback document noting the error
                   docs = [Document(
                       page_content=f"[Error loading PDF content: {str(pdf_error)}]",
                       metadata={
                           'source': os.path.basename(file_path),
                           'filetype': 'pdf',
                           'department': department or 'general',
                           'page_number': 1,
                           'error': True
                       }
                   )]
                   
           # DOCX files
           elif file_ext == '.docx':
               try:
                   loader = Docx2txtLoader(file_path)
                   docs = loader.load()
                   for doc in docs:
                       doc.metadata['source'] = os.path.basename(file_path)
                       doc.metadata['filetype'] = 'docx'
                       doc.metadata['department'] = department or 'general'
                   logger.info(f"Successfully loaded DOCX document")
               except Exception as docx_error:
                   logger.error(f"Error loading DOCX {file_path}: {docx_error}")
                   # Create a fallback document noting the error
                   docs = [Document(
                       page_content=f"[Error loading DOCX content: {str(docx_error)}]",
                       metadata={
                           'source': os.path.basename(file_path),
                           'filetype': 'docx',
                           'department': department or 'general',
                           'page_number': 1,
                           'error': True
                       }
                   )]
               
           else:
               logger.warning(f"Unsupported file type: {file_ext}")
               # Create a placeholder document for unsupported file type
               docs = [Document(
                   page_content=f"[Unsupported file type: {file_ext}]",
                   metadata={
                       'source': os.path.basename(file_path),
                       'filetype': file_ext.lstrip('.'),
                       'department': department or 'general',
                       'page_number': 1,
                       'error': True
                   }
               )]
               
           # Add the file path to metadata for all documents
           for doc in docs:
               doc.metadata['file_path'] = file_path
               
           logger.info(f"Loaded document with {len(docs)} sections/pages")
           return docs
           
       except Exception as e:
           logger.error(f"Error loading {file_path}: {e}")
           logger.error(traceback.format_exc())
           
           # Return a placeholder document noting the error
           return [Document(
               page_content=f"[Error loading document: {str(e)}]",
               metadata={
                   'source': os.path.basename(file_path),
                   'filetype': file_ext.lstrip('.') if file_ext else 'unknown',
                   'department': department or 'general',
                   'page_number': 1,
                   'error': True,
                   'file_path': file_path
               }
           )]

   def load_documents_from_folder(self, folder_path: str, department: Optional[str] = None) -> List[Document]:
       """
       Load all documents from a specific folder.
       
       Args:
           folder_path: Path to the folder containing documents
           department: Optional department classification override
           
       Returns:
           List of Document objects
       """
       pdf_files = glob.glob(os.path.join(folder_path, "*.pdf"))
       docx_files = glob.glob(os.path.join(folder_path, "*.docx"))
       txt_files = glob.glob(os.path.join(folder_path, "*.txt"))
       
       all_files = pdf_files + docx_files + txt_files
       
       if not all_files:
           logger.warning(f"No supported document files found in {folder_path}")
           return []
           
       all_docs = []
       for file_path in all_files:
           try:
               # Determine department from path if not specified
               if department is None:
                   rel_path = os.path.relpath(file_path, self.data_folder)
                   parts = rel_path.split(os.sep)
                   file_department = parts[0] if len(parts) > 1 and parts[0] in self.default_departments else "general"
               else:
                   file_department = department
                   
               # Load the document
               docs = self.load_document(file_path, file_department)
               all_docs.extend(docs)
               
           except Exception as e:
               logger.error(f"Error loading {file_path}: {e}")
               logger.error(traceback.format_exc())
               
       return all_docs

   def load_all_documents(self) -> List[Document]:
       """
       Load all documents from all departments and the root data folder.
       
       Returns:
           List of Document objects
       """
       all_docs = []
       
       # Load documents from root data folder
       root_docs = self.load_documents_from_folder(self.data_folder)
       all_docs.extend(root_docs)
       
       # Load documents from each department folder
       for dept in self.default_departments:
           dept_path = os.path.join(self.data_folder, dept)
           if os.path.exists(dept_path):
               dept_docs = self.load_documents_from_folder(dept_path, dept)
               all_docs.extend(dept_docs)
               
       logger.info(f"Loaded {len(all_docs)} total documents from all folders")
       return all_docs

   def split_documents(self, docs: List[Document]) -> List[Document]:
       """
       Split documents into chunks for better processing.
       
       Args:
           docs: List of Document objects
           
       Returns:
           List of chunked Document objects
       """
       if not docs:
           logger.warning("No documents to split")
           return []
       
       # Filter out error documents
       valid_docs = [doc for doc in docs if not doc.metadata.get('error', False)]
       
       if not valid_docs:
           logger.warning("No valid documents to split")
           return docs  # Return the original error documents
           
       try:    
           chunks = self.text_splitter.split_documents(valid_docs)
           
           # Add metadata to each chunk
           for i, chunk in enumerate(chunks):
               chunk.metadata['chunk_id'] = i
               if 'source' not in chunk.metadata and valid_docs:
                   chunk.metadata['source'] = valid_docs[0].metadata.get('source', 'unknown')
               if 'department' not in chunk.metadata and valid_docs:
                   chunk.metadata['department'] = valid_docs[0].metadata.get('department', 'general')
                   
           logger.info(f"Split {len(valid_docs)} documents into {len(chunks)} chunks")
           
           # Add back any error documents without splitting them
           error_docs = [doc for doc in docs if doc.metadata.get('error', False)]
           chunks.extend(error_docs)
               
           return chunks
       except Exception as e:
           logger.error(f"Error splitting documents: {e}")
           logger.error(traceback.format_exc())
           return docs  # Return original documents if splitting fails

   def get_document_info(self, file_path: str) -> Dict[str, Any]:
       """
       Get basic information about a document.
       
       Args:
           file_path: Path to the document
           
       Returns:
           Dictionary with document metadata
       """
       if not os.path.exists(file_path):
           return {}
           
       try:
           file_size = os.path.getsize(file_path)
           file_type = os.path.splitext(file_path)[1].lstrip('.').lower()
           filename = os.path.basename(file_path)
           
           # Determine department from path
           rel_path = os.path.relpath(file_path, self.data_folder)
           parts = rel_path.split(os.sep)
           department = parts[0] if len(parts) > 1 and parts[0] in self.default_departments else "general"
           
           info = {
               "filename": filename,
               "file_path": file_path,
               "file_size": file_size,
               "file_type": file_type,
               "department": department
           }
           
           return info
       except Exception as e:
           logger.error(f"Error getting document info for {file_path}: {e}")
           return {}

   def enhanced_context_window(self, docs: List[Document], query: str, window_size: int = 3) -> List[Document]:
       """
       Create enhanced context windows around the most relevant parts of documents.
       
       Args:
           docs: List of Document objects
           query: The query string
           window_size: Number of sentences to include before and after the most relevant sentence
           
       Returns:
           List of Document objects with enhanced context windows
       """
       if not docs:
           return []

       enhanced_docs = []
       
       for doc in docs:
           try:
               # Skip error documents
               if doc.metadata.get('error', False):
                   enhanced_docs.append(doc)
                   continue

               # Split into sentences
               sentences = re.split(r'(?<=[.!?])\s+', doc.page_content)
               if not sentences or len(sentences) <= window_size * 2:
                   enhanced_docs.append(doc)
                   continue
                   
               # Simple relevance scoring - count term overlaps
               query_terms = set(query.lower().split())
               
               # Score each sentence by term overlap
               scores = []
               for sentence in sentences:
                   sentence_terms = set(sentence.lower().split())
                   common_terms = query_terms.intersection(sentence_terms)
                   score = len(common_terms) / max(len(query_terms), 1) if query_terms else 0
                   scores.append(score)
               
               # If no good match, keep original document
               if max(scores) < 0.1 and len(sentences) > 10:
                   # For longer documents with no good match, take the first portion
                   new_content = " ".join(sentences[:min(10, len(sentences))])
                   new_doc = Document(
                       page_content=new_content, 
                       metadata=doc.metadata.copy()
                   )
                   new_doc.metadata['window_type'] = 'prefix'
                   enhanced_docs.append(new_doc)
                   continue
               
               # Find most relevant sentence
               most_relevant_idx = scores.index(max(scores))
               
               # Create window
               start_idx = max(0, most_relevant_idx - window_size)
               end_idx = min(len(sentences), most_relevant_idx + window_size + 1)
               window_content = " ".join(sentences[start_idx:end_idx])
               
               # Create new document with window
               new_doc = Document(
                   page_content=window_content,
                   metadata=doc.metadata.copy()
               )
               new_doc.metadata["window_start"] = start_idx
               new_doc.metadata["window_end"] = end_idx
               new_doc.metadata["central_sentence"] = most_relevant_idx
               new_doc.metadata["sentence_relevance"] = scores[most_relevant_idx]
               new_doc.metadata['window_type'] = 'context'
               
               enhanced_docs.append(new_doc)
           except Exception as e:
               logger.error(f"Error processing document for sentence window: {e}")
               enhanced_docs.append(doc)  # Keep original if processing fails
               
       return enhanced_docs