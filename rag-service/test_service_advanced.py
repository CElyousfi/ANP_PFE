# rag-service/test_service_advanced.py
import requests
import json
import os
import sys
import time
from pprint import pprint
import argparse

# Service URL
RAG_URL = "http://localhost:8001/api"

def test_health():
    """Test the health endpoint"""
    try:
        print("Testing health endpoint...")
        response = requests.get(f"{RAG_URL}/health")
        data = response.json()
        print(f"Service status: {data.get('status', 'unknown')}")
        
        # Print index information if available
        if 'index_stats' in data:
            stats = data['index_stats']
            print(f"Main index exists: {stats.get('main_index', {}).get('exists', False)}")
            print(f"Main index size: {stats.get('main_index', {}).get('size', 0)} documents")
            print(f"Total documents: {stats.get('total_documents', 0)}")
            
            # Department indexes
            if 'department_indexes' in stats:
                print("\nDepartment indexes:")
                for dept, info in stats.get('department_indexes', {}).items():
                    print(f"  - {dept}: {'exists' if info.get('exists') else 'missing'}, {info.get('size', 0)} documents")
        
        return response.status_code == 200
    except Exception as e:
        print(f"Error testing health endpoint: {e}")
        return False

def list_documents():
    """List indexed documents"""
    try:
        print("\nListing indexed documents...")
        response = requests.get(f"{RAG_URL}/documents")
        docs = response.json()
        
        print(f"Found {len(docs)} documents:")
        for doc in docs:
            print(f"- {doc.get('filename')} (Department: {doc.get('department')}, Added: {doc.get('added_date')})")
        
        return response.status_code == 200
    except Exception as e:
        print(f"Error listing documents: {e}")
        return False

def list_departments():
    """List available departments"""
    try:
        print("\nListing departments...")
        response = requests.get(f"{RAG_URL}/departments")
        data = response.json()
        
        departments = data.get('departments', [])
        print(f"Found {len(departments)} departments:")
        for dept in departments:
            print(f"- {dept.get('name')}: {dept.get('document_count')} documents")
        
        return response.status_code == 200
    except Exception as e:
        print(f"Error listing departments: {e}")
        return False

def test_query(query_text, department=None):
    """Test the query endpoint"""
    payload = {
        "query": query_text,
        "max_results": 5,
        "include_metadata": True
    }
    
    if department:
        payload["department"] = department
    
    try:
        print(f"\nSending query: '{query_text}'{f' (Department: {department})' if department else ''}")
        response = requests.post(f"{RAG_URL}/query", json=payload)
        result = response.json()
        
        print(f"\nResponse: {result.get('response', 'No response')}")
        
        if 'sources' in result and result['sources']:
            print("\nSources:")
            for source in result.get('sources', []):
                print(f"- {source.get('source')} (Department: {source.get('department')}, Relevance: {source.get('relevance', 0):.2f})")
                if 'content' in source:
                    # Print a preview of the content
                    content_preview = source.get('content', '')
                    if len(content_preview) > 100:
                        content_preview = content_preview[:100] + "..."
                    print(f"  Preview: {content_preview}")
        else:
            print("\nNo sources provided in response")
        
        if 'metrics' in result:
            print(f"\nMetrics:")
            metrics = result.get('metrics', {})
            print(f"- Retrieval time: {metrics.get('retrieval_time', 0):.4f}s")
            print(f"- Generation time: {metrics.get('generation_time', 0):.4f}s")
            print(f"- Context relevance: {metrics.get('context_relevance', 0):.4f}")
        
        return response.status_code == 200
    except Exception as e:
        print(f"Error testing query endpoint: {e}")
        return False

def create_test_documents():
    """Create test documents in the data directory"""
    try:
        print("\nCreating test documents...")
        script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "create_test_documents.py")
        
        if os.path.exists(script_path):
            import subprocess
            result = subprocess.run([sys.executable, script_path], check=True)
            return result.returncode == 0
        else:
            print(f"Error: Test document creation script not found at {script_path}")
            return False
    except Exception as e:
        print(f"Error creating test documents: {e}")
        return False

def upload_document(file_path, department="general"):
    """Upload a document"""
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return False
    
    filename = os.path.basename(file_path)
    metadata = {
        "department": department,
        "tags": ["test", "upload"],
        "description": "Test document upload"
    }
    
    try:
        print(f"\nUploading document: {filename} to department: {department}")
        with open(file_path, 'rb') as f:
            files = {'document': (filename, f)}
            data = {'metadata': json.dumps(metadata)}
            
            response = requests.post(f"{RAG_URL}/documents", files=files, data=data)
            
        result = response.json()
        print(f"Upload result: {result}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error uploading document: {e}")
        return False

def rebuild_indexes():
    """Trigger index rebuild"""
    try:
        print("\nTriggering index rebuild...")
        response = requests.post(f"{RAG_URL}/rebuild")
        result = response.json()
        print(f"Rebuild result: {result}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error triggering index rebuild: {e}")
        return False

def run_test_queries():
    """Run a set of predefined test queries"""
    test_queries = [
        # General questions
        ("What are the main port regulations?", None),
        ("Tell me about port safety regulations", None),
        ("What are the commercial tariffs for containers?", "commercial"),
        ("What are the technical specifications for cranes?", "technical"),
        ("What documents are required for vessels?", "regulatory"),
        
        # Specific questions that should match content in the test documents
        ("quels sont les regles d'accostage au port de casablanca?", "regulatory"),
        ("What are the storage charges for containers?", "commercial"),
        ("How often should port equipment be maintained?", "technical"),
        ("What safety gear is required in the port?", "regulatory"),
        ("What are the pilotage fees for vessels?", "commercial"),
        ("resume moi l'article 3", "regulatory")
    ]
    
    success_count = 0
    for query, department in test_queries:
        if test_query(query, department):
            success_count += 1
        print("\n" + "-"*50)  # separator
        
    print(f"\nCompleted {success_count}/{len(test_queries)} test queries successfully")
    return success_count == len(test_queries)

def main():
    parser = argparse.ArgumentParser(description="Test the RAG service")
    parser.add_argument("--create-docs", action="store_true", help="Create test documents")
    parser.add_argument("--upload", type=str, help="Upload a specific document file")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild the indexes")
    parser.add_argument("--query", type=str, help="Run a specific query")
    parser.add_argument("--department", type=str, help="Department for query or upload")
    parser.add_argument("--all", action="store_true", help="Run all tests")
    args = parser.parse_args()

    print("=== ANP RAG Service Testing ===")
    
    # Wait for service to be ready
    print("Waiting for service to initialize...")
    time.sleep(2)
    
    # Start with health check
    if not test_health():
        print("❌ Health check failed. Make sure the service is running.")
        return False
    print("✅ Health check passed")
    
    # List documents and departments
    list_documents()
    list_departments()
    
    # Create test documents if requested
    if args.create_docs or args.all:
        if create_test_documents():
            print("✅ Test documents created")
        else:
            print("❌ Failed to create test documents")
    
    # Upload specific document if requested
    if args.upload:
        department = args.department or "general"
        if upload_document(args.upload, department):
            print(f"✅ Document {args.upload} uploaded to {department}")
        else:
            print(f"❌ Failed to upload document {args.upload}")
    
    # Rebuild indexes if requested
    if args.rebuild or args.all:
        if rebuild_indexes():
            print("✅ Index rebuild triggered")
            print("Waiting for rebuild to complete...")
            time.sleep(5)  # Give time for the background rebuild
        else:
            print("❌ Failed to trigger index rebuild")
    
    # Run a specific query if requested
    if args.query:
        department = args.department
        if test_query(args.query, department):
            print("✅ Query completed successfully")
        else:
            print("❌ Query failed")
    
    # Run all test queries if requested
    if args.all:
        print("\n=== Running Test Queries ===")
        # Wait a bit for any document processing to complete
        print("Waiting for document processing to complete...")
        time.sleep(5)
        run_test_queries()
    
    print("\n=== Testing Complete ===")
    return True

if __name__ == "__main__":
    main()