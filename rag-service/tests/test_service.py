# rag-service/test_service.py
import requests
import json
import os
import sys
import time
from pprint import pprint

# Service URL
RAG_URL = "http://localhost:8001/api"

def test_health():
    """Test the health endpoint"""
    try:
        response = requests.get(f"{RAG_URL}/health")
        pprint(response.json())
        return response.status_code == 200
    except Exception as e:
        print(f"Error testing health endpoint: {e}")
        return False

def test_query(query_text):
    """Test the query endpoint"""
    payload = {
        "query": query_text,
        "max_results": 5,
        "include_metadata": True
    }
    
    try:
        response = requests.post(f"{RAG_URL}/query", json=payload)
        result = response.json()
        
        print(f"Query: {query_text}")
        print(f"Response: {result.get('response', 'No response')}")
        
        print("\nSources:")
        for source in result.get('sources', []):
            print(f"- {source.get('source')} (Department: {source.get('department')}, Relevance: {source.get('relevance', 0):.2f})")
        
        print(f"\nMetrics: {result.get('metrics', {})}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error testing query endpoint: {e}")
        return False

def list_documents():
    """List indexed documents"""
    try:
        response = requests.get(f"{RAG_URL}/documents")
        docs = response.json()
        
        print(f"Found {len(docs)} documents:")
        for doc in docs:
            print(f"- {doc.get('filename')} (Department: {doc.get('department')}, Added: {doc.get('added_date')})")
        
        return response.status_code == 200
    except Exception as e:
        print(f"Error listing documents: {e}")
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

def create_test_document():
    """Create a simple test document if none is provided"""
    test_file = "test_document.txt"
    with open(test_file, "w") as f:
        f.write("""# Port Regulations and Guidelines

## General Port Information
The port operates 24/7 with specific loading and unloading schedules.
All vessels must register at least 24 hours before arrival.

## Safety Regulations
1. All personnel must wear appropriate safety gear in designated areas.
2. Emergency procedures must be clearly displayed on all vessels.
3. Regular safety drills are conducted monthly.

## Environmental Policies
The port follows strict environmental guidelines to minimize pollution.
Waste disposal must follow the established protocols.

## Contact Information
Port Authority Office: +123-456-7890
Emergency Contact: +123-456-7899
""")
    print(f"Created test document: {test_file}")
    return test_file

def main():
    """Run all tests"""
    print("=== Testing RAG Service ===")
    
    # Wait a moment for the service to fully start
    print("Waiting for service to initialize...")
    time.sleep(2)
    
    print("\n1. Health Check:")
    if not test_health():
        print("❌ Health check failed. Make sure the service is running.")
        sys.exit(1)
    else:
        print("✅ Health check passed")
    
    print("\n2. Document Listing:")
    list_documents()
    
    print("\n3. Query Test (without documents):")
    test_query("What are the main port regulations in Morocco?")
    
    # Create or use test document
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        print(f"\n4. Document Upload Test (user-provided: {file_path}):")
    else:
        file_path = create_test_document()
        print(f"\n4. Document Upload Test (auto-generated: {file_path}):")
    
    if upload_document(file_path):
        print("✅ Document upload successful")
    else:
        print("❌ Document upload failed")
    
    print("\n5. Document Listing (after upload):")
    list_documents()
    
    # Wait for document processing
    print("\nWaiting for document processing...")
    time.sleep(5)
    
    print("\n6. Query Test (with documents):")
    test_query("Learn about common port procedures?")
    
    print("\n=== Test Complete ===")

if __name__ == "__main__":
    main()