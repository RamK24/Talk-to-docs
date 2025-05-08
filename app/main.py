from fastapi import FastAPI, UploadFile, File, HTTPException, Body, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import uvicorn
import subprocess
import sys
import os
from datetime import datetime
import uuid
import logging

# Import from your application modules
from app.services.generate import generate_response
from app.core.config import settings

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# Create FastAPI app
app = FastAPI(
    title="Document Query API",
    description="API for uploading documents and querying their content",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Modify this in production to specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define storage directory for uploaded files
UPLOAD_DIR = os.environ.get("UPLOAD_DIR", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# In-memory storage for uploaded document metadata
# In a production environment, this would be a database
document_store = {}


# Data models
class QueryRequest(BaseModel):
    query: str
    document_ids: Optional[List[str]] = None


class QueryResponse(BaseModel):
    response: str
    document_references: List[Dict[str, Any]]
    query_id: str


class HealthResponse(BaseModel):
    status: str
    timestamp: str


# Helper functions
def store_document(file: UploadFile) -> str:
    """
    Store an uploaded document and return its ID.
    """
    try:
        # Generate a unique ID for the document
        doc_id = str(uuid.uuid4())

        # Create a filename with the original name
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        filename = f"{timestamp}_{doc_id}_{file.filename}"
        filepath = os.path.join(UPLOAD_DIR, filename)

        # Save the file
        with open(filepath, "wb") as f:
            content = file.file.read()
            f.write(content)

        # Store metadata
        document_store[doc_id] = {
            "id": doc_id,
            "filename": file.filename,
            "filepath": filepath,
            "content_type": file.content_type,
            "size": os.path.getsize(filepath),
            "upload_time": datetime.now().isoformat(),
        }

        return doc_id

    except Exception as e:
        logger.error(f"Error storing document: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error storing document: {str(e)}",
        )


# Endpoints
@app.post("/upload", status_code=status.HTTP_201_CREATED)
async def upload_documents(files: List[UploadFile] = File(...)):
    """
    Upload one or more documents to be processed and queried later.
    """
    if not files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No files provided",
        )

    result = []
    for file in files:
        doc_id = store_document(file)
        result.append({
            "document_id": doc_id,
            "filename": file.filename,
        })

    return {
        "message": f"Successfully uploaded {len(files)} document(s)",
        "documents": result,
    }


@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """
    Ask a question about the uploaded documents.
    """
    if not request.query.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query cannot be empty",
        )

    # Filter documents if document_ids are provided
    docs_to_query = []
    if request.document_ids:
        for doc_id in request.document_ids:
            if doc_id not in document_store:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Document with ID {doc_id} not found",
                )
            docs_to_query.append(document_store[doc_id])
    else:
        # If no document_ids provided, use all documents
        docs_to_query = list(document_store.values())

    if not docs_to_query:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No documents available to query",
        )

    try:
        # Call the generate function from the document service
        response_data = generate_response([request.query], docs_to_query)

        return {
            "response": response_data.get("answer", ""),
            "document_references": response_data.get("references", []),
            "query_id": str(uuid.uuid4()),
        }
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing query: {str(e)}",
        )


@app.post("/process")
async def process_uploaded_documents():
    try:
        result = subprocess.run(
            [sys.executable, "scripts/process_documents.py"],
            check=True,
            capture_output=True,
            text=True
        )
        logger.info(result.stdout)
        return {"message": "Processing started successfully", "output": result.stdout}
    except subprocess.CalledProcessError as e:
        logger.error(f"Processing failed: {e.stderr}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {e.stderr}")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Simple status check to verify the API is running.
    """
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
    }


if __name__ == "__main__":
    # Get port from environment variable or use default
    port = int(os.environ.get("PORT", 8000))

    # Run the application
    uvicorn.run(
        "app.main:app",  # Since this file is app/main.py
        host="0.0.0.0",
        port=port,
        reload=True,  # Set to False in production
    )