import os
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Document Retrieval API"

    # Directories
    UPLOAD_DIR: str = "./uploads"
    PROCESSED_CHUNKS_DIR: str = "./processed_chunks"
    EMBEDDING_STORE_DIR: str = "./embedding_store"
    BM25_STORE_DIR: str = "./lexical_store"

    # Document processing settings
    DEFAULT_CHUNK_SIZE: int = 1500
    DEFAULT_OVERLAP: float = 0.2
    DEFAULT_BATCH_SIZE: int = 16


    # Embedding model
    EMBEDDING_MODEL: str = "Linq-AI-Research/Linq-Embed-Mistral"

    # Create necessary directories
    def create_directories(self):
        os.makedirs(self.UPLOAD_DIR, exist_ok=True)
        os.makedirs(self.PROCESSED_CHUNKS_DIR, exist_ok=True)
        os.makedirs(self.EMBEDDING_STORE_DIR, exist_ok=True)


settings = Settings()