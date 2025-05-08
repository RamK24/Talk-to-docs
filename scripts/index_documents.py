import os
import sys
import logging
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from app.services.document_processor import save_chunks
from app.services.vector_store import SemanticEmbedder, ChromaDB, LexicalEmbedder
from app.core.config import settings

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


UPLOAD_DIR = settings.UPLOAD_DIR
PROCESSED_DIR = settings.PROCESSED_CHUNKS_DIR
SUPPORTED_EXTENSIONS = [".pdf"]


def is_processed(file_name):
    """Check if a file has already been processed (chunked)."""
    base = os.path.splitext(file_name)[0]
    return os.path.exists(os.path.join(PROCESSED_DIR, f"{base}.json"))


def process_file(file_name):
    """Process a single file sequentially: chunk → embed → store."""
    try:
        if not file_name.lower().endswith(tuple(SUPPORTED_EXTENSIONS)):
            logger.warning(f"Unsupported file type: {file_name}")
            return False

        if is_processed(file_name):
            logger.info(f"Skipping already processed file: {file_name}")
            return True

        file_path = os.path.join(UPLOAD_DIR, file_name)
        logger.info(f"Processing file: {file_name}")

        json_path = save_chunks(file_name)
        if not json_path:
            logger.error(f"Chunking failed for: {file_name}")
            return False

        embedder = SemanticEmbedder()
        db = ChromaDB()
        success = db.store_embeddings(json_path, embedding_obj=embedder)
        embedder.cleanup()

        return success

    except Exception as e:
        logger.error(f"Failed to process {file_name}: {str(e)}")
        return False


def index_lexical():
    """Build a single BM25 index from all JSON files."""
    try:
        logger.info("Building BM25 index from processed chunks...")
        bm25 = LexicalEmbedder(data_dir=PROCESSED_DIR, persist_dir=settings.BM25_STORE_DIR)
        success = bm25.build_index_from_directory()

        if success:
            logger.info("BM25 index created and saved successfully.")
        else:
            logger.warning("BM25 index creation failed or no data found.")

    except Exception as e:
        logger.error(f"Error building BM25 index: {str(e)}")


def main():
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    doc_files = [
        f for f in os.listdir(UPLOAD_DIR)
        if f.lower().endswith(tuple(SUPPORTED_EXTENSIONS))
    ]

    logger.info(f"Found {len(doc_files)} documents to process.")

    success_count = 0
    for file_name in doc_files:
        if process_file(file_name):
            success_count += 1

    logger.info(f"Processing complete: {success_count}/{len(doc_files)} files processed successfully.")

    index_lexical()


if __name__ == "__main__":
    main()
