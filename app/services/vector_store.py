import torch
import torch.nn.functional as F
from more_itertools import chunked
import chromadb
import json
import os
import logging
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
from app.core.config import settings
import pickle
from typing import List, Optional
from langchain_community.retrievers import BM25Retriever
from langchain.schema import Document


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SemanticEmbedder:
    """Class for creating semantic embeddings from text."""

    def __init__(self, model_name=settings.EMBEDDING_MODEL):
        """Initialize semantic embedder with the specified model."""
        self.model_name = model_name
        self.tokenizer = None
        self.model = None

    def _load_models(self):
        """Load models only when needed to save resources."""
        if self.tokenizer is None or self.model is None:
            logger.info(f"Loading embedding model: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name, device_map='auto')

    @staticmethod
    def last_token_pool(last_hidden_states, attention_mask):
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

    @staticmethod
    def get_detailed_instruct(query: str,
                              task_description='Given a question, retrieve passages that answer the question') -> str:
        """Create a detailed instruction for the embedding model."""
        return f'Instruct: {task_description}\nQuery: {query}'

    def get_embeddings(self, texts, batch_size=8, max_length=4096):
        """
        Get embeddings for a list of texts.

        Args:
            texts (list): List of text strings
            batch_size (int): Batch size for processing
            max_length (int): Maximum token length

        Returns:
            list: List of embeddings
        """
        self._load_models()  # Ensure models are loaded

        if not texts:
            logger.warning("No texts provided for embedding")
            return []

        embeddings_all = []

        try:
            with torch.no_grad():
                for i, passages in enumerate(chunked(texts, batch_size)):
                    batch_dict = self.tokenizer(
                        passages,
                        max_length=max_length,
                        padding=True,
                        truncation=True,
                        return_tensors="pt"
                    ).to(self.model.device)

                    outputs = self.model(**batch_dict)
                    embeddings = self.last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
                    embeddings = F.normalize(embeddings, p=2, dim=1)
                    embeddings = embeddings.detach().cpu().tolist()
                    embeddings_all.extend(embeddings)

                    logger.debug(f"Processed embedding batch {i + 1}")

            logger.info(f"Generated {len(embeddings_all)} embeddings")
            return embeddings_all
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            return []

    def cleanup(self):
        """Clean up resources to prevent memory leaks."""
        if self.model:
            del self.model
            self.model = None
        if self.tokenizer:
            del self.tokenizer
            self.tokenizer = None
        torch.cuda.empty_cache()
        logger.info("Semantic embedder resources cleaned up")


class ChromaDB:
    """Interface for storing and retrieving embeddings using ChromaDB."""

    def __init__(self, persist_dir=settings.EMBEDDING_STORE_DIR, collection_name="docs"):
        """Initialize ChromaDB client with the specified parameters."""
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.client = None
        self.collection = None
        self._initialize_client()

    def _initialize_client(self):
        """Initialize ChromaDB client and collection."""
        try:
            os.makedirs(self.persist_dir, exist_ok=True)
            self.client = chromadb.PersistentClient(path=self.persist_dir)
            self.collection = self.client.get_or_create_collection(name=self.collection_name)
            logger.info(f"ChromaDB initialized with collection '{self.collection_name}'")
        except Exception as e:
            logger.error(f"Error initializing ChromaDB: {str(e)}")
            self.client = None
            self.collection = None

    def store_embeddings(self, json_file_path, embedding_obj=None):
        """
        Store embeddings from a JSON file in ChromaDB.

        Args:
            json_file_path (str): Path to the JSON file with document chunks
            embedding_obj (SemanticEmbedder): Embedder object or None to create a new one

        Returns:
            bool: True if storage was successful, False otherwise
        """
        if not self.collection:
            logger.error("ChromaDB collection not initialized")
            return False

        try:
            # Create embedder if not provided
            cleanup_embedder = False
            if embedding_obj is None:
                embedding_obj = SemanticEmbedder()
                cleanup_embedder = True

            # Load document data
            with open(json_file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if not data:
                logger.warning(f"No data found in {json_file_path}")
                return False

            texts = [item["content"] for item in data]
            ids = [item["id"] for item in data]
            metadatas = [item["metadata"] for item in data]

            # Generate embeddings
            logger.info(f"Generating embeddings for {len(texts)} documents from {json_file_path}")
            embeddings = embedding_obj.get_embeddings(texts)

            if len(embeddings) != len(texts):
                logger.error(f"Embedding count mismatch: {len(embeddings)} embeddings for {len(texts)} texts")
                return False

            # Add to collection
            self.collection.add(
                documents=texts,
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas
            )
            doc_count = self.collection.count()
            logger.info(f"Chroma collection now has {doc_count} documents.")

            # Clean up if we created the embedder
            if cleanup_embedder:
                embedding_obj.cleanup()


            return True
        except Exception as e:
            logger.error(f"Error storing embeddings from {json_file_path}: {str(e)}")
            return False


class LexicalEmbedder:
    def __init__(self, persist_dir=settings.BM25_STORE_DIR, data_dir="processed_chunks"):
        self.persist_dir = persist_dir
        self.data_dir = data_dir
        self.retriever: Optional[BM25Retriever] = None
        os.makedirs(self.persist_dir, exist_ok=True)

    def build_index_from_directory(self) -> bool:
        documents = []

        try:
            json_files = [
                f for f in os.listdir(self.data_dir)
                if f.endswith(".json") and not f.startswith(".")
            ]

            if not json_files:
                logger.warning(f"No JSON files found in {self.data_dir}")
                return False

            logger.info(f"Building BM25 index from {len(json_files)} files")

            for json_file in json_files:
                path = os.path.join(self.data_dir, json_file)
                with open(path, "r", encoding="utf-8") as f:
                    chunks = json.load(f)

                for chunk in chunks:
                    documents.append(
                        Document(
                            page_content=chunk["content"],
                            metadata=chunk.get("metadata", {})
                        )
                    )

            if not documents:
                logger.warning("No valid chunks found in files")
                return False

            self.retriever = BM25Retriever.from_documents(documents)
            logger.info("BM25 index built successfully")

            self._save_index()
            return True

        except Exception as e:
            logger.error(f"Error building BM25 index: {str(e)}")
            return False

    def _save_index(self):
        if self.retriever is None:
            logger.error("No retriever to save")
            return

        index_file = os.path.join(self.persist_dir, "bm25_retriever.pkl")
        with open(index_file, "wb") as f:
            pickle.dump(self.retriever, f)

        logger.info(f"BM25 index saved to {index_file}")

    def load_index(self) -> bool:
        index_file = os.path.join(self.persist_dir, "bm25_retriever.pkl")

        if not os.path.exists(index_file):
            logger.error(f"BM25 index file not found at {index_file}")
            return False

        try:
            with open(index_file, "rb") as f:
                self.retriever = pickle.load(f)

            logger.info(f"BM25 index loaded from {index_file}")
            return True
        except Exception as e:
            logger.error(f"Failed to load BM25 index: {str(e)}")
            return False

