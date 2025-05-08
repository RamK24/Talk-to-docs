import logging
from typing import List, Dict, Any, Union
import torch
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import json
from app.services.vector_store import SemanticEmbedder, ChromaDB, LexicalEmbedder
from app.core.config import settings
from FlagEmbedding import FlagLLMReranker

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Retriever:
    """Class for retrieving relevant documents using hybrid search and reranking."""

    def __init__(self):
        """Initialize the retriever components."""
        self.semantic_embedder = None
        self.lexical_embedder = None
        self.chroma_client = None
        self.reranker = None
        self._initialize_components()
        self._initialize_reranker()

    def _initialize_components(self):
        """Initialize all components for retrieval and reranking."""
        self._initialize_semantic()
        self._initialize_lexical()
        self._initialize_reranker()

    def _initialize_semantic(self):
        """Initialize semantic retrieval components if not already initialized."""
        if self.semantic_embedder is None:
            logger.info("Initializing semantic embedder")
            self.semantic_embedder = SemanticEmbedder()
            self.chroma_client = ChromaDB()

    def _initialize_lexical(self):
        """Initialize lexical retrieval components if not already initialized."""
        if self.lexical_embedder is None:
            logger.info("Initializing lexical embedder")
            self.lexical_embedder = LexicalEmbedder()
            self.lexical_embedder.load_index()  # Load the existing BM25 index

    def _initialize_reranker(self):
        """Initialize the reranker if not already initialized."""
        if self.reranker is None:
            logger.info("Initializing FlagReranker")
            try:
                self.reranker = FlagLLMReranker('BAAI/bge-reranker-v2-gemma', use_fp16=True)
                logger.info("FlagReranker initialized successfully")
            except Exception as e:
                logger.error(f"Error initializing FlagReranker: {str(e)}")
                self.reranker = None

    def retrieve_hybrid_and_rerank(self,
                                   queries: List[str],
                                   initial_top_k: int = 100,
                                   final_top_k: int = 5,
                                   alpha: float = 0.5,
                                   normalize_scores: bool = True) -> List[List[Dict[str, Any]]]:
        """
        Retrieve documents using hybrid search and rerank them using cross-encoder.
        Args:
            queries (List[str]): List of query strings
            initial_top_k (int): Number of documents to retrieve initially with hybrid search
            final_top_k (int): Number of documents to return after reranking
            alpha (float): Weight for hybrid retrieval (semantic vs lexical)
            normalize_scores (bool): Whether to normalize reranker scores to [0,1] range

        Returns:
            List[List[Dict[str, Any]]]: Lists of reranked documents for each query
        """

        if not queries:
            logger.warning("No queries provided for retrieval")
            return []

        # Step 1: Hybrid Retrieval
        logger.info(f"Starting hybrid retrieval for {len(queries)} queries")
        hybrid_results = self._hybrid_retrieve(queries, initial_top_k, alpha)

        # Step 2: Reranking
        logger.info(f"Starting reranking for {len(queries)} queries")
        reranked_results = self._rerank_documents(queries, hybrid_results, final_top_k, normalize_scores)

        return reranked_results

    def _hybrid_retrieve(self, queries: List[str], top_k: int = 100, alpha: float = 0.5) -> List[List[Dict[str, Any]]]:
        """
        Perform hybrid retrieval by combining semantic and lexical search results.

        Args:
            queries (List[str]): List of query strings
            top_k (int): Total number of documents to retrieve per query
            alpha (float): Weight for semantic scores (1-alpha for lexical), between 0 and 1

        Returns:
            List[List[Dict[str, Any]]]: List of document lists for each query
        """
        # Run both retrievers in parallel
        # with ThreadPoolExecutor(max_workers=2) as executor:
        #     semantic_future = executor.submit(self._retrieve_semantic, queries, top_k)
        #     lexical_future = executor.submit(self._retrieve_lexical, queries, top_k)
        #
        #     semantic_results = semantic_future.result()
        #     lexical_results = lexical_future.result()
        semantic_results = self._retrieve_semantic(queries, top_k)
        lexical_results = self._retrieve_lexical(queries, top_k)

        combined_results = []

        for i, query in enumerate(queries):
            semantic_docs = semantic_results[i] if i < len(semantic_results) else []
            lexical_docs = lexical_results[i] if i < len(lexical_results) else []

            semantic_scores = {doc["id"]: doc["score"] for doc in semantic_docs}
            lexical_scores = {doc["id"]: doc["score"] for doc in lexical_docs}

            all_doc_ids = set(semantic_scores.keys()) | set(lexical_scores.keys())

            if semantic_scores:
                max_semantic = max(semantic_scores.values())
                min_semantic = min(semantic_scores.values())
                semantic_range = max_semantic - min_semantic if max_semantic > min_semantic else 1.0
            else:
                semantic_range = 1.0
                min_semantic = 0.0

            if lexical_scores:
                max_lexical = max(lexical_scores.values())
                min_lexical = min(lexical_scores.values())
                lexical_range = max_lexical - min_lexical if max_lexical > min_lexical else 1.0
            else:
                lexical_range = 1.0
                min_lexical = 0.0

            combined_docs = []
            for doc_id in all_doc_ids:
                semantic_doc = next((doc for doc in semantic_docs if doc["id"] == doc_id), None)
                lexical_doc = next((doc for doc in lexical_docs if doc["id"] == doc_id), None)

                doc = semantic_doc if semantic_doc else lexical_doc

                # Normalize and combine scores
                semantic_score = 0.0
                if doc_id in semantic_scores:
                    semantic_score = (semantic_scores[
                                          doc_id] - min_semantic) / semantic_range if semantic_range > 0 else 0.5

                lexical_score = 0.0
                if doc_id in lexical_scores:
                    lexical_score = (lexical_scores[doc_id] - min_lexical) / lexical_range if lexical_range > 0 else 0.5

                # Combined score with alpha weighting
                combined_score = alpha * semantic_score + (1 - alpha) * lexical_score

                # Create combined document
                combined_doc = dict(doc)  # Copy the document
                combined_doc["score"] = combined_score
                combined_doc["semantic_score"] = semantic_score if doc_id in semantic_scores else None
                combined_doc["lexical_score"] = lexical_score if doc_id in lexical_scores else None

                combined_docs.append(combined_doc)

            # Sort by combined score and take top_k
            combined_docs.sort(key=lambda x: x["score"], reverse=True)
            combined_docs = combined_docs[:top_k]

            combined_results.append(combined_docs)

        logger.info(f"Completed hybrid retrieval for {len(queries)} queries")
        return combined_results

    def _retrieve_semantic(self, queries: List[str], top_k: int = 100) -> List[List[Dict[str, Any]]]:
        """Internal semantic retrieval implementation."""
        if not self.semantic_embedder or not self.chroma_client:
            logger.error("Semantic embedder not initialized")
            return [[] for _ in queries]

        try:
            # Create detailed instructions for each query
            instructed_queries = [
                self.semantic_embedder.get_detailed_instruct(query)
                for query in queries
            ]

            # Generate embeddings for the queries
            query_embeddings = self.semantic_embedder.get_embeddings(instructed_queries)

            # Perform batch retrieval
            query_results = self.chroma_client.collection.query(
                query_embeddings=query_embeddings,
                n_results=top_k,
                include=["documents", "metadatas", "distances"]
            )

            results = []
            for i in range(len(queries)):
                documents = []
                for j in range(len(query_results['ids'][i])):
                    doc_id = query_results['ids'][i][j]
                    document = {
                        "id": doc_id,
                        "content": query_results['documents'][i][j],
                        "metadata": query_results['metadatas'][i][j],
                        "score": 1.0 - query_results['distances'][i][j]
                    }
                    documents.append(document)

                results.append(documents)

            return results

        except Exception as e:
            logger.error(f"Error in semantic retrieval: {str(e)}")
            return [[] for _ in queries]

    def _retrieve_lexical(self, queries: List[str], top_k: int = 100) -> List[List[Dict[str, Any]]]:
        """Internal lexical retrieval implementation."""
        if not self.lexical_embedder or not self.lexical_embedder.retriever:
            logger.error("Lexical embedder not initialized")
            return [[] for _ in queries]

        try:
            results = []
            for query in queries:
                retrieved_docs = self.lexical_embedder.retriever.get_relevant_documents(query, k=top_k)

                # Format the results
                documents = []
                for doc in retrieved_docs:
                    document = {
                        "id": doc.metadata.get("id", "unknown"),
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                        "score": doc.metadata.get("score", 1.0)  # BM25 score if available
                    }
                    documents.append(document)

                results.append(documents)

            return results

        except Exception as e:
            logger.error(f"Error in lexical retrieval: {str(e)}")
            return [[] for _ in queries]

    def _rerank_documents(self,
                          queries: List[str],
                          doc_lists: List[List[Dict[str, Any]]],
                          top_k: int = 5,
                          normalize: bool = True) -> List[List[Dict[str, Any]]]:
        """
        Rerank retrieved documents using the cross-encoder reranker.

        Args:
            queries (List[str]): List of query strings
            doc_lists (List[List[Dict[str, Any]]]): List of document lists for each query
            top_k (int): Number of top documents to return after reranking
            normalize (bool): Whether to normalize scores to [0,1] range using sigmoid

        Returns:
            List[List[Dict[str, Any]]]: List of reranked document lists
        """
        if not self.reranker:
            logger.error("Reranker not initialized, returning original results")
            return doc_lists

        if len(queries) != len(doc_lists):
            logger.error(
                f"Number of queries ({len(queries)}) doesn't match number of document lists ({len(doc_lists)})")
            return doc_lists

        reranked_results = []

        for i, (query, docs) in enumerate(zip(queries, doc_lists)):
            if not docs:
                reranked_results.append([])
                continue

            try:
                # Create pairs of [query, document] for all documents
                pairs = [[query, doc["content"]] for doc in docs]

                # Get scores using FlagReranker
                logger.info(f"Reranking {len(pairs)} documents for query {i + 1}/{len(queries)}")
                scores = self.reranker.compute_score(pairs, normalize=normalize)

                # Create reranked documents with new scores
                reranked_docs = []
                for j, doc in enumerate(docs):
                    reranked_doc = dict(doc)  # Create a copy
                    reranked_doc["hybrid_score"] = doc["score"]  # Store original hybrid score
                    reranked_doc["score"] = float(scores[j])  # Reranker score becomes the primary score
                    reranked_docs.append(reranked_doc)

                # Sort by new scores
                reranked_docs.sort(key=lambda x: x["score"], reverse=True)

                # Take top_k
                reranked_docs = reranked_docs[:top_k]

                reranked_results.append(reranked_docs)
                logger.debug(f"Reranked {len(reranked_docs)} documents for query: {query[:50]}...")

            except Exception as e:
                logger.error(f"Error in reranking for query {i + 1}: {str(e)}")
                # Take top_k from original docs if reranking fails
                sorted_docs = sorted(docs, key=lambda x: x["score"], reverse=True)[:top_k]
                reranked_results.append(sorted_docs)

        logger.info(f"Completed reranking for {len(queries)} queries")
        return reranked_results

    def cleanup(self):
        """Clean up resources to prevent memory leaks."""
        if self.semantic_embedder:
            self.semantic_embedder.cleanup()
            self.semantic_embedder = None

        if self.lexical_embedder:
            self.lexical_embedder = None

        if self.chroma_client:
            self.chroma_client = None

        self.reranker = None

        logger.info("Retriever resources cleaned up")


_retriever_instance = None


def get_retriever():
    """Get or create a retriever instance."""
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = Retriever()
    return _retriever_instance


# Convenient external function
def retrieve_hybrid_and_rerank(queries: List[str],
                               initial_top_k: int = 100,
                               final_top_k: int = 5,
                               alpha: float = 0.5) -> List[List[Dict[str, Any]]]:
    """
    Retrieve documents using hybrid search and rerank them.

    Args:
        queries (List[str]): List of query strings
        initial_top_k (int): Number of documents to retrieve initially with hybrid search
        final_top_k (int): Number of documents to return after reranking
        alpha (float): Weight for hybrid retrieval (semantic vs lexical)

    Returns:
        List[List[Dict[str, Any]]]: Lists of reranked documents for each query
    """
    retriever = get_retriever()
    return retriever.retrieve_hybrid_and_rerank(queries, initial_top_k, final_top_k, alpha)


# For a single query convenience
def retrieve_and_rerank_single(query: str,
                               initial_top_k: int = 100,
                               final_top_k: int = 5,
                               alpha: float = 0.5) -> List[Dict[str, Any]]:
    """
    Retrieve documents for a single query using hybrid search and rerank them.

    Args:
        query (str): Query string
        initial_top_k (int): Number of documents to retrieve initially with hybrid search
        final_top_k (int): Number of documents to return after reranking
        alpha (float): Weight for hybrid retrieval (semantic vs lexical)

    Returns:
        List[Dict[str, Any]]: List of reranked documents
    """
    retriever = get_retriever()
    results = retriever.retrieve_hybrid_and_rerank([query], initial_top_k, final_top_k, alpha)
    return results[0] if results else []


# queries = ['what are the rules for leave', 'how many signatures for a check']
# docs = retrieve_hybrid_and_rerank(queries, 100, 20,0.5)
# import pickle
# with open('data.pkl', 'wb') as f:
#     pickle.dump(docs, f)

