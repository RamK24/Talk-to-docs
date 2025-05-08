import torch
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from app.services.retrieve import retrieve_hybrid_and_rerank
from transformers import AutoModelForCausalLM, AutoTokenizer
from app.services.retrieve import Retriever

# Global variables for model and tokenizer
model = None
tokenizer = None


def format_context(retrieved_docs):
    """
    Format retrieved documents into readable context strings.

    Args:
        retrieved_docs (list): List of retrieved document chunks grouped by document

    Returns:
        list: List of formatted context strings, one per document set
    """
    contexts = []
    for doc in retrieved_docs:
        curr_context = ''
        for chunk in doc:
            curr_context += chunk['content'] + '\n'
        contexts.append(curr_context)
    return contexts


def craft_prompt(prompts):
    """
    Create properly formatted prompts with system message for the language model.

    Args:
        prompts (list): List of user prompt strings

    Returns:
        list: List of formatted message sequences ready for the model
    """
    system_message = """You are an assistant providing accurate, helpful responses based on the retrieved documents. 
        When answering:
        1. Use only information from the provided documents
        2. If the answer is not in the documents, say so
        3. Keep responses clear, concise, and directly relevant to the query
        4. Cite specific document numbers that you used (e.g., "According to Document 1...")
        5. Do not include information not present in the documents"""

    prompts_ = [
        [{'role': "system", "content": system_message}, {'role': "user", "content": prompt}]
        for prompt in prompts
    ]
    return prompts_


def tokenize(messages):
    """
    Tokenize messages for the model.

    Args:
        messages (list): List of message sequences to tokenize

    Returns:
        dict: Tokenized inputs ready for model processing
    """
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )

    model_inputs = tokenizer(text, return_tensors="pt", padding=True).to(model.device)
    return model_inputs


def generate_response(queries, retrieve_k=40, rerank_k=20, hybrid_weight=0.6,
                      max_new_tokens=2000, temperature=0.7, top_p=0.8, top_k=20, min_p=0):
    """
    Generate responses for queries using RAG (Retrieval-Augmented Generation).

    This function retrieves relevant documents, formats them as context,
    creates prompts with the context and queries, and generates responses
    using the language model.

    Args:
        queries (list): List of query strings
        retrieve_k (int, optional): Number of documents to retrieve initially. Defaults to 40.
        rerank_k (int, optional): Number of documents to keep after reranking. Defaults to 20.
        hybrid_weight (float, optional): Weight for hybrid search between vector and lexical. Defaults to 0.6.
        max_new_tokens (int, optional): Maximum number of tokens to generate. Defaults to 2000.
        temperature (float, optional): Sampling temperature. Defaults to 0.7.
        top_p (float, optional): Nucleus sampling parameter. Defaults to 0.8.
        top_k (int, optional): Top-k sampling parameter. Defaults to 20.
        min_p (float, optional): Minimum probability for token consideration. Defaults to 0.

    Returns:
        list: Generated responses for each query
    """
    # Retrieve relevant documents
    docs = retrieve_hybrid_and_rerank(queries, retrieve_k, rerank_k, hybrid_weight)

    # Format context from documents
    context = format_context(docs)

    # Create user prompts with context
    user_content = [f"""Here's some context:

    {context[i]}

    My question is: {queries[i]}""" for i in range(len(queries))]

    # Prepare prompts for the model
    prompts = craft_prompt(user_content)

    # Tokenize inputs
    model_inputs = tokenize(prompts)

    # Generate responses
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=max_new_tokens,
        top_p=top_p,
        top_k=top_k,
        min_p=min_p,
        temperature=temperature
    )

    # Extract and decode the generated text
    output_ids = generated_ids[:, len(model_inputs.input_ids[0]):].tolist()
    responses = tokenizer.batch_decode(output_ids, skip_special_tokens=True)

    return {'response': responses}


if __name__ == "__main__":
    retriever = Retriever()  # embedding model, vector and lexical dbs, reranker

    model_name = "Qwen/Qwen3-4B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    # Example query
    query = "What are the policies for parental leave?"
    response = generate_response([query])
    print("Answer:", response[0])

