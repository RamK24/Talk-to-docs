import logging
import fitz
import argparse
import gc
import os
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
from itertools import chain
import torch
from langchain.text_splitter import RecursiveCharacterTextSplitter
from app.core.config import settings


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DocumentProcessor:

    def __init__(self, dir_path=settings.UPLOAD_DIR, chunk_size=settings.DEFAULT_CHUNK_SIZE,
                 overlap=settings.DEFAULT_OVERLAP):
        """Initialize document processor with specified parameters."""
        self.dir_path = dir_path
        self.chunk_size = chunk_size
        self.overlap_ratio = overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=int(chunk_size * overlap)
        )

        self.tokenizer = None
        self.model = None

    def _load_models(self):
        """Load tokenizer and model only when needed."""
        if self.tokenizer is None or self.model is None:
            logger.info("Loading language model and tokenizer")
            model_id = 'Qwen/Qwen3-4B'
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            self.tokenizer.padding_side = 'left'
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map='auto'
            )

    def extract_information(self, file_path):
        """
        Identify the type of document using multiple methods.

        Args:
            self.file_path (str): Path to the document file

        Returns:
            str: The identified file type ('pdf', 'txt', 'docx', or 'unknown')
        """
        file_path = os.path.join(self.dir_path, file_path)
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return

        _, file_extension = os.path.splitext(file_path.lower())

        if file_extension == '.pdf':
            return self.process_file(file_path)
        return

    def process_file(self, file_path):
        """
        Process PDF file to extract text and create chunks.

        Args:
            file_path (str): Path to the file relative to dir_path

        Returns:
            tuple: (pages, chunks) where pages are text content and chunks are split text
        """
        pages = []
        chunks_all = []
        full_path = os.path.join(self.dir_path, file_path)

        if not os.path.exists(full_path):
            logger.error(f"File not found: {full_path}")
            return None

        _, file_extension = os.path.splitext(full_path.lower())

        if file_extension == '.pdf':
            try:
                with fitz.open(full_path) as doc:
                    for i, page in enumerate(doc, start=1):
                        text = page.get_text()
                        pages.append(text)
                        chunks = self.create_chunks(text)
                        chunks_all.append(chunks)
                logger.info(
                    f"Successfully processed {file_path}: {len(pages)} pages, {sum(len(c) for c in chunks_all)} chunks")
                return pages, chunks_all
            except Exception as e:
                logger.error(f"Error processing PDF {file_path}: {str(e)}")
                return None
        else:
            logger.warning(f"Unsupported file format: {file_extension}")
            return None

    def create_chunks(self, text):
        """Split text into chunks using the configured text splitter."""
        chunks = self.splitter.split_text(text)
        chunks = [chunk.strip() for chunk in chunks]
        return chunks
    def craft_prompt(self, doc, chunk):
        """Create prompt for contextualizing a chunk within the document."""
        prompt = [{
            "role": "user",
            "content": f"""<document> 
            {doc} 
            </document> 
            Here is the chunk we want to situate within the whole document 
            <chunk> 
            {chunk} 
            </chunk> 
            Please give a short succinct context to situate this chunk within the overall document for the purposes of 
            improving search retrieval of the chunk. Answer only with the succinct context and nothing else. """
        }]
        return prompt

    def tokenize(self, messages):
        """Tokenize messages for the model."""
        self._load_models()  # Ensure models are loaded

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )

        model_inputs = self.tokenizer(text, return_tensors="pt", padding=True).to(self.model.device)
        return model_inputs

    def get_context(self, inputs):
        """Generate context for chunks using the language model."""

        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=2000,
            top_p=0.8,
            top_k=20,
            min_p=0,
            temperature=0.7
        )

        output_ids = generated_ids[:, len(inputs.input_ids[0]):].tolist()
        content = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        return content

    def process_chunks(self, file_path):
        prompts = []
        doc = ''
        pages, chunks = self.process_file(file_path)
        for i, chunk in enumerate(chunks):
            if i % 3 == 0 and i != 0:
                doc = pages[i]
            else:
                doc += pages[i]
            prompt = [self.craft_prompt(doc, chu) for chu in chunk]

            prompts.extend(prompt)
        return prompts, chunks


    def create_contextualised_chunks(self, file_path, batch_size=16):
        filename = file_path.split('/')[-1]
        prompts, chunks = self.process_chunks(file_path)
        chunks = list(chain.from_iterable(chunks))
        contextualized_chunks = []
        for i in range(0, len(prompts), batch_size):
            window_end = i+batch_size
            inst = prompts[i: window_end]
            chunk = chunks[i: window_end]
            inputs = self.tokenize(inst)
            contexts = self.get_context(inputs)
            contextualized_chunks.extend([
                {
                    'id': f"{filename}_{idx + i}",
                    'content': a + '\n' + b,
                    'metadata': {'filename': filename}
                }
                for idx, (a, b) in enumerate(zip(contexts, chunk))
            ])

        return contextualized_chunks

    def cleanup(self):
        """Clean up resources to prevent memory leaks."""
        if self.model:
            del self.model
            self.model = None
        if self.tokenizer:
            del self.tokenizer
            self.tokenizer = None
        torch.cuda.empty_cache()
        gc.collect()
        logger.info("Document processor resources cleaned up")


def save_chunks(file_path, chunk_size=settings.DEFAULT_CHUNK_SIZE,
                overlap=settings.DEFAULT_OVERLAP, batch_size=settings.DEFAULT_BATCH_SIZE,
                output_dir=settings.PROCESSED_CHUNKS_DIR):
    """
    Process a document file and save chunks as JSON.

    Args:
        file_path (str): Path to the document file
        chunk_size (int): Size of text chunks
        overlap (float): Overlap ratio between chunks
        batch_size (int): Batch size for processing
        output_dir (str): Output directory for processed chunks

    Returns:
        str: Path to the output JSON file or None if processing failed
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        filename = os.path.basename(file_path)

        logger.info(f"Processing {filename} with chunk_size={chunk_size}, overlap={overlap}, batch_size={batch_size}")

        dc = DocumentProcessor(settings.UPLOAD_DIR, chunk_size, overlap)
        chunks = dc.create_contextualised_chunks(file_path, batch_size)

        if not chunks:
            logger.error(f"No chunks were created for {filename}")
            dc.cleanup()
            return None

        base_name = os.path.splitext(filename)[0]
        output_path = os.path.join(output_dir, f"{base_name}.json")

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)

        logger.info(f"Saved {len(chunks)} chunks to {output_path}")

        dc.cleanup()
        del dc

        return output_path
    except Exception as e:
        logger.error(f"Error saving chunks for {file_path}: {str(e)}")
        return None
