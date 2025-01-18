import os
import logging
import pickle
from typing import Dict, Tuple, Callable, List
from tqdm import tqdm
from transformers import pipeline
from transformers import BartForConditionalGeneration, BartTokenizer
import logging

CHUNKING_METHODS = {
    "sentence": lambda text, chunk_size, **kwargs: split_text_sentence_based(text, chunk_size),
    "token": lambda text, chunk_size, **kwargs: split_text_token_based(text, chunk_size),
    "sliding": lambda text, chunk_size, **kwargs: split_text_sliding_window(text, chunk_size, kwargs.get('overlap', 50)),
    "semantic": lambda text, chunk_size, **kwargs: split_text_semantic(text, chunk_size)
}

def split_text_sentence_based(text: str, chunk_size: int = 200) -> List[str]:
    """Split text into chunks based on sentence boundaries."""
    sentences = text.split('. ')
    chunks = []
    current_chunk = []
    current_length = 0
    for sentence in sentences:
        sentence_length = len(sentence.split())
        if current_length + sentence_length <= chunk_size:
            current_chunk.append(sentence)
            current_length += sentence_length
        else:
            chunks.append('. '.join(current_chunk))
            current_chunk = [sentence]
            current_length = sentence_length
    if current_chunk:
        chunks.append('. '.join(current_chunk))
    return chunks

def split_text_token_based(text: str, chunk_size: int = 500) -> List[str]:
    """Split text into chunks based on token count."""
    words = text.split()
    chunks = [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
    return chunks

def split_text_sliding_window(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """Split text using a sliding window approach."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        chunks.append(chunk)
        if i + chunk_size >= len(words):
            break
    return chunks



from transformers import BartForConditionalGeneration, BartTokenizer
import logging
from typing import List

def split_text_semantic(text: str, chunk_size: int = 500) -> List[str]:
    """Split text based on semantic boundaries using a smaller pre-trained model."""
    try:
        # Use a smaller model for summarization (Distilled BART)
        model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")
        tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
    except Exception as e:
        logging.error(f"Error loading model or tokenizer: {e}")
        return split_text_sentence_based(text, chunk_size)  # Fallback to sentence-based splitting

    sentences = text.split('. ')
    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        sentence_length = len(sentence.split())
        if current_length + sentence_length <= chunk_size:
            current_chunk.append(sentence)
            current_length += sentence_length
        else:
            chunk_text = '. '.join(current_chunk)
            try:
                # Tokenize and summarize the chunk directly
                inputs = tokenizer(chunk_text, return_tensors="pt", max_length=1024, truncation=True)
                summary_ids = model.generate(inputs["input_ids"], max_length=chunk_size, min_length=chunk_size // 2, length_penalty=2.0, num_beams=4, early_stopping=True)
                summarized = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
                chunks.append(summarized)
            except Exception as e:
                logging.error(f"Error summarizing chunk: {e}")
                chunks.append(chunk_text)  # Fallback to original chunk
            current_chunk = [sentence]
            current_length = sentence_length

    if current_chunk:
        chunk_text = '. '.join(current_chunk)
        try:
            # Tokenize and summarize the last chunk directly
            inputs = tokenizer(chunk_text, return_tensors="pt", max_length=1024, truncation=True)
            summary_ids = model.generate(inputs["input_ids"], max_length=chunk_size, min_length=chunk_size // 2, length_penalty=2.0, num_beams=4, early_stopping=True)
            summarized = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            chunks.append(summarized)
        except Exception as e:
            logging.error(f"Error summarizing last chunk: {e}")
            chunks.append(chunk_text)  # Fallback to original chunk

    return chunks



def chunk_markdown_files(md_dir: str, chunking_method: str, chunk_size: int = 500, **kwargs) -> Dict[Tuple[int, int], str]:
    """Chunk markdown files from a directory using the chosen method and save them as separate files."""
    if chunking_method not in CHUNKING_METHODS:
        raise ValueError(f"Invalid chunking method. Choose from: {list(CHUNKING_METHODS.keys())}")

    chunk_mapping = {}
    md_files = sorted([f for f in os.listdir(md_dir) if f.lower().endswith('.md')])

    for doc_id, md_file in enumerate(tqdm(md_files), 1):
        md_path = os.path.join(md_dir, md_file)
        try:
            with open(md_path, 'r', encoding='utf-8') as f:
                content = f.read()

            chunks = CHUNKING_METHODS[chunking_method](content, chunk_size, **kwargs)
            for chunk_idx, chunk in enumerate(chunks):
                chunk_filename = f"document_{doc_id}_chunk_{chunk_idx}.md"
                chunk_path = os.path.join(md_dir, chunk_filename)
                with open(chunk_path, 'w', encoding='utf-8') as f:
                    f.write(chunk)
                chunk_mapping[(doc_id, chunk_idx)] = chunk_path

        except Exception as e:
            logging.error(f"Error chunking document {md_path}: {str(e)}")

    mapping_path = os.path.join(md_dir, "chunk_mapping.pkl")
    with open(mapping_path, 'wb') as f:
        pickle.dump(chunk_mapping, f)

    return chunk_mapping