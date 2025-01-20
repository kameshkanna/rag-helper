# file_chunker.py
import os
import json
import csv
import docx
import PyPDF2
from typing import List, Optional
from pathlib import Path

class FileChunker:
    """Simple utility to split various file types into smaller text chunks."""
    
    def __init__(
        self, 
        input_file: str, 
        chunk_size: int = 500, 
        method: str = "sentence",
        output_dir: Optional[str] = None
    ):
        """
        Initialize the file chunker.
        
        Args:
            input_file: Path to the file to chunk
            chunk_size: Approximate number of words per chunk
            method: Chunking method - "sentence" or "token"
            output_dir: Custom output directory (optional)
        """
        self.input_file = Path(input_file)
        self.chunk_size = chunk_size
        self.method = method
        
        # Validate input file
        if not self.input_file.exists():
            raise FileNotFoundError(f"File not found: {input_file}")
            
        if method not in ["sentence", "token"]:
            raise ValueError("Method must be either 'sentence' or 'token'")
            
        # Set up output directory
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            # Create default output directory next to the input file
            default_dir = self.input_file.parent / "chunks" / self.input_file.stem
            self.output_dir = default_dir
            
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def read_file(self) -> str:
        """Read content from various file formats."""
        suffix = self.input_file.suffix.lower()
        
        try:
            if suffix == '.txt':
                return self._read_text_file()
            elif suffix == '.pdf':
                return self._read_pdf_file()
            elif suffix == '.docx':
                return self._read_docx_file()
            elif suffix == '.json':
                return self._read_json_file()
            elif suffix == '.csv':
                return self._read_csv_file()
            elif suffix in ['.md', '.markdown']:
                return self._read_text_file()
            else:
                raise ValueError(f"Unsupported file format: {suffix}")
        except Exception as e:
            raise Exception(f"Error reading {self.input_file}: {str(e)}")
    
    def _read_text_file(self) -> str:
        """Read content from text files (txt, md)."""
        with open(self.input_file, 'r', encoding='utf-8') as f:
            return f.read()
    
    def _read_pdf_file(self) -> str:
        """Read content from PDF files."""
        text = []
        with open(self.input_file, 'rb') as f:
            pdf = PyPDF2.PdfReader(f)
            for page in pdf.pages:
                text.append(page.extract_text())
        return ' '.join(text)
    
    def _read_docx_file(self) -> str:
        """Read content from Word documents."""
        doc = docx.Document(self.input_file)
        return ' '.join([paragraph.text for paragraph in doc.paragraphs])
    
    def _read_json_file(self) -> str:
        """Read content from JSON files."""
        with open(self.input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # Handle different JSON structures
            if isinstance(data, str):
                return data
            elif isinstance(data, dict):
                return ' '.join(str(value) for value in data.values())
            elif isinstance(data, list):
                return ' '.join(str(item) for item in data)
            else:
                return str(data)
    
    def _read_csv_file(self) -> str:
        """Read content from CSV files."""
        text = []
        with open(self.input_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                text.append(' '.join(str(cell) for cell in row))
        return ' '.join(text)
    
    def chunk_by_sentences(self, text: str) -> List[str]:
        """Split text into chunks at sentence boundaries."""
        sentences = text.split('. ')
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence.split())
            if current_length + sentence_length <= self.chunk_size:
                current_chunk.append(sentence)
                current_length += sentence_length
            else:
                chunks.append('. '.join(current_chunk) + '.')
                current_chunk = [sentence]
                current_length = sentence_length
                
        if current_chunk:
            chunks.append('. '.join(current_chunk) + '.')
        return chunks
    
    def chunk_by_tokens(self, text: str) -> List[str]:
        """Split text into chunks of fixed word count."""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), self.chunk_size):
            chunk = ' '.join(words[i:i + self.chunk_size])
            chunks.append(chunk)
        return chunks
    
    def save_chunks(self) -> List[Path]:
        """
        Process the input file and save chunks.
        
        Returns:
            List of paths to the created chunk files
        """
        # Read input file
        print(f"Reading {self.input_file}...")
        text = self.read_file()
        
        # Generate chunks
        print("Generating chunks...")
        if self.method == "sentence":
            chunks = self.chunk_by_sentences(text)
        else:
            chunks = self.chunk_by_tokens(text)
        
        # Save chunks
        chunk_files = []
        print(f"Saving chunks to {self.output_dir}...")
        
        for i, chunk in enumerate(chunks, 1):
            chunk_path = self.output_dir / f"chunk_{i}.txt"
            chunk_path.write_text(chunk, encoding='utf-8')
            chunk_files.append(chunk_path)
            print(f"Created: {chunk_path}") #yess
        
        return chunk_files
