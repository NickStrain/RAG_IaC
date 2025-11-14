import os
from pathlib import Path
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
from bs4 import BeautifulSoup
import markdown
from typing import List, Dict
import hashlib
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configuration - SET THESE VALUES IN .env FILE
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")
INDEX_NAME = os.getenv("INDEX_NAME", "terraform-aws-docs")

# Validate API key
if not PINECONE_API_KEY:
    raise ValueError(
        "Please set your Pinecone API key in the .env file!\n"
        "Create a .env file in the same directory with:\n"
        "PINECONE_API_KEY=your-api-key-here\n"
        "Get your API key from: https://app.pinecone.io/"
    )

# Initialize the embedding model
print("Loading embedding model...")
model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize Pinecone
print("Connecting to Pinecone...")
pc = Pinecone(api_key=PINECONE_API_KEY)

# Create or connect to index
index_name = INDEX_NAME
print(f"Checking for index '{index_name}'...")
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,  # all-MiniLM-L6-v2 produces 384-dimensional embeddings
        metric='cosine',
        spec=ServerlessSpec(cloud='aws', region=PINECONE_ENVIRONMENT)
    )
    print(f"Created new index '{index_name}'")
else:
    print(f"Using existing index '{index_name}'")

index = pc.Index(index_name)

def extract_text_from_html(html_content: str) -> str:
    """Extract clean text from HTML content."""
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Remove script and style elements
    for script in soup(["script", "style"]):
        script.decompose()
    
    # Get text
    text = soup.get_text()
    
    # Clean up whitespace
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    text = ' '.join(chunk for chunk in chunks if chunk)
    
    return text

def extract_text_from_markdown(md_content: str) -> str:
    """Convert markdown to HTML and extract text."""
    html = markdown.markdown(md_content)
    return extract_text_from_html(html)

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """Split text into overlapping chunks."""
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        if chunk:
            chunks.append(chunk)
    
    return chunks

def generate_id(text: str, file_path: str, chunk_idx: int) -> str:
    """Generate a unique ID for each chunk."""
    content = f"{file_path}_{chunk_idx}_{text[:50]}"
    return hashlib.md5(content.encode()).hexdigest()

def process_file(file_path: Path, base_path: Path) -> List[Dict]:
    """Process a single file and return chunks with metadata."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Determine file type and extract text
        if file_path.suffix.lower() in ['.html', '.htm']:
            text = extract_text_from_html(content)
        elif file_path.suffix.lower() in ['.md', '.markdown']:
            text = extract_text_from_markdown(content)
        else:
            text = content  # Plain text fallback
        
        # Skip empty files
        if not text.strip():
            return []
        
        # Chunk the text
        chunks = chunk_text(text)
        
        # Prepare data with metadata
        relative_path = str(file_path.relative_to(base_path))
        
        chunk_data = []
        for idx, chunk in enumerate(chunks):
            chunk_data.append({
                'id': generate_id(chunk, relative_path, idx),
                'text': chunk,
                'metadata': {
                    'file_path': relative_path,
                    'chunk_index': idx,
                    'total_chunks': len(chunks),
                    'file_type': file_path.suffix
                }
            })
        
        return chunk_data
    
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return []

def embed_and_upload(docs_folder: str, batch_size: int = 100):
    """Read all files, embed them, and upload to Pinecone."""
    base_path = Path(docs_folder)
    
    # Find all HTML and Markdown files
    file_patterns = ['**/*.html', '**/*.htm', '**/*.md', '**/*.markdown']
    all_files = []
    
    for pattern in file_patterns:
        all_files.extend(base_path.glob(pattern))
    
    print(f"Found {len(all_files)} files to process")
    
    all_chunks = []
    
    # Process all files
    for i, file_path in enumerate(all_files, 1):
        print(f"Processing {i}/{len(all_files)}: {file_path.name}")
        chunks = process_file(file_path, base_path)
        all_chunks.extend(chunks)
    
    print(f"\nTotal chunks created: {len(all_chunks)}")
    
    # Batch embed and upload
    print("\nEmbedding and uploading to Pinecone...")
    
    for i in range(0, len(all_chunks), batch_size):
        batch = all_chunks[i:i + batch_size]
        
        # Extract texts for embedding
        texts = [chunk['text'] for chunk in batch]
        
        # Generate embeddings
        embeddings = model.encode(texts, show_progress_bar=True)
        
        # Prepare vectors for Pinecone
        vectors = [
            {
                'id': chunk['id'],
                'values': embedding.tolist(),
                'metadata': {
                    **chunk['metadata'],
                    'text': chunk['text'][:1000]  # Store first 1000 chars in metadata
                }
            }
            for chunk, embedding in zip(batch, embeddings)
        ]
        
        # Upload to Pinecone
        index.upsert(vectors=vectors)
        
        print(f"Uploaded batch {i // batch_size + 1}/{(len(all_chunks) + batch_size - 1) // batch_size}")
    
    print("\n All documents embedded and uploaded to Pinecone!")
    print(f"Index stats: {index.describe_index_stats()}")

if __name__ == "__main__":
    # Replace with your actual folder path
    DOCS_FOLDER = "D:/RAG_IaC/terraform-provider-aws/website/docs"
    
    embed_and_upload(DOCS_FOLDER)