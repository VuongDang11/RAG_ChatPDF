import sys
import uuid # Moved to top for cleaner imports
import pymupdf4llm # Required for structural chunking
from typing import List, Dict, Tuple, Union
import numpy as np
from pypdf import PdfReader
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
from config import settings

# --- LangChain Splitters ---
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings # Wrapper for your local model


# --- Hierarchy of chunking strategies ---
def chunk_hierarchical(full_text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> tuple[List[Dict], List[Dict]]:
    # 1. Setup Parent and Child splitters
    # Parent chunk is automatically 4x the user's selected chunk size
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size * 4, chunk_overlap=0)
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    
    parents_to_insert = []
    children_to_insert = []
    
    # 2. Create Parent Chunks
    parent_chunks = parent_splitter.split_text(full_text)
    
    for parent_text in parent_chunks:
        parent_id = str(uuid.uuid4()) # Generate unique ID for the parent
        
        # Save the parent (NO embedding needed)
        parents_to_insert.append({
            "doc_id": parent_id,
            "text": parent_text,
            "type": "parent"
        })
        
        # 3. Create Child Chunks from this specific Parent
        child_chunks = child_splitter.split_text(parent_text)
        
        for child_text in child_chunks:
            # You will embed the child_text later in your main() function
            children_to_insert.append({
                "chunk_id": str(uuid.uuid4()),
                "parent_id": parent_id, # Crucial: Link back to parent
                "text": child_text,
                "type": "child" 
            })
            
    return parents_to_insert, children_to_insert


# --- Structural Chunking Function ---
def chunk_structural(pdf_path: str) -> List[Dict]:
    """Converts PDF to Markdown and splits based on headers."""
    # 1. Convert PDF to Markdown
    md_text = pymupdf4llm.to_markdown(pdf_path)
    
    # 2. Define headers to split on
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    md_splits = markdown_splitter.split_text(md_text)
    
    # 3. Return as dictionaries so we keep the header metadata
    return [{"text": doc.page_content, "metadata": doc.metadata} for doc in md_splits]


# --- Unified Chunking Function ---
def chunk_text(text: str, strategy: str = "recursive", model=None, pdf_path: str = None, chunk_size: int = 500, chunk_overlap: int = 50):
    """
    Routes the text to the appropriate chunking strategy.
    """
    if not text.strip():
        return []

    if strategy == "fixed-size":
        # Standard fixed size with overlap
        splitter = CharacterTextSplitter(
            separator="",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        chunks = splitter.split_text(text)
        return chunks
        
    elif strategy == "recursive":
        # Tries to keep paragraphs and sentences intact
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        chunks = splitter.split_text(text)
        return chunks
        
    elif strategy == "semantic":
        if model is None:
            raise ValueError("Semantic chunking requires an embedding model.")
        
        class LocalHuggingFaceEmbeddings:
            def __init__(self, embedder):
                self.embedder = embedder
            def embed_documents(self, texts):
                return self.embedder.encode(texts).tolist()
            def embed_query(self, text):
                return self.embedder.encode([text])[0].tolist()
                
        lc_embeddings = LocalHuggingFaceEmbeddings(model)
        
        splitter = SemanticChunker(
            lc_embeddings, 
            breakpoint_threshold_type="percentile", 
            breakpoint_threshold_amount=80 
        )
        
        docs = splitter.create_documents([text])
        chunks = [doc.page_content for doc in docs]
        return chunks
        
    elif strategy == "hierarchical":
        # Return the tuple of dictionaries directly. 
        parents, children = chunk_hierarchical(text, chunk_size, chunk_overlap)
        print(f"Strategy 'hierarchical' created {len(parents)} parents and {len(children)} children.")
        return parents, children 
        
    elif strategy == "structural":
        # NOTE: This requires pdf_path to be passed into the chunk_text function
        if not pdf_path:
            raise ValueError("Structural chunking requires the pdf_path argument.")
            
        structural_chunks = chunk_structural(pdf_path)
        print(f"Strategy 'structural' created {len(structural_chunks)} chunks.")
        return structural_chunks

    else:
        raise ValueError(f"Unknown chunking strategy: {strategy}")


# --- Embedding function ---
def embed_texts(model: SentenceTransformer, texts: List[str]) -> List[List[float]]:
    if not texts:
        return []
    vecs = model.encode(texts, normalize_embeddings=True)
    # Ensure float32 lists for MongoDB 
    if isinstance(vecs, list) or isinstance(vecs, np.ndarray):
        return [np.asarray(v, dtype=np.float32).tolist() for v in vecs]
    return []
    
    
# --- Main ingestion function ---
def main(pdf_path: str, doc_id: str, chunk_strategy: str = "recursive", chunk_size: int = 500, chunk_overlap: int = 50):
    coll = settings.col
    model = settings.embedder
    
    coll.delete_many({"doc_id": doc_id}) 
    print(f"Cleared existing chunks for {doc_id}. Starting fresh ingestion...")

    reader = PdfReader(pdf_path)
    docs_to_insert: List[Dict] = []
    chunk_global_id = 0

    # For Semantic chunking, it's often better to extract all text first rather than page-by-page,
    # so we don't accidentally split a continuous thought at a page break.
    full_text = ""
    for page in reader.pages:
        full_text += (page.extract_text() or "") + "\n"

    # 1. Chunk the text (pass UI parameters)
    chunks_data = chunk_text(full_text, strategy=chunk_strategy, model=model, pdf_path=pdf_path, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    
    # 2 & 3. Embed chunks and shape documents based on the strategy returned
    if chunk_strategy == "hierarchical":
        parents, children = chunks_data
        
        # Embed ONLY the children texts
        child_texts = [child["text"] for child in children]
        child_embeddings = embed_texts(model, child_texts)
        
        for child, emb in zip(children, child_embeddings):
            child["embedding"] = emb
            child["strategy_used"] = chunk_strategy
            docs_to_insert.append(child)
            
        # Parents are just stored for context retrieval, no vectors needed
        for parent in parents:
            parent["strategy_used"] = chunk_strategy
            docs_to_insert.append(parent)
            
    elif chunk_strategy == "structural":
        # Structural returns a list of dictionaries with metadata
        texts_to_embed = [doc["text"] for doc in chunks_data]
        embeddings = embed_texts(model, texts_to_embed)
        
        for doc, emb in zip(chunks_data, embeddings):
            docs_to_insert.append({
                "doc_id": doc_id,
                "chunk_id": chunk_global_id,
                "strategy_used": chunk_strategy,
                "text": doc["text"],
                "metadata": doc["metadata"], # Preserves the Markdown headers
                "embedding": emb
            })
            chunk_global_id += 1
            
    else:
        # Fixed, Recursive, and Semantic return a standard List of strings
        embeddings = embed_texts(model, chunks_data)
        
        for text, emb in zip(chunks_data, embeddings):
            docs_to_insert.append({
                "doc_id": doc_id,
                "chunk_id": chunk_global_id,
                "strategy_used": chunk_strategy,
                "text": text,
                "embedding": emb
            })
            chunk_global_id += 1

    # 4. Insert into MongoDB
    if docs_to_insert:
        coll.insert_many(docs_to_insert)
        print(f"Inserted {len(docs_to_insert)} documents using {chunk_strategy} strategy.")
        
        # NEW: Return the list of text strings for the UI to display
        # We filter for 'child' or 'text' fields depending on the strategy
        return [doc.get("text", "") for doc in docs_to_insert if "text" in doc]
    else:
        return []


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ingest_pdf.py <pdf_path> [doc_id] [strategy]")
        raise SystemExit(1)
        
    pdf = sys.argv[1]
    docid = sys.argv[2] if len(sys.argv) >= 3 else pdf.split("/")[-1]
    strategy = sys.argv[3] if len(sys.argv) >= 4 else "recursive" # Default to recursive
    
    main(pdf, docid, strategy)