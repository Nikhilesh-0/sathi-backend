"""
FAISS RAG Service — Sathi Backend

What this does:
1. On startup, reads 3 knowledge text files (jargon, RBI guidelines, comparison guide)
2. Splits them into chunks of ~200 words each
3. Converts each chunk to a vector embedding using fastembed (ONNX-based, no PyTorch)
   Model: BAAI/bge-small-en-v1.5 — ~25MB, fast, CPU-only, works well for multilingual
   retrieval since our knowledge base is in English (the LLM handles language output).
4. Stores all embeddings in a FAISS index (in-memory, no database needed)
5. At query time: converts the user's query to an embedding, finds top-2 closest chunks
6. Returns those chunks as grounding context for the Gemini prompt

Why fastembed instead of sentence-transformers:
- sentence-transformers pulls PyTorch (~700MB) → image too large for Railway build timeout
- fastembed uses ONNX Runtime (~50MB total) — same quality embeddings, 14x smaller
- The knowledge base is English text, so an English embedding model retrieves correctly
  regardless of what language the user typed — Gemini handles the language output.
"""

import os
import faiss
import numpy as np
from fastembed import TextEmbedding
from typing import List, Tuple

# Load once at module import — Railway keeps this in memory
# BAAI/bge-small-en-v1.5: ~25MB ONNX model, 384-dim embeddings, excellent retrieval quality
MODEL_NAME = "BAAI/bge-small-en-v1.5"
model = TextEmbedding(model_name=MODEL_NAME)

DATA_DIR = os.path.join(os.path.dirname(__file__), "../data")


def load_and_chunk_file(filepath: str, chunk_size: int = 200) -> List[str]:
    """
    Reads a text file and splits it into chunks by word count.
    We split on the '---' separator first (each term/section), then
    further chunk if a section is too long.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    # Split by our separator marker
    sections = [s.strip() for s in content.split("---") if s.strip()]

    chunks = []
    for section in sections:
        words = section.split()
        if len(words) <= chunk_size:
            chunks.append(section)
        else:
            # Further split long sections
            for i in range(0, len(words), chunk_size):
                chunk = " ".join(words[i:i + chunk_size])
                chunks.append(chunk)

    return chunks


def build_faiss_index() -> Tuple[faiss.Index, List[str]]:
    """
    Builds the FAISS index from all 3 knowledge files.
    Returns (index, list_of_chunks) — we keep chunks so we can return
    the actual text when a chunk is retrieved.

    FAISS uses L2 (Euclidean) distance by default. We use IndexFlatL2
    which does exact nearest-neighbor search. For 100-200 chunks this
    is fast enough; no need for approximate methods (IVF, HNSW).
    """
    files = [
        os.path.join(DATA_DIR, "fd_jargon.txt"),
        os.path.join(DATA_DIR, "rbi_guidelines.txt"),
        os.path.join(DATA_DIR, "fd_comparison_guide.txt"),
    ]

    all_chunks = []
    for filepath in files:
        chunks = load_and_chunk_file(filepath)
        all_chunks.extend(chunks)

    print(f"[FAISS] Total chunks to index: {len(all_chunks)}")

    # fastembed returns a generator of numpy arrays — collect into a list first
    # Shape: (n_chunks, 384) for bge-small-en-v1.5
    embeddings = np.array(list(model.embed(all_chunks))).astype("float32")

    # Build FAISS index
    dimension = embeddings.shape[1]  # 384
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    print(f"[FAISS] Index built. {index.ntotal} vectors stored.")
    return index, all_chunks


def retrieve_relevant_chunks(query: str, index: faiss.Index, chunks: List[str], top_k: int = 2) -> List[str]:
    """
    Given a user query, find the top_k most relevant chunks.

    Steps:
    1. Encode the query to a 384-dim vector
    2. Search the FAISS index for the nearest vectors (by L2 distance)
    3. Return the actual text of the closest chunks

    The model is multilingual, so Hindi/Punjabi/Bengali queries will map
    to the same embedding space as the English knowledge chunks — this is
    the key reason we chose paraphrase-multilingual-MiniLM-L12-v2.
    """
    # fastembed.embed() returns a generator — take the first (and only) result
    query_embedding = np.array(list(model.embed([query]))).astype("float32")
    distances, indices = index.search(query_embedding, top_k)

    retrieved = []
    for idx in indices[0]:
        if idx < len(chunks):
            retrieved.append(chunks[idx])

    return retrieved


# Singleton — built once when the module is first imported
_index, _chunks = build_faiss_index()


def get_context_for_query(query: str) -> Tuple[List[str], bool]:
    """
    Public interface for the chat router.
    Returns (list_of_relevant_chunks, was_retrieval_used).

    We only use RAG for conceptual/jargon questions.
    The caller (gemini_service) decides whether to use these chunks.
    """
    chunks = retrieve_relevant_chunks(query, _index, _chunks)
    return chunks, True
