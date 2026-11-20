"""
ingest.py — Run this ONCE to:
  1. Load the PDF and extract text page-by-page (preserving page numbers).
  2. Split the text into overlapping chunks for better retrieval.
  3. Generate FREE local embeddings via HuggingFace sentence-transformers.
  4. Save the FAISS vector index to disk so we never re-embed at query time.

No API key needed for embeddings — model runs entirely on-device.

Usage:
    python ingest.py
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# LangChain — correct import paths (fully free, no OpenAI)
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# ── 1. Load environment variables from .env ───────────────────────────────────
load_dotenv()

# Embedding model runs locally — no API key required
# Must match the model used in rag.py exactly
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)

# ── 2. File paths ─────────────────────────────────────────────────────────────
BASE_DIR         = Path(__file__).parent
PDF_PATH         = BASE_DIR / "data" / "arthashastra_and_gita_compendium_50p.pdf"
VECTOR_STORE_DIR = BASE_DIR / "vector_store" / "faiss_index"


def load_pdf(pdf_path: Path) -> list:
    """
    Load a PDF file and return a list of LangChain Document objects.
    Each Document represents one page and carries 'page' metadata
    so we can cite page numbers in answers later.
    """
    print(f"📖 Loading PDF: {pdf_path.name}")
    loader = PyPDFLoader(str(pdf_path))
    documents = loader.load()
    print(f"   ✅ Loaded {len(documents)} pages.")
    return documents


def split_documents(documents: list) -> list:
    """
    Split pages into smaller overlapping chunks.

    Why chunk?
      - LLMs have limited context windows.
      - Smaller chunks improve retrieval precision.

    Why overlap?
      - An overlap of 200 characters means consecutive chunks share a little
        text, so sentences that sit at a boundary don't get cut off and lost.
    """
    print("✂️  Splitting pages into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,     # ~200 words per chunk — good balance
        chunk_overlap=200,  # 200-char overlap to avoid losing context at edges
        separators=["\n\n", "\n", ".", " ", ""],  # try paragraph breaks first
    )
    chunks = splitter.split_documents(documents)
    print(f"   ✅ Created {len(chunks)} chunks from {len(documents)} pages.")
    return chunks


def build_and_save_vector_store(chunks: list, save_path: Path) -> None:
    """
    Generate embeddings for every chunk and persist the FAISS index to disk.

    Uses HuggingFaceEmbeddings (sentence-transformers/all-MiniLM-L6-v2):
      - Runs fully locally — zero API cost, zero rate limits.
      - 384-dimensional dense vectors; excellent for semantic search.
      - First run downloads ~90 MB model weights (cached in ~/.cache/huggingface).

    The Streamlit app simply loads the saved index — no re-embedding at query time.
    """
    print(f"🔢 Generating embeddings with '{EMBEDDING_MODEL}' (local, free)...")
    print("   First run downloads ~90 MB model weights — cached after that.")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},   # change to "cuda" if GPU available
        encode_kwargs={"normalize_embeddings": True},  # cosine similarity
    )

    # Build the FAISS index from chunks + their embeddings
    vector_store = FAISS.from_documents(chunks, embeddings)

    # Persist to disk
    save_path.mkdir(parents=True, exist_ok=True)
    vector_store.save_local(str(save_path))
    print(f"   ✅ FAISS index saved to: {save_path}")


def main():
    # Validate the PDF exists before doing anything
    if not PDF_PATH.exists():
        raise FileNotFoundError(
            f"PDF not found at: {PDF_PATH}\n"
            "Please place 'arthashastra_and_gita_compendium_50p.pdf' "
            "inside the 'data/' folder."
        )

    documents = load_pdf(PDF_PATH)
    chunks    = split_documents(documents)
    build_and_save_vector_store(chunks, VECTOR_STORE_DIR)

    print("\n🎉 Ingestion complete! You can now run the app:")
    print("   streamlit run app.py")


if __name__ == "__main__":
    main()
