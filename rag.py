"""
rag.py — The RAG (Retrieval-Augmented Generation) engine.

Responsibilities:
  1. Load the pre-built FAISS index from disk (no re-embedding).
  2. Accept a user question and retrieve the top-k most relevant chunks.
  3. Send those chunks as context to the LLM (Gemini Flash via Google AI).
  4. Return the answer AND the source chunks (with page numbers) so the UI
     can display references.
  5. If the answer is not in the PDF, the LLM is instructed to say so clearly
     rather than hallucinate.

Free stack:
  Embeddings → HuggingFace sentence-transformers/all-MiniLM-L6-v2 (local, no key)
  LLM        → Gemini 2.0 Flash via Google AI Studio (free tier, needs GOOGLE_API_KEY)

This module is imported by app.py — it has no UI code of its own.
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from typing import Any

# Embeddings: local HuggingFace model — no API key, no cost
from langchain_huggingface import HuggingFaceEmbeddings

# LLM: Google Gemini Flash — free tier at aistudio.google.com
from langchain_google_genai import ChatGoogleGenerativeAI

# Vector store + chain primitives
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

# ── Load environment variables ────────────────────────────────────────────────
load_dotenv()

GOOGLE_API_KEY  = os.getenv("GOOGLE_API_KEY")          # required for Gemini LLM
EMBEDDING_MODEL = os.getenv(                            # local HuggingFace model
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)
CHAT_MODEL = os.getenv("CHAT_MODEL", "gemini-2.0-flash")  # free on Google AI Studio

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR         = Path(__file__).parent
VECTOR_STORE_DIR = BASE_DIR / "vector_store" / "faiss_index"

# ── How many chunks to retrieve per question ──────────────────────────────────
TOP_K = 4

# ── Custom prompt — instructs the LLM to stay grounded in the PDF ─────────────
RAG_PROMPT_TEMPLATE = """You are a knowledgeable assistant specializing in \
classical Indian philosophy and statecraft, specifically the Arthashastra \
and the Bhagavad Gita.

IMPORTANT RULE: Answer ONLY from the context provided below.
You have absolutely no knowledge outside the provided context.
If the answer cannot be found in the context, respond with exactly:
"This information is not available in the provided text."

Do NOT make up facts or draw from outside knowledge.

---
Context from the document:
{context}

---
Question: {question}

Answer (be clear and concise, cite relevant concepts from the texts):"""

RAG_PROMPT = PromptTemplate(
    template=RAG_PROMPT_TEMPLATE,
    input_variables=["context", "question"],
)


def load_vector_store() -> FAISS:
    """
    Load the persisted FAISS index from disk.
    Raises a helpful error if ingest.py has not been run yet.

    Uses HuggingFaceEmbeddings to encode queries at search time —
    MUST match the model used during ingest.py exactly.
    No API key needed; model runs locally.
    """
    if not VECTOR_STORE_DIR.exists():
        raise FileNotFoundError(
            f"Vector store not found at: {VECTOR_STORE_DIR}\n"
            "Please run  'python ingest.py'  first to build the index."
        )

    # Same model as ingest.py — encodes the user's query for similarity search
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    vector_store = FAISS.load_local(
        str(VECTOR_STORE_DIR),
        embeddings,
        allow_dangerous_deserialization=True,  # safe: we wrote this file ourselves
    )
    return vector_store


def _format_docs(docs: list) -> str:
    """Join retrieved document chunks into a single context string."""
    return "\n\n".join(doc.page_content for doc in docs)


def build_qa_chain(vector_store: FAISS) -> Any:
    """
    Wire together the retriever, prompt, and LLM using LangChain Expression
    Language (LCEL) — no deprecated RetrievalQA needed.

    Pipeline:
        question ──► HuggingFace retriever (top-K chunks, local)
                 ──► RAG_PROMPT (context + question)
                 ──► Gemini 2.0 Flash (Google AI Studio, free)
                 ──► StrOutputParser

    Returns a dict with 'answer' and 'source_documents' for ask_question().
    """
    if not GOOGLE_API_KEY:
        raise EnvironmentError(
            "GOOGLE_API_KEY not found in .env.\n"
            "Get a free key at https://aistudio.google.com/app/apikey\n"
            "Then add it to your .env file:  GOOGLE_API_KEY=AIza..."
        )

    llm = ChatGoogleGenerativeAI(
        model=CHAT_MODEL,
        temperature=0,          # 0 = deterministic, factual answers
        google_api_key=GOOGLE_API_KEY,
    )

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K},
    )

    # Build the LCEL chain — parallel branches for context text + raw docs
    chain = (
        {
            "context": retriever | RunnableLambda(_format_docs),
            "question": RunnablePassthrough(),
            "source_documents": retriever,          # captured for citation
        }
        | RunnableLambda(
            lambda x: {
                "answer": (
                    RAG_PROMPT
                    | llm
                    | StrOutputParser()
                ).invoke({"context": x["context"], "question": x["question"]}),
                "source_documents": x["source_documents"],
            }
        )
    )
    return chain


def ask_question(qa_chain: Any, question: str) -> dict:
    """
    Run a question through the RAG chain and return a structured result.

    Returns:
        {
            "answer": str,
            "sources": [
                {"page": int, "snippet": str},
                ...
            ]
        }
    """
    if not question.strip():
        return {"answer": "Please enter a question.", "sources": []}

    result = qa_chain.invoke(question)

    answer = result.get("answer", "No answer generated.")

    # Build a deduplicated list of source references
    sources = []
    seen_pages = set()
    for doc in result.get("source_documents", []):
        page_num = doc.metadata.get("page", "?")
        # PDF pages are 0-indexed by PyPDFLoader → display as 1-indexed
        display_page = page_num + 1 if isinstance(page_num, int) else page_num
        if display_page not in seen_pages:
            seen_pages.add(display_page)
            sources.append({
                "page": display_page,
                "snippet": doc.page_content[:300].strip(),  # preview of the chunk
            })

    return {"answer": answer, "sources": sources}
