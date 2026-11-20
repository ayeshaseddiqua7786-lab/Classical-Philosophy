"""
app.py — Streamlit user interface for the Classical Philosophy Text Explorer.

Features:
  - Clean, themed UI with a sidebar description.
  - Text input for questions.
  - Displays the LLM-generated answer prominently.
  - Collapsible "Source References" section showing page numbers and snippets.
  - Loads the FAISS index and QA chain once (cached) so repeated questions
    don't reload heavy resources.

Run with:
    streamlit run app.py
"""

import streamlit as st
from rag import load_vector_store, build_qa_chain, ask_question

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="Classical Philosophy & Statecraft Explorer",
    page_icon="📜",
    layout="centered",
)

# ── Cache expensive resources so they load only once per session ───────────────
@st.cache_resource(show_spinner="📚 Loading knowledge base…")
def get_qa_chain():
    """
    Load the FAISS vector store and build the QA chain.
    @st.cache_resource ensures this runs only once — not on every question.
    """
    vector_store = load_vector_store()
    return build_qa_chain(vector_store)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    # FIX: use_container_width replaces deprecated use_column_width
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/5/5b/Arthashastra_Manuscript_page_sample.jpg/320px-Arthashastra_Manuscript_page_sample.jpg",
        caption="Arthashastra manuscript (Wikipedia)",
        use_container_width=True,
    )
    st.markdown("## 📜 About")
    st.markdown(
        """
        This tool lets you explore two foundational texts of classical
        Indian thought:

        - **Arthashastra** — Kautilya's treatise on statecraft, economics,
          and political strategy (~4th century BCE).
        - **Bhagavad Gita** — A philosophical dialogue on duty, righteousness,
          and the nature of the self.

        Ask any question and the system will search the document and
        provide a grounded answer with page references.
        """
    )
    st.divider()
    st.caption("🔒 Powered by OpenAI + LangChain + FAISS")
    st.caption("📄 Source: arthashastra_and_gita_compendium_50p.pdf")


# ── Main Page ─────────────────────────────────────────────────────────────────
st.title("📜 Classical Philosophy & Statecraft")
st.subheader("Text Explorer — Arthashastra & Bhagavad Gita")
st.markdown(
    "Ask a question about the texts and receive an answer grounded "
    "in the document, with source page references."
)
st.divider()

# ── Load QA chain (cached) ────────────────────────────────────────────────────
try:
    qa_chain = get_qa_chain()
except FileNotFoundError as e:
    st.error(
        f"⚠️ **Knowledge base not found.**\n\n"
        f"Please run the ingestion script first:\n\n"
        f"```bash\npython ingest.py\n```\n\n"
        f"Details: {e}"
    )
    st.stop()
except Exception as e:
    st.error(f"⚠️ Failed to load the knowledge base:\n\n{e}")
    st.stop()

# ── Question Input ────────────────────────────────────────────────────────────
st.markdown("### 💬 Ask a Question")

# Example questions to help first-time users
example_questions = [
    "What does the Arthashastra say about the duties of a king?",
    "How does the Bhagavad Gita describe the concept of dharma?",
    "What is Kautilya's advice on diplomacy and foreign policy?",
    "How should a king deal with corrupt ministers according to the Arthashastra?",
    "What does the Gita say about performing one's duty without attachment to results?",
]

with st.expander("💡 Example questions to get started", expanded=False):
    for q in example_questions:
        st.markdown(f"- *{q}*")

question = st.text_area(
    label="Your question:",
    placeholder="e.g. What does the Arthashastra say about taxation?",
    height=100,
    key="question_input",
)

ask_button = st.button("🔍 Search the Texts", type="primary", use_container_width=True)

# ── Answer Section ────────────────────────────────────────────────────────────
if ask_button:
    if not question.strip():
        st.warning("Please type a question before searching.")
    else:
        with st.spinner("🔍 Searching the texts and generating answer…"):
            result = ask_question(qa_chain, question)

        st.divider()

        # ── Answer ────────────────────────────────────────────────────────────
        st.markdown("### 📖 Answer")
        st.markdown(result["answer"])

        # ── Source References ─────────────────────────────────────────────────
        if result["sources"]:
            st.divider()
            st.markdown("### 🗂️ Source References")
            st.caption(
                f"The answer was drawn from {len(result['sources'])} "
                f"section(s) of the document:"
            )
            for i, src in enumerate(result["sources"], start=1):
                with st.expander(f"📄 Source {i} — Page {src['page']}"):
                    st.markdown(f"**Page {src['page']}**")
                    st.markdown(
                        f"> {src['snippet']}{'…' if len(src['snippet']) == 300 else ''}"
                    )
        else:
            st.info("No specific source sections were retrieved for this question.")

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "⚠️ This tool answers only from the provided 50-page compendium. "
    "For scholarly research, consult the full texts and academic editions."
)
