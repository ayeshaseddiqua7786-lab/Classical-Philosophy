"""
test_rag.py — Run the 5 assessment questions against the live RAG chain.

Usage (with venv active):
    python test_rag.py

Expected behaviour:
  Q1, Q3, Q4 → answered from PDF with page citations
  Q2, Q5     → "This information is not available in the provided text."
"""

from rag import load_vector_store, build_qa_chain, ask_question

QUESTIONS = [
    (
        "Q1",
        "According to the Saptanga theory section, what are the seven constituent "
        "limbs (Prakritis) of a sovereign state?",
        "IN_CORPUS",
    ),
    (
        "Q2",
        "In which year was the Battle of Plassey fought between the British East "
        "India Company and the Nawab of Bengal?",
        "OUT_OF_CORPUS",
    ),
    (
        "Q3",
        "How many categories of envoys (Dutas) are identified in the text according "
        "to their level of authority, and what are their designations?",
        "IN_CORPUS",
    ),
    (
        "Q4",
        "What metaphor does Chapter 2 use to explain how a person of steady wisdom "
        "(Sthitaprajna) withdraws their senses from sense objects?",
        "IN_CORPUS",
    ),
    (
        "Q5",
        "Which political coalition won the 2024 Indian General Election?",
        "OUT_OF_CORPUS",
    ),
]

OUT_OF_CORPUS_REPLY = "This information is not available in the provided text."


def main():
    print("Loading vector store…")
    vs = load_vector_store()
    chain = build_qa_chain(vs)
    print("Chain ready. Running 5 questions.\n" + "=" * 60)

    all_pass = True
    for label, question, kind in QUESTIONS:
        result = ask_question(chain, question)
        answer  = result["answer"]
        sources = result["sources"]
        pages   = [str(s["page"]) for s in sources]

        print(f"\n{label} [{kind}]")
        print(f"Q: {question}")
        print(f"A: {answer}")
        print(f"Pages cited: {', '.join(pages) if pages else 'none'}")

        # Automated pass/fail check
        if kind == "OUT_OF_CORPUS":
            passed = OUT_OF_CORPUS_REPLY.lower() in answer.lower()
            status = "✅ PASS" if passed else "❌ FAIL — model leaked outside knowledge"
        else:
            passed = bool(pages)  # at least one page citation means retrieval worked
            status = "✅ PASS" if passed else "⚠️  WARN — no pages cited (check retrieval)"

        all_pass = all_pass and passed
        print(f"Status: {status}")
        print("-" * 60)

    print("\n" + ("🎉 ALL CHECKS PASSED" if all_pass else "⚠️  SOME CHECKS FAILED"))


if __name__ == "__main__":
    main()

