"""Small rule-based RAG demonstration with local text documents.

This is a framework starter, not a production-grade or embedding-based RAG system.
It retrieves relevant document chunks using simple keyword overlap so the flow is
easy to run and understand. Replace retrieve() with vector search for a full RAG stack.
"""
from pathlib import Path
import re

DOCS_DIR = Path("documents")
TOP_K = 3
MIN_OVERLAP = 1

def tokenize(text):
    return set(re.findall(r"[a-zA-Z0-9]+", text.lower()))

def load_documents():
    DOCS_DIR.mkdir(exist_ok=True)
    docs = []
    for path in sorted(DOCS_DIR.glob("*.txt")):
        docs.append({"source": path.name, "text": path.read_text(encoding="utf-8")})
    return docs

def retrieve(question, documents):
    q_terms = tokenize(question)
    ranked = []
    for doc in documents:
        score = len(q_terms & tokenize(doc["text"]))
        if score >= MIN_OVERLAP:
            ranked.append((score, doc))
    ranked.sort(key=lambda item: item[0], reverse=True)
    return [doc for _, doc in ranked[:TOP_K]]

def answer(question, documents):
    if not question.strip():
        return "Please enter a question."
    matches = retrieve(question, documents)
    if not matches:
        return "I could not find relevant information in the provided documents."
    excerpts = "\\n\\n".join(
        f"Source: {doc['source']}\\n{doc['text'][:1200]}" for doc in matches
    )
    return (
        "Answer draft based only on the retrieved excerpts:\\n"
        "Review the source excerpts below and use them to formulate a supported answer.\\n\\n"
        + excerpts
    )

def main():
    documents = load_documents()
    if not documents:
        print("Add .txt files to retrieval_augmented_generation/documents/ first.")
        return
    print("RAG demo ready. Type 'exit' to stop.")
    while True:
        question = input("\\nQuestion: ").strip()
        if question.lower() in {"exit", "quit"}:
            break
        print("\\n" + answer(question, documents))

if __name__ == "__main__":
    main()
