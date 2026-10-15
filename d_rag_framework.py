"""
AIML Assignment (d): Retrieval-Augmented Generation (RAG)
This educational demo retrieves relevant lines from local .txt files and
uses a text-generation model to answer questions using the retrieved context.

Folder: knowledge_base/ (add one or more .txt files)
Install: pip install sentence-transformers transformers torch sentencepiece
Run: python d_rag_framework.py
The first run downloads pretrained models and requires internet access.
"""

from pathlib import Path
import torch
from sentence_transformers import SentenceTransformer
from transformers import pipeline

KNOWLEDGE_DIR = Path("knowledge_base")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GENERATION_MODEL = "google/flan-t5-base"


def load_documents():
    KNOWLEDGE_DIR.mkdir(exist_ok=True)
    chunks = []
    for file_path in sorted(KNOWLEDGE_DIR.glob("*.txt")):
        content = file_path.read_text(encoding="utf-8")
        # Simple educational chunking by non-empty lines.
        for line in content.splitlines():
            line = line.strip()
            if line:
                chunks.append({"source": file_path.name, "text": line})
    if not chunks:
        raise ValueError(
            f"No text found. Add a non-empty .txt file to {KNOWLEDGE_DIR.resolve()}."
        )
    return chunks


def main():
    chunks = load_documents()
    embedder = SentenceTransformer(EMBEDDING_MODEL)
    texts = [item["text"] for item in chunks]
    embeddings = embedder.encode(
        texts, normalize_embeddings=True, convert_to_tensor=True
    )

    device = 0 if torch.cuda.is_available() else -1
    generator = pipeline(
        "text2text-generation",
        model=GENERATION_MODEL,
        device=device
    )

    print("RAG assistant ready. Type 'exit' to quit.")
    while True:
        question = input("\nQuestion: ").strip()
        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            continue

        q_embedding = embedder.encode(
            question, normalize_embeddings=True, convert_to_tensor=True
        )
        scores = embeddings @ q_embedding
        k = min(3, len(chunks))
        indices = torch.topk(scores, k=k).indices.tolist()
        retrieved = [chunks[i] for i in indices]
        context = "\n".join(item["text"] for item in retrieved)

        prompt = (
            "Answer the question using only the context. "
            "If the answer is not in the context, say you could not find it.\n\n"
            f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
        )
        response = generator(
            prompt, max_new_tokens=150, do_sample=False
        )[0]["generated_text"]

        print("\nRetrieved sources:")
        for item in retrieved:
            print(f"- {item['source']}: {item['text']}")
        print("\nAnswer:")
        print(response)


if __name__ == "__main__":
    main()
