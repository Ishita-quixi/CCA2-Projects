"""
Topic 4: ML Framework - RETRIEVAL AUGMENTED GENERATION (RAG)
-------------------------------------------------------------
RAG = Retrieve relevant text first, then Generate an answer using it.

Pipeline:
  1. Load documents        (built-in sample knowledge base, or your .txt files)
  2. Chunk them            (split into small overlapping pieces)
  3. Embed chunks          (sentence-transformers -> vectors)
  4. Retrieve              (cosine similarity between question and chunks)
  5. Generate              (flan-t5 answers using ONLY the retrieved context)

Runs fully locally - no API key needed.

Modes:
  python 4_rag.py                       -> demo questions on sample knowledge
  python 4_rag.py --docs my_folder      -> use all .txt files in my_folder
  python 4_rag.py --interactive         -> ask your own questions

Needs: sentence-transformers, transformers, torch, numpy
  pip install sentence-transformers transformers torch numpy
First run downloads two small models (~80 MB + ~1 GB for flan-t5-base).
"""
import argparse
import glob
import os

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    from transformers import pipeline
except ImportError:
    raise SystemExit(
        "Missing packages. Run:\n"
        "  pip install sentence-transformers transformers torch numpy"
    )

EMBED_MODEL = "all-MiniLM-L6-v2"
GEN_MODEL = "google/flan-t5-base"
CHUNK_SIZE = 60     # words per chunk
CHUNK_OVERLAP = 15  # words shared between neighbouring chunks
TOP_K = 3           # chunks retrieved per question

SAMPLE_DOCS = {
    "machine_learning.txt": (
        "Machine learning is a branch of artificial intelligence in which "
        "computers learn patterns from data instead of being explicitly "
        "programmed with rules. Supervised learning uses labelled examples to "
        "train a model, for instance predicting whether an email is spam. "
        "Unsupervised learning finds structure in unlabelled data, such as "
        "grouping customers into clusters. Reinforcement learning trains an "
        "agent to take actions that maximise a reward."
    ),
    "transformers.txt": (
        "The Transformer is a neural network architecture introduced in 2017 "
        "in the paper 'Attention Is All You Need'. It relies on a mechanism "
        "called self-attention, which lets every word in a sentence look at "
        "every other word to understand context. Transformers power modern "
        "language models such as BERT and GPT. Unlike older recurrent "
        "networks, they process all tokens in parallel, which makes training "
        "much faster on GPUs."
    ),
    "huggingface.txt": (
        "Hugging Face is a company and open-source community that hosts "
        "thousands of pretrained models and datasets. Its transformers "
        "library provides a simple pipeline API for tasks like sentiment "
        "analysis, translation, summarisation and question answering. The "
        "Hugging Face Hub lets developers share models and download them "
        "with a single line of code."
    ),
    "rag.txt": (
        "Retrieval Augmented Generation, or RAG, combines a retriever with a "
        "language model. Documents are split into chunks and converted into "
        "embedding vectors stored in a vector index. When a user asks a "
        "question, the system retrieves the most similar chunks and passes "
        "them to the language model as context. This reduces hallucination "
        "and lets the model answer using private or up-to-date information "
        "without retraining."
    ),
}


# ---------------------------------------------------------------
# Step 1 & 2: load + chunk
# ---------------------------------------------------------------
def load_documents(folder=None):
    if folder:
        paths = sorted(glob.glob(os.path.join(folder, "*.txt")))
        if not paths:
            raise SystemExit(f"[error] No .txt files found in '{folder}'")
        docs = {}
        for p in paths:
            with open(p, encoding="utf-8") as f:
                docs[os.path.basename(p)] = f.read()
        return docs
    return SAMPLE_DOCS


def chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    words = text.split()
    if len(words) <= size:
        return [" ".join(words)]
    step = size - overlap
    chunks = []
    for start in range(0, len(words), step):
        piece = words[start:start + size]
        chunks.append(" ".join(piece))
        if start + size >= len(words):
            break
    return chunks


def build_chunks(docs):
    chunks, sources = [], []
    for name, text in docs.items():
        for c in chunk_text(text):
            chunks.append(c)
            sources.append(name)
    return chunks, sources


# ---------------------------------------------------------------
# The RAG system
# ---------------------------------------------------------------
class RAG:
    def __init__(self, docs):
        print(f"[info] Loading embedding model '{EMBED_MODEL}' ...")
        self.embedder = SentenceTransformer(EMBED_MODEL)
        print(f"[info] Loading generator model '{GEN_MODEL}' ...")
        self.generator = pipeline("text2text-generation", model=GEN_MODEL)

        self.chunks, self.sources = build_chunks(docs)
        print(f"[info] {len(docs)} documents -> {len(self.chunks)} chunks")
        # normalised vectors => dot product == cosine similarity
        self.vectors = self.embedder.encode(
            self.chunks, normalize_embeddings=True, show_progress_bar=False
        )

    def retrieve(self, question, k=TOP_K):
        q_vec = self.embedder.encode([question], normalize_embeddings=True)[0]
        scores = self.vectors @ q_vec
        top = np.argsort(scores)[::-1][:k]
        return [(self.chunks[i], self.sources[i], float(scores[i])) for i in top]

    def answer(self, question):
        hits = self.retrieve(question)
        context = "\n".join(h[0] for h in hits)
        prompt = (
            "Answer the question using only the context below. "
            "If the answer is not in the context, say \"I don't know\".\n\n"
            f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
        )
        out = self.generator(prompt, max_new_tokens=120, do_sample=False)
        return out[0]["generated_text"].strip(), hits


def show(question, answer, hits):
    print("\n" + "-" * 65)
    print(f"Q: {question}")
    print(f"A: {answer}")
    print("Sources used:")
    for text, src, score in hits:
        print(f"  - {src} (similarity {score:.2f}): {text[:80]}...")


DEMO_QUESTIONS = [
    "What is self-attention in a Transformer?",
    "How does RAG reduce hallucination?",
    "What is the Hugging Face pipeline API used for?",
    "What is the difference between supervised and unsupervised learning?",
    "Who won the football world cup in 2010?",   # not in docs -> should say I don't know
]


def main():
    parser = argparse.ArgumentParser(description="Simple local RAG demo")
    parser.add_argument("--docs", help="folder containing .txt files")
    parser.add_argument("--interactive", action="store_true",
                        help="ask your own questions")
    args = parser.parse_args()

    rag = RAG(load_documents(args.docs))

    if args.interactive:
        print("\nAsk questions (empty line or 'quit' to stop).")
        while True:
            try:
                q = input("\nQuestion> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if q.lower() in ("", "quit", "exit"):
                break
            ans, hits = rag.answer(q)
            show(q, ans, hits)
    else:
        for q in DEMO_QUESTIONS:
            ans, hits = rag.answer(q)
            show(q, ans, hits)


if __name__ == "__main__":
    main()
