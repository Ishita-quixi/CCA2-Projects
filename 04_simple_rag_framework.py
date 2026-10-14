# Simple Retrieval-Augmented Generation (RAG) demo
# This beginner-friendly example retrieves the most relevant local note
# and uses it as context. It demonstrates the retrieval part of RAG without
# requiring a paid API key or downloading a large language model.

# Install first: pip install scikit-learn

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

documents = [
    "Machine learning allows computers to learn patterns from data.",
    "Traditional programming uses rules explicitly written by a programmer.",
    "Sentiment analysis identifies whether text expresses positive or negative feelings.",
    "Retrieval-Augmented Generation searches relevant documents and uses them as context for an answer.",
    "Excel files can be read in Python using the pandas library and read_excel function."
]

vectorizer = TfidfVectorizer()
document_vectors = vectorizer.fit_transform(documents)

print("Simple RAG Demo (type 'quit' to exit)")
while True:
    question = input("\nAsk a question: ").strip()
    if question.lower() in {"quit", "exit"}:
        break
    if not question:
        print("Please enter a question.")
        continue

    question_vector = vectorizer.transform([question])
    scores = cosine_similarity(question_vector, document_vectors)[0]
    best_index = scores.argmax()
    retrieved_context = documents[best_index]

    print("\nRetrieved context:", retrieved_context)
    print("Similarity score:", round(float(scores[best_index]), 3))
    print("Demo answer:", retrieved_context)
    print("\nNote: This minimal demo retrieves context; a full RAG system also sends")
    print("the retrieved context to a generative language model to create an answer.")
