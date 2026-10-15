"""
AIML Assignment (a): Traditional Programming vs Machine Learning Programming
Run: python a_traditional_vs_ml.py

Traditional part uses manually written keyword rules.
ML part uses a pretrained Hugging Face sentiment classifier.
Install: pip install transformers torch
"""

def traditional_sentiment(text):
    """Rule-based example: simple and limited."""
    text = text.lower()
    positive_words = ["good", "great", "excellent", "love", "happy", "amazing"]
    negative_words = ["bad", "poor", "terrible", "hate", "disappointing", "broken"]

    positive_count = sum(word in text for word in positive_words)
    negative_count = sum(word in text for word in negative_words)

    if positive_count > negative_count:
        return "POSITIVE"
    if negative_count > positive_count:
        return "NEGATIVE"
    return "NEUTRAL"


def machine_learning_sentiment(text):
    """Pretrained transformer model learns patterns from training data."""
    from transformers import pipeline

    classifier = pipeline(
        "sentiment-analysis",
        model="distilbert/distilbert-base-uncased-finetuned-sst-2-english"
    )
    result = classifier(text, truncation=True)[0]
    return f'{result["label"]} (confidence={result["score"]:.3f})'


if __name__ == "__main__":
    samples = [
        "I love this excellent product!",
        "The service was terrible and disappointing.",
        "The package arrived today."
    ]

    print("TRADITIONAL PROGRAMMING (manual rules)")
    for sentence in samples:
        print(f"{sentence} -> {traditional_sentiment(sentence)}")

    print("\nMACHINE LEARNING PROGRAMMING (pretrained Hugging Face model)")
    try:
        for sentence in samples:
            print(f"{sentence} -> {machine_learning_sentiment(sentence)}")
    except ImportError:
        print("Install dependencies with: pip install transformers torch")
