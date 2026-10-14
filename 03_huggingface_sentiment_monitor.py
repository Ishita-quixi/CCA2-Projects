# Hugging Face Sentiment Monitor
# Install first: pip install transformers torch
# The first run downloads the model and may take some time.

from transformers import pipeline

sentiment_monitor = pipeline("sentiment-analysis")

print("Sentiment Monitor (type 'quit' to exit)")
while True:
    text = input("\nEnter a sentence or review: ").strip()
    if text.lower() in {"quit", "exit"}:
        break
    if not text:
        print("Please enter some text.")
        continue

    result = sentiment_monitor(text)[0]
    print("Sentiment:", result["label"])
    print("Confidence:", round(result["score"] * 100, 2), "%")
