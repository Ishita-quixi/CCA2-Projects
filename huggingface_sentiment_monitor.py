"""Hugging Face sentiment monitor that reads reviews from Excel."""
from pathlib import Path
import pandas as pd
from transformers import pipeline

INPUT_FILE = Path("data/sentiment_input.xlsx")
OUTPUT_FILE = Path("data/huggingface_sentiment_output.xlsx")
MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"

def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Could not find {INPUT_FILE}. Place the Excel file in the data folder."
        )
    df = pd.read_excel(INPUT_FILE)
    if "review_text" not in df.columns:
        raise ValueError("Excel input must contain a 'review_text' column.")

    # The model downloads the first time it runs; internet access may be required.
    sentiment_model = pipeline("sentiment-analysis", model=MODEL_NAME)
    texts = df["review_text"].fillna("").astype(str).tolist()
    predictions = sentiment_model(texts, truncation=True)
    df["ml_sentiment"] = [p["label"] for p in predictions]
    df["confidence"] = [round(float(p["score"]), 4) for p in predictions]
    df.to_excel(OUTPUT_FILE, index=False)
    print(f"Processed {len(df)} reviews. Results saved to {OUTPUT_FILE}")
    print("\\nSentiment summary:")
    print(df["ml_sentiment"].value_counts().to_string())

if __name__ == "__main__":
    main()
