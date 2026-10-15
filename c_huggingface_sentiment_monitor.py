"""
AIML Assignment (c): Hugging Face Sentiment Monitor
Input: b_ml_sentiment_input.xlsx (or another Excel file with a 'text' column)
Output: sentiment_results.xlsx
Install: pip install pandas openpyxl transformers torch
Run: python c_huggingface_sentiment_monitor.py
"""

from pathlib import Path
import pandas as pd
from transformers import pipeline

INPUT_FILE = "b_ml_sentiment_input.xlsx"
OUTPUT_FILE = "sentiment_results.xlsx"
MODEL_NAME = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"


def main():
    input_path = Path(INPUT_FILE)
    if not input_path.exists():
        raise FileNotFoundError(
            f"Cannot find {INPUT_FILE}. Put it in the same folder as this script."
        )

    df = pd.read_excel(input_path)
    if "text" not in df.columns:
        raise ValueError("Input Excel file must contain a column named 'text'.")

    df = df.dropna(subset=["text"]).copy()
    df["text"] = df["text"].astype(str)

    classifier = pipeline("sentiment-analysis", model=MODEL_NAME)
    results = classifier(df["text"].tolist(), batch_size=16, truncation=True)

    df["predicted_sentiment"] = [r["label"].lower() for r in results]
    df["confidence"] = [round(float(r["score"]), 4) for r in results]

    output_path = Path(OUTPUT_FILE)
    df.to_excel(output_path, index=False)

    print(df.to_string(index=False))
    print(f"\nSaved predictions to: {output_path.resolve()}")
    print("Note: this model predicts positive/negative, not a separate neutral class.")


if __name__ == "__main__":
    main()
