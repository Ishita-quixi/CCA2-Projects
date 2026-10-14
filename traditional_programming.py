"""Rule-based sentiment analysis: reads review text from an Excel file."""
from pathlib import Path
import pandas as pd

INPUT_FILE = Path("data/sentiment_input.xlsx")
OUTPUT_FILE = Path("data/traditional_sentiment_output.xlsx")

POSITIVE_WORDS = {
    "excellent", "fast", "helpful", "polite", "amazing", "good", "easy",
    "great", "love", "happy", "perfect", "friendly", "wonderful"
}
NEGATIVE_WORDS = {
    "terrible", "refund", "crashing", "frustrating", "damaged", "disappointed",
    "late", "bad", "poor", "hate", "angry", "broken", "high"
}

def classify_sentiment(text: str) -> str:
    """Classify sentiment using a transparent word-count rule."""
    words = set(str(text).lower().replace(".", "").replace(",", "").split())
    score = len(words & POSITIVE_WORDS) - len(words & NEGATIVE_WORDS)
    if score > 0:
        return "POSITIVE"
    if score < 0:
        return "NEGATIVE"
    return "NEUTRAL"

def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Could not find {INPUT_FILE}. Place the Excel file in the data folder."
        )
    df = pd.read_excel(INPUT_FILE)
    if "review_text" not in df.columns:
        raise ValueError("Excel input must contain a 'review_text' column.")
    df["traditional_sentiment"] = df["review_text"].fillna("").apply(classify_sentiment)
    df.to_excel(OUTPUT_FILE, index=False)
    print(f"Processed {len(df)} reviews. Results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
