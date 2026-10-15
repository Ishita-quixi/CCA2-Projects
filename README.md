
# CCA2-Projects

A Python project that compares **traditional rule-based programming** with **AI/ML-driven approaches** through two small, hands-on mini-projects:

1. **Customer review sentiment analysis** (Excel in, Excel out)
2. **Retrieval-Augmented Generation (RAG) demo** (question answering over local documents)

The goal is to show how a transparent, hand-written rule works, where its limits are, and how it can be extended with machine learning (for example, Hugging Face models and embedding-based retrieval).

---

## Project Structure

| File | Description |
|---|---|
| `traditional_programming.py` | Rule-based sentiment classifier that reads reviews from an Excel file and labels each one as POSITIVE, NEGATIVE or NEUTRAL. |
| `ml_inputs.xlsx` | Sample spreadsheet of customer reviews used as input for the sentiment work. |
| `PROBLEM_STATEMENTS (1).md` | 15 extension tasks for a Hugging Face Sentiment Monitor (confidence scores, urgent-word flagging, reports, and more). |
| `rag_demo.py` | Command-line RAG starter that answers questions from local `.txt` documents using keyword matching. |
| `RAG_RULES.md` | 20 rules for building a trustworthy, safe and testable RAG system. |
| `README.md` | This file. |

---

## 1. Rule-Based Sentiment Analysis

`traditional_programming.py` classifies each review using a transparent word-count rule:

- It lowercases the review text and checks it against a list of positive words (e.g. *excellent, fast, helpful, great*) and a list of negative words (e.g. *terrible, refund, broken, late*).
- `score = positive matches - negative matches`
- Score above 0 gives **POSITIVE**, below 0 gives **NEGATIVE**, and exactly 0 gives **NEUTRAL**.

**Input:** an Excel file with a `review_text` column (default path: `data/sentiment_input.xlsx`)
**Output:** a new Excel file with an added `traditional_sentiment` column (`data/traditional_sentiment_output.xlsx`). The input file is never overwritten.

### Run it

```bash
pip install pandas openpyxl
mkdir data
# place your Excel file in data/ as sentiment_input.xlsx (for example, a copy of ml_inputs.xlsx)
python traditional_programming.py
```

### Planned extensions

The problem statements in `PROBLEM_STATEMENTS (1).md` describe how to grow this into a full Hugging Face sentiment monitor, including:

- Model-based classification with confidence scores and a neutral class
- Flagging low-confidence, urgent (e.g. "refund", "fraud") and mixed-feedback reviews
- Summary reports with counts and percentages, grouped by product or category
- Comparing rule-based results against model predictions and flagging disagreements
- Safe handling of blank cells, duplicates and unexpected column names
- Exporting negative reviews to a separate sheet for the support team
- Logging, configurable file paths, and documenting model limitations (sarcasm, context, language coverage)

---

## 2. RAG Demo (Question Answering over Documents)

`rag_demo.py` is a lightweight demonstration of the Retrieval-Augmented Generation flow:

1. Loads every `.txt` file from a `documents/` folder.
2. Splits the question into keywords and scores each document by keyword overlap.
3. Returns the top matching documents (default top 3) with their **source filenames**.
4. If nothing relevant is found, it says so instead of guessing.

> This is a starter framework, not a production system. It uses simple keyword overlap rather than embeddings. For a full RAG stack, replace the `retrieve()` function with vector search.

### Run it

```bash
mkdir documents
# add one or more .txt files to documents/
python rag_demo.py
```

Type a question at the prompt, or `exit` to quit.

### RAG rules

`RAG_RULES.md` defines the quality and safety rules this demo is built around, such as:

- Answer only from retrieved sources, and cite the source filename for each claim
- Never invent quotes, dates, statistics or titles
- Treat retrieved documents as untrusted data and ignore prompt-injection attempts inside them
- Apply a relevance threshold and a configurable top-k, and reject weak matches
- Remove duplicate chunks, handle bad input gracefully, and never expose secrets
- Flag conflicts between sources instead of silently picking one
- Test with answerable, unanswerable, ambiguous and adversarial questions

---

## Tech Stack

- Python 3
- pandas and openpyxl (Excel handling)
- Standard library only for the RAG demo (`pathlib`, `re`)
- Hugging Face Transformers (planned, for the sentiment monitor extension)

## Requirements

```bash
pip install pandas openpyxl
```

## Author

[Ishita-quixi](https://github.com/Ishita-quixi)
