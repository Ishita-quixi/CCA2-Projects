"""
Topic 3: ML Framework program - Hugging Face SENTIMENT MONITOR
---------------------------------------------------------------
Uses the Hugging Face `transformers` pipeline to classify text as
POSITIVE / NEGATIVE and "monitors" a stream of messages:

  * logs every message with label + confidence to sentiment_log.csv
  * keeps a rolling window of recent messages
  * raises an ALERT when the share of negative messages in the window
    crosses a threshold (e.g. a customer-feedback / brand monitor)

Modes:
  python 3_hf_sentiment_monitor.py              -> demo feed (built-in sample)
  python 3_hf_sentiment_monitor.py --file x.txt -> one message per line
  python 3_hf_sentiment_monitor.py --interactive-> type messages live

Needs: transformers, torch   (pip install transformers torch)
First run downloads the model (~250 MB) -> internet required once.
"""
import argparse
import csv
import os
import time
from collections import deque
from datetime import datetime

try:
    from transformers import pipeline
except ImportError:
    raise SystemExit(
        "transformers is not installed. Run:  pip install transformers torch"
    )

MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
LOG_FILE = "sentiment_log.csv"

WINDOW_SIZE = 5            # how many recent messages to look at
NEGATIVE_ALERT_RATIO = 0.6  # alert if >= 60% of the window is negative

DEMO_MESSAGES = [
    "I love this product, it works perfectly!",
    "Delivery was fast and the packaging was great.",
    "Customer support was very helpful and polite.",
    "The app keeps crashing, this is so frustrating.",
    "Worst purchase ever. Total waste of money.",
    "I am really disappointed with the quality.",
    "It stopped working after two days. Terrible!",
    "Not happy at all, I want a refund.",
    "Pretty good value for the price.",
    "Amazing experience, I will buy again!",
]


def load_classifier():
    print(f"[info] Loading model '{MODEL_NAME}' ... (first run downloads it)")
    return pipeline("sentiment-analysis", model=MODEL_NAME)


def init_log():
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["timestamp", "text", "label", "score"])


def log_row(text, label, score):
    with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(
            [datetime.now().isoformat(timespec="seconds"), text, label,
             f"{score:.4f}"]
        )


class SentimentMonitor:
    def __init__(self, classifier):
        self.clf = classifier
        self.window = deque(maxlen=WINDOW_SIZE)
        self.total = 0
        self.neg_total = 0
        self.alerts = 0

    def process(self, text):
        text = text.strip()
        if not text:
            return
        # truncation=True protects against texts longer than the model limit
        result = self.clf(text, truncation=True)[0]
        label, score = result["label"], float(result["score"])

        self.total += 1
        self.window.append(label)
        if label == "NEGATIVE":
            self.neg_total += 1
        log_row(text, label, score)

        icon = "[+]" if label == "POSITIVE" else "[-]"
        print(f"{icon} {label:<8} ({score:.2f})  {text}")

        # Check alert condition once the window is full
        if len(self.window) == WINDOW_SIZE:
            ratio = self.window.count("NEGATIVE") / WINDOW_SIZE
            if ratio >= NEGATIVE_ALERT_RATIO:
                self.alerts += 1
                print(f"    !!! ALERT: {ratio:.0%} of the last {WINDOW_SIZE} "
                      f"messages are NEGATIVE !!!")

    def summary(self):
        print("\n" + "=" * 55)
        print("SENTIMENT MONITOR SUMMARY")
        print("=" * 55)
        if self.total == 0:
            print("No messages processed.")
            return
        pos = self.total - self.neg_total
        print(f"Total messages : {self.total}")
        print(f"Positive       : {pos} ({pos / self.total:.0%})")
        print(f"Negative       : {self.neg_total} "
              f"({self.neg_total / self.total:.0%})")
        print(f"Alerts raised  : {self.alerts}")
        print(f"Log saved to   : {LOG_FILE}")


def run_demo(monitor, delay=0.3):
    print("[info] Running demo feed...\n")
    for msg in DEMO_MESSAGES:
        monitor.process(msg)
        time.sleep(delay)  # simulates messages arriving over time


def run_file(monitor, path):
    if not os.path.exists(path):
        raise SystemExit(f"[error] File not found: {path}")
    with open(path, encoding="utf-8") as f:
        for line in f:
            monitor.process(line)


def run_interactive(monitor):
    print("Type a message and press Enter (empty line or 'quit' to stop).\n")
    while True:
        try:
            text = input("> ")
        except (EOFError, KeyboardInterrupt):
            break
        if text.strip().lower() in ("", "quit", "exit"):
            break
        monitor.process(text)


def main():
    parser = argparse.ArgumentParser(description="Hugging Face Sentiment Monitor")
    parser.add_argument("--file", help="text file with one message per line")
    parser.add_argument("--interactive", action="store_true",
                        help="type messages live")
    args = parser.parse_args()

    init_log()
    monitor = SentimentMonitor(load_classifier())

    if args.file:
        run_file(monitor, args.file)
    elif args.interactive:
        run_interactive(monitor)
    else:
        run_demo(monitor)

    monitor.summary()


if __name__ == "__main__":
    main()
