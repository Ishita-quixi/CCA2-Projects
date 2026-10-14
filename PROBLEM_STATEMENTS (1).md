# Hugging Face Sentiment Monitor — Additional Problem Statements

1. Read customer reviews from an Excel file and classify each review as positive or negative.
2. Extend the monitor to include a neutral category using a suitable model or a documented confidence rule.
3. Save the predicted sentiment and confidence score to a new Excel file without overwriting the input.
4. Produce a summary showing the number and percentage of positive, negative, and neutral reviews.
5. Flag low-confidence predictions for manual review using a configurable threshold.
6. Identify reviews containing urgent words such as "refund", "fraud", "unsafe", or "broken".
7. Group reviews by product/category when a category column is available in the input workbook.
8. Compare rule-based sentiment with Hugging Face predictions and flag disagreements.
9. Handle blank cells, duplicate reviews, and unexpected Excel column names safely.
10. Create a daily monitoring report showing sentiment counts and the most common complaint topics.
11. Detect mixed-feedback reviews that contain both praise and complaints, and mark them for review.
12. Add a configurable input/output file path so the program can be reused with other spreadsheets.
13. Export a separate Excel sheet containing only negative reviews for the customer-support team.
14. Add basic logging and clear error messages when the file or required columns are missing.
15. Document model limitations, including sarcasm, context, language coverage, and domain-specific wording.
