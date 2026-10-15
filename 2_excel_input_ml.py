"""
Topic 2: Excel (.xlsx) file as INPUT to a Machine Learning program
-------------------------------------------------------------------
Flow:
  ml_input.xlsx  --(pandas)-->  train model  --> predict new rows --> ml_output.xlsx

ml_input.xlsx has two sheets:
  * training_data : study_hours, attendance, prev_score, passed (0/1)
  * new_students  : study_hours, attendance, prev_score  (we predict 'passed')

If ml_input.xlsx does not exist, this script creates a sample one first,
so you can run it straight away. Replace it with your own file if needed
(keep the same sheet names and column names).

Run:  python 2_excel_input_ml.py
Needs: pandas, numpy, scikit-learn, openpyxl
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

INPUT_FILE = "ml_input.xlsx"
OUTPUT_FILE = "ml_output.xlsx"
FEATURES = ["study_hours", "attendance", "prev_score"]
TARGET = "passed"


def create_sample_input(path):
    """Build a sample input workbook with training data + new students."""
    rng = np.random.default_rng(7)
    n = 300
    hours = rng.uniform(0, 10, n).round(1)
    attendance = rng.uniform(40, 100, n).round(0)
    prev = rng.uniform(30, 100, n).round(0)
    score = 5 * hours + 0.3 * attendance + 0.4 * prev + rng.normal(0, 4, n)
    passed = (score >= 55).astype(int)
    train_df = pd.DataFrame(
        {"study_hours": hours, "attendance": attendance,
         "prev_score": prev, "passed": passed}
    )
    new_df = pd.DataFrame(
        {
            "student_name": ["Aarav", "Diya", "Rohan", "Sneha", "Kabir", "Meera"],
            "study_hours": [2.0, 6.5, 8.0, 1.0, 4.5, 9.0],
            "attendance": [60, 85, 92, 45, 78, 70],
            "prev_score": [45, 70, 88, 35, 60, 55],
        }
    )
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        train_df.to_excel(writer, sheet_name="training_data", index=False)
        new_df.to_excel(writer, sheet_name="new_students", index=False)
    print(f"[info] Sample input file created: {path}")


def load_input(path):
    """Read both sheets from the Excel file and validate columns."""
    try:
        train_df = pd.read_excel(path, sheet_name="training_data")
        new_df = pd.read_excel(path, sheet_name="new_students")
    except ValueError as e:
        sys.exit(f"[error] Sheet missing in {path}: {e}")

    missing = [c for c in FEATURES + [TARGET] if c not in train_df.columns]
    if missing:
        sys.exit(f"[error] training_data is missing columns: {missing}")
    missing = [c for c in FEATURES if c not in new_df.columns]
    if missing:
        sys.exit(f"[error] new_students is missing columns: {missing}")

    train_df = train_df.dropna(subset=FEATURES + [TARGET])
    new_df = new_df.dropna(subset=FEATURES)
    return train_df, new_df


def main():
    if not os.path.exists(INPUT_FILE):
        create_sample_input(INPUT_FILE)

    train_df, new_df = load_input(INPUT_FILE)
    print(f"[info] Loaded {len(train_df)} training rows and "
          f"{len(new_df)} new rows from {INPUT_FILE}")

    X = train_df[FEATURES]
    y = train_df[TARGET].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("\n=== Model evaluation on held-out test data ===")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2%}")
    print(classification_report(y_test, y_pred, target_names=["Fail", "Pass"]))

    print("=== Feature importance ===")
    for name, imp in sorted(zip(FEATURES, model.feature_importances_),
                            key=lambda t: -t[1]):
        print(f"  {name:<12} {imp:.3f}")

    # Predict for the new students read from Excel
    result = new_df.copy()
    result["prediction"] = model.predict(new_df[FEATURES])
    result["prediction"] = result["prediction"].map({1: "Pass", 0: "Fail"})
    result["pass_probability"] = (
        model.predict_proba(new_df[FEATURES])[:, 1].round(3)
    )

    print("\n=== Predictions for new students ===")
    print(result.to_string(index=False))

    with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
        result.to_excel(writer, sheet_name="predictions", index=False)
    print(f"\n[done] Predictions saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
