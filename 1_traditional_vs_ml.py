"""
Topic 1: Traditional Programming vs Machine Learning
-----------------------------------------------------
Problem: Predict whether a student PASSES, given study hours and attendance.

Traditional programming : Human writes the RULES  -> Rules + Data   -> Answers
Machine learning        : Computer learns RULES   -> Data + Answers -> Rules

Run:  python 1_traditional_vs_ml.py
Needs: numpy, scikit-learn
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# ---------------------------------------------------------------
# 1) TRADITIONAL PROGRAMMING: a programmer hand-writes the rules
# ---------------------------------------------------------------
def traditional_program(study_hours, attendance):
    """Fixed, hand-written IF/ELSE rules. Returns 1 (Pass) or 0 (Fail)."""
    if study_hours >= 4 and attendance >= 75:
        return 1
    return 0


# ---------------------------------------------------------------
# 2) DATA (the real world follows a hidden pattern + some noise)
# ---------------------------------------------------------------
def make_dataset(n=500, seed=42):
    rng = np.random.default_rng(seed)
    hours = rng.uniform(0, 10, n)           # study hours per day
    attendance = rng.uniform(40, 100, n)    # attendance %
    # hidden real-world rule (the program does NOT know this formula)
    score = 8 * hours + 0.5 * attendance + rng.normal(0, 5, n)
    passed = (score >= 62).astype(int)
    X = np.column_stack([hours, attendance])
    return X, passed


def main():
    X, y = make_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=1
    )

    # ----- Traditional approach: apply hand-written rules -----
    trad_pred = np.array([traditional_program(h, a) for h, a in X_test])
    trad_acc = accuracy_score(y_test, trad_pred)

    # ----- Machine learning approach: learn rules from data -----
    model = LogisticRegression()
    model.fit(X_train, y_train)          # learning step
    ml_pred = model.predict(X_test)
    ml_acc = accuracy_score(y_test, ml_pred)

    print("=" * 60)
    print("TRADITIONAL PROGRAMMING vs MACHINE LEARNING")
    print("=" * 60)
    print(f"Traditional (hand-written rules) accuracy : {trad_acc:.2%}")
    print(f"Machine Learning (learned rules) accuracy : {ml_acc:.2%}")
    print()
    print("What the ML model learned (weights):")
    print(f"  study_hours coefficient : {model.coef_[0][0]:.3f}")
    print(f"  attendance  coefficient : {model.coef_[0][1]:.3f}")
    print(f"  intercept               : {model.intercept_[0]:.3f}")

    # ----- Try some new students with both approaches -----
    print()
    print("Predictions for new students")
    print("-" * 60)
    print(f"{'Hours':>6} {'Attend%':>8} | {'Traditional':>12} | {'ML model':>9}")
    samples = [(2, 90), (5, 70), (7, 60), (3, 95), (9, 50), (1, 45)]
    for h, a in samples:
        t = "Pass" if traditional_program(h, a) else "Fail"
        m = "Pass" if model.predict([[h, a]])[0] else "Fail"
        print(f"{h:>6} {a:>8} | {t:>12} | {m:>9}")

    print()
    print("Summary table")
    print("-" * 60)
    print("| Feature          | Traditional         | Machine Learning      |")
    print("| Rules come from  | Programmer          | Data (learned)        |")
    print("| Input            | Data + Rules        | Data + Answers        |")
    print("| Output           | Answers             | Rules (model)         |")
    print("| Adapts to change | No, edit code       | Yes, retrain on data  |")


if __name__ == "__main__":
    main()
