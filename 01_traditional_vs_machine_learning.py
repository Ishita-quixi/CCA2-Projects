# Traditional Programming vs Machine Learning
# Run: python 01_traditional_vs_machine_learning.py

# Traditional programming: rules are written by the programmer.
def traditional_result(marks):
    if marks >= 40:
        return "Pass"
    return "Fail"

print("TRADITIONAL PROGRAMMING")
for marks in [32, 55, 78]:
    print(f"Marks: {marks} -> {traditional_result(marks)}")

# Machine learning: the model learns a pattern from example data.
# Install first: pip install scikit-learn
from sklearn.tree import DecisionTreeClassifier

# Example training data: [hours_studied]
X = [[1], [2], [3], [4], [5], [6]]
y = ["Fail", "Fail", "Fail", "Pass", "Pass", "Pass"]

model = DecisionTreeClassifier(random_state=42)
model.fit(X, y)

print("\nMACHINE LEARNING")
for hours in [2, 4, 5]:
    prediction = model.predict([[hours]])[0]
    print(f"Hours studied: {hours} -> Predicted result: {prediction}")
