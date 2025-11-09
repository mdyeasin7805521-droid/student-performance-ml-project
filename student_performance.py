"""
Student Performance Prediction
Auto-generated code for the Berea College application project.
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

df = pd.read_csv("student_performance_dataset.csv")
X = df[["StudyHours","Attendance","AssignmentMarks","MidtermMarks"]]
y = df["Pass"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:\n", cm)

# Save confusion matrix plot
fig, ax = plt.subplots(figsize=(4,4))
ax.imshow(cm, interpolation='nearest')
ax.set_title('Confusion Matrix')
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
ax.set_xticks([0,1]); ax.set_yticks([0,1])
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, cm[i,j], ha='center', va='center', color='white', fontsize=14)
plt.tight_layout()
fig.savefig("confusion_matrix.py.png")
print("Saved confusion_matrix.py.png")
