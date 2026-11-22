
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

# 1. Load Dataset
data = pd.read_csv("dataset/train.csv")

print("\n========== DATASET INFORMATION ==========")
print("Total Records:", len(data))
print("Total Features:", len(data.columns) - 1)
print("Missing Values:", data.isnull().sum().sum())

# 2. Remove Missing Values
data = data.dropna()

# 3. Separate Input and Output
X = data.drop("price_range", axis=1)
y = data["price_range"]

# 4. Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\n========== DATA SPLIT ==========")
print("Training Records:", len(X_train))
print("Testing Records:", len(X_test))

# 5. Create KNN Model
model = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier(n_neighbors=5))
])

# 6. Train Model
model.fit(X_train, y_train)

# 7. Predictions
train_pred = model.predict(X_train)
y_pred = model.predict(X_test)

# 8. Calculate Performance
train_accuracy = accuracy_score(y_train, train_pred)
accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test, y_pred, average="weighted"
)

recall = recall_score(
    y_test, y_pred, average="weighted"
)

f1 = f1_score(
    y_test, y_pred, average="weighted"
)

error_rate = 1 - accuracy

# 9. Cross Validation
cv_scores = cross_val_score(
    model, X_train, y_train, cv=5, scoring="accuracy"
)

# 10. Display Results
print("\n========== KNN MODEL PERFORMANCE ==========")

print("Training Accuracy:", round(train_accuracy * 100, 2), "%")
print("Testing Accuracy:", round(accuracy * 100, 2), "%")
print("Precision:", round(precision, 4))
print("Recall:", round(recall, 4))
print("F1 Score:", round(f1, 4))
print("Error Rate:", round(error_rate * 100, 2), "%")

print("\n========== CROSS VALIDATION ==========")
print("Individual Scores:", cv_scores)
print("Average CV Accuracy:", round(cv_scores.mean() * 100, 2), "%")

# 11. Classification Report
print("\n========== CLASSIFICATION REPORT ==========")

print(classification_report(y_test, y_pred))

# 12. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

print("\n========== CONFUSION MATRIX ==========")
print(cm)

ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred,
    cmap="Blues"
)

plt.title("KNN Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()

# 13. Performance Graph
metrics = {
    "Train Accuracy": train_accuracy,
    "Test Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "F1 Score": f1
}

plt.figure(figsize=(9, 5))

plt.bar(metrics.keys(), metrics.values())

plt.title("KNN Model Performance Comparison")
plt.ylabel("Score")
plt.ylim(0, 1.1)
plt.xticks(rotation=15)

plt.tight_layout()
plt.savefig("performance_graph.png")
plt.show()

# 14. Save Model
joblib.dump(model, "knn_model.pkl")

print("\n========== PROJECT COMPLETED ==========")
print("KNN Model Saved Successfully!")
print("File: knn_model.pkl")
