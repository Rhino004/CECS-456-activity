# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# --------------------------
# 1. Data Acquisition and Familiarization
# --------------------------
# Load the dataset (ensure 'data.csv' is in your working directory)
data = pd.read_csv("data.csv")

print("First 5 rows of the dataset:")
print(data.head())
print("\nDataset Information:")
print(data.info())

# --------------------------
# 2. Data Preprocessing
# --------------------------
# Remove non-useful columns (e.g., the ID column)
if 'id' in data.columns:
    data = data.drop(columns=['id'])
elif 'ID' in data.columns:
    data = data.drop(columns=['ID'])

# Drop columns that are completely empty (all NaN), e.g., "Unnamed: 32"
data = data.dropna(axis=1, how='all')

# Convert the diagnosis column into a binary format:
# Map malignant ('M') to 1 and benign ('B') to 0
data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})

# Check for any remaining missing values
print("\nMissing values in each column:")
print(data.isnull().sum())

# Optionally, drop any remaining rows with missing values
data = data.dropna()

# --------------------------
# 3. Feature Scaling and Data Splitting
# --------------------------
# Separate features (X) and target (y)
X = data.drop(columns=['diagnosis'])
y = data['diagnosis']

# Normalize features using StandardScaler (important for KNN)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the dataset into training and testing sets (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42
)

# --------------------------
# 4. Model Implementation
# --------------------------
# -- KNN Classifier --
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)

# -- Logistic Regression --
logreg = LogisticRegression(max_iter=10000, random_state=42)
logreg.fit(X_train, y_train)
y_pred_logreg = logreg.predict(X_test)

# --------------------------
# 5. Evaluation of Models
# --------------------------
def evaluate_model(y_true, y_pred, model_name):
    print(f"\nEvaluation Metrics for {model_name}:")
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print("Confusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))
    
# Evaluate KNN
evaluate_model(y_test, y_pred_knn, "KNN")

# Evaluate Logistic Regression
evaluate_model(y_test, y_pred_logreg, "Logistic Regression")

# --------------------------
# 6. Visualization of Confusion Matrices
# --------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sns.heatmap(confusion_matrix(y_test, y_pred_knn), annot=True, fmt="d", cmap='Blues', ax=axes[0])
axes[0].set_title("KNN Confusion Matrix")
axes[0].set_xlabel("Predicted")
axes[0].set_ylabel("True")

sns.heatmap(confusion_matrix(y_test, y_pred_logreg), annot=True, fmt="d", cmap='Greens', ax=axes[1])
axes[1].set_title("Logistic Regression Confusion Matrix")
axes[1].set_xlabel("Predicted")
axes[1].set_ylabel("True")

plt.tight_layout()
plt.show()
