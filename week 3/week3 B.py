#step 1: import libarys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Step 2: Loading the Dataset from the url
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data"
columns = ["class", "cap-shape", "cap-surface", "cap-color", "bruises", "odor",
           "gill-attachment", "gill-spacing", "gill-size", "gill-color",
           "stalk-shape", "stalk-root", "stalk-surface-above-ring",
           "stalk-surface-below-ring", "stalk-color-above-ring",
           "stalk-color-below-ring", "veil-type", "veil-color",
           "ring-number", "ring-type", "spore-print-color", "population", "habitat"]
#putting the dataset into pandas
data = pd.read_csv(url, header=None, names=columns)

# Step 3: Preprocess the Data
label = {}
for column in data.columns:
    labelencoder = LabelEncoder()
    data[column] = labelencoder.fit_transform(data[column])
    label[column] = labelencoder

# Step 4: Split the Data
X = data.drop(columns=["class"])
y = data["class"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Train the Model
classifier = DecisionTreeClassifier(criterion="entropy", max_depth=5, random_state=42)
classifier.fit(X_train, y_train)

# Step 6: Evaluate the Model
y_pred = classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
class_report = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"Accuracy of the ML: {accuracy:.4f}")
print("Classification Report on the ML:\n", class_report)
print("Confusion Matrix:\n", conf_matrix)

# Step 7: Visualize the Tree
plt.figure(figsize=(10, 7))
plot_tree(classifier, feature_names=X.columns, class_names=["edible", "poisonous"], filled=True)
plt.show()

"""
1. We would have to label the data because the dataset contains categorical features. So, we need
to convert them into numerical values using LabelEncoder()
2.The max_depth parameter is to control how deep the decision tree can be.
3.
4.
5.
"""