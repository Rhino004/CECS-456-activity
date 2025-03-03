import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import confusion_matrix, accuracy_score

# Verify TensorFlow version
print("TensorFlow Version:", tf.__version__)

# Load the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values  # Selecting relevant columns
y = dataset.iloc[:, 13].values  # Target variable

# Encoding categorical data
label_encoder = LabelEncoder()
X[:, 2] = label_encoder.fit_transform(X[:, 2])  # Encoding 'Gender'

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Building the ANN
ann = tf.keras.models.Sequential([
    tf.keras.layers.Dense(units=6, activation='relu'),
    tf.keras.layers.Dense(units=6, activation='relu'),
    tf.keras.layers.Dense(units=1, activation='sigmoid')
])

# Compiling the ANN
ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Training the ANN
ann.fit(X_train, y_train, batch_size=32, epochs=100)

# Making predictions on test set
y_pred = (ann.predict(X_test) > 0.5)

# Evaluating model performance
cm = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
print("Confusion Matrix:\n", cm)
print("Accuracy Score:", accuracy)

# Predicting a single customer
new_customer = np.array([[600, 'Male', 40, 3, 60000, 2, 1, 1, 50000]])
new_customer[:, 1] = label_encoder.transform(new_customer[:, 1])
new_customer = ct.transform(new_customer)
new_customer = scaler.transform(new_customer)
prediction = ann.predict(new_customer) > 0.5

print("Will the customer leave?", "Yes" if prediction else "No")
