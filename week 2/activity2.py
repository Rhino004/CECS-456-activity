import pandas as pd
import numpy as np

#step 1: Loading the data from Loan_Default file into pandas
df = pd.read_csv('Loan_Default.csv')

# Display info about the dataset
print("Dataset Information:\n", df.info())
print("\nSummary Statistics:\n", df.describe())

#step 2: checks for missing values in the dataset
missing_values = df.isnull().sum()
print("\nMissing Values\n", missing_values[missing_values > 0])

#treatment of missing values with imputation
for col in df.columns:
    if df[col].dtype == 'object':  # Categorical columns
        df[col] = df[col].fillna(df[col].mode()[0])
    else:
        df[col] = df[col].fillna(df[col].median())
#checks if the missing values are handled
print("\nMissing Values After Imputation:\n", df.isnull().sum().sum())

#step 3: Outlier Detection and Treatment
numerical_cols = df.select_dtypes(include=[np.number]).columns
Q1 = df[numerical_cols].quantile(0.25)
Q3 = df[numerical_cols].quantile(0.75)
IQR = Q3 - Q1

# Define lower and upper caps
lower_cap = Q1 - 1.5 * IQR
upper_cap = Q3 + 1.5 * IQR

for col in numerical_cols:
    df[col] = np.where(df[col] < lower_cap[col], lower_cap[col], df[col])
    df[col] = np.where(df[col] > upper_cap[col], upper_cap[col], df[col])

print("\nDataset Shape After Outlier Capping:", df.shape)
#Step 4: Feature Engineering using One-Hot Encoding
categorical_cols = df.select_dtypes(include=['object']).columns
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

print("\nFeature Engineering Completed:\n", df.head())
