import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('amazon_sales_data 2025.csv')  # Replace with your actual filename

# Display basic info
print("Number of rows:", df.shape[0])
print("Number of columns:", df.shape[1])

# Descriptive statistics for continuous variables
continuous_vars = ['Price', 'Quantity']

for var in continuous_vars:
    print(f"\nStatistics for {var}:")
    print(f"  Min: {df[var].min()}")
    print(f"  Max: {df[var].max()}")
    print(f"  Mean: {df[var].mean()}")
    print(f"  Median: {df[var].median()}")
    print(f"  Standard Deviation: {df[var].std()}")
    # Counts and percentages for categorical variables
categorical_vars = ['Category', 'Payment Method']

for var in categorical_vars:
    print(f"\nCategory breakdown for {var}:")
    counts = df[var].value_counts()
    percentages = df[var].value_counts(normalize=True) * 100
    summary = pd.DataFrame({'Count': counts, 'Percentage': percentages.round(2)})
    print(summary)
# Bar chart for 'Category' variable
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='Category', palette='Set2')
plt.title('Sales Distribution by Category')
plt.xlabel('Category')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Bar chart for 'Payment Method' variable
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='Payment Method', palette='Set2')
plt.title('Sales Distribution by Payment Method')
plt.xlabel('Payment Method')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()