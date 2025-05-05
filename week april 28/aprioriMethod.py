# Step 1: Import necessary libraries
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# Step 2: Load the dataset
df = pd.read_csv('amazon_sales_data 2025.csv')  # <-- replace with your actual file name

# Step 3: Preprocessing
# Filter only completed orders
df = df[df['Status'] == 'Completed']

# Group products by Order ID
basket = df.groupby('Order ID')['Product'].apply(list).reset_index()

# Step 4: Encode the transactions
te = TransactionEncoder()
te_ary = te.fit(basket['Product']).transform(basket['Product'])
transaction_df = pd.DataFrame(te_ary, columns=te.columns_)

# Step 5: Apply Apriori Algorithm
frequent_itemsets = apriori(transaction_df, min_support=0.02, use_colnames=True)

# Step 6: Generate Association Rules
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)

# Step 7: Display the results
print("Frequent Itemsets:")
print(frequent_itemsets)

print("\nAssociation Rules:")
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])

# Optional: Save results to CSV
frequent_itemsets.to_csv('frequent_itemsets.csv', index=False)
rules.to_csv('association_rules.csv', index=False)
