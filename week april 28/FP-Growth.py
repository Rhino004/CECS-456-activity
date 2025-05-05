# Step 1: Import necessary libraries
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth, association_rules


df = pd.read_csv('amazon_sales_data 2025.csv')
# Combine Customer Name + Date to simulate a shopping session
df['Session'] = df['Customer Name'] + '_' + df['Date'].astype(str)
basket = df.groupby('Session')['Product'].apply(list).reset_index()
# After grouping, filter to keep only sessions with 2 or more products
basket = basket[basket['Product'].apply(lambda x: len(x) > 1)]

te = TransactionEncoder()
te_ary = te.fit(basket['Product']).transform(basket['Product'])
transaction_df = pd.DataFrame(te_ary, columns=te.columns_)

frequent_itemsets = fpgrowth(transaction_df, min_support=0.02, use_colnames=True)

rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)

rules_sorted = rules.sort_values(by='confidence', ascending=False)

# Select the top 5 rules
top_5_rules = rules_sorted.head(5)

# Display nicely
print("\nTop 5 Association Rules based on Confidence:")
print(top_5_rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])

# Step 8: Display the results
print("Frequent Itemsets:")
print(frequent_itemsets)

print("\nAssociation Rules:")
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])

# Optional: Save results to CSV
frequent_itemsets.to_csv('frequent_itemsets_fpgrowth.csv', index=False)
rules.to_csv('association_rules_fpgrowth.csv', index=False)

rules_sorted.to_csv('association_rules_sorted_by_confidence.csv', index=False)

print("Rules have been sorted by confidence and saved to 'association_rules_sorted_by_confidence.csv'.")