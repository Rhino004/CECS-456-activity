import pandas as pd
import apyori

df = pd.read_csv('Market_Basket_Optimisation.csv', header=None)

transactions = []
for _, row in df.iterrows():
    transaction = [item for item in row if pd.notna(item)]  # Remove NaN values
    transactions.append(transaction)

# Print sample transactions
print(transactions[:5])  # Display first 5 transactions
transactions.apriori()