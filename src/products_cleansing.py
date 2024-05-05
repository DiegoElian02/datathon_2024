import pandas as pd

# Load data
flights = pd.read_csv(r"../data/Filghts TEC_Valid.csv")
sales = pd.read_csv(r"../data/Sales TEC_Valid.csv")

# Remove rows with missing values
sales_no_combo = sales[sales["ProductName"].str.contains("Combo") == False]

# Group by Flight_ID and ProductName
sales_no_combo = sales_no_combo.groupby(['Flight_ID', 'ProductName']).sum()
sales_no_combo = sales_no_combo.reset_index()

# Pivot table
pivot_productos = sales_no_combo.pivot_table(index='Flight_ID', columns='ProductName', values='Quantity', fill_value=0).reset_index()

# Merge flights and pivot_productos
flights_completo = flights.merge(pivot_productos, on='Flight_ID', how='inner')

# Save data
flights_completo.to_parquet(r"../data/flights_with_products.parquet", index=False)
