import pandas as pd
from pathlib import Path

# Construct the file path relative to this script's location
current_dir = Path(__file__).parent
file_path = current_dir / '..' / '..' / '..' / 'Data' / 'Order_ERP.csv'
customer_file_path = current_dir / '..' / '..' / '..' / 'Data' / 'Costumer_Master.csv'

# Load the ERP Order Data
df = pd.read_csv(file_path.resolve(), sep=';', encoding='utf-8')

# Load the Customer Master Data to filter by E-commerce channel
customers_df = pd.read_csv(customer_file_path.resolve(), sep=';', encoding='utf-8')
ecommerce_customers = customers_df[customers_df['Channel'] == 'E-commerce']['Customer ID']

# Filter the orders for E-commerce customers only
df = df[df['Customer ID'].isin(ecommerce_customers)]

# Convert Order Date to datetime format to extract Month and Year
# Order Date format observed from sample: DD/MM/YYYY HH:MM
df['Order Date'] = pd.to_datetime(df['Order Date'], format='%d/%m/%Y %H:%M')

# Create a Year-Month column for exactly matching months from 2023 to 2025
df['YearMonth'] = df['Order Date'].dt.to_period('M')

# The user specified: "an order is place form the same costumer ID, on the same order date. 
# So group by those 2 atributs... in order to get correctly the quantities for each month."
# First, aggregate at the Order/SKU level to ensure quantities are correctly consolidated
order_sku_qty = df.groupby(['Customer ID', 'Order Date', 'SKU'], as_index=False)['Quantity'].sum()

# Convert Order Date to YearMonth period for the final grouping
order_sku_qty['YearMonth'] = order_sku_qty['Order Date'].dt.to_period('M')

# Pivot table: Rows as YearMonth (Dates from 2023-2025), Columns as SKUs, values are the sum of Quantities
matrix_table = pd.pivot_table(
    order_sku_qty, 
    values='Quantity', 
    index='YearMonth', 
    columns='SKU', 
    aggfunc='sum',
    fill_value=0
)

# Sorting index to assure chronicle order
matrix_table = matrix_table.sort_index()

# Display the matrix output
print("Monthly SKU Quantity Matrix (2023 - 2025):")
print(matrix_table)

# Save the matrix to an Excel sheet inside the output folder
output_dir = current_dir.parent.parent / 'output'
output_dir.mkdir(parents=True, exist_ok=True)
output_file = output_dir / 'sku_monthly_quantities.xlsx'

matrix_table.to_excel(output_file.resolve())
print(f"\nMatrix successfully saved to Excel sheet: {output_file.resolve()}")
