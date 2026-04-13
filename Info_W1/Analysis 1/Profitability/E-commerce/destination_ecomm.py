import pandas as pd
import os

# Load the data
base_path = os.path.join(os.path.dirname(__file__), '../../Data')
orders_df = pd.read_csv(os.path.join(base_path, 'Order_ERP.csv'), sep=';')
customers_df = pd.read_csv(os.path.join(base_path, 'Costumer_Master.csv'), sep=';')
ship_to_df = pd.read_csv(os.path.join(base_path, 'Ship_to_master.csv'), sep=';')

# Merge orders with customers and ship_to data
df = orders_df.merge(customers_df[['Customer ID', 'Channel']], on='Customer ID', how='left')
df = df.merge(ship_to_df[['Ship To ID', 'Country']], on='Ship To ID', how='left')

# Format dates and extract year
df['Order Date'] = pd.to_datetime(df['Order Date'], format='%d/%m/%Y %H:%M')
df['Year'] = df['Order Date'].dt.year

# Filter for E-commerce
ecomm_df = df[df['Channel'] == 'E-commerce']

# Group by Year and Destination (Country)
df_ecomm_destination = ecomm_df.groupby(['Year', 'Country']).agg(
    Orders=('Order ID', 'nunique'),
    Quantity_of_SKUs=('Quantity', 'sum')
).reset_index()

# Rename Country to Destination to match the requested format
df_ecomm_destination.rename(columns={'Country': 'Destination'}, inplace=True)

# Rearrange columns to matches the required format: Year, Orders, Quantity of SKU's, Destination
df_ecomm_destination = df_ecomm_destination[['Year', 'Orders', 'Quantity_of_SKUs', 'Destination']]

# Sort by Year (ascending) and Quantity of SKUs (descending highest to lowest)
df_ecomm_destination = df_ecomm_destination.sort_values(by=['Year', 'Quantity_of_SKUs'], ascending=[True, False])

# Calculate yearly totals
yearly_totals = df_ecomm_destination.groupby('Year').agg(
    Orders=('Orders', 'sum'),
    Quantity_of_SKUs=('Quantity_of_SKUs', 'sum')
).reset_index()
yearly_totals['Destination'] = 'Total'

# Rearrange columns for consistency
yearly_totals = yearly_totals[['Year', 'Orders', 'Quantity_of_SKUs', 'Destination']]

# Combine and sort to place the 'Total' row at the end of each year group
df_ecomm_destination = pd.concat([df_ecomm_destination, yearly_totals], ignore_index=True)
df_ecomm_destination['Is_Total'] = df_ecomm_destination['Destination'] == 'Total'
df_ecomm_destination = df_ecomm_destination.sort_values(by=['Year', 'Is_Total', 'Quantity_of_SKUs'], ascending=[True, True, False])
df_ecomm_destination.drop(columns=['Is_Total'], inplace=True)
df_ecomm_destination = df_ecomm_destination.reset_index(drop=True)

print("E-commerce Destination Orders Analysis:")
print(df_ecomm_destination)

# Save the results to a CSV file in the output folder
output_path = os.path.join(os.path.dirname(__file__), '../output/ecomm_destination_results.csv')
df_ecomm_destination.to_csv(output_path, index=False, sep=';')
print(f"\nResults saved successfully to: {output_path}")
