import pandas as pd
from pathlib import Path

# Construct the file paths relative to this script's location
current_dir = Path(__file__).parent
data_dir = current_dir / '..' / '..' / '..' / 'Data'

# Load the data
orders_df = pd.read_csv((data_dir / 'Order_ERP.csv').resolve(), sep=';')
customers_df = pd.read_csv((data_dir / 'Costumer_Master.csv').resolve(), sep=';')
ship_to_df = pd.read_csv((data_dir / 'Ship_to_master.csv').resolve(), sep=';')

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

# Add Rank column by Quantity of SKUs within each year
df_ecomm_destination['Rank'] = df_ecomm_destination.groupby('Year')['Quantity_of_SKUs'].rank(method='min', ascending=False).astype(int)

# Calculate yearly totals
yearly_totals = df_ecomm_destination.groupby('Year').agg(
    Orders=('Orders', 'sum'),
    Quantity_of_SKUs=('Quantity_of_SKUs', 'sum')
).reset_index()
yearly_totals['Destination'] = 'Total for Year'
yearly_totals['Rank'] = None

# Calculate the grand total across all years
grand_total = pd.DataFrame([{
    'Year': 'All Years',
    'Orders': yearly_totals['Orders'].sum(),
    'Quantity_of_SKUs': yearly_totals['Quantity_of_SKUs'].sum(),
    'Destination': 'Grand Total',
    'Rank': None
}])

# Combine and sort to place the 'Total' row at the end of each year group
df_ecomm_destination = pd.concat([df_ecomm_destination, yearly_totals], ignore_index=True)
df_ecomm_destination['Is_Total'] = df_ecomm_destination['Destination'] == 'Total for Year'
df_ecomm_destination = df_ecomm_destination.sort_values(by=['Year', 'Is_Total', 'Quantity_of_SKUs'], ascending=[True, True, False])
df_ecomm_destination.drop(columns=['Is_Total'], inplace=True)

# Append the grand total at the very end
df_ecomm_destination = pd.concat([df_ecomm_destination, grand_total], ignore_index=True)
df_ecomm_destination = df_ecomm_destination.reset_index(drop=True)

# Rearrange columns to put Rank properly
df_ecomm_destination = df_ecomm_destination[['Year', 'Rank', 'Destination', 'Orders', 'Quantity_of_SKUs']]

print("E-commerce Destination Orders Analysis:")
print(df_ecomm_destination)

# Save the results to a CSV and Excel file in the output folder
output_dir = current_dir.parent.parent / 'output'
output_dir.mkdir(parents=True, exist_ok=True)
csv_output_path = output_dir / 'ecomm_destination_results.csv'
excel_output_path = output_dir / 'ecomm_destination_results.xlsx'

df_ecomm_destination.to_csv(csv_output_path.resolve(), index=False, sep=';')
df_ecomm_destination.to_excel(excel_output_path.resolve(), index=False)
print(f"\nResults saved successfully to: {csv_output_path.resolve()}")
print(f"Results successfully saved to Excel sheet: {excel_output_path.resolve()}")
