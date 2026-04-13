import pandas as pd
import numpy as np
import os

base_path = r"c:\ULB\MA2\Q2\Supply Chain Performance Analytics\Course 2\Data"

# 1. Load Data
orders = pd.read_csv(os.path.join(base_path, "Orders_ERP_Parent_company.csv"), sep=';')
products = pd.read_csv(os.path.join(base_path, "Products_New_Company.csv"), sep=';', decimal=',')
ship_to = pd.read_csv(os.path.join(base_path, "Ship_To_New_Company.csv"), sep=';')
customers = pd.read_csv(os.path.join(base_path, "Master_Data_New_Company.csv"), sep=';')

# Merge to get Channel and Country
orders = orders.merge(ship_to[['Ship To ID', 'Country']], on='Ship To ID', how='left')
orders = orders.merge(customers[['Customer ID', 'Channel']], on='Customer ID', how='left')
orders = orders.merge(products[['SKU', 'Net Weight kg']], on='SKU', how='left')

# Date columns
date_cols = ['Order Date', 'ERP Entry Date', 'Customer Requested Date', 'Promised Date', 'Actual Receipt Date']
for col in date_cols:
    orders[col] = pd.to_datetime(orders[col], errors='coerce')

print("--- NEW METRIC: COMMANDS BY COUNTRY ---")
# Unique orders per country
orders_per_country = orders.drop_duplicates(subset=['Order ID']).groupby('Country').size().reset_index(name='Orders')
orders_per_country['% of Total'] = (orders_per_country['Orders'] / orders_per_country['Orders'].sum() * 100).round(2)
orders_per_country = orders_per_country.sort_values(by='Orders', ascending=False).reset_index(drop=True)
orders_per_country.index += 1
orders_per_country['Rank'] = orders_per_country.index
print(orders_per_country.to_string(index=False))

print("\n--- 1. ORDER ENTRY LAG (J+1 Tolerance) ---")
# Lag Days = ERP Entry Date - Order Date
# Tolerance: J+1 means lag <= 1 is essentially "OK", so real delay implies > 1
orders_unique = orders.drop_duplicates('Order ID')
lag_days = (orders_unique['ERP Entry Date'] - orders_unique['Order Date']).dt.days
delayed_orders_strict = lag_days > 0
delayed_orders_tolerant = lag_days > 1

print(f"Total Unique Orders: {len(orders_unique)}")
print(f"Average Lag: {lag_days.mean():.2f} days")
print(f"Max Lag: {lag_days.max()} days")
print(f"Delay Percentage (0 tolerance, > 0 days): {(delayed_orders_strict.mean() * 100):.2f}%")
print(f"Delay Percentage (J+1 tolerance, > 1 day): {(delayed_orders_tolerant.mean() * 100):.2f}%")

print("\n--- 2. ORDER SPLITTING ---")
# Group by Customer ID and Order Date
basket_groups = orders.groupby(['Customer ID', 'Order Date'])
baskets_df = basket_groups.agg({
    'Order ID': 'nunique',
    'Branch/DC': 'nunique'
}).reset_index()
total_baskets = len(baskets_df)
split_baskets = len(baskets_df[baskets_df['Order ID'] > 1])
print(f"Total Customer-Day Baskets: {total_baskets}")
print(f"Total Baskets split into multiple Order IDs: {split_baskets}")
print(f"Split Percentage: {(split_baskets / total_baskets * 100):.2f}%")

print("\n--- 3. DELIVERY RELIABILITY (OTIF) ---")
valid_receipts = orders.dropna(subset=['Actual Receipt Date', 'Customer Requested Date']).drop_duplicates(subset=['Order ID'])
on_time = valid_receipts['Actual Receipt Date'] <= valid_receipts['Customer Requested Date']
print(f"Global On-Time vs Requested: {(on_time.mean() * 100):.2f}%")
print("By Channel Breakdown:")
channel_otif = valid_receipts.groupby('Channel').apply(lambda x: (x['Actual Receipt Date'] <= x['Customer Requested Date']).mean() * 100)
for channel, otif in channel_otif.items():
    print(f"  {channel}: {otif:.2f}%")

print("\n--- 4. LOGISTICS COST (Pallets & <50kg Shipments) ---")
# Calculate weight per line
orders['Line Weight'] = orders['Quantity'] * orders['Net Weight kg']
# Group by Order ID and DC 
shipments = orders.groupby(['Order ID', 'Branch/DC']).agg(
    Total_Weight=('Line Weight', 'sum')
).reset_index()

shipments['Pallets'] = np.ceil(shipments['Total_Weight'] / 200)
shipments['Pallets'] = shipments['Pallets'].replace(0, 1) # Minimum 1 pallet

total_shipments = len(shipments)
total_pallets = shipments['Pallets'].sum()
avg_weight = shipments['Total_Weight'].mean()
fill_rate = avg_weight / 200
under_50kg = len(shipments[shipments['Total_Weight'] < 50])

print(f"Total Physical Shipments: {total_shipments}")
print(f"Total physical pallets billed: {total_pallets}")
print(f"Average Weight per Shipment: {avg_weight:.2f} kg")
print(f"Overall Pallet Fill Rate: {(fill_rate * 100):.2f}%")
print(f"Shipments < 50kg: {under_50kg} ({(under_50kg / total_shipments * 100):.2f}%)")
