import pandas as pd
import numpy as np
import os

def main():
    # Set up paths relative to the script location
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'Data')
    
    orders_path = os.path.join(data_dir, 'Order_ERP.csv')
    customers_path = os.path.join(data_dir, 'Costumer_Master.csv')
    products_path = os.path.join(data_dir, 'Product_master.csv')
    
    print("Loading data...")
    # Read the data, specifying the separator and decimal character
    orders = pd.read_csv(orders_path, sep=';', encoding='utf-8')
    customers = pd.read_csv(customers_path, sep=';', encoding='utf-8')
    products = pd.read_csv(products_path, sep=';', decimal=',', encoding='utf-8')
    
    print("Merging data...")
    # Merge datasets to get Channel and Net Weight per order line
    merged_df = orders.merge(customers[['Customer ID', 'Channel']], on='Customer ID', how='left')
    merged_df = merged_df.merge(products[['SKU', 'Net Weight kg', 'Primary DC']], on='SKU', how='left')
    
    # Process dates to extract Year
    # Dates are formatted as DD/MM/YYYY HH:MM
    merged_df['Order Date'] = pd.to_datetime(merged_df['Order Date'], format='%d/%m/%Y %H:%M')
    merged_df['Year'] = merged_df['Order Date'].dt.year
    
    # Calculate Total Weight for each order line
    merged_df['Total Weight kg'] = merged_df['Quantity'] * merged_df['Net Weight kg']
    
    print("Computing metrics...")
    
    # 1. Number of orders for each year and channel (unique Order IDs)
    orders_per_channel_year = merged_df.groupby(['Year', 'Channel'])['Order ID'].nunique().reset_index(name='Number of Orders')
    
    # 2. Total quantity for each year and channel
    quantity_per_channel_year = merged_df.groupby(['Year', 'Channel'])['Quantity'].sum().reset_index(name='Total Quantity')
    
    # 3. Total weight in one year for each channel (Grouped by Channel and Year)
    weight_per_channel_year = merged_df.groupby(['Year', 'Channel'])['Total Weight kg'].sum().reset_index(name='Total Weight (kg)')
    
    # 4. Pallet Fulfillment calculations
    # Max pallet weight is 200 kg. Calculate total weight per order to find pallets required.
    order_totals = merged_df.groupby(['Year', 'Channel', 'Order ID'])['Total Weight kg'].sum().reset_index()
    # Number of pallets is the ceiling of Total Weight / 200kg (minimum 1 if weight > 0)
    order_totals['Pallets Required'] = np.ceil(order_totals['Total Weight kg'] / 200).replace(0, 1) # Ensure min 1 pallet per non-zero order
    
    # Ignore 0 weight orders that might skew the calculation slightly
    order_totals = order_totals[order_totals['Total Weight kg'] > 0]
    
    # Helper to calculate fill rate for specific data slices
    def get_fill_rate(data_slice):
        weight = data_slice['Total Weight kg'].sum()
        capacity = data_slice['Pallets Required'].sum() * 200
        return (weight / capacity) * 100 if capacity > 0 else 0

    # Scenarios for ALL years
    scen_a_data = order_totals[order_totals['Channel'] == 'E-commerce']
    scen_b_data = order_totals[order_totals['Channel'] == 'Retail']
    scen_c_data = order_totals[order_totals['Channel'] == 'Pharmacy']
    scen_d_data = order_totals[order_totals['Channel'] == 'Retail Sport']
    
    scen_a_fill_rate = get_fill_rate(scen_a_data)
    scen_b_fill_rate = get_fill_rate(scen_b_data)
    scen_c_fill_rate = get_fill_rate(scen_c_data)
    scen_d_fill_rate = get_fill_rate(scen_d_data)
    
    # Yearly breakdown for Scenarios A, B, C, D
    def yearly_calc(data):
        return data.groupby('Year').apply(get_fill_rate)

    yearly_scen_a = yearly_calc(scen_a_data).reset_index(name='Scenario A (E-commerce) %')
    yearly_scen_b = yearly_calc(scen_b_data).reset_index(name='Scenario B (Retail) %')
    yearly_scen_c = yearly_calc(scen_c_data).reset_index(name='Scenario C (Pharmacy) %')
    yearly_scen_d = yearly_calc(scen_d_data).reset_index(name='Scenario D (Retail Sport) %')
    
    yearly_fill_rates = pd.merge(yearly_scen_a, yearly_scen_b, on='Year', how='outer')
    yearly_fill_rates = pd.merge(yearly_fill_rates, yearly_scen_c, on='Year', how='outer')
    yearly_fill_rates = pd.merge(yearly_fill_rates, yearly_scen_d, on='Year', how='outer')
    
    # 5. Orders split across multiple distinct DCs (Logical Orders)
    # A logical order is defined by Customer ID and Order Date
    dc_per_logical_order = merged_df.groupby(['Year', 'Channel', 'Customer ID', 'Order Date'])['Primary DC'].nunique().reset_index()
    
    # Group by Year to get total logical orders and multi DC logical orders
    total_logical_orders = dc_per_logical_order.groupby(['Year', 'Channel']).size().reset_index(name='Total Logical Orders')
    
    # Filter for logical orders with > 1 distinct DCs
    multi_dc_orders = dc_per_logical_order[dc_per_logical_order['Primary DC'] > 1]
    multi_dc_split = multi_dc_orders.groupby(['Year', 'Channel']).size().reset_index(name='Logical Orders with Multiple DCs')
    
    # Merge and format
    multi_dc_summary = pd.merge(total_logical_orders, multi_dc_split, on=['Year', 'Channel'], how='left').fillna(0)
    multi_dc_summary['Logical Orders with Multiple DCs'] = multi_dc_summary['Logical Orders with Multiple DCs'].astype(int)
    multi_dc_summary['% Multi DC'] = (multi_dc_summary['Logical Orders with Multiple DCs'] / multi_dc_summary['Total Logical Orders'] * 100).round(2)
    
    print("\n--- RESULTS ---")
    print("\n[NUMBER OF ORDERS PER YEAR AND CHANNEL]")
    print(orders_per_channel_year.to_string(index=False))
    
    print("\n[TOTAL QUANTITY PER YEAR AND CHANNEL]")
    print(quantity_per_channel_year.to_string(index=False))
    
    print("\n[TOTAL WEIGHT PER YEAR AND CHANNEL]")
    print(weight_per_channel_year.to_string(index=False))
    
    print("\n[PALLET FULFILLMENT RATES (Overall 3-Year Average)]")
    print(f"Scenario A (E-commerce):  {scen_a_fill_rate:.2f}%")
    print(f"Scenario B (Retail):      {scen_b_fill_rate:.2f}%")
    print(f"Scenario C (Pharmacy):    {scen_c_fill_rate:.2f}%")
    print(f"Scenario D (Retail Sport):{scen_d_fill_rate:.2f}%")
    
    print("\n[PALLET FULFILLMENT RATES (Per Year)]")
    print(yearly_fill_rates.to_string(index=False, float_format="%.2f"))
    
    print("\n[LOGICAL ORDERS (BY CUSTOMER & DATE) SPLIT ACROSS DCs (Per Year and Channel)]")
    print(multi_dc_summary.to_string(index=False))

if __name__ == "__main__":
    main()
