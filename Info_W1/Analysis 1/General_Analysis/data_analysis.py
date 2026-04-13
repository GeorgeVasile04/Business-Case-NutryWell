import pandas as pd
import os

def main():
    data_dir = r"c:\ULB\MA2\Q2\Supply Chain Performance Analytics\Course 1\Data"
    
    # 1. Load the data 
    print("Loading data...")
    orders = pd.read_csv(os.path.join(data_dir, "Order_ERP.csv"), sep=";")
    products = pd.read_csv(os.path.join(data_dir, "Product_master.csv"), sep=";", decimal=",")
    shipto = pd.read_csv(os.path.join(data_dir, "Ship_to_master.csv"), sep=";")
    customers = pd.read_csv(os.path.join(data_dir, "Costumer_Master.csv"), sep=";")

    # Convert dates safely
    date_cols = ['Order Date', 'ERP Entry Date', 'Customer Requested Date', 'Promised Date', 'Actual Receipt Date']
    for col in date_cols:
        orders[col] = pd.to_datetime(orders[col], dayfirst=True)
    
    # Ensure numeric fields are clean before weight calculations
    orders['Quantity'] = pd.to_numeric(orders['Quantity'], errors='coerce').fillna(0)
    products['Net Weight kg'] = pd.to_numeric(products['Net Weight kg'], errors='coerce').fillna(0)

    # Merge order data 
    df = orders.merge(products[['SKU', 'Net Weight kg', 'Range']], on='SKU', how='left')
    df = df.merge(shipto[['Ship To ID', 'Country']], on='Ship To ID', how='left')
    df = df.merge(customers[['Customer ID', 'Channel']], on='Customer ID', how='left')
    df['Total Line Weight'] = df['Quantity'] * df['Net Weight kg']

    print("\n" + "="*50)
    print("ANALYSIS RESULTS: ORDER-TO-DELIVERY (STEP-BY-STEP)")
    print("="*50 + "\n")

    # --- 0. Country Overview ---
    print("0. ORDER VOLUME BY COUNTRY")
    print("-" * 35)
    orders_unique_country = df.drop_duplicates(subset=['Order ID'])
    country_counts = orders_unique_country.groupby('Country').size().reset_index(name='Orders')
    country_counts['% of Total'] = (country_counts['Orders'] / country_counts['Orders'].sum() * 100).round(2)
    country_counts = country_counts.sort_values(by='Orders', ascending=False).reset_index(drop=True)
    country_counts.index += 1
    country_counts['Rank'] = country_counts.index
    print(country_counts.to_string(index=False))
    print("\n")

    # --- 1. Order Entry & Information Flow Analysis ---
    print("1. ORDER ENTRY & INFORMATION FLOW")
    print("-" * 35)
    print("Step 1: Calculate 'Entry Lag Days' (ERP Entry Date - Order Date) for each line.")
    df['Entry Lag Days'] = (df['ERP Entry Date'] - df['Order Date']).dt.days
    
    total_unique_orders = df['Order ID'].nunique()
    print(f"  -> Total unique orders evaluated: {total_unique_orders}")
    print(f"  -> Total order lines evaluated: {len(df)}")
    
    avg_lag = df['Entry Lag Days'].mean()
    max_lag = df['Entry Lag Days'].max()
    delayed_entries_strict = (df['Entry Lag Days'] > 0).mean() * 100
    delayed_entries_tolerant = (df['Entry Lag Days'] > 1).mean() * 100
    
    # Calculate the average lag ONLY for the orders that are delayed beyond J+1
    avg_lag_delayed_tolerant = df[df['Entry Lag Days'] > 1]['Entry Lag Days'].mean()
    
    print(f"\nStep 2: Compute averages and maximums.")
    print(f"  -> Average Lag (All Orders): {avg_lag:.2f} days")
    print(f"  -> Average Lag (Of those failing J+1 tolerance): {avg_lag_delayed_tolerant:.2f} days")
    print(f"  -> Maximum Lag: {max_lag:.0f} days")
    print(f"  -> Percentage of delayed entries (>0 days lag): {delayed_entries_strict:.2f}%")
    print(f"  -> Percentage of delayed entries with J+1 tolerance (>1 day lag): {delayed_entries_tolerant:.2f}%\n")

    # --- 2. Order Splitting & Network Complexity ---
    print("2. ORDER SPLITTING (CUSTOMER-DAY SPLITS)")
    print("-" * 35)
    print("Step 1: Group orders by 'Customer ID' and 'Order Date'.")
    # We look at baskets created by the same customer on the same day, instead of just Order ID
    customer_daily_baskets = df.groupby(['Customer ID', 'Order Date'])['Branch/DC'].nunique()
    total_baskets = len(customer_daily_baskets)
    split_baskets = (customer_daily_baskets > 1).sum()
    
    print(f"  -> Total unique Customer-Day baskets found: {total_baskets}")
    print("Step 2: Count how many of these baskets were fulfilled by more than 1 distinct DC.")
    print(f"  -> Baskets split across multiple DCs: {split_baskets}")
    print(f"  -> Split percentage: {(split_baskets/total_baskets)*100:.2f}%\n")

    # --- 3. Delivery Reliability & Service Level (OTIF) ---
    print("3. DELIVERY RELIABILITY & SERVICE LEVEL")
    print("-" * 35)
    print("Step 1: Calculate days late vs Promised and Requested dates.")
    df['Lateness vs Promised'] = (df['Actual Receipt Date'] - df['Promised Date']).dt.days
    df['Lateness vs Requested'] = (df['Actual Receipt Date'] - df['Customer Requested Date']).dt.days
    
    on_time_promised = (df['Lateness vs Promised'] <= 0).mean() * 100
    on_time_requested = (df['Lateness vs Requested'] <= 0).mean() * 100
    print(f"  -> Lines on or before Promised Date: {on_time_promised:.2f}%")
    print(f"  -> Lines on or before Requested Date: {on_time_requested:.2f}%")
    
    print("Step 2: Breakdown by Channel.")
    channel_performance = df.groupby('Channel')['Lateness vs Requested'].apply(lambda x: (x <= 0).mean() * 100)
    for channel, perf in channel_performance.items():
        print(f"  -> {channel}: {perf:.2f}%")
    print("\n")

    # --- 4. Average Pallet Utilization (Order-Level) ---
    print("4. AVERAGE PALLET UTILIZATION (ORDER-LEVEL)")
    print("-" * 35)
    print("Step A: Compute order weight = sum(Quantity * Net Weight kg) for each Order ID and year.")
    df['Year'] = df['Order Date'].dt.year
    order_weights = (
        df.groupby(['Year', 'Order ID'], as_index=False)['Total Line Weight']
        .sum()
        .rename(columns={'Total Line Weight': 'Order Weight kg'})
    )

    print("Step B: Compute pallet fill rate per order as (Order Weight kg / 200).")
    order_weights['Pallet Fill Rate %'] = (order_weights['Order Weight kg'] / 200) * 100

    yearly_fill = (
        order_weights.groupby('Year', as_index=False)
        .agg(
            Orders=('Order ID', 'nunique'),
            Avg_Fill_Rate_Pct=('Pallet Fill Rate %', 'mean'),
            Min_Order_Weight_kg=('Order Weight kg', 'min'),
            Max_Order_Weight_kg=('Order Weight kg', 'max'),
        )
        .sort_values('Year')
    )

    print("  -> Pallet fill evolution by year:")
    for _, row in yearly_fill.iterrows():
        print(
            f"     {int(row['Year'])}: Orders={int(row['Orders']):,} | "
            f"Avg Fill={row['Avg_Fill_Rate_Pct']:.2f}% | "
            f"Min Weight={row['Min_Order_Weight_kg']:.2f} kg | "
            f"Max Weight={row['Max_Order_Weight_kg']:.2f} kg"
        )
    print("")

if __name__ == "__main__":
    main()
