import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

def main():
    data_dir = r"c:\ULB\MA2\Q2\Supply Chain Performance Analytics\Course 1\Data"
    
    # 1. Load data
    print("Loading data...")
    orders = pd.read_csv(os.path.join(data_dir, "Order_ERP.csv"), sep=";")
    products = pd.read_csv(os.path.join(data_dir, "Product_master.csv"), sep=";", decimal=",")
    customers = pd.read_csv(os.path.join(data_dir, "Costumer_Master.csv"), sep=";")
    
    # 2. Clean and format
    orders['Order Date'] = pd.to_datetime(orders['Order Date'], dayfirst=True)
    orders['Year'] = orders['Order Date'].dt.year
    orders['Quantity'] = pd.to_numeric(orders['Quantity'], errors='coerce').fillna(0)
    
    # Clean the prices safely
    if products['Selling Price EUR'].dtype == 'O':
        products['Selling Price EUR'] = products['Selling Price EUR'].str.replace(',', '.')
    products['Selling Price EUR'] = pd.to_numeric(products['Selling Price EUR'], errors='coerce').fillna(0)
    
    # 3. Merge Datasets
    # Order_ERP already has 'Product Name', we just need Selling Price EUR
    df = orders.merge(products[['SKU', 'Selling Price EUR']], on='SKU', how='left')
    df['Product Name'] = df['Product Name'].fillna("Unknown")
    df = df.merge(customers[['Customer ID', 'Channel']], on='Customer ID', how='left')
    
    # Calculate revenue per line
    df['Line Revenue'] = df['Quantity'] * df['Selling Price EUR']
    
    years = sorted(df['Year'].dropna().unique())
    channels = sorted(df['Channel'].dropna().unique())
    
    print("\n" + "="*80)
    print("GRANULAR PORTFOLIO ANALYSIS: PER YEAR & PER CHANNEL")
    print("="*80)
    
    results = []
    
    # Process the data iteratively to respect constraints
    for year in years:
        for channel in channels:
            subset = df[(df['Year'] == year) & (df['Channel'] == channel)].copy()
            if subset.empty: continue
            
            total_orders_in_channel_year = subset['Order ID'].nunique()
            
            # Group by SKU for this specific Year and Channel
            sku_grp = subset.groupby(['SKU', 'Product Name'])
            
            for (sku, name), group in sku_grp:
                qty = group['Quantity'].sum()
                rev = group['Line Revenue'].sum()
                unq_orders = group['Order ID'].nunique()
                
                # Demand Metrics
                order_rate = (unq_orders / total_orders_in_channel_year) * 100 if total_orders_in_channel_year > 0 else 0
                
                # Repeat Metrics
                cust_counts = group.groupby('Customer ID')['Order ID'].nunique()
                repeat_custs = (cust_counts > 1).sum()
                total_custs = len(cust_counts)
                repeat_rate = (repeat_custs / total_custs) * 100 if total_custs > 0 else 0
                
                # Niche metrics - concentration risk
                top_cust_vol = group.groupby('Customer ID')['Quantity'].sum().max()
                concentration = (top_cust_vol / qty) * 100 if qty > 0 else 0
                
                results.append({
                    'Year': int(year),
                    'Channel': channel,
                    'SKU': sku,
                    'Product Name': name,
                    'Sold Quantity': qty,
                    'Revenue EUR': rev,
                    'Order Rate %': order_rate,
                    'Repeat Rate %': repeat_rate,
                    'Total Customers': total_custs,
                    'Top Cust Vol %': concentration
                })
                
    res_df = pd.DataFrame(results)
    
    # 4. Generate the rankings per Year and Channel exactly as requested
    for year in years:
        print(f"\n" + "#"*80)
        print(f" YEAR: {int(year)}")
        print("#"*80)
        
        for channel in channels:
            c_df = res_df[(res_df['Year'] == year) & (res_df['Channel'] == channel)]
            if c_df.empty: continue
            
            print(f"\n*** CHANNEL: {channel} ***")
            print("-" * 50)
            
            # 1. Volume Ranking
            print("1) ORDER VOLUME RANKING (Sold Quantity):")
            vol_df = c_df.sort_values('Sold Quantity', ascending=False)
            for i, (_, r) in enumerate(vol_df.head(3).iterrows(), 1): 
                print(f"   {i}. {r['Product Name'][:30]:<30} | {r['Sold Quantity']:,.0f} units")
            
            # 2. Revenue Ranking
            print("\n2) REVENUE RANKING:")
            rev_df = c_df.sort_values('Revenue EUR', ascending=False)
            for i, (_, r) in enumerate(rev_df.head(3).iterrows(), 1): 
                print(f"   {i}. {r['Product Name'][:30]:<30} | {r['Revenue EUR']:,.0f} EUR")
                
            # 3. Order Rate Ranking
            print("\n3) ORDER RATE RANKING (Penetration):")
            or_df = c_df.sort_values('Order Rate %', ascending=False)
            for i, (_, r) in enumerate(or_df.head(3).iterrows(), 1): 
                print(f"   {i}. {r['Product Name'][:30]:<30} | {r['Order Rate %']:.1f}%")
                
            # 4. Repeat Rate Ranking
            print("\n4) REPEAT RATE RANKING:")
            rr_df = c_df.sort_values('Repeat Rate %', ascending=False)
            for i, (_, r) in enumerate(rr_df.head(3).iterrows(), 1): 
                print(f"   {i}. {r['Product Name'][:30]:<30} | {r['Repeat Rate %']:.1f}%")
            
            # 5. NICHE ANALYSIS (< 5% Order Rate detailed check)
            niche_df = c_df[c_df['Order Rate %'] < 5.0].sort_values('Order Rate %', ascending=True)
            print("\n5) NICHE PRODUCT ANALYSIS (< 5% Order Rate):")
            if niche_df.empty:
                print("   - No products with < 5% order rate found in this channel.")
            else:
                for _, r in niche_df.iterrows():
                    print(f"   > [REVIEW] {r['Product Name'][:25]:<25} | Order Rate: {r['Order Rate %']:.1f}%")
                    
                    # Logic to identify *WHY* it's a niche
                    if r['Total Customers'] <= 3:
                        print(f"     -> Insight: HIGHLY SPECIFIC. Only bought by {r['Total Customers']} distinct customers total.")
                    elif r['Top Cust Vol %'] > 50:
                        print(f"     -> Insight: CLIENT DEPENDENT. A single client buys {r['Top Cust Vol %']:.0f}% of all units sold.")
                    else:
                        print(f"     -> Insight: GENERIC/WEAK. {r['Total Customers']} customers buy it, but randomly in low priority.")
            print("")

if __name__ == "__main__":
    main()
