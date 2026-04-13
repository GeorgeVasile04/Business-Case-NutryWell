import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from math import pi

def scale_1_to_10(series, use_rank=True):
    """
    Scales a pandas Series from 1 to 10.
    use_rank=True uses percentile ranking to ignore massive outliers.
    """
    if use_rank:
        # Ranks from 0.0 to 1.0, then multiply by 9 and add 1 -> range [1, 10]
        return 1 + (series.rank(pct=True) * 9)
    else:
        # Standard min-max normalization
        min_v = series.min()
        max_v = series.max()
        if max_v == min_v: return pd.Series(5.5, index=series.index)
        return 1 + ((series - min_v) / (max_v - min_v)) * 9

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'Data')
    
    print("Loading data...")
    orders = pd.read_csv(os.path.join(data_dir, "Order_ERP.csv"), sep=";")
    products = pd.read_csv(os.path.join(data_dir, "Product_master.csv"), sep=";", decimal=",")
    
    orders['Quantity'] = pd.to_numeric(orders['Quantity'], errors='coerce').fillna(0)
    
    if products['Selling Price EUR'].dtype == 'O':
        products['Selling Price EUR'] = products['Selling Price EUR'].str.replace(',', '.')
    products['Selling Price EUR'] = pd.to_numeric(products['Selling Price EUR'], errors='coerce').fillna(0)
    
    df = orders.merge(products[['SKU', 'Selling Price EUR']], on='SKU', how='left')
    df['Product Name'] = df['Product Name'].fillna(df['SKU'])
    df['Line Revenue'] = df['Quantity'] * df['Selling Price EUR']
    
    total_distinct_orders = df['Order ID'].nunique()
    
    # Aggregate data by Product
    print("Aggregating metrics...")
    prod_grp = df.groupby('Product Name')
    
    agg_df = pd.DataFrame()
    agg_df['Volume'] = prod_grp['Quantity'].sum()
    agg_df['Revenue'] = prod_grp['Line Revenue'].sum()
    
    # Order Rate (% of total orders containing this product)
    agg_df['Order Rate'] = (prod_grp['Order ID'].nunique() / total_distinct_orders) * 100
    
    # Repeat Rate
    def calc_repeat(group):
        custs = group.groupby('Customer ID')['Order ID'].nunique()
        if len(custs) == 0: return 0
        return (sum(custs > 1) / len(custs)) * 100
        
    agg_df['Repeat Rate'] = prod_grp.apply(calc_repeat)
    agg_df = agg_df.reset_index()
    
    # Scale from 1 to 10
    # We use rank scaling here because Volume/Revenue usually have huge outliers
    agg_df['Volume_1_10'] = scale_1_to_10(agg_df['Volume'], use_rank=True)
    agg_df['Revenue_1_10'] = scale_1_to_10(agg_df['Revenue'], use_rank=True)
    agg_df['Order_Rate_1_10'] = scale_1_to_10(agg_df['Order Rate'], use_rank=True)
    agg_df['Repeat_Rate_1_10'] = scale_1_to_10(agg_df['Repeat Rate'], use_rank=True)

    # Calculate overall shares (% of the total across all 3 years)
    total_volume_overall = agg_df['Volume'].sum()
    total_revenue_overall = agg_df['Revenue'].sum()
    
    agg_df['Volume Share %'] = (agg_df['Volume'] / total_volume_overall) * 100
    agg_df['Revenue Share %'] = (agg_df['Revenue'] / total_revenue_overall) * 100

    # ---------------------------------------------------------
    # OUTPUT ALL PRODUCTS METRICS
    # ---------------------------------------------------------
    # Save the aggregated and scaled metrics to a CSV file
    output_path = os.path.join(base_dir, 'Analysis 1', 'output', 'product_metrics_scaled.csv')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    agg_df.to_csv(output_path, index=False, float_format='%.2f')
    
    print(f"\nMetrics for all products successfully calculated and saved to:\n{output_path}\n")
    
    # Selecting the columns to display
    display_cols = ['Product Name', 'Volume', 'Volume Share %', 'Revenue', 'Revenue Share %', 
                    'Order Rate', 'Repeat Rate', 
                    'Volume_1_10', 'Revenue_1_10', 'Order_Rate_1_10', 'Repeat_Rate_1_10']
    
    # We round up the numbers for display purposes
    format_dict = {
        'Volume': '{:.0f}'.format, 'Revenue': '{:.0f}'.format,
        'Volume Share %': '{:.2f}%'.format, 'Revenue Share %': '{:.2f}%'.format,
        'Order Rate': '{:.1f}'.format, 'Repeat Rate': '{:.1f}'.format, 
        'Order_Rate_1_10': '{:.1f}'.format, 'Repeat_Rate_1_10': '{:.1f}'.format, 
        'Revenue_1_10': '{:.1f}'.format, 'Volume_1_10': '{:.1f}'.format
    }
    
    # Print out nicely to the terminal
    print(agg_df[display_cols].to_string(formatters=format_dict, index=False))

if __name__ == "__main__":
    main()
