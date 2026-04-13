import os
import pandas as pd
import numpy as np
import math

# =============================================================================
# DATA LOADING
# =============================================================================
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
data_dir = os.path.join(base_dir, "Data")
if not os.path.exists(data_dir):
    data_dir = "."

try:
    order_erp = pd.read_csv(f"{data_dir}/Order_ERP.csv", sep=';', decimal=',')
    ship_to   = pd.read_csv(f"{data_dir}/Ship_to_master.csv", sep=';', decimal=',')
    customer  = pd.read_csv(f"{data_dir}/Costumer_Master.csv", sep=';', decimal=',')
    product   = pd.read_csv(f"{data_dir}/Product_master.csv", sep=';', decimal=',')
    delivery  = pd.read_csv(f"{data_dir}/Delivery_logistics.csv", sep=';', decimal=',')
    order_erp['Order Date']     = pd.to_datetime(order_erp['Order Date'], format='%d/%m/%Y %H:%M')
    order_erp['ERP Entry Date'] = pd.to_datetime(order_erp['ERP Entry Date'], format='%d/%m/%Y %H:%M')
    print("Loaded from CSV files.\n")
except FileNotFoundError:
    xlsx = "NutrYWell_DATASET_W1.xlsx"
    if not os.path.exists(xlsx):
        xlsx = os.path.join(data_dir, xlsx)
    order_erp = pd.read_excel(xlsx, 'ORDER_ERP')
    ship_to   = pd.read_excel(xlsx, 'SHIP_TO_MASTER')
    customer  = pd.read_excel(xlsx, 'CUSTOMER_MASTER')
    product   = pd.read_excel(xlsx, 'PRODUCT_MASTER')
    delivery  = pd.read_excel(xlsx, 'DELIVERY_LOGISTICS')
    order_erp['Order Date']     = pd.to_datetime(order_erp['Order Date'])
    order_erp['ERP Entry Date'] = pd.to_datetime(order_erp['ERP Entry Date'])
    print("Loaded from XLSX file.\n")

order_erp['Year']    = order_erp['Order Date'].dt.year
order_erp['ERP_Day'] = order_erp['ERP Entry Date'].dt.date

# =============================================================================
# PREPARE BASE DATA
# =============================================================================
df = order_erp.merge(ship_to[['Ship To ID', 'Country']], on='Ship To ID', how='left')
df = df.merge(customer[['Customer ID', 'Channel']], on='Customer ID', how='left')
df = df.merge(product[['SKU', 'Net Weight kg', 'Selling Price EUR']], on='SKU', how='left')
df['Line_Weight']  = df['Net Weight kg'] * df['Quantity']
df['Line_Revenue'] = df['Selling Price EUR'] * df['Quantity']

delivery_renamed = delivery.rename(columns={'From DC': 'Branch/DC', 'To Country': 'Country'})

# =============================================================================
# PARCEL CARRIER PRICING (DPD Belgium — published B2C export rates)
# Source: DPD Belgium public rate card for parcels shipped from Belgium
# "To address" (home delivery) pricing used as default for B2C e-commerce
# =============================================================================
# Zone definitions
PARCEL_ZONES = {
    'BE': 1,
    'NL': 2, 'LU': 2,
    'DE': 3,
    'DK': 4, 'FR': 4, 'AT': 4, 'PL': 4,
    'HU': 5, 'HR': 5, 'LV': 5, 'LT': 5, 'CZ': 5,
    'EE': 6, 'FI': 6, 'PT': 6, 'RO': 6, 'SI': 6, 'SK': 6,
    'BG': 7, 'GR': 7, 'IE': 7, 'IT': 7, 'ES': 7, 'SE': 7, 'CY': 7, 'MT': 7,
}

# Price per parcel by zone and weight bracket (EUR, home delivery)
# Weight brackets: 0-1kg, 1-10kg, 10-20kg
PARCEL_PRICES = {
    #           0-1kg  1-10kg 10-20kg
    1: {1: 5.95,  10: 6.25,  20: 10.00},
    2: {1: 10.00, 10: 14.00, 20: 24.00},
    3: {1: 10.50, 10: 15.50, 20: 25.50},
    4: {1: 11.00, 10: 16.00, 20: 26.00},
    5: {1: 15.00, 10: 20.00, 20: 30.00},
    6: {1: 16.00, 10: 21.00, 20: 31.00},
    7: {1: 21.00, 10: 26.00, 20: 36.00},
}

def get_parcel_cost(country, weight_kg):
    """Compute DPD parcel cost for a given country and weight."""
    zone = PARCEL_ZONES.get(country, 7)  # default to zone 7 if unknown
    brackets = PARCEL_PRICES[zone]
    for max_w in sorted(brackets.keys()):
        if weight_kg <= max_w:
            return brackets[max_w]
    # Over 20kg: split into multiple parcels
    n_parcels = math.ceil(weight_kg / 20.0)
    return n_parcels * brackets[20.0]

# Surcharge parameters
SURCHARGE_THRESHOLD = 50.0  # EUR — orders below this get surcharge
SURCHARGE_AMOUNT    = 4.0   # EUR

def fmt(x):
    """Format number as EUR with space thousands separator."""
    return f"{x:,.2f}".replace(",", " ") + " EUR"

# =============================================================================
# HELPER: compute pallet-based transport cost for a set of shipments
# =============================================================================
def compute_pallet_cost(shipments_df, delivery_tariffs):
    """Add transport columns to a shipment-level DataFrame using pallet tariffs."""
    merged = shipments_df.merge(
        delivery_tariffs[['Branch/DC','Country','Channel',
                          'Fixed Cost per Pallet EUR','Variable Cost EUR per kg',
                          'Last-mile Share %']],
        on=['Branch/DC','Country','Channel'], how='left'
    )
    merged['Pallets'] = np.ceil(merged['Order_Total_Weight_kg'] / 200).clip(lower=1)
    merged['Transport_Cost'] = (
        merged['Pallets'] * merged['Fixed Cost per Pallet EUR']
        + merged['Order_Total_Weight_kg'] * merged['Variable Cost EUR per kg']
    )
    return merged

# =============================================================================
# SCENARIO A — CURRENT STATE
# 1 order = 1 shipment = 1 pallet (minimum), no consolidation
# =============================================================================
print("Computing Scenario A (current state)...")

current = df.groupby(
    ['Order ID','Branch/DC','Country','Channel','Year'], as_index=False
).agg(
    Order_Total_Weight_kg = ('Line_Weight','sum'),
    Total_SKU_Qty         = ('Quantity','sum'),
    Order_Revenue         = ('Line_Revenue','sum'),
)

current = compute_pallet_cost(current, delivery_renamed)

# =============================================================================
# SCENARIO B — CONSOLIDATION + ALL DCs FOR E-COMMERCE
# - E-commerce: all products in all DCs, pick cheapest DC, consolidate by
#   Customer + Ship-To + ERP Entry Day
# - Others: keep original DC, consolidate by Customer + Ship-To + ERP Day + DC
# =============================================================================
print("Computing Scenario B (consolidation + all DCs for e-commerce)...")

df_ecom   = df[df['Channel'] == 'E-commerce'].copy()
df_others = df[df['Channel'] != 'E-commerce'].copy()

# Cheapest DC per country for e-commerce
ecom_tariffs = delivery_renamed[delivery_renamed['Channel'] == 'E-commerce']
idx_min = ecom_tariffs.groupby('Country')['Fixed Cost per Pallet EUR'].idxmin()
dc_best = ecom_tariffs.loc[idx_min, [
    'Country','Branch/DC','Fixed Cost per Pallet EUR',
    'Variable Cost EUR per kg','Last-mile Share %'
]].copy()
dc_best.rename(columns={'Branch/DC':'Best_DC'}, inplace=True)

# Consolidate e-commerce
ecom_b = df_ecom.groupby(
    ['Customer ID','Ship To ID','ERP_Day','Country','Channel','Year'], as_index=False
).agg(
    Order_Total_Weight_kg = ('Line_Weight','sum'),
    Total_SKU_Qty         = ('Quantity','sum'),
    Order_Revenue         = ('Line_Revenue','sum'),
    Num_Original_Orders   = ('Order ID','nunique'),
)
ecom_b = ecom_b.merge(dc_best, on='Country', how='left')
ecom_b['Branch/DC'] = ecom_b['Best_DC']
ecom_b['Pallets'] = np.ceil(ecom_b['Order_Total_Weight_kg'] / 200).clip(lower=1)
ecom_b['Transport_Cost'] = (
    ecom_b['Pallets'] * ecom_b['Fixed Cost per Pallet EUR']
    + ecom_b['Order_Total_Weight_kg'] * ecom_b['Variable Cost EUR per kg']
)

# Consolidate others (keep DC)
others_b = df_others.groupby(
    ['Customer ID','Ship To ID','ERP_Day','Country','Branch/DC','Channel','Year'], as_index=False
).agg(
    Order_Total_Weight_kg = ('Line_Weight','sum'),
    Total_SKU_Qty         = ('Quantity','sum'),
    Order_Revenue         = ('Line_Revenue','sum'),
    Num_Original_Orders   = ('Order ID','nunique'),
)
others_b = compute_pallet_cost(others_b, delivery_renamed)
others_b['Num_Original_Orders'] = others_b.get('Num_Original_Orders',
                                                others_b.get('Num_Original_Orders'))

# =============================================================================
# SCENARIO C — B + SMALL ORDER SURCHARGE (€4 on e-commerce orders < €50)
# Transport costs identical to B; surcharge adds revenue recovery
# =============================================================================
print("Computing Scenario C (B + surcharge)...")

# Surcharge is computed on CONSOLIDATED e-commerce shipments
ecom_c = ecom_b.copy()
ecom_c['Surcharge'] = np.where(ecom_c['Order_Revenue'] < SURCHARGE_THRESHOLD,
                                SURCHARGE_AMOUNT, 0.0)
ecom_c['Surcharge_Revenue'] = ecom_c['Surcharge']

others_c = others_b.copy()
others_c['Surcharge'] = 0.0
others_c['Surcharge_Revenue'] = 0.0

# =============================================================================
# SCENARIO D — C + PARCEL CARRIER FOR E-COMMERCE
# Replace pallet transport with DPD parcel pricing for e-commerce only
# Pharma/Retail/Retail Sport keep pallet transport (with consolidation from B)
# =============================================================================
print("Computing Scenario D (C + parcel carrier for e-commerce)...")

ecom_d = ecom_c.copy()  # keeps surcharge from C
ecom_d['Transport_Cost'] = ecom_d.apply(
    lambda r: get_parcel_cost(r['Country'], r['Order_Total_Weight_kg']),
    axis=1
)
ecom_d['Pallets'] = 0  # no pallets — parcels instead
ecom_d['Parcels'] = ecom_d.apply(
    lambda r: max(1, math.ceil(r['Order_Total_Weight_kg'] / 20.0)),
    axis=1
)

others_d = others_c.copy()  # unchanged from C
others_d['Parcels'] = 0

# =============================================================================
# BUILD SUMMARY FUNCTION
# =============================================================================
def build_summary(ecom_df, others_df, scenario_name, has_surcharge=False, has_parcels=False):
    """Build a per-Year-Channel summary from e-commerce and others DataFrames."""
    
    parts = []
    for label, sub in [('ecom', ecom_df), ('others', others_df)]:
        agg_dict = {
            'Total_Shipments':  ('Total_SKU_Qty', 'count'),  # one row = one shipment
            'Total_Pallets':    ('Pallets', 'sum'),
            'Total_SKU_Qty':    ('Total_SKU_Qty', 'sum'),
            'Transport_Cost':   ('Transport_Cost', 'sum'),
            'Order_Revenue':    ('Order_Revenue', 'sum'),
        }
        if 'Num_Original_Orders' in sub.columns:
            agg_dict['Total_Orders'] = ('Num_Original_Orders', 'sum')
        
        if has_surcharge and 'Surcharge_Revenue' in sub.columns:
            agg_dict['Surcharge_Revenue'] = ('Surcharge_Revenue', 'sum')
        
        if has_parcels and 'Parcels' in sub.columns:
            agg_dict['Total_Parcels'] = ('Parcels', 'sum')

        s = sub.groupby(['Year','Channel'], as_index=False).agg(**agg_dict)
        parts.append(s)
    
    result = pd.concat(parts, ignore_index=True)
    result = result.sort_values(['Year','Channel']).reset_index(drop=True)
    
    # Fill missing columns
    if 'Total_Orders' not in result.columns:
        result['Total_Orders'] = result['Total_Shipments']
    if 'Surcharge_Revenue' not in result.columns:
        result['Surcharge_Revenue'] = 0.0
    if 'Total_Parcels' not in result.columns:
        result['Total_Parcels'] = 0
    
    result['Scenario'] = scenario_name
    return result

# Build all summaries
# Scenario A: current uses order-level data
sumA_parts = []
for ch in current['Channel'].unique():
    sub = current[current['Channel']==ch]
    s = sub.groupby(['Year','Channel'], as_index=False).agg(
        Total_Orders     = ('Order ID','count'),
        Total_Shipments  = ('Order ID','count'),
        Total_Pallets    = ('Pallets','sum'),
        Total_SKU_Qty    = ('Total_SKU_Qty','sum'),
        Transport_Cost   = ('Transport_Cost','sum'),
        Order_Revenue    = ('Order_Revenue','sum'),
    )
    sumA_parts.append(s)
sumA = pd.concat(sumA_parts, ignore_index=True).sort_values(['Year','Channel']).reset_index(drop=True)
sumA['Scenario'] = 'A'
sumA['Surcharge_Revenue'] = 0.0
sumA['Total_Parcels'] = 0

sumB = build_summary(ecom_b, others_b, 'B')
sumC = build_summary(ecom_c, others_c, 'C', has_surcharge=True)
sumD = build_summary(ecom_d, others_d, 'D', has_surcharge=True, has_parcels=True)

# =============================================================================
# PRINT FUNCTION
# =============================================================================
def print_scenario(summary, title, subtitle=""):
    """Print a formatted scenario table."""
    print(f"\n{'=' * 140}")
    print(f"  {title}")
    if subtitle:
        print(f"  {subtitle}")
    print(f"{'=' * 140}")
    
    has_surcharge = summary['Surcharge_Revenue'].sum() > 0
    has_parcels   = summary['Total_Parcels'].sum() > 0
    
    hdr = (f"  {'Year':<6} {'Channel':<14} {'Orders':>8} {'Shipments':>10} "
           f"{'Pallets':>8}")
    if has_parcels:
        hdr += f" {'Parcels':>8}"
    hdr += f" {'SKU Qty':>10} {'Transport':>18}"
    if has_surcharge:
        hdr += f" {'Surcharge':>12} {'Net Transport':>18}"
    print(f"\n{hdr}")
    print(f"  {'─' * (len(hdr) - 2)}")
    
    for _, r in summary.iterrows():
        line = (f"  {r['Year']:<6} {r['Channel']:<14} {r['Total_Orders']:>8,.0f} "
                f"{r['Total_Shipments']:>10,.0f} {r['Total_Pallets']:>8,.0f}")
        if has_parcels:
            line += f" {r['Total_Parcels']:>8,.0f}"
        line += f" {r['Total_SKU_Qty']:>10,.0f} {fmt(r['Transport_Cost']):>18}"
        if has_surcharge:
            net = r['Transport_Cost'] - r['Surcharge_Revenue']
            line += f" {fmt(r['Surcharge_Revenue']):>12} {fmt(net):>18}"
        print(line)
    
    t = summary[['Total_Orders','Total_Shipments','Total_Pallets','Total_Parcels',
                  'Total_SKU_Qty','Transport_Cost','Surcharge_Revenue']].sum()
    print(f"\n  {'TOTAL':<21} {t['Total_Orders']:>8,.0f} {t['Total_Shipments']:>10,.0f} "
          f"{t['Total_Pallets']:>8,.0f}", end="")
    if has_parcels:
        print(f" {t['Total_Parcels']:>8,.0f}", end="")
    print(f" {t['Total_SKU_Qty']:>10,.0f} {fmt(t['Transport_Cost']):>18}", end="")
    if has_surcharge:
        print(f" {fmt(t['Surcharge_Revenue']):>12} {fmt(t['Transport_Cost']-t['Surcharge_Revenue']):>18}", end="")
    print()

# =============================================================================
# PRINT ALL SCENARIOS
# =============================================================================
print_scenario(sumA,
    "SCENARIO A — CURRENT STATE",
    "1 order = 1 shipment = 1 pallet. No consolidation. Pallet-based carrier for all channels.")

print_scenario(sumB,
    "SCENARIO B — CONSOLIDATION + ALL DCs FOR E-COMMERCE",
    "Consolidate by Customer+ShipTo+ERP Day. E-commerce: all products in all DCs, cheapest DC per route.")

print_scenario(sumC,
    "SCENARIO C — B + SMALL ORDER SURCHARGE",
    f"Same as B + €{SURCHARGE_AMOUNT:.0f} surcharge on e-commerce orders below €{SURCHARGE_THRESHOLD:.0f}. "
    "Surcharge partially offsets transport cost.")

print_scenario(sumD,
    "SCENARIO D — C + PARCEL CARRIER FOR E-COMMERCE",
    "Same as C + replace pallet carrier with DPD parcel service for e-commerce. "
    "Pharma/Retail/Retail Sport keep pallet carrier (with consolidation).")

# =============================================================================
# COMPARISON TABLE: ALL SCENARIOS VS CURRENT
# =============================================================================
print(f"\n\n{'=' * 140}")
print(f"  COMPARISON — ALL SCENARIOS vs CURRENT (A)")
print(f"{'=' * 140}")

# Aggregate by scenario + channel
def agg_scenario(s):
    return s.groupby('Channel', as_index=False).agg(
        Orders          = ('Total_Orders','sum'),
        Shipments       = ('Total_Shipments','sum'),
        Pallets         = ('Total_Pallets','sum'),
        Parcels         = ('Total_Parcels','sum'),
        SKU_Qty         = ('Total_SKU_Qty','sum'),
        Transport       = ('Transport_Cost','sum'),
        Surcharge       = ('Surcharge_Revenue','sum'),
    )

agg_a = agg_scenario(sumA)
agg_b = agg_scenario(sumB)
agg_c = agg_scenario(sumC)
agg_d = agg_scenario(sumD)

# Net cost = Transport - Surcharge recovery
for a in [agg_a, agg_b, agg_c, agg_d]:
    a['Net_Cost'] = a['Transport'] - a['Surcharge']

print(f"\n  {'Channel':<14} {'Metric':<22} {'A (Current)':>16} {'B (Consol.)':>16} "
      f"{'C (+Surchg)':>16} {'D (+Parcel)':>16}")
print(f"  {'─' * 102}")

for ch in ['E-commerce','Pharmacy','Retail','Retail Sport']:
    ra = agg_a[agg_a['Channel']==ch].iloc[0]
    rb = agg_b[agg_b['Channel']==ch].iloc[0]
    rc = agg_c[agg_c['Channel']==ch].iloc[0]
    rd = agg_d[agg_d['Channel']==ch].iloc[0]
    
    print(f"  {ch:<14} {'Orders':<22} {ra['Orders']:>16,.0f} {rb['Orders']:>16,.0f} "
          f"{rc['Orders']:>16,.0f} {rd['Orders']:>16,.0f}")
    print(f"  {'':<14} {'Shipments':<22} {ra['Shipments']:>16,.0f} {rb['Shipments']:>16,.0f} "
          f"{rc['Shipments']:>16,.0f} {rd['Shipments']:>16,.0f}")
    print(f"  {'':<14} {'Pallets':<22} {ra['Pallets']:>16,.0f} {rb['Pallets']:>16,.0f} "
          f"{rc['Pallets']:>16,.0f} {rd['Pallets']:>16,.0f}")
    if ch == 'E-commerce':
        print(f"  {'':<14} {'Parcels':<22} {ra['Parcels']:>16,.0f} {rb['Parcels']:>16,.0f} "
              f"{rc['Parcels']:>16,.0f} {rd['Parcels']:>16,.0f}")
    print(f"  {'':<14} {'Transport Cost':<22} {fmt(ra['Transport']):>16} {fmt(rb['Transport']):>16} "
          f"{fmt(rc['Transport']):>16} {fmt(rd['Transport']):>16}")
    if ch == 'E-commerce':
        print(f"  {'':<14} {'Surcharge Revenue':<22} {fmt(ra['Surcharge']):>16} {fmt(rb['Surcharge']):>16} "
              f"{fmt(rc['Surcharge']):>16} {fmt(rd['Surcharge']):>16}")
        print(f"  {'':<14} {'Net Cost':<22} {fmt(ra['Net_Cost']):>16} {fmt(rb['Net_Cost']):>16} "
              f"{fmt(rc['Net_Cost']):>16} {fmt(rd['Net_Cost']):>16}")
    print(f"  {'':<14} {'Saving vs A':<22} {'—':>16} "
          f"{fmt(ra['Net_Cost']-rb['Net_Cost']):>16} "
          f"{fmt(ra['Net_Cost']-rc['Net_Cost']):>16} "
          f"{fmt(ra['Net_Cost']-rd['Net_Cost']):>16}")
    sav_b = (ra['Net_Cost']-rb['Net_Cost'])/ra['Net_Cost']*100 if ra['Net_Cost']>0 else 0
    sav_c = (ra['Net_Cost']-rc['Net_Cost'])/ra['Net_Cost']*100 if ra['Net_Cost']>0 else 0
    sav_d = (ra['Net_Cost']-rd['Net_Cost'])/ra['Net_Cost']*100 if ra['Net_Cost']>0 else 0
    print(f"  {'':<14} {'Saving %':<22} {'—':>16} {sav_b:>15.1f}% {sav_c:>15.1f}% {sav_d:>15.1f}%")
    print()

# Grand totals
print(f"  {'─' * 102}")
ta = agg_a[['Transport','Surcharge','Net_Cost','Shipments','Pallets','Parcels']].sum()
tb = agg_b[['Transport','Surcharge','Net_Cost','Shipments','Pallets','Parcels']].sum()
tc = agg_c[['Transport','Surcharge','Net_Cost','Shipments','Pallets','Parcels']].sum()
td = agg_d[['Transport','Surcharge','Net_Cost','Shipments','Pallets','Parcels']].sum()

print(f"  {'ALL CHANNELS':<14} {'Transport Cost':<22} {fmt(ta['Transport']):>16} {fmt(tb['Transport']):>16} "
      f"{fmt(tc['Transport']):>16} {fmt(td['Transport']):>16}")
print(f"  {'':<14} {'Surcharge Revenue':<22} {fmt(ta['Surcharge']):>16} {fmt(tb['Surcharge']):>16} "
      f"{fmt(tc['Surcharge']):>16} {fmt(td['Surcharge']):>16}")
print(f"  {'':<14} {'NET COST':<22} {fmt(ta['Net_Cost']):>16} {fmt(tb['Net_Cost']):>16} "
      f"{fmt(tc['Net_Cost']):>16} {fmt(td['Net_Cost']):>16}")
print(f"  {'':<14} {'TOTAL SAVING vs A':<22} {'—':>16} "
      f"{fmt(ta['Net_Cost']-tb['Net_Cost']):>16} "
      f"{fmt(ta['Net_Cost']-tc['Net_Cost']):>16} "
      f"{fmt(ta['Net_Cost']-td['Net_Cost']):>16}")
print(f"  {'':<14} {'SAVING %':<22} {'—':>16} "
      f"{(ta['Net_Cost']-tb['Net_Cost'])/ta['Net_Cost']*100:>15.1f}% "
      f"{(ta['Net_Cost']-tc['Net_Cost'])/ta['Net_Cost']*100:>15.1f}% "
      f"{(ta['Net_Cost']-td['Net_Cost'])/ta['Net_Cost']*100:>15.1f}%")

# =============================================================================
# YEARLY PROGRESSION
# =============================================================================
print(f"\n\n{'=' * 140}")
print(f"  YEARLY NET COST PROGRESSION — E-COMMERCE ONLY")
print(f"{'=' * 140}")

def yearly_ecom(summ):
    e = summ[summ['Channel']=='E-commerce'].groupby('Year').agg(
        Transport=('Transport_Cost','sum'),
        Surcharge=('Surcharge_Revenue','sum'),
        Shipments=('Total_Shipments','sum'),
        Orders=('Total_Orders','sum'),
    ).reset_index()
    e['Net'] = e['Transport'] - e['Surcharge']
    e['Cost_Per_Order'] = e['Net'] / e['Orders']
    return e

ya, yb, yc, yd = yearly_ecom(sumA), yearly_ecom(sumB), yearly_ecom(sumC), yearly_ecom(sumD)

print(f"\n  {'Year':<6} {'Orders':>7} "
      f"{'A Net Cost':>16} {'B Net Cost':>16} {'C Net Cost':>16} {'D Net Cost':>16}   "
      f"{'A €/ord':>8} {'B €/ord':>8} {'C €/ord':>8} {'D €/ord':>8}")
print(f"  {'─' * 120}")
for yr in sorted(ya['Year'].unique()):
    a = ya[ya['Year']==yr].iloc[0]
    b = yb[yb['Year']==yr].iloc[0]
    c = yc[yc['Year']==yr].iloc[0]
    d = yd[yd['Year']==yr].iloc[0]
    print(f"  {yr:<6} {a['Orders']:>7,.0f} "
          f"{fmt(a['Net']):>16} {fmt(b['Net']):>16} {fmt(c['Net']):>16} {fmt(d['Net']):>16}   "
          f"{a['Cost_Per_Order']:>7.2f}€ {b['Cost_Per_Order']:>7.2f}€ "
          f"{c['Cost_Per_Order']:>7.2f}€ {d['Cost_Per_Order']:>7.2f}€")

print(f"\n  {'TOTAL':<6} {ya['Orders'].sum():>7,.0f} "
      f"{fmt(ya['Net'].sum()):>16} {fmt(yb['Net'].sum()):>16} "
      f"{fmt(yc['Net'].sum()):>16} {fmt(yd['Net'].sum()):>16}   "
      f"{ya['Net'].sum()/ya['Orders'].sum():>7.2f}€ "
      f"{yb['Net'].sum()/yb['Orders'].sum():>7.2f}€ "
      f"{yc['Net'].sum()/yc['Orders'].sum():>7.2f}€ "
      f"{yd['Net'].sum()/yd['Orders'].sum():>7.2f}€")

# =============================================================================
# EXECUTIVE SUMMARY
# =============================================================================
ecom_a = agg_a[agg_a['Channel']=='E-commerce'].iloc[0]
ecom_d = agg_d[agg_d['Channel']=='E-commerce'].iloc[0]
total_saving_d = ta['Net_Cost'] - td['Net_Cost']

print(f"""

{'=' * 140}
  EXECUTIVE SUMMARY
{'=' * 140}

  SCENARIO OVERVIEW:
  ┌─────────┬──────────────────────────────────────────────────────────────────────────────────────┐
  │ A       │ Current state: 1 order = 1 pallet, no consolidation                                │
  │ B       │ A + consolidate same-day shipments + all DCs for e-commerce                         │
  │ C       │ B + €{SURCHARGE_AMOUNT:.0f} surcharge on e-commerce orders below €{SURCHARGE_THRESHOLD:.0f}                                    │
  │ D       │ C + replace pallet carrier with DPD parcel service for e-commerce                   │
  └─────────┴──────────────────────────────────────────────────────────────────────────────────────┘

  TOTAL NET TRANSPORT COST (2023-2025):
  ┌─────────────────────┬──────────────────┬──────────────────┬──────────────────┐
  │ Scenario            │     Net Cost     │  Saving vs A     │  Saving %        │
  ├─────────────────────┼──────────────────┼──────────────────┼──────────────────┤
  │ A (Current)         │ {fmt(ta['Net_Cost']):>16} │        —         │        —         │
  │ B (Consolidation)   │ {fmt(tb['Net_Cost']):>16} │ {fmt(ta['Net_Cost']-tb['Net_Cost']):>16} │ {(ta['Net_Cost']-tb['Net_Cost'])/ta['Net_Cost']*100:>14.1f}%  │
  │ C (B + Surcharge)   │ {fmt(tc['Net_Cost']):>16} │ {fmt(ta['Net_Cost']-tc['Net_Cost']):>16} │ {(ta['Net_Cost']-tc['Net_Cost'])/ta['Net_Cost']*100:>14.1f}%  │
  │ D (C + Parcel)      │ {fmt(td['Net_Cost']):>16} │ {fmt(ta['Net_Cost']-td['Net_Cost']):>16} │ {(ta['Net_Cost']-td['Net_Cost'])/ta['Net_Cost']*100:>14.1f}%  │
  └─────────────────────┴──────────────────┴──────────────────┴──────────────────┘

  E-COMMERCE COST PER ORDER:
    A (Current):  {ecom_a['Net_Cost']/ecom_a['Orders']:.2f} EUR/order
    D (Optimal):  {ecom_d['Net_Cost']/ecom_d['Orders']:.2f} EUR/order
    Reduction:    {(ecom_a['Net_Cost']/ecom_a['Orders'])-(ecom_d['Net_Cost']/ecom_d['Orders']):.2f} EUR/order ({((ecom_a['Net_Cost']/ecom_a['Orders'])-(ecom_d['Net_Cost']/ecom_d['Orders']))/(ecom_a['Net_Cost']/ecom_a['Orders'])*100:.0f}% less)

  INCREMENTAL VALUE OF EACH LEVER (cumulative):
    Consolidation + All DCs:  {fmt(ta['Net_Cost']-tb['Net_Cost'])} saved
    + Surcharge:              {fmt(tb['Net_Cost']-tc['Net_Cost'])} additional
    + Parcel carrier:         {fmt(tc['Net_Cost']-td['Net_Cost'])} additional
    ─────────────────────────────────────────────────
    TOTAL:                    {fmt(ta['Net_Cost']-td['Net_Cost'])} saved vs current

  IMPLEMENTATION NOTES:
    - Consolidation (B): requires ERP batch processing by ERP entry day
    - Surcharge (C): requires e-commerce platform pricing rule update
    - Parcel carrier (D): requires new carrier contract (DPD or equivalent)
    - All DCs for e-commerce: ~{int(ecom_b['Total_SKU_Qty'].sum()/3):,} units/year to duplicate across DCs
      (minimal vs total portfolio volume of {int(sumA['Total_SKU_Qty'].sum()):,} units)
""")
