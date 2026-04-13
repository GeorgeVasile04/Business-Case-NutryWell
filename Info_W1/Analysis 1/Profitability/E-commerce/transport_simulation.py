import os
import pandas as pd
import numpy as np
import math

# =============================================================================
# DATA LOADING
# =============================================================================
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
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
# BASE DATA PREPARATION
# =============================================================================
df = order_erp.merge(ship_to[['Ship To ID', 'Country']], on='Ship To ID', how='left')
df = df.merge(customer[['Customer ID', 'Channel']], on='Customer ID', how='left')
df = df.merge(product[['SKU', 'Net Weight kg', 'Selling Price EUR', 'Gross Margin %']],
              on='SKU', how='left')
df['Line_Weight']  = df['Net Weight kg'] * df['Quantity']
df['Line_Revenue'] = df['Selling Price EUR'] * df['Quantity']
df['Line_GP']      = df['Line_Revenue'] * df['Gross Margin %']

delivery_renamed = delivery.rename(columns={'From DC': 'Branch/DC', 'To Country': 'Country'})

# =============================================================================
# REFERENCE: total orders per Year x Channel (never changes across scenarios)
# =============================================================================
ORDERS_REF = df.groupby(['Year', 'Channel'], as_index=False).agg(
    Total_Orders=('Order ID', 'nunique'))

# =============================================================================
# PARCEL CARRIER PRICING (DPD Belgium published B2C export rates)
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
SURCHARGE_THRESHOLD = 50.0
SURCHARGE_AMOUNT    = 4.0

def get_parcel_cost(country, weight_kg):
    zone = PARCEL_ZONES.get(country, 6)
    brackets = PARCEL_PRICES[zone]
    for max_w in sorted(brackets.keys()):
        if weight_kg <= max_w:
            return brackets[max_w]
    return math.ceil(weight_kg / 31.5) * brackets[31.5]

def fmt(x):
    return f"{x:,.2f}".replace(",", " ") + " EUR"

# Cheapest DC per country for e-commerce
ecom_tariffs = delivery_renamed[delivery_renamed['Channel'] == 'E-commerce']
idx_min = ecom_tariffs.groupby('Country')['Fixed Cost per Pallet EUR'].idxmin()
dc_best = ecom_tariffs.loc[idx_min, [
    'Country','Branch/DC','Fixed Cost per Pallet EUR','Variable Cost EUR per kg','Last-mile Share %'
]].copy().rename(columns={'Branch/DC':'Best_DC'})

# =============================================================================
# SCENARIO BUILDERS — each returns a shipment-level DataFrame
# =============================================================================

def build_A():
    """A: Current — 1 order = 1 shipment = 1 pallet, no consolidation."""
    s = df.groupby(['Order ID','Branch/DC','Country','Channel','Year'], as_index=False).agg(
        Weight=('Line_Weight','sum'), Revenue=('Line_Revenue','sum'),
        GP=('Line_GP','sum'), SKU_Qty=('Quantity','sum'))
    s = s.merge(delivery_renamed[['Branch/DC','Country','Channel',
        'Fixed Cost per Pallet EUR','Variable Cost EUR per kg']],
        on=['Branch/DC','Country','Channel'], how='left')
    s['Pallets']   = np.ceil(s['Weight']/200).clip(lower=1)
    s['Transport']  = s['Pallets']*s['Fixed Cost per Pallet EUR'] + s['Weight']*s['Variable Cost EUR per kg']
    s['Surcharge'] = 0.0
    s['Parcels']   = 0
    return s

def build_B():
    """B: Consolidation only — same DC, consolidate by Cust+ShipTo+ERPDay+DC."""
    s = df.groupby(['Customer ID','Ship To ID','ERP_Day','Country','Branch/DC','Channel','Year'],
                   as_index=False).agg(
        Weight=('Line_Weight','sum'), Revenue=('Line_Revenue','sum'),
        GP=('Line_GP','sum'), SKU_Qty=('Quantity','sum'))
    s = s.merge(delivery_renamed[['Branch/DC','Country','Channel',
        'Fixed Cost per Pallet EUR','Variable Cost EUR per kg']],
        on=['Branch/DC','Country','Channel'], how='left')
    s['Pallets']   = np.ceil(s['Weight']/200).clip(lower=1)
    s['Transport']  = s['Pallets']*s['Fixed Cost per Pallet EUR'] + s['Weight']*s['Variable Cost EUR per kg']
    s['Surcharge'] = 0.0
    s['Parcels']   = 0
    return s

def build_C():
    """C: Consolidation + all DCs for e-commerce (cheapest DC per country)."""
    df_e = df[df['Channel']=='E-commerce']
    df_o = df[df['Channel']!='E-commerce']

    # E-commerce: consolidate ignoring DC, use cheapest
    e = df_e.groupby(['Customer ID','Ship To ID','ERP_Day','Country','Channel','Year'],
                     as_index=False).agg(
        Weight=('Line_Weight','sum'), Revenue=('Line_Revenue','sum'),
        GP=('Line_GP','sum'), SKU_Qty=('Quantity','sum'))
    e = e.merge(dc_best[['Country','Best_DC','Fixed Cost per Pallet EUR',
                          'Variable Cost EUR per kg']], on='Country', how='left')
    e['Branch/DC'] = e['Best_DC']
    e['Pallets']   = np.ceil(e['Weight']/200).clip(lower=1)
    e['Transport']  = e['Pallets']*e['Fixed Cost per Pallet EUR'] + e['Weight']*e['Variable Cost EUR per kg']

    # Others: consolidate within same DC (same as B)
    o = df_o.groupby(['Customer ID','Ship To ID','ERP_Day','Country','Branch/DC','Channel','Year'],
                     as_index=False).agg(
        Weight=('Line_Weight','sum'), Revenue=('Line_Revenue','sum'),
        GP=('Line_GP','sum'), SKU_Qty=('Quantity','sum'))
    o = o.merge(delivery_renamed[['Branch/DC','Country','Channel',
        'Fixed Cost per Pallet EUR','Variable Cost EUR per kg']],
        on=['Branch/DC','Country','Channel'], how='left')
    o['Pallets']   = np.ceil(o['Weight']/200).clip(lower=1)
    o['Transport']  = o['Pallets']*o['Fixed Cost per Pallet EUR'] + o['Weight']*o['Variable Cost EUR per kg']

    s = pd.concat([e, o], ignore_index=True)
    s['Surcharge'] = 0.0
    s['Parcels']   = 0
    return s

def build_D():
    """D: C + surcharge on e-commerce orders below threshold."""
    s = build_C()
    s['Surcharge'] = np.where(
        (s['Channel']=='E-commerce') & (s['Revenue'] < SURCHARGE_THRESHOLD),
        SURCHARGE_AMOUNT, 0.0)
    return s

def build_E():
    """E: D + parcel carrier for e-commerce (replace pallets with parcels)."""
    s = build_D()
    mask = s['Channel'] == 'E-commerce'
    s.loc[mask, 'Transport'] = s.loc[mask].apply(
        lambda r: get_parcel_cost(r['Country'], r['Weight']), axis=1)
    s.loc[mask, 'Parcels'] = s.loc[mask, 'Weight'].apply(
        lambda w: max(1, math.ceil(w / 31.5)))
    s.loc[mask, 'Pallets'] = 0
    return s

# =============================================================================
# BUILD ALL SCENARIOS
# =============================================================================
print("Computing scenarios...")
scenarios = {
    'A': build_A(),
    'B': build_B(),
    'C': build_C(),
    'D': build_D(),
    'E': build_E(),
}

# Add contribution columns to each
for name, s in scenarios.items():
    s['Net_Transport']  = s['Transport'] - s['Surcharge']
    s['Contribution']   = s['GP'] - s['Net_Transport']
    s['Is_Negative']    = (s['Contribution'] < 0).astype(int)

# =============================================================================
# SUMMARIZE — always attach the reference order count
# =============================================================================
def summarize(ships, scenario_name):
    grp = ships.groupby(['Year','Channel'], as_index=False).agg(
        Total_Shipments = ('SKU_Qty', 'count'),
        Total_Pallets   = ('Pallets', 'sum'),
        Total_Parcels   = ('Parcels', 'sum'),
        Total_SKU_Qty   = ('SKU_Qty', 'sum'),
        Transport       = ('Transport', 'sum'),
        Surcharge       = ('Surcharge', 'sum'),
        Total_Revenue   = ('Revenue', 'sum'),
        Total_GP        = ('GP', 'sum'),
        Neg_Shipments   = ('Is_Negative', 'sum'),
    )
    grp['Net_Transport'] = grp['Transport'] - grp['Surcharge']
    grp['Neg_Rate']      = grp['Neg_Shipments'] / grp['Total_Shipments'] * 100
    grp['Scenario']      = scenario_name

    # Attach reference order count (always the same, regardless of scenario)
    grp = grp.merge(ORDERS_REF, on=['Year','Channel'], how='left')

    return grp

summaries = {name: summarize(s, name) for name, s in scenarios.items()}

# =============================================================================
# PRINT SCENARIO TABLE
# =============================================================================
SCENARIO_LABELS = {
    'A': 'SCENARIO A — CURRENT STATE\n  1 order = 1 shipment = 1 pallet. No consolidation.',
    'B': 'SCENARIO B — CONSOLIDATION ONLY (same DC)\n  Consolidate by Customer + Ship-To + ERP Entry Day + DC. Original DC kept for all channels.',
    'C': 'SCENARIO C — CONSOLIDATION + ALL DCs FOR E-COMMERCE\n  B + E-commerce: all products in all DCs, ship from cheapest DC per country.',
    'D': f'SCENARIO D — C + SMALL ORDER SURCHARGE\n  C + €{SURCHARGE_AMOUNT:.0f} surcharge on e-commerce orders below €{SURCHARGE_THRESHOLD:.0f}.',
    'E': 'SCENARIO E — D + PARCEL CARRIER FOR E-COMMERCE\n  D + replace pallet carrier with DPD parcel service for e-commerce.',
}

def print_scenario(s, name):
    has_sur = s['Surcharge'].sum() > 0
    has_par = s['Total_Parcels'].sum() > 0

    print(f"\n{'='*155}")
    for line in SCENARIO_LABELS[name].split('\n'):
        print(f"  {line}")
    print(f"{'='*155}")

    hdr = f"  {'Year':<6} {'Channel':<14} {'Orders':>8} {'Shipments':>10} {'Pallets':>8}"
    if has_par: hdr += f" {'Parcels':>8}"
    hdr += f" {'SKU Qty':>10} {'Transport':>18}"
    if has_sur: hdr += f" {'Surcharge':>12} {'Net Transp':>18}"
    hdr += f" {'Neg Ships':>10} {'Neg %':>7}"
    print(f"\n{hdr}")
    print(f"  {'─'*(len(hdr)-2)}")

    for _, r in s.iterrows():
        line = (f"  {r['Year']:<6} {r['Channel']:<14} {r['Total_Orders']:>8,.0f} "
                f"{r['Total_Shipments']:>10,.0f} {r['Total_Pallets']:>8,.0f}")
        if has_par: line += f" {r['Total_Parcels']:>8,.0f}"
        line += f" {r['Total_SKU_Qty']:>10,.0f} {fmt(r['Transport']):>18}"
        if has_sur: line += f" {fmt(r['Surcharge']):>12} {fmt(r['Net_Transport']):>18}"
        line += f" {r['Neg_Shipments']:>10,.0f} {r['Neg_Rate']:>6.1f}%"
        print(line)

    t = s[['Total_Orders','Total_Shipments','Total_Pallets','Total_Parcels',
           'Total_SKU_Qty','Transport','Surcharge','Net_Transport','Neg_Shipments']].sum()
    nr = t['Neg_Shipments']/t['Total_Shipments']*100

    print(f"\n  {'TOTAL':<21} {t['Total_Orders']:>8,.0f} {t['Total_Shipments']:>10,.0f} "
          f"{t['Total_Pallets']:>8,.0f}", end="")
    if has_par: print(f" {t['Total_Parcels']:>8,.0f}", end="")
    print(f" {t['Total_SKU_Qty']:>10,.0f} {fmt(t['Transport']):>18}", end="")
    if has_sur: print(f" {fmt(t['Surcharge']):>12} {fmt(t['Net_Transport']):>18}", end="")
    print(f" {t['Neg_Shipments']:>10,.0f} {nr:>6.1f}%")

for name in ['A','B','C','D','E']:
    print_scenario(summaries[name], name)

# =============================================================================
# MASTER COMPARISON TABLE
# =============================================================================
def agg_ch(s):
    return s.groupby('Channel', as_index=False).agg(
        Orders=('Total_Orders','sum'), Shipments=('Total_Shipments','sum'),
        Pallets=('Total_Pallets','sum'), Parcels=('Total_Parcels','sum'),
        SKU_Qty=('Total_SKU_Qty','sum'), Transport=('Transport','sum'),
        Surcharge=('Surcharge','sum'), Net=('Net_Transport','sum'),
        Neg=('Neg_Shipments','sum'))

aggs = {name: agg_ch(s) for name, s in summaries.items()}

print(f"\n\n{'='*155}")
print(f"  COMPARISON — ALL SCENARIOS vs CURRENT (A)")
print(f"{'='*155}")

print(f"\n  {'Channel':<14} {'Metric':<26} {'A (Current)':>16} {'B (Consol.)':>16} "
      f"{'C (+AllDC)':>16} {'D (+Surchg)':>16} {'E (+Parcel)':>16}")
print(f"  {'─'*122}")

for ch in ['E-commerce','Pharmacy','Retail','Retail Sport']:
    rows = {n: aggs[n][aggs[n]['Channel']==ch].iloc[0] for n in 'ABCDE'}
    ra = rows['A']

    print(f"  {ch:<14} {'Orders (received)':<26} ", end="")
    for n in 'ABCDE': print(f"{rows[n]['Orders']:>16,.0f}", end="")
    print()

    print(f"  {'':<14} {'Shipments (sent)':<26} ", end="")
    for n in 'ABCDE': print(f"{rows[n]['Shipments']:>16,.0f}", end="")
    print()

    print(f"  {'':<14} {'Pallets':<26} ", end="")
    for n in 'ABCDE': print(f"{rows[n]['Pallets']:>16,.0f}", end="")
    print()

    if ch == 'E-commerce':
        print(f"  {'':<14} {'Parcels':<26} ", end="")
        for n in 'ABCDE': print(f"{rows[n]['Parcels']:>16,.0f}", end="")
        print()

    print(f"  {'':<14} {'Transport Cost':<26} ", end="")
    for n in 'ABCDE': print(f"{fmt(rows[n]['Transport']):>16}", end="")
    print()

    if ch == 'E-commerce':
        print(f"  {'':<14} {'Surcharge Revenue':<26} ", end="")
        for n in 'ABCDE': print(f"{fmt(rows[n]['Surcharge']):>16}", end="")
        print()
        print(f"  {'':<14} {'Net Transport Cost':<26} ", end="")
        for n in 'ABCDE': print(f"{fmt(rows[n]['Net']):>16}", end="")
        print()

    neg_rates = {n: rows[n]['Neg']/rows[n]['Shipments']*100 for n in 'ABCDE'}
    print(f"  {'':<14} {'Neg. Contribution Rate':<26} ", end="")
    for n in 'ABCDE': print(f"{neg_rates[n]:>15.1f}%", end="")
    print()

    print(f"  {'':<14} {'Saving vs A':<26} {'—':>16}", end="")
    for n in 'BCDE': print(f"{fmt(ra['Net']-rows[n]['Net']):>16}", end="")
    print()

    if ra['Net'] > 0:
        print(f"  {'':<14} {'Saving %':<26} {'—':>16}", end="")
        for n in 'BCDE': print(f"{(ra['Net']-rows[n]['Net'])/ra['Net']*100:>15.1f}%", end="")
        print()
    print()

# Grand totals
print(f"  {'─'*122}")
tots = {n: aggs[n][['Transport','Surcharge','Net','Shipments','Pallets','Parcels','Neg','Orders']].sum()
        for n in 'ABCDE'}

print(f"  {'ALL CHANNELS':<14} {'Orders (received)':<26} ", end="")
for n in 'ABCDE': print(f"{tots[n]['Orders']:>16,.0f}", end="")
print()

print(f"  {'':<14} {'Shipments (sent)':<26} ", end="")
for n in 'ABCDE': print(f"{tots[n]['Shipments']:>16,.0f}", end="")
print()

print(f"  {'':<14} {'Net Transport Cost':<26} ", end="")
for n in 'ABCDE': print(f"{fmt(tots[n]['Net']):>16}", end="")
print()

print(f"  {'':<14} {'Neg. Contribution Rate':<26} ", end="")
for n in 'ABCDE': print(f"{tots[n]['Neg']/tots[n]['Shipments']*100:>15.1f}%", end="")
print()

print(f"  {'':<14} {'SAVING vs A':<26} {'—':>16}", end="")
for n in 'BCDE': print(f"{fmt(tots['A']['Net']-tots[n]['Net']):>16}", end="")
print()

print(f"  {'':<14} {'SAVING %':<26} {'—':>16}", end="")
for n in 'BCDE': print(f"{(tots['A']['Net']-tots[n]['Net'])/tots['A']['Net']*100:>15.1f}%", end="")
print()

# =============================================================================
# E-COMMERCE YEARLY PROGRESSION
# =============================================================================
print(f"\n\n{'='*155}")
print(f"  E-COMMERCE — YEARLY NET COST AND NEGATIVE CONTRIBUTION PROGRESSION")
print(f"{'='*155}")

def yearly_ecom(s):
    e = s[s['Channel']=='E-commerce'].groupby('Year', as_index=False).agg(
        Orders=('Total_Orders','sum'), Ships=('Total_Shipments','sum'),
        Transport=('Transport','sum'), Surcharge=('Surcharge','sum'),
        Net=('Net_Transport','sum'), Neg=('Neg_Shipments','sum'))
    e['NegRate'] = e['Neg']/e['Ships']*100
    e['CostPerOrd'] = e['Net']/e['Orders']
    return e

yr = {n: yearly_ecom(summaries[n]) for n in 'ABCDE'}

print(f"\n  {'Year':<6} {'Ord':>5}", end="")
for n in 'ABCDE': print(f"  {n+' Net':>14} {n+' Neg%':>7}", end="")
print()
print(f"  {'─'*135}")

for y in sorted(yr['A']['Year'].unique()):
    rows = {n: yr[n][yr[n]['Year']==y].iloc[0] for n in 'ABCDE'}
    print(f"  {y:<6} {rows['A']['Orders']:>5,.0f}", end="")
    for n in 'ABCDE': print(f"  {fmt(rows[n]['Net']):>14} {rows[n]['NegRate']:>6.1f}%", end="")
    print()

# Totals
t = {n: yr[n].sum() for n in 'ABCDE'}
print(f"\n  {'TOT':<6} {t['A']['Orders']:>5,.0f}", end="")
for n in 'ABCDE':
    net = t[n]['Net']; neg_r = t[n]['Neg']/t[n]['Ships']*100
    print(f"  {fmt(net):>14} {neg_r:>6.1f}%", end="")
print()

print(f"\n  Cost per e-commerce order:")
for n in 'ABCDE':
    print(f"    {n}: {t[n]['Net']/t[n]['Orders']:>8.2f} EUR", end="")
print()

# =============================================================================
# EXECUTIVE SUMMARY
# =============================================================================
ea = aggs['A'][aggs['A']['Channel']=='E-commerce'].iloc[0]
ee = aggs['E'][aggs['E']['Channel']=='E-commerce'].iloc[0]
ta = tots['A']; te = tots['E']

print(f"""

{'='*155}
  EXECUTIVE SUMMARY
{'='*155}

  SCENARIOS:
  ┌──────┬──────────────────────────────────────────────────────────────────────────────────────────────────────┐
  │  A   │ Current: 1 order = 1 pallet, no consolidation, pallet carrier for all channels                     │
  │  B   │ A + consolidate same-day shipments within same DC (all channels)                                    │
  │  C   │ B + stock all products in all DCs for e-commerce (ship from cheapest DC per country)                │
  │  D   │ C + €{SURCHARGE_AMOUNT:.0f} surcharge on e-commerce orders below €{SURCHARGE_THRESHOLD:.0f}                                                           │
  │  E   │ D + replace pallet carrier with DPD parcel service for e-commerce                                   │
  └──────┴──────────────────────────────────────────────────────────────────────────────────────────────────────┘

  Note: Total orders received = {ta['Orders']:,.0f} in ALL scenarios (orders don't change, only shipments do).

  TOTAL NET TRANSPORT COST (2023-2025):
  ┌────────────────────────────┬──────────────────┬──────────────────┬──────────────────┬────────────────────┐
  │ Scenario                   │     Net Cost     │  Saving vs A     │  Saving %        │ Neg. Contrib. Rate │
  ├────────────────────────────┼──────────────────┼──────────────────┼──────────────────┼────────────────────┤
  │ A  Current                 │ {fmt(tots['A']['Net']):>16} │        —         │       —          │ {tots['A']['Neg']/tots['A']['Shipments']*100:>16.1f}%  │
  │ B  Consolidation           │ {fmt(tots['B']['Net']):>16} │ {fmt(ta['Net']-tots['B']['Net']):>16} │ {(ta['Net']-tots['B']['Net'])/ta['Net']*100:>14.1f}%  │ {tots['B']['Neg']/tots['B']['Shipments']*100:>16.1f}%  │
  │ C  B + All DCs e-com       │ {fmt(tots['C']['Net']):>16} │ {fmt(ta['Net']-tots['C']['Net']):>16} │ {(ta['Net']-tots['C']['Net'])/ta['Net']*100:>14.1f}%  │ {tots['C']['Neg']/tots['C']['Shipments']*100:>16.1f}%  │
  │ D  C + Surcharge           │ {fmt(tots['D']['Net']):>16} │ {fmt(ta['Net']-tots['D']['Net']):>16} │ {(ta['Net']-tots['D']['Net'])/ta['Net']*100:>14.1f}%  │ {tots['D']['Neg']/tots['D']['Shipments']*100:>16.1f}%  │
  │ E  D + Parcel carrier      │ {fmt(tots['E']['Net']):>16} │ {fmt(ta['Net']-tots['E']['Net']):>16} │ {(ta['Net']-tots['E']['Net'])/ta['Net']*100:>14.1f}%  │ {tots['E']['Neg']/tots['E']['Shipments']*100:>16.1f}%  │
  └────────────────────────────┴──────────────────┴──────────────────┴──────────────────┴────────────────────┘

  INCREMENTAL VALUE OF EACH LEVER:
    B  Consolidation only:         {fmt(tots['A']['Net']-tots['B']['Net']):>20}  (Neg rate: {tots['A']['Neg']/tots['A']['Shipments']*100:.1f}% → {tots['B']['Neg']/tots['B']['Shipments']*100:.1f}%)
    C  + All DCs for e-commerce:   {fmt(tots['B']['Net']-tots['C']['Net']):>20}  (Neg rate: {tots['B']['Neg']/tots['B']['Shipments']*100:.1f}% → {tots['C']['Neg']/tots['C']['Shipments']*100:.1f}%)
    D  + Surcharge:                {fmt(tots['C']['Net']-tots['D']['Net']):>20}  (Neg rate: {tots['C']['Neg']/tots['C']['Shipments']*100:.1f}% → {tots['D']['Neg']/tots['D']['Shipments']*100:.1f}%)
    E  + Parcel carrier:           {fmt(tots['D']['Net']-tots['E']['Net']):>20}  (Neg rate: {tots['D']['Neg']/tots['D']['Shipments']*100:.1f}% → {tots['E']['Neg']/tots['E']['Shipments']*100:.1f}%)
    ─────────────────────────────────────────────────────────
    TOTAL:                         {fmt(tots['A']['Net']-tots['E']['Net']):>20}

  E-COMMERCE:
    Cost per order:      A: {ea['Net']/ea['Orders']:.2f}€  →  E: {ee['Net']/ee['Orders']:.2f}€  (−{(ea['Net']/ea['Orders'])-(ee['Net']/ee['Orders']):.2f}€/order, {((ea['Net']/ea['Orders'])-(ee['Net']/ee['Orders']))/(ea['Net']/ea['Orders'])*100:.0f}% reduction)
    Neg. contribution:   A: {ea['Neg']/ea['Shipments']*100:.1f}%  →  E: {ee['Neg']/ee['Shipments']*100:.1f}%

{'='*155}
  HOW IS NEGATIVE CONTRIBUTION COMPUTED?
{'='*155}

  For each shipment, the contribution is calculated as:

    Contribution = Gross Profit − Net Transport Cost

  Where:
    Gross Profit      = Revenue − COGS
                      = (Selling Price × Quantity) − (Standard Cost × Quantity)
                      = Revenue × Gross Margin %
                        (since Gross Margin % = (Selling Price − Standard Cost) / Selling Price)

    Net Transport Cost = Transport Cost − Surcharge
    Transport Cost     = Pallets × Fixed Cost per Pallet + Weight × Variable Cost per kg
                         (or DPD parcel rate in Scenario E for e-commerce)

  A shipment has NEGATIVE contribution when:

    Gross Profit < Net Transport Cost

  This means the margin earned from selling the products (after subtracting production
  costs) is not enough to cover the delivery cost.

  Example — a typical e-commerce order in Scenario A:
    1 unit of Ashwagandha Balance (60 caps) shipped to Spain
    Revenue:          1 × €14.00                    = €14.00
    COGS:             1 × €5.04                     = € 5.04
    Gross Profit:     €14.00 − €5.04                = € 8.96  (64% margin)
""")

# =============================================================================
# E-COMMERCE SPECIFIC SUMMARY TABLE (2023-2025)
# =============================================================================
print("\n" + "="*100)
print("  E-COMMERCE ONLY — TOTAL NET TRANSPORT COST & NEGATIVE RATE (2023-2025)")
print("="*100 + "\n")

ecom_a = aggs['A'][aggs['A']['Channel']=='E-commerce'].iloc[0]
ecom_b = aggs['B'][aggs['B']['Channel']=='E-commerce'].iloc[0]
ecom_c = aggs['C'][aggs['C']['Channel']=='E-commerce'].iloc[0]
ecom_d = aggs['D'][aggs['D']['Channel']=='E-commerce'].iloc[0]
ecom_e = aggs['E'][aggs['E']['Channel']=='E-commerce'].iloc[0]

e_rows = {
    'A': ('A  Current',            ecom_a['Net'], 0,                           ecom_a['Neg'] / ecom_a['Shipments'] * 100),
    'B': ('B  Consolidation',      ecom_b['Net'], ecom_a['Net'] - ecom_b['Net'], ecom_b['Neg'] / ecom_b['Shipments'] * 100),
    'C': ('C  B + All DCs e-com',  ecom_c['Net'], ecom_a['Net'] - ecom_c['Net'], ecom_c['Neg'] / ecom_c['Shipments'] * 100),
    'D': ('D  C + Surcharge',      ecom_d['Net'], ecom_a['Net'] - ecom_d['Net'], ecom_d['Neg'] / ecom_d['Shipments'] * 100),
    'E': ('E  D + Parcel carrier', ecom_e['Net'], ecom_a['Net'] - ecom_e['Net'], ecom_e['Neg'] / ecom_e['Shipments'] * 100)
}

print("  ┌────────────────────────────┬────────────────────┬────────────────────┬──────────────────────┐")
print("  │ Scenario                   │      Net Cost      │    Saving vs A     │ Negative Orders Rate │")
print("  ├────────────────────────────┼────────────────────┼────────────────────┼──────────────────────┤")
for k, (name, net, sav, neg) in e_rows.items():
    sav_str = fmt(sav) if k != 'A' else "—"
    print(f"  │ {name:<26} │ {fmt(net):>18} │ {sav_str:>18} │ {neg:>19.1f}% │")
print("  └────────────────────────────┴────────────────────┴────────────────────┴──────────────────────┘")
print("\n")

print("""
    Transport:        1 pallet × ~€57 + 0.06kg × ~€0.07 = €57.00
    Contribution:     €8.96 − €57.00                = −€48.04  ← NEGATIVE

  Same order in Scenario E (parcel carrier):
    Revenue:          €14.00
    COGS:             € 5.04
    Gross Profit:     € 8.96
    Transport:        DPD 0-3kg to Spain             = €18.95
    Surcharge:        order < €50 → €4.00 recovered
    Net Transport:    €18.95 − €4.00                 = €14.95
    Contribution:     €8.96 − €14.95                 = −€5.99  ← still negative, but 8x less loss

  The negative contribution rate is the percentage of shipments where this happens.
  In Scenario A, 91% of e-commerce shipments lose money.
  In Scenario E, this drops to 32% — because the parcel cost (€8-19) is much closer
  to the gross profit (€5-25) than the pallet cost (€43-65) ever was.
""")
