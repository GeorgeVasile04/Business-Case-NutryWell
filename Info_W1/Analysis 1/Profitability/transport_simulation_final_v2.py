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

def pallet_cost(row):
    pallets = max(math.ceil(row['Weight'] / 200), 1)
    cost = pallets * row['Fixed Cost per Pallet EUR'] + row['Weight'] * row['Variable Cost EUR per kg']
    return pd.Series({'Pallets': pallets, 'Transport': cost})

# =============================================================================
# Cheapest DC lookup (for e-commerce all-DC scenarios)
# =============================================================================
ecom_tariffs = delivery_renamed[delivery_renamed['Channel'] == 'E-commerce']
idx_min = ecom_tariffs.groupby('Country')['Fixed Cost per Pallet EUR'].idxmin()
dc_best = ecom_tariffs.loc[idx_min, [
    'Country','Branch/DC','Fixed Cost per Pallet EUR','Variable Cost EUR per kg','Last-mile Share %'
]].copy().rename(columns={'Branch/DC':'Best_DC'})

# =============================================================================
# BUILD SHIPMENT-LEVEL DATA FOR EACH SCENARIO
# =============================================================================

def build_shipments_A():
    """Scenario A: 1 order = 1 shipment, no consolidation."""
    s = df.groupby(['Order ID','Branch/DC','Country','Channel','Year'], as_index=False).agg(
        Weight=('Line_Weight','sum'), Revenue=('Line_Revenue','sum'),
        GP=('Line_GP','sum'), SKU_Qty=('Quantity','sum'))
    s = s.merge(delivery_renamed[['Branch/DC','Country','Channel',
        'Fixed Cost per Pallet EUR','Variable Cost EUR per kg']], 
        on=['Branch/DC','Country','Channel'], how='left')
    s['Pallets'] = np.ceil(s['Weight']/200).clip(lower=1)
    s['Transport'] = s['Pallets']*s['Fixed Cost per Pallet EUR'] + s['Weight']*s['Variable Cost EUR per kg']
    s['Surcharge'] = 0.0
    s['Parcels'] = 0
    return s

def build_shipments_B1():
    """Scenario B1: consolidation only (keep original DC for all channels)."""
    s = df.groupby(['Customer ID','Ship To ID','ERP_Day','Country','Branch/DC','Channel','Year'],
                   as_index=False).agg(
        Weight=('Line_Weight','sum'), Revenue=('Line_Revenue','sum'),
        GP=('Line_GP','sum'), SKU_Qty=('Quantity','sum'),
        Num_Orders=('Order ID','nunique'))
    s = s.merge(delivery_renamed[['Branch/DC','Country','Channel',
        'Fixed Cost per Pallet EUR','Variable Cost EUR per kg']],
        on=['Branch/DC','Country','Channel'], how='left')
    s['Pallets'] = np.ceil(s['Weight']/200).clip(lower=1)
    s['Transport'] = s['Pallets']*s['Fixed Cost per Pallet EUR'] + s['Weight']*s['Variable Cost EUR per kg']
    s['Surcharge'] = 0.0
    s['Parcels'] = 0
    return s

def build_shipments_B2():
    """Scenario B2: consolidation + all DCs for e-commerce (cheapest DC)."""
    df_ecom = df[df['Channel']=='E-commerce']
    df_others = df[df['Channel']!='E-commerce']

    # E-commerce: consolidate without DC constraint, use cheapest DC
    e = df_ecom.groupby(['Customer ID','Ship To ID','ERP_Day','Country','Channel','Year'],
                        as_index=False).agg(
        Weight=('Line_Weight','sum'), Revenue=('Line_Revenue','sum'),
        GP=('Line_GP','sum'), SKU_Qty=('Quantity','sum'),
        Num_Orders=('Order ID','nunique'))
    e = e.merge(dc_best[['Country','Best_DC','Fixed Cost per Pallet EUR',
                          'Variable Cost EUR per kg']], on='Country', how='left')
    e['Branch/DC'] = e['Best_DC']
    e['Pallets'] = np.ceil(e['Weight']/200).clip(lower=1)
    e['Transport'] = e['Pallets']*e['Fixed Cost per Pallet EUR'] + e['Weight']*e['Variable Cost EUR per kg']

    # Others: consolidate within same DC
    o = df_others.groupby(['Customer ID','Ship To ID','ERP_Day','Country','Branch/DC','Channel','Year'],
                          as_index=False).agg(
        Weight=('Line_Weight','sum'), Revenue=('Line_Revenue','sum'),
        GP=('Line_GP','sum'), SKU_Qty=('Quantity','sum'),
        Num_Orders=('Order ID','nunique'))
    o = o.merge(delivery_renamed[['Branch/DC','Country','Channel',
        'Fixed Cost per Pallet EUR','Variable Cost EUR per kg']],
        on=['Branch/DC','Country','Channel'], how='left')
    o['Pallets'] = np.ceil(o['Weight']/200).clip(lower=1)
    o['Transport'] = o['Pallets']*o['Fixed Cost per Pallet EUR'] + o['Weight']*o['Variable Cost EUR per kg']

    s = pd.concat([e, o], ignore_index=True)
    s['Surcharge'] = 0.0
    s['Parcels'] = 0
    return s

def build_shipments_C():
    """Scenario C: B2 + surcharge on e-commerce orders below threshold."""
    s = build_shipments_B2()
    s['Surcharge'] = np.where(
        (s['Channel']=='E-commerce') & (s['Revenue'] < SURCHARGE_THRESHOLD),
        SURCHARGE_AMOUNT, 0.0)
    return s

def build_shipments_D():
    """Scenario D: C + parcel carrier for e-commerce."""
    s = build_shipments_C()
    ecom_mask = s['Channel'] == 'E-commerce'
    s.loc[ecom_mask, 'Transport'] = s.loc[ecom_mask].apply(
        lambda r: get_parcel_cost(r['Country'], r['Weight']), axis=1)
    s.loc[ecom_mask, 'Parcels'] = s.loc[ecom_mask, 'Weight'].apply(
        lambda w: max(1, math.ceil(w / 31.5)))
    s.loc[ecom_mask, 'Pallets'] = 0
    return s

print("Computing all scenarios...")
ships_A  = build_shipments_A()
ships_B1 = build_shipments_B1()
ships_B2 = build_shipments_B2()
ships_C  = build_shipments_C()
ships_D  = build_shipments_D()

# =============================================================================
# CONTRIBUTION ANALYSIS (per shipment)
# =============================================================================
for s in [ships_A, ships_B1, ships_B2, ships_C, ships_D]:
    s['Net_Transport'] = s['Transport'] - s['Surcharge']
    s['Contribution'] = s['GP'] - s['Net_Transport']
    s['Is_Negative'] = (s['Contribution'] < 0).astype(int)

# =============================================================================
# SUMMARY BUILDER
# =============================================================================
def summarize(ships, scenario_name):
    # Count original orders
    if 'Num_Orders' in ships.columns:
        order_col = 'Num_Orders'
    else:
        order_col = None

    grp = ships.groupby(['Year','Channel'], as_index=False).agg(
        Total_Shipments = ('SKU_Qty','count'),
        Total_Pallets   = ('Pallets','sum'),
        Total_Parcels   = ('Parcels','sum'),
        Total_SKU_Qty   = ('SKU_Qty','sum'),
        Transport       = ('Transport','sum'),
        Surcharge       = ('Surcharge','sum'),
        Total_Revenue   = ('Revenue','sum'),
        Total_GP        = ('GP','sum'),
        Neg_Shipments   = ('Is_Negative','sum'),
        **({'Total_Orders': (order_col,'sum')} if order_col else {}),
    )
    if order_col is None:
        grp['Total_Orders'] = grp['Total_Shipments']
    
    grp['Net_Transport']  = grp['Transport'] - grp['Surcharge']
    grp['Neg_Rate']       = grp['Neg_Shipments'] / grp['Total_Shipments'] * 100
    grp['Scenario']       = scenario_name
    return grp

sumA  = summarize(ships_A,  'A')
sumB1 = summarize(ships_B1, 'B1')
sumB2 = summarize(ships_B2, 'B2')
sumC  = summarize(ships_C,  'C')
sumD  = summarize(ships_D,  'D')

# =============================================================================
# PRINT SCENARIO TABLE
# =============================================================================
def print_scenario(s, title, subtitle=""):
    has_sur = s['Surcharge'].sum() > 0
    has_par = s['Total_Parcels'].sum() > 0

    print(f"\n{'='*150}")
    print(f"  {title}")
    if subtitle: print(f"  {subtitle}")
    print(f"{'='*150}")

    hdr = (f"  {'Year':<6} {'Channel':<14} {'Orders':>8} {'Ships':>8} {'Pallets':>8}")
    if has_par: hdr += f" {'Parcels':>8}"
    hdr += f" {'SKU Qty':>10} {'Transport':>18}"
    if has_sur: hdr += f" {'Surcharge':>12} {'Net Transp':>18}"
    hdr += f" {'Neg Ships':>10} {'Neg %':>7}"
    print(f"\n{hdr}")
    print(f"  {'─'*(len(hdr)-2)}")

    for _, r in s.iterrows():
        line = (f"  {r['Year']:<6} {r['Channel']:<14} {r['Total_Orders']:>8,.0f} "
                f"{r['Total_Shipments']:>8,.0f} {r['Total_Pallets']:>8,.0f}")
        if has_par: line += f" {r['Total_Parcels']:>8,.0f}"
        line += f" {r['Total_SKU_Qty']:>10,.0f} {fmt(r['Transport']):>18}"
        if has_sur: line += f" {fmt(r['Surcharge']):>12} {fmt(r['Net_Transport']):>18}"
        line += f" {r['Neg_Shipments']:>10,.0f} {r['Neg_Rate']:>6.1f}%"
        print(line)

    t = s[['Total_Orders','Total_Shipments','Total_Pallets','Total_Parcels',
           'Total_SKU_Qty','Transport','Surcharge','Net_Transport','Neg_Shipments']].sum()
    neg_rate = t['Neg_Shipments']/t['Total_Shipments']*100

    print(f"\n  {'TOTAL':<21} {t['Total_Orders']:>8,.0f} {t['Total_Shipments']:>8,.0f} "
          f"{t['Total_Pallets']:>8,.0f}", end="")
    if has_par: print(f" {t['Total_Parcels']:>8,.0f}", end="")
    print(f" {t['Total_SKU_Qty']:>10,.0f} {fmt(t['Transport']):>18}", end="")
    if has_sur: print(f" {fmt(t['Surcharge']):>12} {fmt(t['Net_Transport']):>18}", end="")
    print(f" {t['Neg_Shipments']:>10,.0f} {neg_rate:>6.1f}%")

# Print all scenarios
print_scenario(sumA,  "SCENARIO A — CURRENT STATE",
    "1 order = 1 shipment = 1 pallet. No consolidation.")

print_scenario(sumB1, "SCENARIO B1 — CONSOLIDATION ONLY (same DC)",
    "Consolidate by Customer+ShipTo+ERP Day+DC. All channels keep their original DC.")

print_scenario(sumB2, "SCENARIO B2 — CONSOLIDATION + ALL DCs FOR E-COMMERCE",
    "B1 + E-commerce: all products in all DCs, ship from cheapest DC per country.")

print_scenario(sumC,  "SCENARIO C — B2 + SMALL ORDER SURCHARGE",
    f"B2 + €{SURCHARGE_AMOUNT:.0f} surcharge on e-commerce orders below €{SURCHARGE_THRESHOLD:.0f}.")

print_scenario(sumD,  "SCENARIO D — C + PARCEL CARRIER FOR E-COMMERCE",
    "C + replace pallet carrier with DPD parcel service for e-commerce.")

# =============================================================================
# MASTER COMPARISON TABLE
# =============================================================================
print(f"\n\n{'='*150}")
print(f"  COMPARISON — ALL SCENARIOS vs CURRENT (A)")
print(f"{'='*150}")

def agg_ch(s):
    return s.groupby('Channel', as_index=False).agg(
        Orders=('Total_Orders','sum'), Shipments=('Total_Shipments','sum'),
        Pallets=('Total_Pallets','sum'), Parcels=('Total_Parcels','sum'),
        SKU_Qty=('Total_SKU_Qty','sum'), Transport=('Transport','sum'),
        Surcharge=('Surcharge','sum'), Net=('Net_Transport','sum'),
        Neg=('Neg_Shipments','sum'))

aa, ab1, ab2, ac, ad = agg_ch(sumA), agg_ch(sumB1), agg_ch(sumB2), agg_ch(sumC), agg_ch(sumD)

print(f"\n  {'Channel':<14} {'Metric':<24} {'A (Current)':>16} {'B1(Consol.)':>16} "
      f"{'B2(+AllDC)':>16} {'C(+Surchg)':>16} {'D(+Parcel)':>16}")
print(f"  {'─'*120}")

for ch in ['E-commerce','Pharmacy','Retail','Retail Sport']:
    ra  = aa[aa['Channel']==ch].iloc[0]
    rb1 = ab1[ab1['Channel']==ch].iloc[0]
    rb2 = ab2[ab2['Channel']==ch].iloc[0]
    rc  = ac[ac['Channel']==ch].iloc[0]
    rd  = ad[ad['Channel']==ch].iloc[0]

    print(f"  {ch:<14} {'Shipments':<24} {ra['Shipments']:>16,.0f} {rb1['Shipments']:>16,.0f} "
          f"{rb2['Shipments']:>16,.0f} {rc['Shipments']:>16,.0f} {rd['Shipments']:>16,.0f}")
    print(f"  {'':<14} {'Pallets':<24} {ra['Pallets']:>16,.0f} {rb1['Pallets']:>16,.0f} "
          f"{rb2['Pallets']:>16,.0f} {rc['Pallets']:>16,.0f} {rd['Pallets']:>16,.0f}")
    if ch == 'E-commerce':
        print(f"  {'':<14} {'Parcels':<24} {ra['Parcels']:>16,.0f} {rb1['Parcels']:>16,.0f} "
              f"{rb2['Parcels']:>16,.0f} {rc['Parcels']:>16,.0f} {rd['Parcels']:>16,.0f}")
    print(f"  {'':<14} {'Transport Cost':<24} {fmt(ra['Transport']):>16} {fmt(rb1['Transport']):>16} "
          f"{fmt(rb2['Transport']):>16} {fmt(rc['Transport']):>16} {fmt(rd['Transport']):>16}")
    if ch == 'E-commerce':
        print(f"  {'':<14} {'Surcharge Revenue':<24} {fmt(ra['Surcharge']):>16} {fmt(rb1['Surcharge']):>16} "
              f"{fmt(rb2['Surcharge']):>16} {fmt(rc['Surcharge']):>16} {fmt(rd['Surcharge']):>16}")
        print(f"  {'':<14} {'Net Transport Cost':<24} {fmt(ra['Net']):>16} {fmt(rb1['Net']):>16} "
              f"{fmt(rb2['Net']):>16} {fmt(rc['Net']):>16} {fmt(rd['Net']):>16}")

    neg_a = ra['Neg']/ra['Shipments']*100
    neg_b1 = rb1['Neg']/rb1['Shipments']*100
    neg_b2 = rb2['Neg']/rb2['Shipments']*100
    neg_c = rc['Neg']/rc['Shipments']*100
    neg_d = rd['Neg']/rd['Shipments']*100
    print(f"  {'':<14} {'Neg. Contribution Rate':<24} {neg_a:>15.1f}% {neg_b1:>15.1f}% "
          f"{neg_b2:>15.1f}% {neg_c:>15.1f}% {neg_d:>15.1f}%")

    sav_b1 = ra['Net']-rb1['Net']; sav_b2 = ra['Net']-rb2['Net']
    sav_c  = ra['Net']-rc['Net'];  sav_d  = ra['Net']-rd['Net']
    print(f"  {'':<14} {'Saving vs A':<24} {'—':>16} {fmt(sav_b1):>16} "
          f"{fmt(sav_b2):>16} {fmt(sav_c):>16} {fmt(sav_d):>16}")
    if ra['Net'] > 0:
        print(f"  {'':<14} {'Saving %':<24} {'—':>16} {sav_b1/ra['Net']*100:>15.1f}% "
              f"{sav_b2/ra['Net']*100:>15.1f}% {sav_c/ra['Net']*100:>15.1f}% {sav_d/ra['Net']*100:>15.1f}%")
    print()

# Grand totals
print(f"  {'─'*120}")
ta  = aa[['Transport','Surcharge','Net','Shipments','Pallets','Parcels','Neg']].sum()
tb1 = ab1[['Transport','Surcharge','Net','Shipments','Pallets','Parcels','Neg']].sum()
tb2 = ab2[['Transport','Surcharge','Net','Shipments','Pallets','Parcels','Neg']].sum()
tc  = ac[['Transport','Surcharge','Net','Shipments','Pallets','Parcels','Neg']].sum()
td  = ad[['Transport','Surcharge','Net','Shipments','Pallets','Parcels','Neg']].sum()

print(f"  {'ALL CHANNELS':<14} {'Net Transport Cost':<24} {fmt(ta['Net']):>16} {fmt(tb1['Net']):>16} "
      f"{fmt(tb2['Net']):>16} {fmt(tc['Net']):>16} {fmt(td['Net']):>16}")
print(f"  {'':<14} {'Neg. Contribution Rate':<24} {ta['Neg']/ta['Shipments']*100:>15.1f}% "
      f"{tb1['Neg']/tb1['Shipments']*100:>15.1f}% {tb2['Neg']/tb2['Shipments']*100:>15.1f}% "
      f"{tc['Neg']/tc['Shipments']*100:>15.1f}% {td['Neg']/td['Shipments']*100:>15.1f}%")
print(f"  {'':<14} {'SAVING vs A':<24} {'—':>16} {fmt(ta['Net']-tb1['Net']):>16} "
      f"{fmt(ta['Net']-tb2['Net']):>16} {fmt(ta['Net']-tc['Net']):>16} {fmt(ta['Net']-td['Net']):>16}")
print(f"  {'':<14} {'SAVING %':<24} {'—':>16} {(ta['Net']-tb1['Net'])/ta['Net']*100:>15.1f}% "
      f"{(ta['Net']-tb2['Net'])/ta['Net']*100:>15.1f}% {(ta['Net']-tc['Net'])/ta['Net']*100:>15.1f}% "
      f"{(ta['Net']-td['Net'])/ta['Net']*100:>15.1f}%")

# =============================================================================
# E-COMMERCE YEARLY PROGRESSION
# =============================================================================
print(f"\n\n{'='*150}")
print(f"  E-COMMERCE — YEARLY NET COST AND NEGATIVE CONTRIBUTION PROGRESSION")
print(f"{'='*150}")

def yearly_ecom(s):
    e = s[s['Channel']=='E-commerce'].groupby('Year', as_index=False).agg(
        Orders=('Total_Orders','sum'), Ships=('Total_Shipments','sum'),
        Transport=('Transport','sum'), Surcharge=('Surcharge','sum'),
        Net=('Net_Transport','sum'), Neg=('Neg_Shipments','sum'))
    e['NegRate'] = e['Neg']/e['Ships']*100
    e['CostPerOrd'] = e['Net']/e['Orders']
    return e

ya, yb1, yb2, yc, yd = yearly_ecom(sumA), yearly_ecom(sumB1), yearly_ecom(sumB2), yearly_ecom(sumC), yearly_ecom(sumD)

print(f"\n  {'Year':<6} {'Ord':>5}  {'A Net':>14} {'A Neg%':>7}  "
      f"{'B1 Net':>14} {'B1 Neg%':>7}  {'B2 Net':>14} {'B2 Neg%':>7}  "
      f"{'C Net':>14} {'C Neg%':>7}  {'D Net':>14} {'D Neg%':>7}")
print(f"  {'─'*140}")
for yr in sorted(ya['Year'].unique()):
    a=ya[ya['Year']==yr].iloc[0]; b1=yb1[yb1['Year']==yr].iloc[0]; b2=yb2[yb2['Year']==yr].iloc[0]
    c=yc[yc['Year']==yr].iloc[0]; d=yd[yd['Year']==yr].iloc[0]
    print(f"  {yr:<6} {a['Orders']:>5,.0f}  {fmt(a['Net']):>14} {a['NegRate']:>6.1f}%  "
          f"{fmt(b1['Net']):>14} {b1['NegRate']:>6.1f}%  {fmt(b2['Net']):>14} {b2['NegRate']:>6.1f}%  "
          f"{fmt(c['Net']):>14} {c['NegRate']:>6.1f}%  {fmt(d['Net']):>14} {d['NegRate']:>6.1f}%")

# Totals
a_t=ya.sum(); b1_t=yb1.sum(); b2_t=yb2.sum(); c_t=yc.sum(); d_t=yd.sum()
print(f"\n  {'TOT':<6} {a_t['Orders']:>5,.0f}  {fmt(a_t['Net']):>14} "
      f"{a_t['Neg']/a_t['Ships']*100:>6.1f}%  {fmt(b1_t['Net']):>14} "
      f"{b1_t['Neg']/b1_t['Ships']*100:>6.1f}%  {fmt(b2_t['Net']):>14} "
      f"{b2_t['Neg']/b2_t['Ships']*100:>6.1f}%  {fmt(c_t['Net']):>14} "
      f"{c_t['Neg']/c_t['Ships']*100:>6.1f}%  {fmt(d_t['Net']):>14} "
      f"{d_t['Neg']/d_t['Ships']*100:>6.1f}%")

print(f"\n  Cost per e-commerce order:")
print(f"  {'A':>4}: {a_t['Net']/a_t['Orders']:>8.2f}€   "
      f"{'B1':>4}: {b1_t['Net']/b1_t['Orders']:>8.2f}€   "
      f"{'B2':>4}: {b2_t['Net']/b2_t['Orders']:>8.2f}€   "
      f"{'C':>4}: {c_t['Net']/c_t['Orders']:>8.2f}€   "
      f"{'D':>4}: {d_t['Net']/d_t['Orders']:>8.2f}€")

# =============================================================================
# EXECUTIVE SUMMARY
# =============================================================================
ecom_a = aa[aa['Channel']=='E-commerce'].iloc[0]
ecom_d = ad[ad['Channel']=='E-commerce'].iloc[0]

print(f"""

{'='*150}
  EXECUTIVE SUMMARY
{'='*150}

  SCENARIOS:
  ┌──────┬────────────────────────────────────────────────────────────────────────────────────────────────┐
  │ A    │ Current: 1 order = 1 pallet, no consolidation, pallet carrier for all channels               │
  │ B1   │ A + consolidate same-day shipments within same DC (all channels)                              │
  │ B2   │ B1 + stock all products in all DCs for e-commerce (ship from cheapest DC)                     │
  │ C    │ B2 + €{SURCHARGE_AMOUNT:.0f} surcharge on e-commerce orders below €{SURCHARGE_THRESHOLD:.0f}                                             │
  │ D    │ C + replace pallet carrier with DPD parcel service for e-commerce                             │
  └──────┴────────────────────────────────────────────────────────────────────────────────────────────────┘

  TOTAL NET TRANSPORT COST (2023-2025):
  ┌─────────────────────┬──────────────────┬──────────────────┬──────────────────┬────────────────────┐
  │ Scenario            │     Net Cost     │  Saving vs A     │  Saving %        │  Neg. Contrib. %   │
  ├─────────────────────┼──────────────────┼──────────────────┼──────────────────┼────────────────────┤
  │ A  (Current)        │ {fmt(ta['Net']):>16} │        —         │       —          │ {ta['Neg']/ta['Shipments']*100:>16.1f}%  │
  │ B1 (Consolidation)  │ {fmt(tb1['Net']):>16} │ {fmt(ta['Net']-tb1['Net']):>16} │ {(ta['Net']-tb1['Net'])/ta['Net']*100:>14.1f}%  │ {tb1['Neg']/tb1['Shipments']*100:>16.1f}%  │
  │ B2 (B1 + All DCs)   │ {fmt(tb2['Net']):>16} │ {fmt(ta['Net']-tb2['Net']):>16} │ {(ta['Net']-tb2['Net'])/ta['Net']*100:>14.1f}%  │ {tb2['Neg']/tb2['Shipments']*100:>16.1f}%  │
  │ C  (B2 + Surcharge) │ {fmt(tc['Net']):>16} │ {fmt(ta['Net']-tc['Net']):>16} │ {(ta['Net']-tc['Net'])/ta['Net']*100:>14.1f}%  │ {tc['Neg']/tc['Shipments']*100:>16.1f}%  │
  │ D  (C + Parcel)     │ {fmt(td['Net']):>16} │ {fmt(ta['Net']-td['Net']):>16} │ {(ta['Net']-td['Net'])/ta['Net']*100:>14.1f}%  │ {td['Neg']/td['Shipments']*100:>16.1f}%  │
  └─────────────────────┴──────────────────┴──────────────────┴──────────────────┴────────────────────┘

  INCREMENTAL VALUE OF EACH LEVER:
    B1  Consolidation only:     {fmt(ta['Net']-tb1['Net']):>20}  (Neg rate: {ta['Neg']/ta['Shipments']*100:.1f}% → {tb1['Neg']/tb1['Shipments']*100:.1f}%)
    B2  + All DCs e-commerce:   {fmt(tb1['Net']-tb2['Net']):>20}  (Neg rate: {tb1['Neg']/tb1['Shipments']*100:.1f}% → {tb2['Neg']/tb2['Shipments']*100:.1f}%)
    C   + Surcharge:            {fmt(tb2['Net']-tc['Net']):>20}  (Neg rate: {tb2['Neg']/tb2['Shipments']*100:.1f}% → {tc['Neg']/tc['Shipments']*100:.1f}%)
    D   + Parcel carrier:       {fmt(tc['Net']-td['Net']):>20}  (Neg rate: {tc['Neg']/tc['Shipments']*100:.1f}% → {td['Neg']/td['Shipments']*100:.1f}%)
    ────────────────────────────────────────────────────────
    TOTAL:                      {fmt(ta['Net']-td['Net']):>20}

  E-COMMERCE COST PER ORDER:
    A:  {ecom_a['Net']/ecom_a['Orders']:.2f}€  →  D:  {ecom_d['Net']/ecom_d['Orders']:.2f}€  (−{(ecom_a['Net']/ecom_a['Orders'])-(ecom_d['Net']/ecom_d['Orders']):.2f}€/order, {((ecom_a['Net']/ecom_a['Orders'])-(ecom_d['Net']/ecom_d['Orders']))/(ecom_a['Net']/ecom_a['Orders'])*100:.0f}% reduction)

  E-COMMERCE NEGATIVE CONTRIBUTION:
    A:  {aa[aa['Channel']=='E-commerce'].iloc[0]['Neg']/aa[aa['Channel']=='E-commerce'].iloc[0]['Shipments']*100:.1f}% of shipments lose money
    D:  {ad[ad['Channel']=='E-commerce'].iloc[0]['Neg']/ad[ad['Channel']=='E-commerce'].iloc[0]['Shipments']*100:.1f}% of shipments lose money
""")
