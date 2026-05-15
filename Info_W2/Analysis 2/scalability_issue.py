"""
Scalability Analysis – NutryWell
Question: Does NutryWell have enough capacity for the next year to be considered scalable?

Step 1 – Current situation: Sept 2024 → Sept 2025
  - NutryWell core orders    : Info_W1/Data/Order_ERP.csv
  - Acquired-line orders     : Info_W2/Data/Orders_ERP_Parent_company.csv
Output: total orders, total SKUs sold (units), total revenue – with and without the acquired line.
"""

import pandas as pd

# ── paths ─────────────────────────────────────────────────────────────────────
NW_ORDERS   = "Info_W1/Data/Order_ERP.csv"
PAR_ORDERS  = "Info_W2/Data/Orders_ERP_Parent_company.csv"
NW_PRODUCTS = "Info_W1/Data/Product_master.csv"
PAR_PRODUCTS= "Info_W2/Data/Products_New_Company.csv"

PERIOD_START = pd.Timestamp("2024-09-01")
PERIOD_END   = pd.Timestamp("2025-09-30")

# ── helpers ───────────────────────────────────────────────────────────────────
def load_orders(path, date_col="Order Date", dayfirst=False):
    df = pd.read_csv(path, sep=";")
    df[date_col] = pd.to_datetime(df[date_col], dayfirst=dayfirst, errors="coerce")
    return df

def load_prices(path):
    df = pd.read_csv(path, sep=";")
    # handle European decimal comma (e.g. "6,5" → 6.5)
    df["Selling Price EUR"] = (
        df["Selling Price EUR"]
        .astype(str)
        .str.replace(",", ".", regex=False)
        .astype(float)
    )
    df["Quantity"] = pd.to_numeric(df.get("Quantity", pd.Series(dtype=float)), errors="coerce")
    return df[["SKU", "Selling Price EUR"]]

def filter_period(df, col="Order Date"):
    return df[(df[col] >= PERIOD_START) & (df[col] <= PERIOD_END)].copy()

def summarise(df, label):
    units    = df["Quantity"].sum()
    revenue  = df["Revenue EUR"].sum()
    n_orders = df["Order ID"].nunique()
    n_lines  = len(df)
    n_skus   = df["SKU"].nunique()

    print(f"\n{'-'*60}")
    print(f"  {label}")
    print(f"{'-'*60}")
    print(f"  Unique orders       : {n_orders:>10,.0f}")
    print(f"  Order lines         : {n_lines:>10,.0f}")
    print(f"  Distinct SKUs sold  : {n_skus:>10,.0f}")
    print(f"  Units sold          : {units:>10,.0f}")
    print(f"  Revenue generated   : EUR {revenue:>12,.2f}")

# ── 1. load & clean ───────────────────────────────────────────────────────────
nw_orders  = load_orders(NW_ORDERS,  dayfirst=True)
par_orders = load_orders(PAR_ORDERS, dayfirst=False)

nw_prices  = load_prices(NW_PRODUCTS)
par_prices = load_prices(PAR_PRODUCTS)

# ── 2. filter to Sept 2024 – Sept 2025 ────────────────────────────────────────
nw_period  = filter_period(nw_orders)
par_period = filter_period(par_orders)

# ── 3. attach prices ──────────────────────────────────────────────────────────
nw_period["Quantity"]    = pd.to_numeric(nw_period["Quantity"],  errors="coerce")
par_period["Quantity"]   = pd.to_numeric(par_period["Quantity"], errors="coerce")

nw_period  = nw_period.merge(nw_prices,  on="SKU", how="left")
par_period = par_period.merge(par_prices, on="SKU", how="left")

nw_period["Revenue EUR"]  = nw_period["Quantity"]  * nw_period["Selling Price EUR"]
par_period["Revenue EUR"] = par_period["Quantity"] * par_period["Selling Price EUR"]

# ── 4. combined dataset ───────────────────────────────────────────────────────
par_period["Source"] = "Acquired Line"
nw_period["Source"]  = "NutryWell Core"
combined = pd.concat([nw_period, par_period], ignore_index=True)

# ── 5. results ────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  CURRENT SITUATION  -  Sept 2024 to Sept 2025")
print("="*60)
print(f"  Analysis period : {PERIOD_START.date()}  to  {PERIOD_END.date()}")

summarise(nw_period,  "NutryWell Core products ONLY")
summarise(combined,   "NutryWell Core + Acquired Line (full picture)")

# -- breakdown by source for the combined view --------------------------------
print(f"\n{'-'*60}")
print("  Breakdown by source (combined period)")
print(f"{'-'*60}")
for src, grp in combined.groupby("Source"):
    units   = grp["Quantity"].sum()
    revenue = grp["Revenue EUR"].sum()
    lines   = len(grp)
    print(f"  {src:<22}  lines: {lines:>6,.0f}   "
          f"units: {units:>8,.0f}   revenue: EUR {revenue:>12,.2f}")

# -- monthly trend (combined) -------------------------------------------------
combined["Month"] = combined["Order Date"].dt.to_period("M")
monthly = (
    combined
    .groupby(["Month", "Source"])
    .agg(Units=("Quantity", "sum"), Revenue=("Revenue EUR", "sum"), Lines=("Order ID", "count"))
    .reset_index()
    .sort_values(["Month", "Source"])
)

print(f"\n{'-'*60}")
print("  Monthly order lines by source")
print(f"{'-'*60}")
pivot = monthly.pivot_table(index="Month", columns="Source", values="Lines",
                            aggfunc="sum", fill_value=0)
pivot["TOTAL"] = pivot.sum(axis=1)
print(pivot.to_string())

# -- top 10 SKUs by revenue (combined) ----------------------------------------
top_sku = (
    combined.groupby(["SKU", "Product Name", "Source"])
    .agg(Units=("Quantity", "sum"), Revenue=("Revenue EUR", "sum"))
    .reset_index()
    .sort_values("Revenue", ascending=False)
    .head(10)
)
print(f"\n{'-'*60}")
print("  Top 10 SKUs by Revenue (Sept 2024 – Sept 2025)")
print(f"{'-'*60}")
print(top_sku[["SKU", "Product Name", "Source", "Units", "Revenue"]].to_string(index=False))

print("\n" + "="*60)
print("  END OF CURRENT-SITUATION ANALYSIS")
print("="*60)
