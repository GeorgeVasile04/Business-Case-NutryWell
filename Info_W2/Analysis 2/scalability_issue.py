"""
Scalability Analysis - NutryWell
Question: Does NutryWell have enough capacity for the next year to be considered scalable?

Step 1 - Current situation: Sept 2024 to Sept 2025
  - NutryWell core orders    : Info_W1/Data/Order_ERP.csv
  - Acquired-line orders     : Info_W2/Data/Orders_ERP_Parent_company.csv
Output: total orders, total SKUs sold (units), total revenue - with and without the acquired line.

Step 2 - Capacity projection with 15% annual growth
  Logic:
    - Compute the PEAK monthly demand (units) observed in the current year (worst-case month)
    - Compute the AVERAGE monthly demand as the expected-case baseline
    - Apply 15% compound growth year-over-year to both
    - Max warehouse capacity = 178,000 units
    - CAUTION threshold = 80% of capacity = 142,400 units (time to plan expansion)
    - CRITICAL threshold = 100% of capacity = 178,000 units (hard limit reached)
  Why peak AND average?
    - Peak tells you WHEN you need to act (infrastructure / capacity planning horizon)
    - Average tells you WHEN the typical month becomes unsustainable
    Both scenarios are shown for NutryWell Core only and for Core + Acquired Line.
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

# =============================================================================
# STEP 2 – CAPACITY PROJECTION (15% annual growth)
# =============================================================================

MAX_CAPACITY   = 250_000   # BE-DC1 + BE-DC2 (178k) + BE-DC3/ACQ-DC1 (72k)
WARN_PCT       = 0.75      # caution threshold
CRIT_PCT       = 0.85      # critical threshold
GROWTH_RATE    = 0.15      # annual growth rate
WARN_LIMIT     = int(MAX_CAPACITY * WARN_PCT)   # 187,500 units
CRIT_LIMIT     = int(MAX_CAPACITY * CRIT_PCT)   # 212,500 units

# ── monthly unit demand per scenario ─────────────────────────────────────────
# "Month" was added to combined in Step 1; derive it for nw_period separately
nw_period_month  = nw_period.copy()
nw_period_month["Month"] = nw_period_month["Order Date"].dt.to_period("M")

monthly_units_nw  = nw_period_month.groupby("Month")["Quantity"].sum()
monthly_units_all = combined.groupby("Month")["Quantity"].sum()

# ── baseline metrics ──────────────────────────────────────────────────────────
peak_nw   = monthly_units_nw.max()
avg_nw    = monthly_units_nw.mean()
peak_month_nw = monthly_units_nw.idxmax()

peak_all  = monthly_units_all.max()
avg_all   = monthly_units_all.mean()
peak_month_all = monthly_units_all.idxmax()

def capacity_projection(peak, avg, label, start_year=2025):
    """
    Print a year-by-year projection table.
    Year 0 = observed current year (Sept 2024 – Sept 2025).
    Year N = current year + N calendar years.
    """
    print(f"\n{'-'*72}")
    print(f"  Scenario: {label}")
    print(f"{'-'*72}")
    print(f"  Max warehouse capacity        : {MAX_CAPACITY:>10,} units  (DC1+DC2: 178k  +  DC3/ACQ-DC1: 72k)")
    print(f"  Caution threshold  (75% cap) : {WARN_LIMIT:>10,} units")
    print(f"  Critical threshold (85% cap) : {CRIT_LIMIT:>10,} units")
    print(f"  Baseline peak month     : {peak:>10,.0f} units")
    print(f"  Baseline avg  month     : {avg:>10,.0f} units")
    print(f"  Annual growth rate      : {GROWTH_RATE*100:.0f}%")
    print()
    header = (f"  {'Year':<6} {'Calendar':^10} {'Peak Month':>12}"
              f" {'% Cap':>7} {'Avg Month':>12} {'% Cap':>7} {'Status':<10}")
    print(header)
    print(f"  {'-'*68}")

    caution_year_peak = caution_year_avg = critical_year_peak = None

    for yr in range(0, 25):
        proj_peak = peak * (1 + GROWTH_RATE) ** yr
        proj_avg  = avg  * (1 + GROWTH_RATE) ** yr
        pct_peak  = proj_peak / MAX_CAPACITY * 100
        pct_avg   = proj_avg  / MAX_CAPACITY * 100
        cal_year  = start_year + yr

        if proj_peak >= CRIT_LIMIT:
            status = "!! CRITICAL"
        elif proj_peak >= WARN_LIMIT:
            status = "!  CAUTION "
        else:
            status = "   OK      "

        # record first-hit years
        if caution_year_peak is None and proj_peak >= WARN_LIMIT:
            caution_year_peak = cal_year
        if caution_year_avg is None and proj_avg >= WARN_LIMIT:
            caution_year_avg = cal_year
        if critical_year_peak is None and proj_peak >= CRIT_LIMIT:
            critical_year_peak = cal_year

        print(f"  {yr:<6} {cal_year:^10} {proj_peak:>12,.0f}"
              f" {pct_peak:>6.1f}% {proj_avg:>12,.0f} {pct_avg:>6.1f}% {status}")

        if proj_peak >= MAX_CAPACITY:
            break  # no point showing beyond hard limit

    print()
    if caution_year_peak:
        print(f"  >> Peak month hits CAUTION (75%) in year {caution_year_peak}"
              f" (+{caution_year_peak - start_year} years from now)")
    if caution_year_avg:
        print(f"  >> Avg  month hits CAUTION (75%) in year {caution_year_avg}"
              f" (+{caution_year_avg - start_year} years from now)")
    if critical_year_peak:
        print(f"  >> Peak month hits CRITICAL (85%) in year {critical_year_peak}"
              f" (+{critical_year_peak - start_year} years from now)")

# ── monthly unit detail for reference ────────────────────────────────────────
print("\n\n" + "="*72)
print("  STEP 2 – CAPACITY PROJECTION  |  15% annual growth  |  Max 178,000 units")
print("="*72)

print(f"\n{'-'*72}")
print("  Monthly units observed in the current period (Year 0 baseline)")
print(f"{'-'*72}")
unit_pivot = combined.groupby(["Month", "Source"])["Quantity"].sum().unstack(fill_value=0)
unit_pivot["TOTAL"] = unit_pivot.sum(axis=1)
unit_pivot["% of Capacity"] = (unit_pivot["TOTAL"] / MAX_CAPACITY * 100).map("{:.1f}%".format)
print(unit_pivot.to_string())

# ── projections ───────────────────────────────────────────────────────────────
capacity_projection(peak_nw,  avg_nw,  "NutryWell CORE only")
capacity_projection(peak_all, avg_all, "NutryWell CORE + Acquired Line")

# ── key takeaway ──────────────────────────────────────────────────────────────
print(f"\n{'='*72}")
print("  KEY TAKEAWAY")
print(f"{'='*72}")
print(f"  Current peak month (core only)     : {peak_nw:>8,.0f} units"
      f"  ({peak_nw/MAX_CAPACITY*100:.1f}% of capacity)  [{peak_month_nw}]")
print(f"  Current peak month (incl. acquired): {peak_all:>8,.0f} units"
      f"  ({peak_all/MAX_CAPACITY*100:.1f}% of capacity)  [{peak_month_all}]")
print()
print(f"  Total capacity : {MAX_CAPACITY:,} units  (DC1+DC2: 178k  +  DC3/ACQ-DC1: 72k)")
print(f"  Caution (75%)  : {WARN_LIMIT:,} units   |   Critical (85%): {CRIT_LIMIT:,} units")
print()
print("  At 15% annual growth (Core + Acquired Line, peak-month basis):")
print("    CAUTION zone (75%)  reached in 2029  (+4 years)")
print("    CRITICAL zone (85%) reached in 2030  (+5 years)")
print()
print("  With the expanded DC network (250k) NutryWell gains ~2 extra years of")
print("  runway vs the original 178k footprint. Plan DC expansion from 2028.")
print(f"{'='*72}")

# =============================================================================
# STEP 3 – GRAPH: Core + Acquired Line – capacity utilisation over 7 years
# Y-axis is % of max capacity so every bar grows to its true relative height
# and the threshold lines sit at exact 80% and 100% with no capping needed.
# =============================================================================
import matplotlib
matplotlib.use("Agg")           # non-interactive backend (no display needed)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

HORIZON    = 7
START_YEAR = 2025

years     = list(range(START_YEAR, START_YEAR + HORIZON + 1))   # 2025..2032
peak_vals = [peak_all * (1 + GROWTH_RATE) ** yr for yr in range(HORIZON + 1)]
pct_vals  = [v / MAX_CAPACITY * 100 for v in peak_vals]         # % of capacity

Y_MAX_PCT = max(pct_vals) * 1.10    # 10% headroom above the tallest bar

# ── figure & axes ─────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 7), facecolor="white")
ax.set_facecolor("white")
# leave room: top for title block, right for threshold labels
fig.subplots_adjust(left=0.08, right=0.79, top=0.84, bottom=0.14)

# ── zone shading in % coordinates ─────────────────────────────────────────────
ax.axhspan(0,    75,        color="#27ae60", alpha=0.08, zorder=0)   # safe
ax.axhspan(75,   85,        color="#f39c12", alpha=0.10, zorder=0)   # caution
ax.axhspan(85,   Y_MAX_PCT, color="#e74c3c", alpha=0.10, zorder=0)   # critical

# ── bars ──────────────────────────────────────────────────────────────────────
BAR_COLOR = "#2471A3"
x = np.arange(len(years))
bars = ax.bar(x, pct_vals, color=BAR_COLOR, width=0.52,
              zorder=3, edgecolor="white", linewidth=0.6)

# ── % label centred inside each bar ──────────────────────────────────────────
for xi, (val, pct) in enumerate(zip(peak_vals, pct_vals)):
    ax.text(xi, pct * 0.50, f"{pct:.1f}%",
            ha="center", va="center",
            fontsize=11, fontweight="bold", color="white", zorder=5)

# ── unit count as a second row below each year tick label ─────────────────────
for xi, val in enumerate(peak_vals):
    ax.annotate(f"{val:,.0f} units",
                xy=(xi, 0), xycoords="data",
                xytext=(0, -26), textcoords="offset points",
                ha="center", va="top",
                fontsize=8.5, color="#666666", clip_on=False)

# ── threshold lines at 75% and 85% ──────────────────────────────────────────
ax.axhline(85, color="#c0392b", linewidth=1.5, linestyle="--", zorder=4)
ax.axhline(75,  color="#d35400", linewidth=1.5, linestyle="--", zorder=4)

# right-margin labels pinned to the threshold lines (clip_on=False lets them
# draw outside the axes box into the right margin)
common = dict(xycoords=("axes fraction", "data"), textcoords="offset points",
              xytext=(10, 0), va="center", ha="left", clip_on=False)
ax.annotate(f"Critical — {CRIT_LIMIT:,} units (85%)",
            xy=(1.0, 85), fontsize=9, color="#c0392b",
            fontweight="bold", **common)
ax.annotate(f"Caution — {WARN_LIMIT:,} units (75%)",
            xy=(1.0, 75), fontsize=9, color="#d35400",
            fontweight="bold", **common)

# ── axes formatting ───────────────────────────────────────────────────────────
ax.set_xticks(x)
ax.set_xticklabels(years, fontsize=11)
ax.set_xlabel("Year", fontsize=11, labelpad=40)
ax.set_ylabel("Utilisation of max warehouse capacity (%)", fontsize=11, labelpad=10)
ax.set_ylim(0, Y_MAX_PCT)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
ax.grid(axis="y", linestyle=":", linewidth=0.7, alpha=0.35, color="#aaaaaa", zorder=0)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color("#dddddd")
ax.spines["bottom"].set_color("#dddddd")
ax.tick_params(colors="#555555")

# ── legend (zone patches only — 3 items, clean) ───────────────────────────────
legend_handles = [
    mpatches.Patch(facecolor="#27ae60", alpha=0.45, label="Safe  (< 75%)"),
    mpatches.Patch(facecolor="#f39c12", alpha=0.50, label="Caution  (75 – 85%)"),
    mpatches.Patch(facecolor="#e74c3c", alpha=0.50, label="Critical  (> 85%)"),
]
ax.legend(handles=legend_handles, loc="upper left",
          fontsize=9.5, frameon=True, framealpha=0.95,
          edgecolor="#dddddd", handlelength=1.2, borderpad=0.8)

# ── title block (placed in figure coords — never overlaps the axes) ───────────
fig.text(0.08, 0.96,
         "Warehouse Capacity Projection",
         fontsize=16, fontweight="bold", va="top", ha="left", color="#1a1a2e")
fig.text(0.08, 0.915,
         f"NutryWell Core + Acquired Line   |   "
         f"Baseline peak: {peak_all:,.0f} units / month ({peak_month_all})   |   "
         f"+{GROWTH_RATE*100:.0f}% annual growth   |   "
         f"Total capacity: {MAX_CAPACITY:,} units (DC1+DC2: 178k  +  DC3/ACQ-DC1: 72k)",
         fontsize=9.5, va="top", ha="left", color="#666666")

plt.savefig("Info_W2/Analysis 2/capacity_projection.png", dpi=150, bbox_inches="tight")
print("\nGraph saved to: Info_W2/Analysis 2/capacity_projection.png")
plt.show()
