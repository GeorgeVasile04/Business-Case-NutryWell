"""
comparison_scalability_solutions.py
====================================
Compares two solutions to NutryWell's warehouse capacity constraint:

  Solution 1 – Extend 3PL contract
    When peak monthly demand exceeds 75% of the current 250k capacity
    (= 187,500 units), the overflow is handled by a 3PL partner at
    EUR 20 / pallet / month.  Pallet density: 750 units / pallet
    (conservative; computed blended average is ~853).

  Solution 2 – Buy a new DC
    Warehouse listing: ~1,000 m², EUR 500k purchase + EUR 100k ERP /
    infrastructure setup = EUR 600k one-time cost.
    Capacity derived from physical parameters (45% usable, 2-level
    racking, 750 units / pallet).

Output: per-SKU pallet table, 3PL year-by-year cost, DC capacity
        calculation, break-even comparison table + chart.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── paths ─────────────────────────────────────────────────────────────────────
NW_PRODUCTS  = "Info_W1/Data/Product_master.csv"
PAR_PRODUCTS = "Info_W2/Data/Products_New_Company.csv"
NW_ORDERS    = "Info_W1/Data/Order_ERP.csv"
PAR_ORDERS   = "Info_W2/Data/Orders_ERP_Parent_company.csv"

# ── constants ─────────────────────────────────────────────────────────────────
MAX_CAPACITY   = 250_000        # DC1+DC2 (178k) + DC3/ACQ-DC1 (72k)
WARN_LIMIT     = int(MAX_CAPACITY * 0.75)   # 187,500 – 3PL trigger
GROWTH_RATE    = 0.15
PERIOD_START   = pd.Timestamp("2024-09-01")
PERIOD_END     = pd.Timestamp("2025-09-30")
START_YEAR     = 2025
HORIZON        = 16             # project up to 2040 for break-even analysis

# pallet & 3PL
PALLET_MAX_KG            = 200   # kg load per pallet (nutraceutical products)
UNITS_PER_PALLET         = 750   # conservative blended average
TPL_EUR_PER_PALLET_MONTH = 20    # EUR / pallet stored / month

# new DC
DC_AREA_M2           = 1_000    # m² total footprint
DC_USABLE_PCT        = 0.45     # fraction usable for storage
DC_PALLET_FOOTPRINT  = 0.96     # m² per Euro-pallet (1.2 x 0.8 m)
DC_STACKING_LEVELS   = 2        # double-deep racking
DC_PURCHASE_EUR      = 500_000
DC_SETUP_EUR         = 100_000  # ERP integration + fit-out
DC_TOTAL_EUR         = DC_PURCHASE_EUR + DC_SETUP_EUR

SEP = "=" * 72

# ── helpers ───────────────────────────────────────────────────────────────────
def read_csv_eu(path):
    df = pd.read_csv(path, sep=";")
    for col in df.columns:
        if df[col].dtype == object:
            try:
                df[col] = df[col].str.replace(",", ".", regex=False).astype(float)
            except (ValueError, AttributeError):
                pass
    return df

def load_orders(path, dayfirst=False):
    df = pd.read_csv(path, sep=";")
    df["Order Date"] = pd.to_datetime(df["Order Date"], dayfirst=dayfirst, errors="coerce")
    df["Quantity"]   = pd.to_numeric(df["Quantity"], errors="coerce")
    return df

# ── load data ─────────────────────────────────────────────────────────────────
nw_prod  = read_csv_eu(NW_PRODUCTS)
par_prod = read_csv_eu(PAR_PRODUCTS)

nw_orders  = load_orders(NW_ORDERS,  dayfirst=True)
par_orders = load_orders(PAR_ORDERS, dayfirst=False)

# filter to analysis period
nw_p  = nw_orders[(nw_orders["Order Date"] >= PERIOD_START) & (nw_orders["Order Date"] <= PERIOD_END)].copy()
par_p = par_orders[(par_orders["Order Date"] >= PERIOD_START) & (par_orders["Order Date"] <= PERIOD_END)].copy()

nw_p["Month"]  = nw_p["Order Date"].dt.to_period("M")
par_p["Month"] = par_p["Order Date"].dt.to_period("M")

# peak monthly demand per SKU (each SKU's best month)
nw_peak_sku  = nw_p.groupby(["SKU","Month"])["Quantity"].sum().groupby("SKU").max()
par_peak_sku = par_p.groupby(["SKU","Month"])["Quantity"].sum().groupby("SKU").max()

# =============================================================================
# SECTION 1 – Units / pallet per SKU
# =============================================================================
print("\n" + SEP)
print("  SECTION 1 – Units per pallet analysis")
print(SEP)
print("  Formula: units_per_pallet = floor(200 kg / net_weight_kg)")
print("  (200 kg conservative pallet weight limit for nutraceutical products)\n")

def build_pallet_table(prod_df, peak_series, demand_col_fallback, label):
    df = prod_df[["SKU","Product Name","Net Weight kg"]].copy()
    df["Net Weight kg"] = pd.to_numeric(
        df["Net Weight kg"].astype(str).str.replace(",", ".", regex=False),
        errors="coerce"
    )
    # drop rows with missing or zero weight to avoid NaN / inf in division
    df = df[df["Net Weight kg"].notna() & (df["Net Weight kg"] > 0)].copy()
    df["Units/pallet"] = (PALLET_MAX_KG / df["Net Weight kg"]).apply(np.floor).astype(int)
    # demand: observed peak, fall back to forecast
    df = df.merge(peak_series.rename("Peak demand"), on="SKU", how="left")
    if demand_col_fallback and demand_col_fallback in prod_df.columns:
        fallback = pd.to_numeric(
            prod_df.set_index("SKU")[demand_col_fallback]
            .astype(str).str.replace(",", ".", regex=False),
            errors="coerce"
        )
        df["Peak demand"] = df.set_index("SKU")["Peak demand"].fillna(fallback).values
    df["Pallets/month"] = (df["Peak demand"] / df["Units/pallet"]).round(1)
    df["Source"] = label
    return df

nw_tbl  = build_pallet_table(nw_prod,  nw_peak_sku,  None,
                              "NutryWell Core")
par_tbl = build_pallet_table(par_prod, par_peak_sku,
                             "Forecast Monthly Demand (units)",
                             "Acquired Line")

# combined table
all_tbl = pd.concat([nw_tbl, par_tbl], ignore_index=True)

# ── print per-source tables ───────────────────────────────────────────────────
for label, grp in all_tbl.groupby("Source", sort=False):
    print(f"  --- {label} ---")
    display = grp[["SKU","Product Name","Net Weight kg","Units/pallet",
                   "Peak demand","Pallets/month"]].copy()
    display["Peak demand"]   = display["Peak demand"].apply(lambda x: f"{x:,.0f}")
    display["Pallets/month"] = display["Pallets/month"].apply(lambda x: f"{x:.1f}")
    print(display.to_string(index=False))
    tot_demand  = grp["Peak demand"].sum()
    tot_pallets = grp["Pallets/month"].sum()
    print(f"\n  Sub-total: {tot_demand:>10,.0f} units   {tot_pallets:>6.1f} pallets")
    blended = tot_demand / tot_pallets if tot_pallets else 0
    print(f"  Blended units/pallet: {blended:.0f}")
    print()

grand_demand  = all_tbl["Peak demand"].sum()
grand_pallets = all_tbl["Pallets/month"].sum()
grand_blended = grand_demand / grand_pallets

print(f"  TOTAL  {grand_demand:,.0f} units  /  {grand_pallets:.1f} pallets")
print(f"  Blended average : {grand_blended:.0f} units/pallet")
print(f"  Conservative    : {UNITS_PER_PALLET} units/pallet  <-- used in all cost calculations")

# =============================================================================
# SECTION 2 – DC capacity if new warehouse is purchased
# =============================================================================
print("\n\n" + SEP)
print("  SECTION 2 – New DC capacity (1,000 m² warehouse)")
print(SEP)

dc_usable   = DC_AREA_M2 * DC_USABLE_PCT
dc_pallets1 = dc_usable / DC_PALLET_FOOTPRINT
dc_pallets2 = dc_pallets1 * DC_STACKING_LEVELS
dc_max_units = dc_pallets2 * UNITS_PER_PALLET
dc_safe_units = dc_max_units * 0.75   # 75% safe operating level

rows_dc = [
    ("Total DC surface",         f"{DC_AREA_M2:,} m2",               "Given"),
    ("Usable for storage (45%)", f"{dc_usable:,.0f} m2",             f"{DC_AREA_M2} x 45%"),
    ("Pallets on 1 level",       f"{dc_pallets1:,.0f}",              f"{dc_usable:.0f} / {DC_PALLET_FOOTPRINT}"),
    ("Pallets on 2 levels",      f"{dc_pallets2:,.0f}",              f"{dc_pallets1:.0f} x 2"),
    ("Units per pallet",         f"{UNITS_PER_PALLET}",              "Conservative blended average"),
    ("Max unit capacity",        f"{dc_max_units:,.0f} units",       f"{dc_pallets2:.0f} x {UNITS_PER_PALLET}"),
    ("Safe capacity (75%)",      f"{dc_safe_units:,.0f} units",      f"{dc_max_units:,.0f} x 75%"),
]
print(f"\n  {'Step':<30} {'Formula':<35} {'Result'}")
print(f"  {'-'*28} {'-'*33} {'-'*20}")
for step, result, formula in rows_dc:
    print(f"  {step:<30} {formula:<35} {result}")

print(f"\n  Purchase cost  : EUR {DC_PURCHASE_EUR:>10,}")
print(f"  Setup cost     : EUR {DC_SETUP_EUR:>10,}  (ERP + infrastructure)")
print(f"  Total one-time : EUR {DC_TOTAL_EUR:>10,}")
print(f"\n  New TOTAL capacity if DC purchased : {MAX_CAPACITY:,} + {dc_max_units:,.0f}"
      f" = {MAX_CAPACITY + dc_max_units:,.0f} units/month")

# =============================================================================
# SECTION 3 – 3PL year-by-year cost projection
# =============================================================================
print("\n\n" + SEP)
print("  SECTION 3 – 3PL extension: year-by-year cost")
print(SEP)
print(f"  3PL triggered when peak demand > {WARN_LIMIT:,} units (75% of {MAX_CAPACITY:,})")
print(f"  Cost: EUR {TPL_EUR_PER_PALLET_MONTH}/pallet/month  |  {UNITS_PER_PALLET} units/pallet\n")

# baseline peak from the combined scenario (mirrors scalability_issue.py)
# compute it from the filtered data
combined_monthly = pd.concat([nw_p, par_p], ignore_index=True).groupby("Month")["Quantity"].sum()
peak_all = combined_monthly.max()

years = list(range(START_YEAR, START_YEAR + HORIZON))
tpl_rows = []
cum_cost = 0.0

print(f"  {'Year':<6} {'Peak units':>12} {'% cap':>7} {'Overflow':>11}"
      f" {'Pallets':>8} {'Annual 3PL':>12} {'Cumulative':>12}")
print(f"  {'-'*72}")

for yr_idx, year in enumerate(years):
    proj = peak_all * (1 + GROWTH_RATE) ** yr_idx
    pct  = proj / MAX_CAPACITY * 100
    overflow = max(0.0, proj - WARN_LIMIT)
    pallets  = overflow / UNITS_PER_PALLET
    annual   = pallets * TPL_EUR_PER_PALLET_MONTH * 12
    cum_cost += annual

    overflow_str = f"{overflow:,.0f}" if overflow > 0 else "—"
    pallets_str  = f"{pallets:.0f}"  if overflow > 0 else "—"
    annual_str   = f"EUR {annual:>9,.0f}" if annual > 0 else "—"
    cum_str      = f"EUR {cum_cost:>9,.0f}" if cum_cost > 0 else "—"

    print(f"  {year:<6} {proj:>12,.0f} {pct:>6.1f}%"
          f" {overflow_str:>11} {pallets_str:>8} {annual_str:>12} {cum_str:>12}")

    tpl_rows.append({
        "year": year, "peak_units": proj, "pct_cap": pct,
        "overflow": overflow, "pallets": pallets,
        "annual_tpl": annual, "cum_tpl": cum_cost
    })

tpl_df = pd.DataFrame(tpl_rows)

# break-even year
be_row = tpl_df[tpl_df["cum_tpl"] >= DC_TOTAL_EUR]
if not be_row.empty:
    be_year = int(be_row.iloc[0]["year"])
    be_cum  = be_row.iloc[0]["cum_tpl"]
else:
    be_year, be_cum = None, None

# =============================================================================
# SECTION 4 – Break-even comparison
# =============================================================================
print("\n\n" + SEP)
print("  SECTION 4 – Break-even comparison")
print(SEP)
print(f"\n  DC purchase cost (one-time)  : EUR {DC_TOTAL_EUR:,}")
if be_year:
    print(f"  3PL cumulative cost crosses DC cost in : {be_year}"
          f"  (+{be_year - START_YEAR} years from now)")
    print(f"  At that point cumulative 3PL spend     : EUR {be_cum:,.0f}")
    print()
    print(f"  Conclusion: buying the DC becomes financially preferable from {be_year}.")
    print(f"  Before {be_year}, extending the 3PL contract is the cheaper option.")
else:
    print("  3PL cumulative cost does not exceed DC cost within the horizon.")

# =============================================================================
# SECTION 5 – Comparison chart
# =============================================================================
fig, ax = plt.subplots(figsize=(13, 7), facecolor="white")
ax.set_facecolor("white")
fig.subplots_adjust(left=0.09, right=0.78, top=0.84, bottom=0.10)

plot_years = [r["year"] for r in tpl_rows]
cum_costs  = [r["cum_tpl"] for r in tpl_rows]

# zone shading: left of break-even = 3PL cheaper, right = DC cheaper
if be_year:
    be_idx = plot_years.index(be_year)
    ax.axvspan(plot_years[0] - 0.5, be_year - 0.5,
               color="#2471A3", alpha=0.05, zorder=0, label="_")
    ax.axvspan(be_year - 0.5, plot_years[-1] + 0.5,
               color="#c0392b", alpha=0.05, zorder=0, label="_")

# 3PL cumulative cost line
ax.plot(plot_years, cum_costs,
        color="#2471A3", linewidth=2.5, marker="o", markersize=5,
        zorder=4, label="Cumulative 3PL cost")

# DC purchase horizontal line
ax.axhline(DC_TOTAL_EUR, color="#c0392b", linewidth=1.8,
           linestyle="--", zorder=3)
ax.annotate(f"DC purchase cost\nEUR {DC_TOTAL_EUR:,}",
            xy=(1.0, DC_TOTAL_EUR),
            xycoords=("axes fraction", "data"),
            xytext=(10, 0), textcoords="offset points",
            va="center", ha="left", fontsize=9,
            color="#c0392b", fontweight="bold", clip_on=False)

# break-even annotation
if be_year and be_year in plot_years:
    be_cum_val = tpl_df[tpl_df["year"] == be_year]["cum_tpl"].values[0]
    ax.annotate(
        f"Break-even\n{be_year}",
        xy=(be_year, be_cum_val),
        xytext=(be_year + 0.3, be_cum_val + DC_TOTAL_EUR * 0.10),
        arrowprops=dict(arrowstyle="->", color="#555555", lw=1.3),
        fontsize=9, fontweight="bold", color="#1a1a2e",
        bbox=dict(boxstyle="round,pad=0.3", fc="white",
                  ec="#cccccc", lw=0.8)
    )
    ax.axvline(be_year, color="#888888", linewidth=1.0,
               linestyle=":", zorder=2)

# value labels on every other point to avoid clutter
for i, (yr, cum) in enumerate(zip(plot_years, cum_costs)):
    if cum > 0 and i % 2 == 0:
        ax.text(yr, cum + DC_TOTAL_EUR * 0.03,
                f"EUR {cum:,.0f}",
                ha="center", va="bottom", fontsize=7.5,
                color="#2471A3")

# axes
ax.set_xticks(plot_years)
ax.set_xticklabels(plot_years, fontsize=10, rotation=0)
ax.set_xlabel("Year", fontsize=11, labelpad=8)
ax.set_ylabel("Cumulative cost (EUR)", fontsize=11, labelpad=10)
ax.yaxis.set_major_formatter(
    plt.FuncFormatter(lambda v, _: f"EUR {v:,.0f}")
)
ax.set_xlim(plot_years[0] - 0.5, plot_years[-1] + 0.5)
ax.set_ylim(0, max(max(cum_costs), DC_TOTAL_EUR) * 1.18)
ax.grid(axis="y", linestyle=":", linewidth=0.7, alpha=0.4,
        color="#aaaaaa", zorder=0)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color("#dddddd")
ax.spines["bottom"].set_color("#dddddd")
ax.tick_params(colors="#555555")

# legend
tpl_line    = plt.Line2D([0],[0], color="#2471A3", linewidth=2.5,
                          marker="o", markersize=5, label="Cumulative 3PL cost")
dc_line     = plt.Line2D([0],[0], color="#c0392b", linewidth=1.8,
                          linestyle="--", label=f"DC purchase cost (EUR {DC_TOTAL_EUR:,})")
cheap_3pl   = mpatches.Patch(facecolor="#2471A3", alpha=0.12,
                              label="3PL cheaper")
cheap_dc    = mpatches.Patch(facecolor="#c0392b", alpha=0.12,
                              label="DC cheaper")
ax.legend(handles=[tpl_line, dc_line, cheap_3pl, cheap_dc],
          loc="upper left", fontsize=9.5, frameon=True,
          framealpha=0.95, edgecolor="#dddddd", handlelength=1.5)

# title block
fig.text(0.09, 0.96, "Make-or-Buy: 3PL Extension vs New DC Purchase",
         fontsize=15, fontweight="bold", va="top", ha="left", color="#1a1a2e")
fig.text(0.09, 0.915,
         f"3PL: EUR {TPL_EUR_PER_PALLET_MONTH}/pallet/month  |  "
         f"{UNITS_PER_PALLET} units/pallet  |  "
         f"Overflow above {WARN_LIMIT:,} units (75% of {MAX_CAPACITY:,})   |   "
         f"DC: EUR {DC_TOTAL_EUR:,} one-time",
         fontsize=9.5, va="top", ha="left", color="#666666")

out_path = "Info_W2/Analysis 2/scalability_comparison.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\nChart saved to: {out_path}")
plt.show()
