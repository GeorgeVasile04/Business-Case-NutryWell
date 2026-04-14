"""
NutrYWell — Purchasing Policy Simulation (Proof of Concept)
============================================================
Replays the last 12 months (Oct 2024 – Sep 2025) under four
purchasing policies and compares the financial outcome:

 Scenario A — CURRENT: flat monthly POs copied from NutrYWell_PO dataset
 Scenario B — FORECAST: Moving Average (3)  forecast drives purchasing
 Scenario C — FORECAST: Simple Exp Smoothing forecast drives purchasing
 Scenario D — FORECAST: Champion method per SKU (best by RMSE on hold-out)

For every (SKU, month) the simulation reports:
    • demand              (actual units sold)
    • purchased_units     (units bought from supplier that month)
    • stock_on_hand_eom   (cumulative purchased − cumulative demand)
    • price_paid_eur      (purchased × standard cost)
    • revenue_eur         (demand × selling price  — user spec: 'sell all demand')
    • profit_eur          (revenue − price_paid)

Aggregates are produced at SKU×month, SKU×year, scenario totals.
The savings vs Scenario A = proof that forecasting releases cash.
"""

import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, Holt, ExponentialSmoothing

warnings.filterwarnings("ignore")

# --------------------------------------------------------------- config
PROJECT_DIR  = Path(__file__).resolve().parents[4]
OUT_DIR      = PROJECT_DIR / "Info_W1" / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ERP_FILE     = PROJECT_DIR / "Info_W1" / "Data" / "Order_ERP.csv"
PO_FILE      = PROJECT_DIR / "Info_W3" / "NutrYWell_PO.csv"
PM_FILE      = PROJECT_DIR / "Info_W1" / "Data" / "Product_master.csv"

SIM_HORIZON  = 12  # simulate the last 12 months
SEASON_PER   = 12

# --------------------------------------------------------------- load
def load_all():
    pm = pd.read_csv(PM_FILE, sep=";", decimal=",")
    skus = pm["SKU"].tolist()
    prices = pm.set_index("SKU")[["Selling Price EUR", "Standard Cost EUR"]]

    erp = pd.read_csv(ERP_FILE, sep=";", decimal=",")
    erp = erp[erp["SKU"].isin(skus)].copy()
    erp["Month"] = pd.to_datetime(erp["Order Date"], dayfirst=True, format="mixed").dt.to_period("M")
    dem = (erp.groupby(["Month", "SKU"])["Quantity"]
              .sum()
              .unstack(fill_value=0)
              .sort_index())
    full_idx = pd.period_range(dem.index.min(), dem.index.max(), freq="M")
    dem = dem.reindex(full_idx, fill_value=0)
    for s in skus:
        if s not in dem.columns:
            dem[s] = 0
    dem = dem[skus]

    po = pd.read_csv(PO_FILE, sep=";", decimal=",")
    po = po[po["SKU"].isin(skus)].copy()
    po["Month"] = pd.to_datetime(po["PO Date"], dayfirst=True, format="mixed").dt.to_period("M")
    po_matrix = (po.groupby(["Month", "SKU"])["Quantity Purchased (units)"]
                   .sum()
                   .unstack(fill_value=0)
                   .reindex(dem.index, fill_value=0))
    for s in skus:
        if s not in po_matrix.columns:
            po_matrix[s] = 0
    po_matrix = po_matrix[skus]

    return dem, po_matrix, prices, skus

# --------------------------------------------------------------- forecast engines
def fcst_moving_avg(train, h, w=3):
    return np.repeat(np.mean(train[-w:]), h)

def fcst_ses(train, h):
    m = SimpleExpSmoothing(train, initialization_method="estimated").fit(optimized=True)
    return np.asarray(m.forecast(h))

def fcst_naive(train, h):
    return np.repeat(train[-1], h)

def fcst_holt(train, h):
    m = Holt(train, initialization_method="estimated").fit(optimized=True)
    return np.asarray(m.forecast(h))

def fcst_hw(train, h, sp=SEASON_PER):
    if len(train) < 2 * sp:
        return fcst_holt(train, h)
    m = ExponentialSmoothing(train, trend="add", seasonal="add",
                             seasonal_periods=sp,
                             initialization_method="estimated").fit(optimized=True)
    return np.asarray(m.forecast(h))

def fcst_linear_seasonal(train, h):
    n = len(train)
    t = np.arange(n)
    months = t % 12
    X = np.ones((n, 13))
    X[:, 1] = t
    for m in range(1, 12):
        X[:, 1 + m] = (months == m).astype(float)
    beta, *_ = np.linalg.lstsq(X, train, rcond=None)
    t_f = np.arange(n, n + h)
    months_f = t_f % 12
    Xf = np.ones((h, 13))
    Xf[:, 1] = t_f
    for m in range(1, 12):
        Xf[:, 1 + m] = (months_f == m).astype(float)
    return Xf @ beta

METHOD_FN = {
    "Naive":          fcst_naive,
    "MovingAvg3":     fcst_moving_avg,
    "SES":            fcst_ses,
    "Holt":           fcst_holt,
    "HoltWinters":    fcst_hw,
    "LinearSeasonal": fcst_linear_seasonal,
}

# --------------------------------------------------------------- champion selection
def compute_champions(demand, skus, test_months):
    """Refit and rank methods on a hold-out to pick a champion per SKU."""
    champions = {}
    sim_start = len(demand) - test_months
    for sku in skus:
        y = demand[sku].values.astype(float)
        first_nz = int(np.argmax(y > 0))
        y_active = y[first_nz:]
        if len(y_active) <= test_months + 3:
            champions[sku] = "MovingAvg3"
            continue
        tr = y_active[:-test_months]
        te = y_active[-test_months:]
        best_name, best_rmse = "MovingAvg3", np.inf
        for name, fn in METHOD_FN.items():
            try:
                p = np.clip(fn(tr, test_months), 0, None)
                r = np.sqrt(np.mean((te - p) ** 2))
                if r < best_rmse:
                    best_rmse, best_name = r, name
            except Exception:
                pass
        champions[sku] = best_name
    return champions

# --------------------------------------------------------------- scenario builders
def build_forecast_matrix(demand, skus, sim_months, method):
    """Re-fit `method` on train (all months before sim_months), forecast sim_months.
    `method` can be a single string or a dict {SKU: method}."""
    h = len(sim_months)
    train_df = demand.loc[demand.index < sim_months[0]]
    out = pd.DataFrame(0.0, index=sim_months, columns=skus)
    for sku in skus:
        y = train_df[sku].values.astype(float)
        first_nz = int(np.argmax(y > 0)) if y.sum() > 0 else 0
        y_active = y[first_nz:]
        m_name = method[sku] if isinstance(method, dict) else method
        try:
            fc = METHOD_FN[m_name](y_active, h)
        except Exception:
            fc = fcst_moving_avg(y_active, h)
        out[sku] = np.clip(np.round(fc), 0, None)
    return out.astype(int)

# --------------------------------------------------------------- simulator
def simulate(name, demand_sim, purchases, prices):
    """Build the month-level ledger for one scenario."""
    rows = []
    skus = demand_sim.columns
    cum_purch = pd.Series(0, index=skus, dtype=float)
    cum_dem   = pd.Series(0, index=skus, dtype=float)
    for month in demand_sim.index:
        for sku in skus:
            d = float(demand_sim.loc[month, sku])
            b = float(purchases.loc[month, sku])
            cum_purch[sku] += b
            cum_dem[sku]   += d
            stock = cum_purch[sku] - cum_dem[sku]
            sp = float(prices.loc[sku, "Selling Price EUR"])
            cp = float(prices.loc[sku, "Standard Cost EUR"])
            price_paid = b * cp
            revenue    = d * sp      # user spec: 'sell all products in demand'
            rows.append({
                "scenario": name,
                "month":    str(month),
                "year":     month.year,
                "SKU":      sku,
                "demand":              d,
                "purchased_units":     b,
                "stock_on_hand_eom":   stock,
                "price_paid_eur":      price_paid,
                "revenue_eur":         revenue,
                "profit_eur":          revenue - price_paid,
            })
    return pd.DataFrame(rows)

# --------------------------------------------------------------- main
def main():
    demand, po_actual, prices, skus = load_all()
    print(f"Demand: {demand.shape[0]} months × {demand.shape[1]} SKUs "
          f"({demand.index.min()} → {demand.index.max()})")

    sim_months = demand.index[-SIM_HORIZON:]
    print(f"Simulation window: {sim_months[0]} → {sim_months[-1]} "
          f"({SIM_HORIZON} months)\n")
    demand_sim = demand.loc[sim_months]

    # Scenario A — current policy = actual POs in that window
    po_scenA = po_actual.loc[sim_months]

    # Scenario B — MovingAvg3
    po_scenB = build_forecast_matrix(demand, skus, sim_months, "MovingAvg3")

    # Scenario C — SES
    po_scenC = build_forecast_matrix(demand, skus, sim_months, "SES")

    # Scenario D — Champion per SKU (picked on earlier hold-out)
    champions = compute_champions(demand, skus, test_months=6)
    print("Champion method per SKU:")
    for k, v in champions.items():
        print(f"  {k:<10}  {v}")
    po_scenD = build_forecast_matrix(demand, skus, sim_months, champions)

    scenarios = {
        "A_Current_FlatPO":   po_scenA,
        "B_MovingAvg3":       po_scenB,
        "C_SES":              po_scenC,
        "D_Champion_per_SKU": po_scenD,
    }

    all_rows = pd.concat([simulate(n, demand_sim, p, prices)
                          for n, p in scenarios.items()], ignore_index=True)

    # ---- aggregate views -------------------------------------------
    monthly = all_rows.copy()

    yearly = (all_rows.groupby(["scenario", "year", "SKU"])
                      [["demand", "purchased_units",
                        "price_paid_eur", "revenue_eur", "profit_eur"]]
                      .sum()
                      .reset_index())

    totals = (all_rows.groupby("scenario")
                       [["demand", "purchased_units",
                         "price_paid_eur", "revenue_eur", "profit_eur"]]
                       .sum()
                       .reset_index())
    totals["over_purchase_units"] = totals["purchased_units"] - totals["demand"]
    totals["cash_tied_in_overstock_eur"] = totals.apply(
        lambda r: max(0, r["purchased_units"] - r["demand"]) *
                  (r["price_paid_eur"] / r["purchased_units"]
                   if r["purchased_units"] else 0),
        axis=1,
    )

    baseline = totals.loc[totals["scenario"] == "A_Current_FlatPO", "profit_eur"].iloc[0]
    totals["savings_vs_A_eur"] = totals["profit_eur"] - baseline
    totals["savings_vs_A_pct"] = (totals["savings_vs_A_eur"] /
                                  abs(baseline) * 100).round(1)

    print("\n=== SCENARIO TOTALS (12-month simulation) ===")
    print(totals.round(0).to_string(index=False))
    print(f"\nBaseline profit (Scenario A): €{baseline:,.0f}")

    # ---- export ----------------------------------------------------
    out_file = OUT_DIR / "simulation_results.xlsx"
    with pd.ExcelWriter(out_file, engine="openpyxl") as w:
        totals.round(2).to_excel(w, sheet_name="Scenario_Totals", index=False)
        yearly.round(2).to_excel(w, sheet_name="By_SKU_Year", index=False)
        monthly.round(2).to_excel(w, sheet_name="By_SKU_Month", index=False)
        for name, po in scenarios.items():
            po.to_excel(w, sheet_name=f"PO_{name[:25]}")
        demand_sim.to_excel(w, sheet_name="Actual_Demand")
        pd.Series(champions, name="Champion").rename_axis("SKU").reset_index()\
            .to_excel(w, sheet_name="Champions", index=False)

    print(f"\nSaved: {out_file}")


if __name__ == "__main__":
    main()
