"""
NutrYWell — Demand Forecasting (17 existing SKUs)
==================================================
Builds monthly demand series from ORDER_ERP, fits multiple forecasting
techniques per SKU, evaluates them on a hold-out test set, and exports
forecasts + accuracy metrics.

Methods implemented
-------------------
1. Naive            — last observed value repeated
2. Moving Average 3 — mean of last 3 months
3. SES              — Simple Exponential Smoothing (optimized alpha)
4. Holt             — Double Exp Smoothing (level + trend)
5. Holt-Winters     — Triple Exp Smoothing, additive, 12-month seasonality
6. Linear + seasonal— OLS on time-trend + month dummies

Hold-out: last 6 months used as test window.
Metric   : RMSE (primary) and MAE.
"""

import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, Holt, ExponentialSmoothing

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------- config
PROJECT_DIR  = Path(__file__).resolve().parents[4]
OUT_DIR      = PROJECT_DIR / "Info_W1" / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ERP_FILE     = PROJECT_DIR / "Info_W1" / "Data" / "Order_ERP.csv"
PM_FILE      = PROJECT_DIR / "Info_W1" / "Data" / "Product_master.csv"
TEST_MONTHS  = 6   # hold-out window for accuracy evaluation
SEASON_PER   = 12  # monthly seasonality

# ---------------------------------------------------------------- load data
def load_demand():
    """Return (demand_matrix, sku_list). demand_matrix: rows=months, cols=SKU."""
    pm = pd.read_csv(PM_FILE, sep=";", decimal=",")
    skus = pm["SKU"].tolist()

    erp = pd.read_csv(ERP_FILE, sep=";", decimal=",")
    erp = erp[erp["SKU"].isin(skus)].copy()
    erp["Month"] = pd.to_datetime(erp["Order Date"], dayfirst=True, format="mixed").dt.to_period("M")

    dem = (erp.groupby(["Month", "SKU"])["Quantity"]
              .sum()
              .unstack(fill_value=0)
              .sort_index())

    # Ensure full month index & all 17 SKUs present
    full_idx = pd.period_range(dem.index.min(), dem.index.max(), freq="M")
    dem = dem.reindex(full_idx, fill_value=0)
    for s in skus:
        if s not in dem.columns:
            dem[s] = 0
    dem = dem[skus]
    return dem, skus

# ---------------------------------------------------------------- forecast methods
def f_naive(train, h):
    return np.repeat(train[-1], h)

def f_moving_avg(train, h, window=3):
    return np.repeat(np.mean(train[-window:]), h)

def f_ses(train, h):
    model = SimpleExpSmoothing(train, initialization_method="estimated").fit(optimized=True)
    return np.asarray(model.forecast(h))

def f_holt(train, h):
    model = Holt(train, initialization_method="estimated").fit(optimized=True)
    return np.asarray(model.forecast(h))

def f_holt_winters(train, h, seasonal_periods=SEASON_PER):
    if len(train) < 2 * seasonal_periods:
        return f_holt(train, h)  # fallback if not enough seasons
    model = ExponentialSmoothing(
        train,
        trend="add",
        seasonal="add",
        seasonal_periods=seasonal_periods,
        initialization_method="estimated",
    ).fit(optimized=True)
    return np.asarray(model.forecast(h))

def f_linear_seasonal(train, h):
    """OLS on time trend + 11 month dummies (closed-form, no sklearn dep)."""
    n = len(train)
    t = np.arange(n)
    months = (t % 12)
    # design matrix: intercept, trend, 11 dummies (drop Jan = 0)
    X = np.ones((n, 1 + 1 + 11))
    X[:, 1] = t
    for m in range(1, 12):
        X[:, 1 + m] = (months == m).astype(float)
    beta, *_ = np.linalg.lstsq(X, train, rcond=None)
    t_f = np.arange(n, n + h)
    months_f = (t_f % 12)
    Xf = np.ones((h, 1 + 1 + 11))
    Xf[:, 1] = t_f
    for m in range(1, 12):
        Xf[:, 1 + m] = (months_f == m).astype(float)
    return Xf @ beta

METHODS = {
    "Naive":            f_naive,
    "MovingAvg3":       f_moving_avg,
    "SES":              f_ses,
    "Holt":             f_holt,
    "HoltWinters":      f_holt_winters,
    "LinearSeasonal":   f_linear_seasonal,
}

# ---------------------------------------------------------------- metrics
def rmse(y_true, y_pred): return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
def mae(y_true, y_pred):  return float(np.mean(np.abs(y_true - y_pred)))
def mape(y_true, y_pred):
    mask = y_true > 0
    if mask.sum() == 0: return np.nan
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)

# ---------------------------------------------------------------- per-SKU evaluation
def evaluate_sku(series, sku):
    """Return dict: method -> (forecast_test, rmse, mae, mape)."""
    y = series.values.astype(float)
    # Start from first non-zero month (product launch)
    first_nz = np.argmax(y > 0)
    y_active = y[first_nz:]

    if len(y_active) <= TEST_MONTHS + 3:
        return None, first_nz  # not enough history

    train = y_active[:-TEST_MONTHS]
    test  = y_active[-TEST_MONTHS:]

    results = {}
    for name, fn in METHODS.items():
        try:
            pred = np.clip(fn(train, TEST_MONTHS), 0, None)
            results[name] = {
                "forecast": pred,
                "rmse": rmse(test, pred),
                "mae":  mae(test, pred),
                "mape": mape(test, pred),
            }
        except Exception as e:
            results[name] = {"forecast": None, "rmse": np.inf,
                             "mae": np.inf, "mape": np.nan, "error": str(e)}
    return results, first_nz

# ---------------------------------------------------------------- full-history forecast
def full_fit_forecast(series, method_name, h):
    """Refit on full active history and produce h-step forecast."""
    y = series.values.astype(float)
    first_nz = np.argmax(y > 0)
    y_active = y[first_nz:]
    pred = METHODS[method_name](y_active, h)
    return np.clip(pred, 0, None)

# ---------------------------------------------------------------- main
def main():
    demand, skus = load_demand()
    print(f"Loaded demand: {demand.shape[0]} months × {demand.shape[1]} SKUs")
    print(f"Range: {demand.index.min()} to {demand.index.max()}")
    print(f"Hold-out for evaluation: last {TEST_MONTHS} months\n")

    test_months = demand.index[-TEST_MONTHS:]

    # Per-SKU metrics
    metrics_rows = []
    # Per-SKU test-period forecasts by method
    test_forecasts = {m: pd.DataFrame(index=test_months, columns=skus, dtype=float)
                      for m in METHODS}
    test_forecasts["Actual"] = demand.loc[test_months].copy()
    champion = {}

    for sku in skus:
        res, first_nz = evaluate_sku(demand[sku], sku)
        if res is None:
            print(f"  {sku}: insufficient history, skipping")
            continue
        for name, r in res.items():
            metrics_rows.append({
                "SKU": sku, "Method": name,
                "RMSE": r["rmse"], "MAE": r["mae"], "MAPE_%": r["mape"],
            })
            if r["forecast"] is not None:
                test_forecasts[name].loc[test_months, sku] = r["forecast"]
        # champion = lowest RMSE
        best = min(res.items(), key=lambda kv: kv[1]["rmse"])
        champion[sku] = best[0]
        print(f"  {sku:<10}  best = {best[0]:<16}  "
              f"RMSE={best[1]['rmse']:,.0f}  MAE={best[1]['mae']:,.0f}")

    metrics = pd.DataFrame(metrics_rows)

    # Aggregate ranking
    agg = (metrics.groupby("Method")[["RMSE", "MAE", "MAPE_%"]]
                   .mean()
                   .sort_values("RMSE"))
    print("\n=== Average accuracy across all SKUs (lower = better) ===")
    print(agg.round(1).to_string())
    print(f"\nGlobal best method by avg RMSE: {agg.index[0]}")

    champion_df = (pd.Series(champion, name="Champion_Method")
                     .rename_axis("SKU").reset_index())
    print("\n=== Champion method per SKU ===")
    print(champion_df.to_string(index=False))

    # -------- export ------------------------------------------------
    out_file = OUT_DIR / "forecasts.xlsx"
    with pd.ExcelWriter(out_file, engine="openpyxl") as w:
        demand.astype(int).to_excel(w, sheet_name="Demand_History")
        metrics.round(2).to_excel(w, sheet_name="Accuracy_Per_SKU", index=False)
        agg.round(2).to_excel(w, sheet_name="Accuracy_Avg_By_Method")
        champion_df.to_excel(w, sheet_name="Champion_Per_SKU", index=False)
        for name, df in test_forecasts.items():
            df.round(1).to_excel(w, sheet_name=f"TestFcst_{name[:20]}")

    print(f"\nSaved: {out_file}")
    return demand, metrics, champion, test_forecasts


if __name__ == "__main__":
    main()
