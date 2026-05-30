"""
NutryWell — Combined Demand Forecasting
========================================
Computes total monthly demand (NutryWell + Acquired portfolio), fits
Holt-Winters (additive) and Linear Additive Seasonal models on the
training window, evaluates both on the Sep 2024–Sep 2025 test window,
selects the best by RMSE, overlays a +10 % safety buffer, and shows
NutryWell's historical flat ordering level (178k units/month).

Output: demand_forecast_combined.png  ->  Info_W1/output/
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from statsmodels.tsa.holtwinters import ExponentialSmoothing, Holt

warnings.filterwarnings("ignore")

# ── paths ─────────────────────────────────────────────────────────────────────
PROJECT_DIR    = Path(__file__).resolve().parents[4]
OUT_DIR        = PROJECT_DIR / "Info_W1" / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

NW_ORDERS_FILE = PROJECT_DIR / "Info_W1" / "Data" / "Order_ERP.csv"
NW_PROD_FILE   = PROJECT_DIR / "Info_W1" / "Data" / "Product_master.csv"
AQ_ORDERS_FILE = PROJECT_DIR / "Info_W2" / "Data" / "Orders_ERP_Parent_company.csv"

# ── config ────────────────────────────────────────────────────────────────────
TRAIN_END   = pd.Period("2024-08", freq="M")   # last month used for training
GRAPH_START = pd.Period("2024-09", freq="M")   # forecast / evaluation window start
GRAPH_END   = pd.Period("2025-09", freq="M")   # forecast / evaluation window end
SEASON_PER  = 12

SAFETY      = 0.10   # +10 %

# ── design tokens ─────────────────────────────────────────────────────────────
C_NW     = "#2E86AB"   # NutryWell demand bars — steel-blue
C_AQ     = "#5C4E8C"   # Acquired demand bars  — purple
C_ORDER  = "#E63946"   # NutryWell ordering level line — red
C_HW     = "#27AE60"   # Holt-Winters — green
C_LS     = "#E67E22"   # Linear Seasonal — orange
C_BUF    = "#5FA8C0"   # +10 % band fill — light blue
BG       = "#F7F9FC"   # figure background
PANEL_BG = "#EEF2F7"   # axes background

ORDER_LEVEL = 178_000  # NutryWell historical flat ordering level (DC1 + DC2)


# ── data loading ──────────────────────────────────────────────────────────────
def load_monthly_demand() -> pd.DataFrame:
    """Return DataFrame with columns [NutryWell, Acquired, Total] by Period."""

    nw_prod = pd.read_csv(NW_PROD_FILE, sep=";", decimal=",")
    nw_skus = set(nw_prod["SKU"])

    nw = pd.read_csv(NW_ORDERS_FILE, sep=";", decimal=",")
    nw = nw[nw["SKU"].isin(nw_skus)].copy()
    nw["Month"] = (
        pd.to_datetime(nw["Order Date"], dayfirst=True, format="mixed")
        .dt.to_period("M")
    )
    nw_monthly = nw.groupby("Month")["Quantity"].sum().rename("NutryWell")

    aq = pd.read_csv(AQ_ORDERS_FILE, sep=";", decimal=",")
    aq["Month"] = (
        pd.to_datetime(aq["Order Date"], dayfirst=False, format="mixed")
        .dt.to_period("M")
    )
    aq_monthly = aq.groupby("Month")["Quantity"].sum().rename("Acquired")

    full_idx = pd.period_range(
        min(nw_monthly.index.min(), aq_monthly.index.min()),
        max(nw_monthly.index.max(), aq_monthly.index.max()),
        freq="M",
    )
    df = pd.DataFrame(index=full_idx)
    df["NutryWell"] = nw_monthly.reindex(full_idx, fill_value=0)
    df["Acquired"]  = aq_monthly.reindex(full_idx, fill_value=0)
    df["Total"]     = df["NutryWell"] + df["Acquired"]
    return df


# ── forecast methods ──────────────────────────────────────────────────────────
def fit_holt_winters(y: np.ndarray, h: int) -> np.ndarray:
    """Triple Exp Smoothing — additive trend + additive 12-month seasonality."""
    for init in ("estimated", "heuristic"):
        try:
            m = ExponentialSmoothing(
                y, trend="add", seasonal="add",
                seasonal_periods=SEASON_PER,
                initialization_method=init,
            ).fit(optimized=True)
            return np.clip(m.forecast(h), 0, None)
        except Exception:
            pass
    # Fallback: Holt (trend only, no seasonality)
    try:
        m = Holt(y, initialization_method="estimated").fit(optimized=True)
        return np.clip(m.forecast(h), 0, None)
    except Exception:
        return np.repeat(y[-1], h)


def fit_linear_seasonal(y: np.ndarray, h: int, first_cal_month: int = 0) -> np.ndarray:
    """OLS on time-trend + 11 calendar-month dummies (additive seasonality)."""
    n  = len(y)
    t  = np.arange(n)
    cm = (first_cal_month + t) % 12           # calendar month index (0 = Jan)
    X  = np.ones((n, 13))                     # intercept + trend + 11 dummies
    X[:, 1] = t
    for m in range(1, 12):
        X[:, 1 + m] = (cm == m).astype(float)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)

    t_f  = np.arange(n, n + h)
    cm_f = (first_cal_month + t_f) % 12
    Xf   = np.ones((h, 13))
    Xf[:, 1] = t_f
    for m in range(1, 12):
        Xf[:, 1 + m] = (cm_f == m).astype(float)
    return np.clip(Xf @ beta, 0, None)


def rmse(actual: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((actual - pred) ** 2)))


# ── plotting helpers ──────────────────────────────────────────────────────────
def _style_ax(ax):
    ax.set_facecolor(PANEL_BG)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0)
    ax.yaxis.grid(True, color="white", linewidth=1.4, zorder=0)
    ax.set_axisbelow(True)


def _fmt_k(x, _):
    return f"{x:,.0f}"


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    # ── 1. demand ─────────────────────────────────────────────────────────────
    demand = load_monthly_demand()

    train      = demand[demand.index <= TRAIN_END]["Total"]
    fcst_idx   = pd.period_range(GRAPH_START, GRAPH_END, freq="M")
    h          = len(fcst_idx)         # 13 months

    actual_tot = demand["Total"].reindex(fcst_idx, fill_value=0)
    actual_nw  = demand["NutryWell"].reindex(fcst_idx, fill_value=0)
    actual_aq  = demand["Acquired"].reindex(fcst_idx, fill_value=0)

    y_train          = train.values.astype(float)
    first_cal_month  = train.index[0].month - 1   # 0-indexed calendar month

    print(f"Training window : {train.index[0]} to {train.index[-1]}  ({len(train)} months)")
    print(f"Forecast window : {fcst_idx[0]} to {fcst_idx[-1]}  ({h} months)")
    print(f"Training total demand: {y_train.sum():,.0f} units\n")

    # ── 2. forecasts ──────────────────────────────────────────────────────────
    hw_fcst = fit_holt_winters(y_train, h)
    ls_fcst = fit_linear_seasonal(y_train, h, first_cal_month=first_cal_month)

    hw_rmse = rmse(actual_tot.values, hw_fcst)
    ls_rmse = rmse(actual_tot.values, ls_fcst)

    print(f"Holt-Winters RMSE      : {hw_rmse:>10,.1f}")
    print(f"Linear Seasonal RMSE   : {ls_rmse:>10,.1f}")

    # ── 3. model selection ────────────────────────────────────────────────────
    if hw_rmse <= ls_rmse:
        best_name, best_fcst, best_rmse = "Holt-Winters", hw_fcst, hw_rmse
        other_name = "Linear + Seasonal"
    else:
        best_name, best_fcst, best_rmse = "Linear + Seasonal", ls_fcst, ls_rmse
        other_name = "Holt-Winters"

    print(f"\nSelected model : {best_name}  (lower RMSE)")

    buf10 = best_fcst * (1 + SAFETY)

    # ── 4. figure ─────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(17, 9.5))
    fig.patch.set_facecolor(BG)

    fig.subplots_adjust(left=0.07, right=0.97, top=0.88, bottom=0.20)
    ax_main = fig.add_subplot(1, 1, 1)
    _style_ax(ax_main)

    x       = np.arange(h)
    bw      = 0.55
    xlabels = [p.strftime("%b\n%Y") for p in fcst_idx]

    # ── main panel ────────────────────────────────────────────────────────────
    # Stacked actual demand bars
    ax_main.bar(x, actual_nw.values, width=bw,
                color=C_NW, alpha=0.85, label="NutryWell — actual demand", zorder=2)
    ax_main.bar(x, actual_aq.values, width=bw, bottom=actual_nw.values,
                color=C_AQ, alpha=0.85, label="Acquired portfolio — actual demand", zorder=2)

    # NutryWell flat ordering level (178k / month)
    ax_main.axhline(y=ORDER_LEVEL, color=C_ORDER, lw=2.2, ls="-",
                    zorder=5, label="NutryWell actual forecast")
    ax_main.text(h - 0.45, ORDER_LEVEL * 1.015, f"{ORDER_LEVEL//1000}k",
                 color=C_ORDER, fontsize=8.5, fontweight="bold", va="bottom", ha="right")

    # +10 % safety buffer fill
    ax_main.fill_between(x, best_fcst, buf10,
                         color=C_BUF, alpha=0.35, zorder=3,
                         label="+10 % safety buffer")
    ax_main.plot(x, buf10, color=C_BUF, lw=1.6, ls="--", zorder=4, alpha=0.85)

    # Forecast lines
    hw_lw = 3.0 if best_name == "Holt-Winters"     else 2.0
    ls_lw = 3.0 if best_name == "Linear + Seasonal" else 2.0
    hw_zo = 7   if best_name == "Holt-Winters"     else 6
    ls_zo = 7   if best_name == "Linear + Seasonal" else 6

    ax_main.plot(x, hw_fcst, color=C_HW, lw=hw_lw,
                 ls="-", marker="o", ms=5, zorder=hw_zo,
                 label=f"Holt-Winters  (RMSE = {hw_rmse:,.0f})"
                       + ("  [selected]" if best_name == "Holt-Winters" else ""))
    ax_main.plot(x, ls_fcst, color=C_LS, lw=ls_lw,
                 ls="--", marker="s", ms=5, zorder=ls_zo,
                 label=f"Linear + Seasonal  (RMSE = {ls_rmse:,.0f})"
                       + ("  [selected]" if best_name == "Linear + Seasonal" else ""))

    # Annotate peak of +10 % buffer
    pk = int(np.argmax(buf10))
    ax_main.annotate(
        f"Peak +10%\n{buf10[pk]:,.0f} units",
        xy=(pk, buf10[pk]),
        xytext=(pk + (1 if pk < h - 2 else -2), buf10[pk] * 1.045),
        fontsize=8, color=C_BUF,
        arrowprops=dict(arrowstyle="-|>", color=C_BUF, lw=1.1),
        ha="center",
    )

    ax_main.set_xticks(x)
    ax_main.set_xticklabels(xlabels, fontsize=9.5)
    ax_main.yaxis.set_major_formatter(mticker.FuncFormatter(_fmt_k))
    ax_main.set_ylabel("Monthly demand (units)", fontsize=11, fontweight="bold", labelpad=10)
    ax_main.set_xlim(-0.5, h - 0.5)

    ax_main.set_title(
        "NutryWell + Acquired Portfolio — Monthly Demand & Forecast\n"
        "September 2024 – September 2025",
        fontsize=15, fontweight="bold", pad=14, color="#1B2A3B",
    )

    legend = ax_main.legend(
        loc="upper center", fontsize=11, framealpha=0.95,
        edgecolor="#CCCCCC", ncol=3, columnspacing=1.4,
        bbox_to_anchor=(0.5, -0.12), borderaxespad=0.0,
    )
    legend.get_frame().set_linewidth(0.8)

    # ── save ──────────────────────────────────────────────────────────────────
    out_path = OUT_DIR / "demand_forecast_combined.png"
    fig.savefig(out_path, dpi=160, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
