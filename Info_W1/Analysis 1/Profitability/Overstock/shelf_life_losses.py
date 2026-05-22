"""
NutryWell — Shelf Life Loss Analysis (FEFO)
============================================
Runs a monthly FEFO inventory simulation for all NutryWell SKUs over a
12-month measurement window (Sep 2024 – Aug 2025), preceded by a 14-month
warm-up (Jul 2023 – Aug 2024) that builds realistic starting inventory.

Two ordering scenarios are compared:
  S1 — Actual NutryWell ordering   (from NutrYWell_PO.csv)
  S2 — Holt-Winters +10 % ordering (per-SKU HW forecast × 1.10)

FEFO shelf-life acceptance rules
  SL remaining >= 75 %           -> all channels (B2B + E-commerce)
  25 % <= SL remaining < 75 %    -> E-commerce only
  SL remaining < 25 %            -> no channel accepts; stock ages to expiry
  SL remaining = 0 %             -> written off

B2B channels  : Pharmacy, Retail, Sport Retail  (from Customer_Master)
E-commerce    : E-commerce channel

Output: shelf_life_losses.png  ->  Info_W1/output/
        shelf_life_losses.xlsx ->  Info_W1/output/
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
from matplotlib.gridspec import GridSpec
from statsmodels.tsa.holtwinters import ExponentialSmoothing, Holt

warnings.filterwarnings("ignore")

# ── paths ─────────────────────────────────────────────────────────────────────
PROJECT_DIR    = Path(__file__).resolve().parents[4]
OUT_DIR        = PROJECT_DIR / "Info_W1" / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

NW_PROD_FILE   = PROJECT_DIR / "Info_W1" / "Data" / "Product_master.csv"
NW_ORDERS_FILE = PROJECT_DIR / "Info_W1" / "Data" / "Order_ERP.csv"
CUST_FILE      = PROJECT_DIR / "Info_W1" / "Data" / "Costumer_Master.csv"
PO_FILE        = PROJECT_DIR / "Info_W3" / "NutrYWell_PO.csv"

# ── config ────────────────────────────────────────────────────────────────────
WARMUP_START = pd.Period("2023-07", freq="M")   # start of PO history
TRAIN_END    = pd.Period("2024-08", freq="M")   # last warm-up month
SIM_START    = pd.Period("2024-09", freq="M")   # measurement start
SIM_END      = pd.Period("2025-08", freq="M")   # measurement end  (12 months)
SAFETY       = 0.10                              # +10 % buffer for S2

SL_B2B_MIN   = 0.75   # B2B refuses stock with SL remaining < 75 %
SL_ECOMM_MIN = 0.25   # E-commerce refuses stock with SL remaining < 25 %

# ── design ────────────────────────────────────────────────────────────────────
C_S1   = "#E63946"   # Scenario 1 — red
C_S2   = "#27AE60"   # Scenario 2 — green
C_RISK = "#F4A261"   # at-risk shade (lighter version)
BG     = "#F7F9FC"
P_BG   = "#EEF2F7"


# ── helpers ───────────────────────────────────────────────────────────────────
def _fmt_k(x, _):
    return f"{x:,.0f}"


def _read_csv(path):
    return pd.read_csv(path, sep=";", decimal=",")


# ── data loading ──────────────────────────────────────────────────────────────
def load_products():
    pm = _read_csv(NW_PROD_FILE)
    pm = pm[["SKU", "Product Name", "Shelf Life (months)",
             "Standard Cost EUR", "Selling Price EUR"]].copy()
    pm.columns = ["SKU", "Name", "SL_months", "Cost", "Price"]
    pm["SL_months"] = pm["SL_months"].astype(int)
    return pm


def load_demand(skus, start, end):
    """Monthly B2B and E-commerce demand per SKU for the given period."""
    cust = _read_csv(CUST_FILE)[["Customer ID", "Channel"]]
    cust_map = dict(zip(cust["Customer ID"], cust["Channel"]))

    orders = _read_csv(NW_ORDERS_FILE)
    orders = orders[orders["SKU"].isin(skus)].copy()
    orders["Month"] = (
        pd.to_datetime(orders["Order Date"], dayfirst=True, format="mixed")
        .dt.to_period("M")
    )
    orders["Channel"] = orders["Customer ID"].map(cust_map).fillna("B2B")
    orders["is_ecomm"] = orders["Channel"] == "E-commerce"

    idx = pd.period_range(start, end, freq="M")

    b2b = (
        orders[~orders["is_ecomm"]]
        .groupby(["Month", "SKU"])["Quantity"].sum()
        .unstack(fill_value=0)
        .reindex(idx, fill_value=0)
        .reindex(columns=skus, fill_value=0)
    )
    ec = (
        orders[orders["is_ecomm"]]
        .groupby(["Month", "SKU"])["Quantity"].sum()
        .unstack(fill_value=0)
        .reindex(idx, fill_value=0)
        .reindex(columns=skus, fill_value=0)
    )
    return b2b, ec


def load_po_orders(skus, start, end):
    po = _read_csv(PO_FILE)
    po = po[po["SKU"].isin(skus)].copy()
    po["Month"] = (
        pd.to_datetime(po["PO Date"], dayfirst=True, format="mixed")
        .dt.to_period("M")
    )
    idx = pd.period_range(start, end, freq="M")
    return (
        po.groupby(["Month", "SKU"])["Quantity Purchased (units)"].sum()
        .unstack(fill_value=0)
        .reindex(idx, fill_value=0)
        .reindex(columns=skus, fill_value=0)
    )


def _fit_hw_sku(y_act, h):
    """Shared HW/Holt fitting helper. Returns (fitted_values, forecast_h)."""
    fitted, fcst = None, None
    if len(y_act) >= 12:
        for init in ("estimated", "heuristic"):
            try:
                m = ExponentialSmoothing(
                    y_act, trend="add", seasonal="add",
                    seasonal_periods=12, initialization_method=init,
                ).fit(optimized=True)
                fitted = np.clip(m.fittedvalues, 0, None)
                fcst   = np.clip(m.forecast(h),  0, None)
                return fitted, fcst
            except Exception:
                pass
    if len(y_act) >= 3:
        try:
            m      = Holt(y_act, initialization_method="estimated").fit(optimized=True)
            fitted = np.clip(m.fittedvalues, 0, None)
            fcst   = np.clip(m.forecast(h),  0, None)
            return fitted, fcst
        except Exception:
            pass
    # Fallback: repeat last observed value
    last   = float(y_act[-1]) if len(y_act) else 0.0
    fitted = np.full(len(y_act), last)
    fcst   = np.full(h, last)
    return fitted, fcst


def compute_hw_warmup_orders(skus, b2b_warmup, ec_warmup):
    """
    S2 warm-up orders: HW in-sample fitted values × 1.10.
    Represents what NutryWell would have ordered from Jul 2023 onward
    if it had always followed the HW model.
    """
    total = b2b_warmup + ec_warmup
    wu_idx = pd.period_range(WARMUP_START, TRAIN_END, freq="M")
    h_dummy = 1   # we only need fitted values here

    orders_df = pd.DataFrame(index=wu_idx, columns=skus, dtype=float)

    for sku in skus:
        y = total[sku].values.astype(float)
        first_nz = int(np.argmax(y > 0)) if y.sum() > 0 else 0
        y_act = y[first_nz:]

        fitted_act, _ = _fit_hw_sku(y_act, h_dummy)

        full_fitted = np.zeros(len(y))
        full_fitted[first_nz:] = fitted_act
        orders_df[sku] = np.clip(full_fitted * (1 + SAFETY), 0, None)

    return orders_df.fillna(0)


def compute_hw_orders(skus, b2b_warmup, ec_warmup):
    """Per-SKU HW +10 % forecast for the 12-month measurement period."""
    total = b2b_warmup + ec_warmup
    h     = len(pd.period_range(SIM_START, SIM_END, freq="M"))

    fcst = pd.DataFrame(
        index=pd.period_range(SIM_START, SIM_END, freq="M"),
        columns=skus, dtype=float,
    )

    for sku in skus:
        y = total[sku].values.astype(float)
        first_nz = int(np.argmax(y > 0)) if y.sum() > 0 else 0
        y_act = y[first_nz:]

        _, pred = _fit_hw_sku(y_act, h)
        fcst[sku] = pred * (1 + SAFETY)

    return fcst.fillna(0)


# ── FEFO simulation ───────────────────────────────────────────────────────────
def simulate_fefo(
    orders_warmup:   np.ndarray,
    b2b_warmup:      np.ndarray,
    ec_warmup:       np.ndarray,
    orders_measure:  np.ndarray,
    b2b_measure:     np.ndarray,
    ec_measure:      np.ndarray,
    sl_months:       int,
) -> dict:
    """
    FEFO simulation split into warm-up (builds inventory) and measurement
    (counts losses) phases.

    inventory[k] = units aged k months (k=0 just received, k=sl_months expired).
    """
    # Age thresholds
    b2b_max_age   = (1 - SL_B2B_MIN)  * sl_months   # age above -> B2B refuses
    ecomm_max_age = (1 - SL_ECOMM_MIN) * sl_months   # age above -> nobody buys

    inventory = np.zeros(sl_months + 1, dtype=float)

    def _run_period(orders, b2b, ec, count_losses):
        nonlocal inventory
        expired = 0.0
        at_risk_snapshot = 0.0

        for t in range(len(orders)):
            # Age
            new_inv = np.zeros(sl_months + 1, dtype=float)
            new_inv[1:] = inventory[:-1]
            inventory = new_inv

            # Expire
            exp = inventory[sl_months]
            inventory[sl_months] = 0.0
            if count_losses:
                expired += exp

            # Receive
            inventory[0] += float(orders[t])

            # FEFO sell (oldest eligible first)
            rem_b2b = float(b2b[t])
            rem_ec  = float(ec[t])

            for age in range(sl_months - 1, -1, -1):
                if inventory[age] < 1e-9:
                    continue
                if age > ecomm_max_age:
                    pass  # nobody buys; ages toward expiry
                elif age > b2b_max_age:
                    sell = min(rem_ec, inventory[age])
                    inventory[age] -= sell
                    rem_ec -= sell
                else:
                    sell_b2b = min(rem_b2b, inventory[age])
                    inventory[age] -= sell_b2b
                    rem_b2b -= sell_b2b
                    sell_ec  = min(rem_ec, inventory[age])
                    inventory[age] -= sell_ec
                    rem_ec  -= sell_ec

        if count_losses:
            # Stock already in no-sell zone at end = will expire after period
            at_risk_snapshot = float(
                np.sum(inventory[int(np.ceil(ecomm_max_age)) + 1: sl_months])
            )

        return expired, at_risk_snapshot

    # Warm-up (don't count losses, just build inventory)
    _run_period(orders_warmup, b2b_warmup, ec_warmup, count_losses=False)

    # Measurement (count losses)
    expired, at_risk = _run_period(
        orders_measure, b2b_measure, ec_measure, count_losses=True
    )

    return {
        "expired_units": expired,
        "at_risk_units": at_risk,
        "total_units":   expired + at_risk,
    }


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    pm   = load_products()
    skus = pm["SKU"].tolist()
    sim_months = len(pd.period_range(SIM_START, SIM_END, freq="M"))

    print(f"Warm-up   : {WARMUP_START} to {TRAIN_END}")
    print(f"Measurement: {SIM_START} to {SIM_END}  ({sim_months} months)\n")

    # ── demand ────────────────────────────────────────────────────────────────
    b2b_wu,  ec_wu  = load_demand(skus, WARMUP_START, TRAIN_END)
    b2b_sim, ec_sim = load_demand(skus, SIM_START, SIM_END)

    # ── orders ────────────────────────────────────────────────────────────────
    po_wu  = load_po_orders(skus, WARMUP_START, TRAIN_END)
    po_sim = load_po_orders(skus, SIM_START, SIM_END)          # S1 measurement
    hw_wu  = compute_hw_warmup_orders(skus, b2b_wu, ec_wu)     # S2 warm-up
    hw_sim = compute_hw_orders(skus, b2b_wu, ec_wu)            # S2 measurement

    # ── simulate per SKU ──────────────────────────────────────────────────────
    rows = []
    for sku in skus:
        info  = pm[pm["SKU"] == sku].iloc[0]
        sl    = int(info["SL_months"])
        cost  = float(info["Cost"])
        price = float(info["Price"])

        s1 = simulate_fefo(
            orders_warmup  = po_wu[sku].values,    # S1: actual PO warm-up
            b2b_warmup     = b2b_wu[sku].values,
            ec_warmup      = ec_wu[sku].values,
            orders_measure = po_sim[sku].values,   # S1: actual PO measurement
            b2b_measure    = b2b_sim[sku].values,
            ec_measure     = ec_sim[sku].values,
            sl_months      = sl,
        )
        s2 = simulate_fefo(
            orders_warmup  = hw_wu[sku].values,    # S2: HW-fitted warm-up
            b2b_warmup     = b2b_wu[sku].values,
            ec_warmup      = ec_wu[sku].values,
            orders_measure = hw_sim[sku].values,   # S2: HW forecast measurement
            b2b_measure    = b2b_sim[sku].values,
            ec_measure     = ec_sim[sku].values,
            sl_months      = sl,
        )

        rows.append({
            "SKU":          sku,
            "Name":         info["Name"],
            "SL_months":    sl,
            "Cost_EUR":     cost,
            "Price_EUR":    price,
            # S1
            "S1_expired":   s1["expired_units"],
            "S1_at_risk":   s1["at_risk_units"],
            "S1_total":     s1["total_units"],
            "S1_loss_EUR":  s1["total_units"] * cost,
            # S2
            "S2_expired":   s2["expired_units"],
            "S2_at_risk":   s2["at_risk_units"],
            "S2_total":     s2["total_units"],
            "S2_loss_EUR":  s2["total_units"] * cost,
        })

        print(
            f"{sku:<10} SL={sl:>2}mo  "
            f"S1: {s1['total_units']:7.0f} units / EUR {s1['total_units']*cost:>9,.0f}  |  "
            f"S2: {s2['total_units']:7.0f} units / EUR {s2['total_units']*cost:>9,.0f}"
        )

    df = pd.DataFrame(rows)

    t_s1_u = df["S1_total"].sum()
    t_s2_u = df["S2_total"].sum()
    t_s1_e = df["S1_loss_EUR"].sum()
    t_s2_e = df["S2_loss_EUR"].sum()

    print(f"\n{'-'*72}")
    print(f"TOTAL  S1 (actual ordering)   : {t_s1_u:>10,.0f} units  /  EUR {t_s1_e:>12,.0f}")
    print(f"TOTAL  S2 (HW +10 %)          : {t_s2_u:>10,.0f} units  /  EUR {t_s2_e:>12,.0f}")
    delta_e = t_s2_e - t_s1_e
    sign    = "+" if delta_e >= 0 else ""
    print(f"DELTA  S2 vs S1               :                    EUR {sign}{delta_e:>12,.0f}")

    # ── figure ────────────────────────────────────────────────────────────────
    # Keep only SKUs where at least one scenario has a loss
    df_loss = (df[df[["S1_loss_EUR", "S2_loss_EUR"]].max(axis=1) > 0]
               .sort_values("S1_loss_EUR", ascending=True)   # ascending → biggest at top
               .reset_index(drop=True))
    n  = len(df_loss)
    y  = np.arange(n)
    bh = 0.32   # bar height

    fig = plt.figure(figsize=(15, 2.5 + n * 0.85))
    fig.patch.set_facecolor(BG)

    gs = GridSpec(2, 1, figure=fig,
                  height_ratios=[1, n],
                  hspace=0.35,
                  left=0.22, right=0.92,
                  top=0.90, bottom=0.08)

    ax_kpi  = fig.add_subplot(gs[0])
    ax_main = fig.add_subplot(gs[1])

    # ── KPI strip ─────────────────────────────────────────────────────────────
    ax_kpi.axis("off")
    ax_kpi.set_xlim(0, 1)
    ax_kpi.set_ylim(0, 1)

    fig.suptitle(
        "Shelf Life Losses — Financial Impact per SKU  (Sep 2024 – Aug 2025)",
        fontsize=14, fontweight="bold", color="#1B2A3B", y=0.97,
    )

    kpis = [
        (0.04, C_S1, "S1 — Actual ordering",
         f"EUR {t_s1_e:,.0f}", f"{t_s1_u:,.0f} units wasted"),
        (0.54, C_S2, "S2 — Holt-Winters +10 %",
         f"EUR {t_s2_e:,.0f}", f"{t_s2_u:,.0f} units wasted"),
    ]
    for bx, color, label, big, sub in kpis:
        card = mpatches.FancyBboxPatch(
            (bx, 0.04), 0.42, 0.88,
            boxstyle="round,pad=0.02",
            transform=ax_kpi.transAxes,
            facecolor=color, alpha=0.12,
            edgecolor=color, linewidth=2, zorder=1,
        )
        ax_kpi.add_patch(card)
        cx = bx + 0.21
        ax_kpi.text(cx, 0.78, label, ha="center", va="center",
                    fontsize=9, fontweight="bold", color=color,
                    transform=ax_kpi.transAxes)
        ax_kpi.text(cx, 0.46, big, ha="center", va="center",
                    fontsize=18, fontweight="bold", color=color,
                    transform=ax_kpi.transAxes)
        ax_kpi.text(cx, 0.16, sub, ha="center", va="center",
                    fontsize=8.5, color="#555555",
                    transform=ax_kpi.transAxes)

    savings = t_s1_e - t_s2_e

    # ── horizontal bar chart ──────────────────────────────────────────────────
    ax_main.set_facecolor(P_BG)
    for sp in ax_main.spines.values():
        sp.set_visible(False)
    ax_main.tick_params(length=0)
    ax_main.xaxis.grid(True, color="white", linewidth=1.3, zorder=0)
    ax_main.set_axisbelow(True)

    # S1 bars (red, top of each pair)
    ax_main.barh(y + bh / 2, df_loss["S1_loss_EUR"],
                 height=bh, color=C_S1, alpha=0.88,
                 label="S1 — Actual ordering", zorder=2)

    # S2 bars (green, bottom of each pair)
    ax_main.barh(y - bh / 2, df_loss["S2_loss_EUR"],
                 height=bh, color=C_S2, alpha=0.88,
                 label="S2 — Holt-Winters +10 %", zorder=2)

    # Value labels — S1
    x_max = df_loss["S1_loss_EUR"].max()
    for i, row in df_loss.iterrows():
        # S1 label
        ax_main.text(row["S1_loss_EUR"] + x_max * 0.012,
                     i + bh / 2,
                     f"EUR {row['S1_loss_EUR']:,.0f}",
                     va="center", ha="left", fontsize=8.5,
                     color=C_S1, fontweight="bold")
        # S2 label (only if non-zero)
        if row["S2_loss_EUR"] > 0:
            ax_main.text(row["S2_loss_EUR"] + x_max * 0.012,
                         i - bh / 2,
                         f"EUR {row['S2_loss_EUR']:,.0f}",
                         va="center", ha="left", fontsize=8.5,
                         color=C_S2, fontweight="bold")
        else:
            ax_main.text(x_max * 0.012,
                         i - bh / 2,
                         "no loss",
                         va="center", ha="left", fontsize=8,
                         color=C_S2, style="italic", alpha=0.7)

    # Y-axis labels: SKU + product name + shelf life
    ylabels = [
        f"{row['SKU']}  —  {row['Name']}\n(shelf life: {int(row['SL_months'])} months)"
        for _, row in df_loss.iterrows()
    ]
    ax_main.set_yticks(y)
    ax_main.set_yticklabels(ylabels, fontsize=9)
    ax_main.set_ylim(-0.7, n - 0.3)
    ax_main.set_xlim(0, x_max * 1.38)
    ax_main.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"EUR {v/1000:.0f}k"))
    ax_main.set_xlabel("Financial loss at standard cost (EUR)", fontsize=10,
                       fontweight="bold", labelpad=8)

    ax_main.legend(loc="lower right", fontsize=9,
                   framealpha=0.9, edgecolor="#CCCCCC")

    # Divider between each SKU pair
    for i in range(n - 1):
        ax_main.axhline(i + 0.5, color="white", lw=1.2, zorder=1)

    fig.text(
        0.5, 0.01,
        "FEFO rules  |  SL >= 75 % : all channels  |  "
        "25-75 % : e-commerce only  |  < 25 % : unsellable  |  "
        "Loss = expired + at-risk stock valued at standard cost",
        ha="center", fontsize=7, color="#999999", style="italic",
    )

    png_path = OUT_DIR / "shelf_life_losses.png"
    fig.savefig(png_path, dpi=160, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"\nSaved: {png_path}")

    # ── CSV export ────────────────────────────────────────────────────────────
    csv_path = OUT_DIR / "shelf_life_losses.csv"
    df.round(1).to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
