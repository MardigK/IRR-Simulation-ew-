# IRRsim.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
from io import StringIO
from fund_backend import (
    FundConfig, Deal, validate_deals,
    annual_cash_flows, investment_calls_by_year,
    management_fees_by_year, exits_by_year, irr
)

# =========================
# GLOBAL SETTINGS & DATASET
# =========================

st.set_page_config(page_title="IRR Simulator — Test & Monte Carlo", layout="wide")
st.title("IRR Simulator — Deterministic Test & Monte Carlo")

st.caption(
    "Yearly CF = Calls (−) + Fees (−) + Exits (+). No leverage, no interim distributions, no partial exits. "
    "Fees: Years 1–4 = 2% of $600m; Years 5–8 = 1.5% of invested capital outstanding at the start of year. "
    "IRR uses year-end timing."
)

# ---- MOIC dataset (reference set; used if you choose 'Dataset' mode) ----
MOIC_POOL: List[float] = [
    1.543333333, 1.35, 1.75, 1.216666667, 1.343333333, 1.8, 1.87, 3.28, 1.426666667, 1.22, 1.445, 1.726666667,
    1.53, 1.67, 2.14, 1.73, 3.33, 0.83, 0.89, 1.22, 2.364, 2.8, 1.76, 1.916666667, 2.27, 2.443333333, 1.71, 1.685,
    7.49, 1.28, 1.57, 1.88, 1.8625, 2.125, 1.536, 1.9, 1.8025, 3.254, 3.413333333, 4.59, 2.363333333, 1.9825,
    2.1475, 1.952, 1.824, 1.03, 2.321666667, 2.661666667, 2.98, 2.966, 2.778, 2.516, 2.128333333, 2.055714286,
    2.163636364, 1.425, 1.84, 1.79, 3.9, 1.343333333, 1.166666667, 2.195, 2.416, 2.025, 1.767142857, 1.968333333,
    1.995714286, 1.801666667, 1.842, 2.13, 2.329, 2.262, 2.858, 2.96, 1.9675, 6.301666667, 2.083333333, 1.663333333,
    1.78, 2.027142857, 1.941111111, 2.366666667, 2.422222222, 3.027, 2.609166667, 1.78, 3.12, 2.15, 1.87, 1.776,
    1.81, 1.82, 1.9925, 2.178333333, 2.606, 1.8375, 1.975, 2.307777778, 2.3725, 2.304, 2.41875, 3.4375, 2.258,
    1.812, 2.461, 1.978, 1.943333333, 2.412, 2.071666667, 1.721428571, 2.034285714, 2.448333333, 1.476666667,
    2.757333333, 2.2975, 1.255, 3.844545455, 1.51, 2.035, 1.716666667, 1.19, 1.53, 1.546666667, 1.63, 1.47, 1.54,
    1.765, 1.8725, 2, 1.79, 1.695, 1.586666667, 2.056666667, 1.6975, 2.02, 1.93, 2.1, 2.015, 3.461428571, 2.975,
    1.96, 1.393333333, 1.957777778, 1.88, 2.006, 1.835714286, 2.718571429, 1.94, 1.665714286, 2.185, 2.070833333,
    1.731666667, 1.911666667, 2, 2.337777778, 1.875, 2.9425, 1.874285714, 2.32, 1.996, 1.763333333, 2.229, 1.892,
    2.366666667, 1.948, 1.9075, 1.903333333, 1.942857143, 1.693333333, 2.2525, 2.568, 1.993333333, 1.79, 1.983333333,
    2.433333333, 1.683333333, 1.84, 1.791666667, 2.238571429, 1.575454545, 1.955454545, 1.40875, 1.71, 1.808, 2.12,
    1.784, 1.746, 1.84, 1.83, 0.99, 1.575, 1.78, 1.755, 1.785, 1.9975, 1.74, 1.7225, 2.10125, 2.068333333, 1.9725,
    1.933333333, 1.781818182, 1.967142857, 1.897142857, 1.666666667, 1.723333333, 1.973333333, 1.43, 1.4525, 1.83,
    1.61, 2.3, 1.7475, 1.78, 1.837777778, 1.45, 1.94, 1.923333333, 1.655, 1.866666667, 1.9975, 2.12, 1.994285714,
    1.805, 1.608333333, 1.653333333, 1.692, 1.652, 1.838, 2.002, 1.625, 1.455, 1.03, 1.735, 1.495, 1.48, 1.516666667,
    1.78, 1.635, 1.685, 1.446666667, 1.483333333, 1.9525, 1.833333333, 2.67, 1.59, 1.336666667, 2.214, 1.7425,
    1.662857143, 1.66, 1.74, 1.582, 1.7325, 1.908, 1.766, 1.903333333, 3.378333333, 1.725, 2.304, 2.014285714,
    1.878571429, 2.778571429, 2.092, 1.572, 1.526666667, 1.706, 1.662, 1.88, 2.412, 1.522857143, 1.706666667,
    1.506666667, 1.376666667, 1.445, 1.565, 1.641428571, 1.653333333, 1.673333333, 2.046666667, 1.713333333,
    1.276666667, 1.788, 1.533333333, 1.743333333, 1.7, 1.685, 1.975, 1.375, 1.6175, 1.8975, 1.605, 1.77875, 1.8525,
    1.725, 1.81, 1.69625, 1.592, 2.375, 1.578, 1.625, 1.845714286, 1.578333333, 2.115, 1.740909091, 1.51, 1.72,
    1.334285714, 1.71125, 1.676666667, 1.442222222, 0.93, 1.496666667, 1.333333333, 1.47, 1.363333333, 1.343333333,
    1.85, 1.223333333, 1.273333333, 0.96, 1.59, 1.53, 1.88375, 1.21, 2.144, 1.6, 1.933333333, 1.836, 1.4825, 0.875,
    1.34, 1.656666667, 1.49, 1.266666667, 1.2275, 1.78, 1.48, 1.475, 1.0225, 1.542, 1.6, 1.56, 1.363333333, 1.56,
    1.386666667, 1.58, 1.648333333, 1.81, 1.61, 1.5625, 1.65, 1.58, 1.612, 1.768, 2.146666667, 1.686, 1.7475, 2.2,
    2.083333333, 1.688, 1.306, 1.288, 1.49, 1.386, 1.475, 1.373333333, 1.436666667, 1.096666667, 1.76, 1.51,
    1.253333333, 1.203333333, 1.523333333, 1.22, 1.053333333, 1.3, 1.493333333, 1.7225, 0.65, 1.24, 1.4575,
    1.123333333, 2.113333333, 1.207142857, 1.136666667, 1.17, 1.36, 1.506666667, 1.375, 1.046666667, 1.12,
    1.263333333, 1.353333333, 0.923333333, 1.476666667, 1.675, 1.586666667, 1.51, 1.1775
]

# Reference MOIC stats (display only)
_moic_arr = np.array(MOIC_POOL, dtype=float)
_moic_mean = float(np.mean(_moic_arr))
_moic_std  = float(np.std(_moic_arr, ddof=0))
_moic_pcts = np.percentile(_moic_arr, [5, 25, 50, 75, 95])

# --------------------------------
# Sidebar: Fund-level parameters
# --------------------------------
st.sidebar.header("Fund Settings")
fund_size = st.sidebar.number_input("Target Fund Size ($)", value=600_000_000.0, step=10_000_000.0, format="%.0f")
investment_period_years = st.sidebar.number_input("Investment Period (years)", min_value=1, max_value=8, value=4, step=1)
fund_life_years = st.sidebar.number_input("Total Fund Life (years)", min_value=1, max_value=20, value=8, step=1)
mgmt_fee_rate_investment = st.sidebar.number_input("Mgmt Fee (Years 1–Investment Period) % of Fund Size",
                                                   min_value=0.0, max_value=10.0, value=2.0, step=0.1) / 100.0
mgmt_fee_rate_post = st.sidebar.number_input("Mgmt Fee (Post-Investment Period) % of Invested Outstanding",
                                             min_value=0.0, max_value=10.0, value=1.5, step=0.1) / 100.0

cfg = FundConfig(
    fund_size=fund_size,
    investment_period_years=investment_period_years,
    fund_life_years=fund_life_years,
    mgmt_fee_rate_investment=mgmt_fee_rate_investment,
    mgmt_fee_rate_post=mgmt_fee_rate_post
)

# -------------------------
# Tabs: Test Case / Monte Carlo
# -------------------------
tab_test, tab_mc = st.tabs(["Deterministic Test Case", "Monte Carlo Simulation"])

# =============================================================================
# DETERMINISTIC TEST CASE
# =============================================================================
with tab_test:
    st.subheader("Deals (Test Case)")
    st.write("Enter deals below. Exit Proceeds = Invested Amount × MOIC, realized fully in the Exit Year.")

    seed_df = pd.DataFrame([
        {"invested_amount": 9_500_000.0,  "invest_year": 1, "exit_year": 6, "moic": 3.0},
        {"invested_amount": 10_000_000.0, "invest_year": 1, "exit_year": 7, "moic": 2.0},
        {"invested_amount": 35_500_000.0, "invest_year": 1, "exit_year": 8, "moic": 2.0},
    ])

    deals_df = st.data_editor(
        seed_df,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "invested_amount": st.column_config.NumberColumn("Invested Amount ($)", step=100_000.0, format="%.0f"),
            "invest_year": st.column_config.NumberColumn("Invest Year (1..N)", step=1, min_value=1, max_value=int(cfg.investment_period_years)),
            "exit_year": st.column_config.NumberColumn("Exit Year (1..N)", step=1, min_value=1, max_value=int(cfg.fund_life_years)),
            "moic": st.column_config.NumberColumn("MOIC", step=0.01, format="%.6f"),
        },
        key="deals_editor_test"
    )

    deals_test: List[Deal] = []
    for _, row in deals_df.iterrows():
        try:
            deals_test.append(
                Deal(
                    invested_amount=float(row.get("invested_amount", 0.0) or 0.0),
                    invest_year=int(row.get("invest_year", 1) or 1),
                    exit_year=int(row.get("exit_year", 1) or 1),
                    moic=float(row.get("moic", 0.0) or 0.0),
                )
            )
        except Exception:
            pass

    errors = validate_deals(deals_test, cfg)
    if errors:
        st.error("Please fix the following issues:")
        for e in errors:
            st.write(f"- {e}")

    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Compute Annual Cash Flows & IRR (Test Case)", use_container_width=True, type="primary",
                     disabled=bool(errors), key="btn_test"):
            try:
                calls = investment_calls_by_year(deals_test, cfg)
                fees  = management_fees_by_year(deals_test, cfg)
                exits = exits_by_year(deals_test, cfg)
                net   = [calls[i] + fees[i] + exits[i] for i in range(cfg.fund_life_years)]

                df_cfs = pd.DataFrame({
                    "Year": list(range(1, cfg.fund_life_years + 1)),
                    "Calls (−)": calls,
                    "Fees (−)": fees,
                    "Exits (+)": exits,
                    "Net CF": net
                })
                st.dataframe(df_cfs, use_container_width=True, hide_index=True)

                irr_val = irr(net)
                st.metric("IRR (annual)", f"{irr_val*100:,.2f}%")
                st.metric("Annualized IRR", f"{irr_val*100:,.2f}%")

            except ValueError as ve:
                st.error(str(ve))
            except Exception as ex:
                st.error(f"Unexpected error: {ex}")

    with col2:
        st.write("**Diagnostics**")
        total_invested = sum(d.invested_amount for d in deals_test)
        st.write(f"- Total Invested (sum of deal costs): **${total_invested:,.0f}**")
        st.write(f"- Number of deals: **{len(deals_test)}**")
        inv_by_year = {t: 0.0 for t in range(1, cfg.fund_life_years + 1)}
        for d in deals_test:
            inv_by_year[d.invest_year] += d.invested_amount
        diag = pd.DataFrame({"Year": list(inv_by_year.keys()),
                             "Invested in Year ($)": list(inv_by_year.values())})
        st.dataframe(diag, use_container_width=True, hide_index=True)

# =============================================================================
# MONTE CARLO (MOIC mode is selectable; MOIC dataset stats shown as reference)
# =============================================================================
with tab_mc:
    st.subheader("Monte Carlo Parameters")

    # ---- Core MC controls (bounds enforced in helpers) ----
    colA, colB = st.columns(2)
    with colA:
        num_iterations = st.number_input("Iterations", min_value=100, max_value=10000, value=2000, step=100)
        mu_deals = st.number_input("Deals ~ Normal(μ, σ) — μ", min_value=1.0, max_value=30.0, value=12.0, step=0.1)
        sd_deals = st.number_input("Deals ~ Normal(μ, σ) — σ", min_value=0.1, max_value=10.0, value=1.6, step=0.1)
    with colB:
        mu_total_m = st.number_input("Total Invested (Y1–Y4) μ ($m)", min_value=100.0, max_value=2000.0, value=520.0, step=10.0)
        sd_total_m = st.number_input("Total Invested (Y1–Y4) σ ($m)", min_value=1.0, max_value=500.0, value=70.0, step=1.0)
        mu_total = mu_total_m * 1_000_000.0
        sd_total = sd_total_m * 1_000_000.0

    st.caption(
        "Bounds: deals ∈ [9, 15]; total invested ∈ [$400m, $600m]; per-deal size ∈ [$20m, $45m]. "
        "Year 1 is fixed at $55m (3 deals). New capital for Y2–Y4 = total − 55m. "
        "Exit year sampled late-tilted and ≥ invest year."
    )

    # ---- MOIC stats (reference only) + MOIC mode for simulation ----
    st.markdown("**MOIC Sample — Reference Stats** (dataset shown below; *simulation can use inputs instead*)")
    c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
    c1.metric("Count", f"{len(_moic_arr)}")
    c2.metric("Mean", f"{_moic_mean:.3f}x")
    c3.metric("Std Dev", f"{_moic_std:.3f}")
    c4.metric("p5", f"{_moic_pcts[0]:.3f}x")
    c5.metric("p25", f"{_moic_pcts[1]:.3f}x")
    c6.metric("p50 (Med)", f"{_moic_pcts[2]:.3f}x")
    c7.metric("p95", f"{_moic_pcts[4]:.3f}x")

    st.markdown("**MOIC Distribution Used in Simulation**")
    moic_mode = st.radio(
        "Choose how to generate MOICs for deals",
        options=["Use provided MOIC dataset", "Use custom MOIC list", "Use parametric (lognormal) — set mean & std"],
        index=0,
        horizontal=False,
    )

    custom_moic_list: List[float] = []
    lognorm_mean = 2.0
    lognorm_std = 0.6

    if moic_mode == "Use custom MOIC list":
        txt = st.text_area(
            "Paste MOICs (comma/space separated). Example: 1.5, 1.9, 2.3, 0.9, 3.1",
            height=100,
            placeholder="1.5, 1.9, 2.3, 0.9, 3.1"
        )
        if txt.strip():
            try:
                parts = [p.strip() for p in txt.replace("\n", " ").replace(",", " ").split(" ") if p.strip()]
                custom_moic_list = [float(x) for x in parts if float(x) > 0]
            except Exception:
                st.warning("Could not parse the list; falling back to dataset mode.")
                custom_moic_list = []
        st.caption(f"Parsed {len(custom_moic_list)} MOIC values.")
    elif moic_mode == "Use parametric (lognormal) — set mean & std":
        colm1, colm2 = st.columns(2)
        lognorm_mean = colm1.number_input("MOIC mean (arithmetic)", min_value=0.01, max_value=20.0, value=2.0, step=0.05)
        lognorm_std  = colm2.number_input("MOIC std (arithmetic)",  min_value=0.0,  max_value=20.0, value=0.6, step=0.05)
        st.caption("We calibrate a lognormal so its arithmetic mean & std match your inputs (no caps).")

    # ---------- MC helpers ----------
    INTERNAL_RANDOM_SEED = None  # None = fresh randomness each run; set an int for reproducible runs

    def truncated_normal_int(mu: float, sigma: float, lo: int, hi: int, rng: np.random.Generator) -> int:
        for _ in range(1000):
            x = rng.normal(mu, sigma)
            n = int(round(x))
            if lo <= n <= hi:
                return n
        return int(np.clip(round(mu), lo, hi))

    def truncated_normal_float(mu: float, sigma: float, lo: float, hi: float, rng: np.random.Generator) -> float:
        for _ in range(1000):
            x = rng.normal(mu, sigma)
            if lo <= x <= hi:
                return float(x)
        return float(np.clip(mu, lo, hi))

    def allocate_per_deal_sizes(total: float, n_deals: int, lo: float, hi: float, rng: np.random.Generator) -> np.ndarray:
        eps = 1e-9
        min_sum, max_sum = lo * n_deals, hi * n_deals
        if total < min_sum:
            return np.full(n_deals, lo)
        if total > max_sum:
            return np.full(n_deals, hi)
        props = rng.dirichlet(np.ones(n_deals))
        sizes = props * total
        for _ in range(100):
            over = sizes > hi + eps
            under = sizes < lo - eps
            if not (over.any() or under.any()):
                break
            excess = np.sum(sizes[over] - hi)
            deficit = np.sum(lo - sizes[under])
            sizes[over] = np.minimum(sizes[over], hi)
            sizes[under] = np.maximum(sizes[under], lo)
            free = excess - deficit
            middle = (~over) & (~under)
            mid_count = np.sum(middle)
            if mid_count > 0:
                sizes[middle] += free / mid_count
        sizes *= total / max(np.sum(sizes), eps)
        sizes = np.clip(sizes, lo, hi)
        sizes *= total / max(np.sum(sizes), eps)
        return sizes

    def assign_invest_years(amounts: np.ndarray, total_invested_all_years: float, rng: np.random.Generator) -> List[int]:
        y1 = 55_000_000.0
        target_cum2 = 0.175 * cfg.fund_size
        target_cum3 = 0.475 * cfg.fund_size
        target_cum4 = 0.775 * cfg.fund_size
        inc2 = max(0.0, target_cum2 - y1)
        inc3 = max(0.0, target_cum3 - target_cum2)
        inc4 = max(0.0, target_cum4 - target_cum3)
        S_desired = inc2 + inc3 + inc4
        remaining = max(0.0, total_invested_all_years - y1)
        if S_desired <= 0 or remaining <= 0:
            return list(map(int, rng.integers(2, 5, size=len(amounts))))
        scale = remaining / S_desired
        budgets = {2: inc2 * scale, 3: inc3 * scale, 4: inc4 * scale}
        years = []
        idx = rng.permutation(len(amounts))
        for i in idx:
            target_year = max(budgets.keys(), key=lambda y: budgets[y])
            years.append(target_year)
            budgets[target_year] = max(0.0, budgets[target_year] - amounts[i])
        out = [0]*len(amounts)
        for k, i in enumerate(idx):
            out[i] = years[k]
        return out

    def sample_exit_year(invest_year: int, rng: np.random.Generator) -> int:
        lo = invest_year
        hi = int(cfg.fund_life_years)
        k = np.arange(lo, hi + 1)
        w = (k - invest_year + 1).astype(float)
        w = w / w.sum()
        return int(rng.choice(k, p=w))

    # --- MOIC sampler based on selected mode ---
    def _lognormal_params_from_mean_std(mean: float, std: float) -> Tuple[float, float]:
        """Calibrate lognormal so arithmetic mean/std match inputs."""
        if mean <= 0:
            mean = 1e-6
        var = std * std
        sigma2 = np.log(1.0 + (var / (mean * mean)))
        sigma = np.sqrt(max(sigma2, 1e-12))
        mu_ln = np.log(mean) - 0.5 * sigma2
        return mu_ln, sigma

    def sample_moic(rng: np.random.Generator) -> float:
        if moic_mode == "Use provided MOIC dataset":
            return float(rng.choice(MOIC_POOL))
        elif moic_mode == "Use custom MOIC list" and len(custom_moic_list) > 0:
            return float(rng.choice(custom_moic_list))
        elif moic_mode == "Use parametric (lognormal) — set mean & std":
            mu_ln, sig_ln = _lognormal_params_from_mean_std(lognorm_mean, lognorm_std)
            return float(rng.lognormal(mean=mu_ln, sigma=sig_ln))
        # Fallback: dataset
        return float(rng.choice(MOIC_POOL))

    def simulate_one_iteration(rng: np.random.Generator) -> Tuple[float, List[Deal]]:
        # 1) Number of deals & total invested (Y1–Y4), with bounds
        n_deals = truncated_normal_int(mu_deals, sd_deals, 9, 15, rng)
        total_invested = truncated_normal_float(mu_total, sd_total, 400_000_000.0, 600_000_000.0, rng)

        # 2) Year-1 fixed deals (amounts fixed; MOIC & exit year randomized)
        deals: List[Deal] = []
        y1_amounts = [9_500_000.0, 10_000_000.0, 35_500_000.0]
        for amt in y1_amounts:
            moic = sample_moic(rng)
            exit_year = sample_exit_year(1, rng)
            deals.append(Deal(invested_amount=amt, invest_year=1, exit_year=exit_year, moic=moic))

        remaining_deals = max(0, n_deals - 3)
        remaining_total = max(0.0, total_invested - sum(y1_amounts))
        if remaining_deals == 0 or remaining_total <= 0:
            cfs = annual_cash_flows(deals, cfg)
            return irr(cfs), deals

        # 3) Sizes for new deals in [$20m, $45m], sum to remaining_total
        sizes = allocate_per_deal_sizes(remaining_total, remaining_deals, 20_000_000.0, 45_000_000.0, rng)

        # 4) Invest years {2,3,4} to softly match cumulative targets
        invest_years = assign_invest_years(sizes, total_invested, rng)

        # 5) MOICs & exit years for new deals
        for amt, inv_y in zip(sizes, invest_years):
            moic = sample_moic(rng)
            exit_year = sample_exit_year(inv_y, rng)
            deals.append(Deal(invested_amount=float(amt), invest_year=int(inv_y), exit_year=exit_year, moic=moic))

        # 6) CFs & IRR
        cfs = annual_cash_flows(deals, cfg)
        r = irr(cfs)
        return r, deals

    # ---- Run MC ----
    if st.button("Run Monte Carlo", type="primary", use_container_width=True, key="btn_run_mc"):
        rng = np.random.default_rng(INTERNAL_RANDOM_SEED)  # None => fresh randomness each run
        irr_list: List[float] = []
        for _ in range(int(num_iterations)):
            try:
                r, _ = simulate_one_iteration(rng)
                irr_list.append(r)
            except Exception:
                pass

        if not irr_list:
            st.error("No successful iterations. Try adjusting parameters.")
        else:
            arr = np.array(irr_list)
            p5, p25, p50, p75, p95 = np.percentile(arr, [5, 25, 50, 75, 95])
            mean = float(np.mean(arr))
            std  = float(np.std(arr))

            k1, k2, k3, k4, k5, k6, k7 = st.columns(7)
            k1.metric("Mean IRR", f"{mean*100:,.2f}%")
            k2.metric("Std Dev", f"{std*100:,.2f}%")
            k3.metric("p5", f"{p5*100:,.2f}%")
            k4.metric("p25", f"{p25*100:,.2f}%")
            k5.metric("p50 (Med)", f"{p50*100:,.2f}%")
            k6.metric("p75", f"{p75*100:,.2f}%")
            k7.metric("p95", f"{p95*100:,.2f}%")

            st.subheader("IRR Histogram")
            fig, ax = plt.subplots()
            ax.hist(arr * 100, bins=40, edgecolor="white")
            ax.set_xlabel("IRR (%)")
            ax.set_ylabel("Frequency")
            ax.set_title("Monte Carlo IRR Distribution")
            st.pyplot(fig)

            df_out = pd.DataFrame({"iteration": np.arange(1, len(arr)+1), "irr": arr})
            csv_buf = StringIO()
            df_out.to_csv(csv_buf, index=False)
            st.download_button("Download IRR by Iteration (CSV)", data=csv_buf.getvalue(),
                               file_name="mc_irr_results.csv", mime="text/csv")