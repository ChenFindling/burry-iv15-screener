import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

st.set_page_config(page_title="Burry IV15 Screener", layout="centered")

st.title("🎯 Burry IV15 Value Screener")
st.caption("Michael Burry's True Owners' Earnings (OE) & AICT Tier Engine")

# AICT Tier Rules
AICT_TIERS = {
    "Fortress (Monopoly / Regulated)":   {"s1": 8, "mult": 0.70, "gt": 0.07, "debt_cap": 3.0, "exit_m": 20.0},
    "Castle (Strong Moat / Dominant)":   {"s1": 7, "mult": 0.55, "gt": 0.05, "debt_cap": 2.5, "exit_m": 16.0},
    "Chapel (High Moat / Real Threat)":  {"s1": 5, "mult": 0.45, "gt": 0.04, "debt_cap": 2.0, "exit_m": 14.5},
    "Stone (Vulnerable / Seat Loss)":    {"s1": 4, "mult": 0.35, "gt": 0.03, "debt_cap": 0.0, "exit_m": 9.0},
    "Wood (Fragile / Wrapper / No R&D)": {"s1": 2, "mult": 0.25, "gt": 0.00, "debt_cap": 0.0, "exit_m": 5.0}
}

ticker = st.text_input("Enter Stock Ticker", value="ADBE").upper().strip()
tier_name = st.selectbox("Company AICT Moat Tier", list(AICT_TIERS.keys()), index=2)
tier = AICT_TIERS[tier_name]

if st.button("Evaluate Stock", type="primary"):
    with st.spinner(f"Fetching data for {ticker}..."):
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            inc = stock.financials
            cf = stock.cashflow
            bs = stock.balance_sheet

            price = info.get("currentPrice", info.get("regularMarketPrice", 0.0))
            shares = info.get("sharesOutstanding", 1.0) / 1e6

            # Income & Cash Flows
            N = inc.loc["Net Income"].iloc[0] / 1e6 if "Net Income" in inc.index else 0.0
            G = cf.loc["Stock Based Compensation"].iloc[0] / 1e6 if "Stock Based Compensation" in cf.index else 0.0
            T = abs(cf.loc["Repurchase Of Capital Stock"].iloc[0] / 1e6) if "Repurchase Of Capital Stock" in cf.index else 0.0
            ebitda = inc.loc["EBITDA"].iloc[0] / 1e6 if "EBITDA" in inc.index else (inc.loc["Operating Income"].iloc[0] / 1e6 if "Operating Income" in inc.index else N * 1.3)

            # Balance Sheet
            cash = (bs.loc["Cash And Cash Equivalents"].iloc[0] / 1e6) if "Cash And Cash Equivalents" in bs.index else 0.0
            st_inv = (bs.loc["Other Short Term Investments"].iloc[0] / 1e6) if "Other Short Term Investments" in bs.index else 0.0
            total_cash = cash + st_inv
            total_debt = (bs.loc["Total Debt"].iloc[0] / 1e6) if "Total Debt" in bs.index else 0.0

            # Forward Growth Baseline
            rev_growth = info.get("revenueGrowth", 0.10)
            if rev_growth is None or np.isnan(rev_growth):
                rev_growth = 0.10
            g1 = max(0.04, min(rev_growth, 0.15))

            # Store in session state for instant calculation & adjustment
            st.session_state["calc_data"] = {
                "ticker": ticker, "price": price, "shares": shares,
                "N": N, "G": G, "T": T, "ebitda": ebitda,
                "total_cash": total_cash, "total_debt": total_debt,
                "g1": g1
            }
        except Exception as e:
            st.error(f"Error fetching data: {e}")

if "calc_data" in st.session_state and st.session_state["calc_data"]["ticker"] == ticker:
    cd = st.session_state["calc_data"]

    # 1. Tragic Algebra Base Owners' Earnings
    # If company has positive buybacks, account for dilution offset
    if cd["T"] > 0:
        Omega = (cd["G"] * 0.20) + min(cd["T"], cd["G"] * 0.90 + cd["T"] * 0.20)
    else:
        Omega = (cd["G"] * 0.20) + (cd["G"] * 1.10)
    
    auto_oe = max(1.0, cd["N"] + cd["G"] - Omega)

    # 2. Transparent Adjustments Bar
    st.markdown("---")
    st.subheader("⚙️ Underlying Data & Adjustments")
    adj_col1, adj_col2, adj_col3 = st.columns(3)
    
    with adj_col1:
        base_oe = st.number_input("Base Owners' Earnings ($M) [OE]", value=float(round(auto_oe, 1)))
        g1_adj = st.number_input("Stage 1 Growth Rate (%)", value=float(round(cd["g1"] * 100, 1))) / 100.0
    with adj_col2:
        shares_adj = st.number_input("Diluted Shares Outstanding (M)", value=float(round(cd["shares"], 1)))
        exit_m_adj = st.number_input("Year 15 Exit Multiple", value=float(tier["exit_m"]))
    with adj_col3:
        cash_adj = st.number_input("Total Cash + ST Investments ($M)", value=float(round(cd["total_cash"], 1)))
        debt_adj = st.number_input("Total Debt ($M)", value=float(round(cd["total_debt"], 1)))

    # 3. Valuation Engine Execution (r = 15%)
    g2 = g1_adj * tier["mult"]
    r = 0.15

    oe_stream = []
    curr_val = base_oe
    for y in range(1, 16):
        curr_val *= (1 + g1_adj) if y <= tier["s1"] else (1 + g2)
        oe_stream.append(curr_val)

    pv_cf = sum([val / ((1 + r) ** yr) for yr, val in enumerate(oe_stream, start=1)])
    
    # Model 1: DCF Perpetual
    tv_dcf = (oe_stream[-1] * (1 + tier["gt"])) / (r - tier["gt"]) if (r > tier["gt"]) else 0.0
    pv_model1 = pv_cf + (tv_dcf / ((1 + r) ** 15))

    # Model 2: Buffett Exit Multiple
    pv_model2 = pv_cf + ((oe_stream[-1] * exit_m_adj) / ((1 + r) ** 15))

    # 50/50 Hybrid Blend
    hybrid_op_pv = (pv_model1 + pv_model2) / 2.0

    # Balance Sheet Net Cash (Cash - Debt)
    net_cash = cash_adj - debt_adj
    total_equity = hybrid_op_pv + net_cash
    iv15 = total_equity / shares_adj if shares_adj > 0 else 0.0

    p_iv15 = cd["price"] / iv15 if iv15 > 0 else 999.0
    iv12 = iv15 * ((1.15 / 1.12) ** 15)
    iv18 = iv15 * ((1.15 / 1.18) ** 15)

    # 4. Results
    st.markdown("---")
    st.subheader(f"📊 Valuation Verdict: {ticker}")

    c1, c2, c3 = st.columns(3)
    c1.metric("Current Market Price", f"${cd['price']:.2f}")
    c2.metric("Estimated IV15", f"${iv15:.2f}", f"P/IV15: {p_iv15:.2f}x")
    c3.metric("Owners' Earnings (OE)", f"${base_oe:,.0f} M", f"Growth: {g1_adj*100:.1f}%")

    if p_iv15 <= 1.0:
        st.success(f"🎯 **FAT PITCH (BUY)**: At ${cd['price']:.2f}, {ticker} is priced below IV15 (${iv15:.2f}) and offers an estimated ≥15% annualized return.")
    elif p_iv15 <= 1.5:
        st.info(f"⚠️ **JUST OUTSIDE (WATCHLIST)**: Approaches value territory. Expected return ~12%–14% (IV12: ${iv12:.2f}).")
    else:
        st.error(f"⛔ **OUT FIELD (OVERVALUED)**: Price (${cd['price']:.2f}) is well above IV15 (${iv15:.2f}).")

    st.write("**Target Entry Bands:**")
    st.write(f"- **18% Annual Return (Deep Margin of Safety):** Buy under **${iv18:.2f}**")
    st.write(f"- **15% Annual Return (IV15 Baseline):** Buy under **${iv15:.2f}**")
    st.write(f"- **12% Annual Return (Fair / Moderate):** Buy under **${iv12:.2f}**")