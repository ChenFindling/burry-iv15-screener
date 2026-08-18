import streamlit as st
import requests
import pandas as pd
import numpy as np
from datetime import datetime

st.set_page_config(page_title="Burry IV15 Screener", layout="centered")

st.title("🎯 Burry IV15 Value Screener")
st.caption("Automated SEC EDGAR Audited Financials & AICT Moat Valuation Engine")

# 1. AICT Moat Tier Rules
AICT_TIERS = {
    "Fortress (Monopoly / Regulated)":   {"s1": 8, "mult": 0.70, "gt": 0.07, "debt_cap": 3.0, "exit_m": 20.0},
    "Castle (Strong Moat / Dominant)":   {"s1": 7, "mult": 0.55, "gt": 0.05, "debt_cap": 2.5, "exit_m": 16.0},
    "Chapel (High Moat / Real Threat)":  {"s1": 5, "mult": 0.45, "gt": 0.04, "debt_cap": 2.0, "exit_m": 14.5},
    "Stone (Vulnerable / Seat Loss)":    {"s1": 4, "mult": 0.35, "gt": 0.03, "debt_cap": 0.0, "exit_m": 9.0},
    "Wood (Fragile / Wrapper / No R&D)": {"s1": 2, "mult": 0.25, "gt": 0.00, "debt_cap": 0.0, "exit_m": 5.0}
}

SEC_HEADERS = {
    "User-Agent": "ValueInvestorResearch admin@valueinvestor.org",
    "Accept-Encoding": "gzip, deflate"
}

@st.cache_data(ttl=86400, show_spinner=False)
def get_sec_ticker_mapping():
    url = "https://www.sec.gov/files/company_tickers.json"
    res = requests.get(url, headers=SEC_HEADERS, timeout=10)
    data = res.json()
    mapping = {}
    for entry in data.values():
        mapping[entry["ticker"].upper()] = str(entry["cik_str"]).zfill(10)
    return mapping

def parse_days_duration(entry):
    if "start" in entry and "end" in entry:
        try:
            d1 = datetime.strptime(entry["start"], "%Y-%m-%d")
            d2 = datetime.strptime(entry["end"], "%Y-%m-%d")
            return (d2 - d1).days
        except Exception:
            return 0
    return 0

def extract_latest_xbrl_facts(facts_json, concept_names, taxonomy="us-gaap", is_flow=False):
    if "facts" not in facts_json or taxonomy not in facts_json["facts"]:
        return 0.0
    
    us_facts = facts_json["facts"][taxonomy]
    for concept in concept_names:
        if concept in us_facts:
            units = us_facts[concept].get("units", {})
            values = units.get("USD", units.get("shares", []))
            
            if is_flow:
                # Strictly isolate full-year (>=330 days) reports to avoid quarterly fragments
                annual_reports = [
                    v for v in values 
                    if v.get("form") in ["10-K", "10-K/A"] 
                    and parse_days_duration(v) >= 330
                ]
                if annual_reports:
                    annual_reports.sort(key=lambda x: x.get("end", ""))
                    return float(annual_reports[-1].get("val", 0.0))
            else:
                # Balance sheet & Shares items: Latest available point-in-time filing
                point_in_time = [v for v in values if v.get("form") in ["10-K", "10-K/A", "10-Q", "10-Q/A"]]
                if point_in_time:
                    point_in_time.sort(key=lambda x: x.get("end", ""))
                    return float(point_in_time[-1].get("val", 0.0))
    return 0.0

def extract_annual_growth_rate(facts_json):
    concepts = ["RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues", "SalesRevenueNet"]
    if "facts" in facts_json and "us-gaap" in facts_json["facts"]:
        us_gaap_facts = facts_json["facts"]["us-gaap"]
        for concept in concepts:
            if concept in us_gaap_facts:
                units = us_gaap_facts[concept].get("units", {}).get("USD", [])
                annual = [
                    v for v in units 
                    if v.get("form") in ["10-K", "10-K/A"] 
                    and parse_days_duration(v) >= 330
                ]
                if len(annual) >= 2:
                    annual.sort(key=lambda x: x.get("end", ""))
                    v_latest = annual[-1].get("val", 0.0)
                    v_prev = annual[-2].get("val", 0.0)
                    if v_prev > 0:
                        g = (v_latest - v_prev) / v_prev
                        return max(0.03, min(g, 0.18))
    return 0.10

def fetch_live_price(ticker_symbol):
    try:
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker_symbol}?interval=1d&range=1d"
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        res = requests.get(url, headers=headers, timeout=5)
        meta = res.json()["chart"]["result"][0]["meta"]
        return float(meta.get("regularMarketPrice", meta.get("chartPreviousClose", 100.0)))
    except Exception:
        return 100.0

def fetch_sec_financials(symbol):
    ticker_map = get_sec_ticker_mapping()
    if symbol not in ticker_map:
        raise ValueError(f"Ticker '{symbol}' not found in SEC EDGAR directory.")
    
    cik = ticker_map[symbol]
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
    res = requests.get(url, headers=SEC_HEADERS, timeout=10)
    if res.status_code != 200:
        raise ConnectionError(f"SEC EDGAR returned status {res.status_code}")
    
    facts = res.json()

    # 1. Income & SBC Items (Full 1-Year Periods)
    N_raw = extract_latest_xbrl_facts(facts, ["NetIncomeLossAvailableToCommonStockholdersBasic", "NetIncomeLoss", "ProfitLoss"], is_flow=True)
    G_raw = extract_latest_xbrl_facts(facts, ["AllocatedShareBasedCompensationExpense", "ShareBasedCompensation", "ShareBasedCompensationArrangementByShareBasedPaymentAwardExpense"], is_flow=True)
    T_raw = extract_latest_xbrl_facts(facts, ["PaymentsForRepurchaseOfCommonStock", "PaymentsForRepurchaseOfEquity", "PaymentsRelatedToTaxWithholdingForShareBasedCompensation"], is_flow=True)

    # 2. Cash & Liquid Securities
    cash_raw = extract_latest_xbrl_facts(facts, ["CashAndCashEquivalentsAtCarryingValue", "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"])
    st_inv_raw = extract_latest_xbrl_facts(facts, ["MarketableSecuritiesCurrent", "AvailableForSaleSecuritiesCurrent", "OtherShortTermInvestments"])

    # 3. Funded Debt (Excludes Payroll Tax Float and Client Funds)
    lt_debt = extract_latest_xbrl_facts(facts, ["LongTermDebtNoncurrent", "LongTermDebtAndCapitalLeaseObligations"])
    st_debt = extract_latest_xbrl_facts(facts, ["DebtCurrent", "ShortTermBorrowings", "CommercialPaper"])
    leases_nc = extract_latest_xbrl_facts(facts, ["OperatingLeaseLiabilityNoncurrent", "FinanceLeaseLiabilityNoncurrent"])
    leases_cur = extract_latest_xbrl_facts(facts, ["OperatingLeaseLiabilityCurrent", "FinanceLeaseLiabilityCurrent"])
    
    total_debt_raw = lt_debt + st_debt + leases_nc + leases_cur

    # 4. Diluted Share Count
    shares_raw = extract_latest_xbrl_facts(facts, ["WeightedAverageNumberOfDilutedSharesOutstanding"], is_flow=True)
    if shares_raw == 0.0:
        shares_raw = extract_latest_xbrl_facts(facts, ["EntityCommonStockSharesOutstanding"], taxonomy="dei")
    if shares_raw == 0.0:
        shares_raw = extract_latest_xbrl_facts(facts, ["CommonStockSharesOutstanding"])

    price = fetch_live_price(symbol)
    g1 = extract_annual_growth_rate(facts)

    return {
        "ticker": symbol, "price": price,
        "shares": (shares_raw / 1e6) if shares_raw > 0 else 100.0,
        "N": N_raw / 1e6, "G": G_raw / 1e6, "T": abs(T_raw) / 1e6,
        "total_cash": (cash_raw + st_inv_raw) / 1e6, "total_debt": total_debt_raw / 1e6,
        "g1": g1
    }

# Search Bar
ticker = st.text_input("Enter Stock Ticker", value="", placeholder="e.g. PAYC, CRM, NOW, ADBE").upper().strip()
tier_name = st.selectbox("Baseline AICT Moat Tier", list(AICT_TIERS.keys()), index=2)
tier = AICT_TIERS[tier_name]

if st.button("Evaluate Stock", type="primary"):
    if not ticker:
        st.warning("Please enter a stock ticker symbol first.")
    else:
        with st.spinner(f"Ingesting audited full-year SEC 10-K facts for {ticker}..."):
            try:
                st.session_state["calc_data"] = fetch_sec_financials(ticker)
            except Exception as e:
                st.error(f"Error loading SEC data: {e}")

if "calc_data" in st.session_state and ticker and st.session_state["calc_data"]["ticker"] == ticker:
    cd = st.session_state["calc_data"]

    # 1. Tragic Algebra & Diagnostic Calculations
    C_val = cd["G"] * 0.20
    V_val = min(cd["T"], cd["G"] * 0.90 + cd["T"] * 0.20) if cd["T"] > 0 else cd["G"] * 1.10
    Omega = C_val + V_val
    auto_oe = max(1.0, cd["N"] + cd["G"] - Omega)

    dilution_tax_rate = ((Omega - cd["G"]) / cd["N"] * 100.0) if cd["N"] > 0 else 0.0
    buyback_treadmill_pct = min(100.0, (Omega / cd["T"] * 100.0)) if cd["T"] > 0 else 0.0

    # 2. Financial Adjustments Bar
    st.markdown("---")
    st.subheader("⚙️ Underlying Financials & Adjustments")
    adj_c1, adj_c2, adj_c3 = st.columns(3)
    with adj_c1:
        base_oe = st.number_input("Base Owners' Earnings ($M) [OE]", value=float(round(auto_oe, 1)))
        g1_adj = st.number_input("Stage 1 Growth Rate (%)", value=float(round(cd["g1"] * 100, 1))) / 100.0
    with adj_c2:
        shares_adj = st.number_input("Diluted Shares (M)", value=float(round(cd["shares"], 1)))
        exit_m_adj = st.number_input("Exit Multiple (M15)", value=float(tier["exit_m"]))
    with adj_c3:
        cash_adj = st.number_input("Cash & ST Inv ($M)", value=float(round(cd["total_cash"], 1)))
        debt_adj = st.number_input("Total Debt ($M)", value=float(round(cd["total_debt"], 1)))

    # 3. Hamster Wheel Diagnostic Card
    st.markdown("---")
    st.subheader("🐹 Tragic Algebra Diagnostic")
    d1, d2, d3 = st.columns(3)
    d1.metric("True SBC Burden (Ω)", f"${Omega:,.1f} M", f"GAAP SBC: ${cd['G']:,.1f} M")
    d2.metric("Dilution Tax Rate", f"{dilution_tax_rate:.1f}%", "Transfer from owners")
    d3.metric("Buyback Offset Drag", f"{buyback_treadmill_pct:.1f}%", "Used to offset SBC")

    if buyback_treadmill_pct > 80.0:
        st.warning("⚠️ **Hamster Wheel Alert:** Over 80% of company buybacks merely neutralize employee stock grants rather than reducing share count!")
    elif dilution_tax_rate > 25.0:
        st.info("ℹ️ **High Dilution Drag:** Stock-based compensation significantly reduces the earnings attributable to common shareholders.")

    # 4. Valuation Engine Helper
    def run_valuation(oe_b, g_rate, t_rules, m_exit, s_count, c_val, d_val):
        g_two = g_rate * t_rules["mult"]
        r_rate = 0.15
        s_stream = []
        c_val_proj = oe_b
        for y_idx in range(1, 16):
            c_val_proj *= (1 + g_rate) if y_idx <= t_rules["s1"] else (1 + g_two)
            s_stream.append(c_val_proj)
        
        pv_cfs = sum([v / ((1 + r_rate) ** y) for y, v in enumerate(s_stream, start=1)])
        tv_dcf_calc = (s_stream[-1] * (1 + t_rules["gt"])) / (r_rate - t_rules["gt"]) if (r_rate > t_rules["gt"]) else 0.0
        pv_m1_calc = pv_cfs + (tv_dcf_calc / ((1 + r_rate) ** 15))
        pv_m2_calc = pv_cfs + ((s_stream[-1] * m_exit) / ((1 + r_rate) ** 15))
        hyb_pv = (pv_m1_calc + pv_m2_calc) / 2.0
        
        n_cash = c_val - d_val
        return (hyb_pv + n_cash) / s_count if s_count > 0 else 0.0

    iv15_baseline = run_valuation(base_oe, g1_adj, tier, exit_m_adj, shares_adj, cash_adj, debt_adj)
    p_iv15 = cd["price"] / iv15_baseline if iv15_baseline > 0 else 999.0
    iv12 = iv15_baseline * ((1.15 / 1.12) ** 15)
    iv18 = iv15_baseline * ((1.15 / 1.18) ** 15)

    # 5. Stress-Testing Engine
    st.markdown("---")
    st.subheader("🧪 Scenario Stress-Testing Engine")
    st.caption("Evaluate what happens to IV15 if growth slows or AI degrades the moat tier.")

    stress_col1, stress_col2 = st.columns(2)
    with stress_col1:
        tier_keys = list(AICT_TIERS.keys())
        def_downgrade_idx = min(len(tier_keys) - 1, tier_keys.index(tier_name) + 1)
        stressed_tier_name = st.selectbox("Stress-Test Moat Downgrade", tier_keys, index=def_downgrade_idx)
        stressed_tier = AICT_TIERS[stressed_tier_name]
    with stress_col2:
        growth_haircut_pct = st.slider("Growth Haircut (%)", min_value=-50, max_value=50, value=0, step=5)
    
    stressed_growth = max(0.01, g1_adj * (1 + (growth_haircut_pct / 100.0)))
    stressed_iv15 = run_valuation(base_oe, stressed_growth, stressed_tier, stressed_tier["exit_m"], shares_adj, cash_adj, debt_adj)
    stressed_p_iv15 = cd["price"] / stressed_iv15 if stressed_iv15 > 0 else 999.0

    # 6. Results Display
    st.markdown("---")
    st.subheader(f"📊 Valuation Verdict: {ticker}")

    res_c1, res_c2, res_c3 = st.columns(3)
    res_c1.metric("Market Price", f"${cd['price']:.2f}")
    res_c2.metric("Baseline IV15", f"${iv15_baseline:.2f}", f"P/IV15: {p_iv15:.2f}x")
    res_c3.metric("Stressed IV15", f"${stressed_iv15:.2f}", f"P/IV15: {stressed_p_iv15:.2f}x")

    # Decision Banner
    if p_iv15 <= 1.0:
        st.success(f"🎯 **FAT PITCH (BUY)**: At **\${cd['price']:.2f}**, {ticker} is priced below baseline IV15 (**\${iv15_baseline:.2f}**) and offers an estimated ≥15% annualized return.")
    elif p_iv15 <= 1.5:
        st.info(f"⚠️ **JUST OUTSIDE (WATCHLIST)**: At **\${cd['price']:.2f}**, price approaches value territory. Expected return ~12%–14% (IV12: **\${iv12:.2f}**).")
    else:
        st.error(f"⛔ **OUT FIELD (OVERVALUED)**: Price (\${cd['price']:.2f}) is well above IV15 (\${iv15_baseline:.2f}).")

    st.write("**Target Entry Bands:**")
    st.write(f"- **18% Annual Return (Deep Margin of Safety):** Buy under **\${iv18:.2f}**")
    st.write(f"- **15% Annual Return (IV15 Baseline):** Buy under **\${iv15_baseline:.2f}**")
    st.write(f"- **12% Annual Return (Fair / Moderate):** Buy under **\${iv12:.2f}**")
