import streamlit as st
import requests
import pandas as pd
import numpy as np

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
    """Maps Tickers to SEC 10-digit CIK numbers."""
    url = "https://www.sec.gov/files/company_tickers.json"
    res = requests.get(url, headers=SEC_HEADERS, timeout=10)
    data = res.json()
    mapping = {}
    for entry in data.values():
        mapping[entry["ticker"].upper()] = str(entry["cik_str"]).zfill(10)
    return mapping

def extract_latest_xbrl_facts(facts_json, concept_names, taxonomy="us-gaap", is_duration=False):
    """
    Extracts the most recent 10-K or 10-Q reported value across matching XBRL concepts.
    If is_duration=True, prefers full-year (10-K) numbers for flow metrics (Income, SBC, Buybacks).
    """
    if "facts" not in facts_json or taxonomy not in facts_json["facts"]:
        return 0.0
    
    for concept in concept_names:
        if concept in facts_json["facts"][taxonomy]:
            units = facts_json["facts"][taxonomy][concept].get("units", {})
            values = units.get("USD", units.get("shares", []))
            
            if is_duration:
                # Prefer full-year 10-K filings for Income/Cash Flow items
                reports = [v for v in values if v.get("form") in ["10-K", "10-K/A"] and v.get("fp") in ["FY", "CY"]]
                if not reports:
                    reports = [v for v in values if v.get("form") in ["10-K", "10-K/A", "10-Q", "10-Q/A"]]
            else:
                # Instantaneous balance sheet / share counts (take latest available date)
                reports = [v for v in values if v.get("form") in ["10-K", "10-K/A", "10-Q", "10-Q/A"]]
            
            if reports:
                reports.sort(key=lambda x: x.get("end", ""))
                return float(reports[-1].get("val", 0.0))
    return 0.0

def extract_annual_growth_rate(facts_json):
    """Calculates historical annual revenue growth rate from the last two 10-K filings."""
    concepts = ["RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues", "SalesRevenueNet"]
    for concept in concepts:
        if "facts" in facts_json and "us-gaap" in facts_json["facts"] and concept in facts_json["facts"]["us-gaap"]:
            units = facts_json["facts"]["us-gaap"][concept].get("units", {}).get("USD", [])
            annual = [v for v in units if v.get("form") in ["10-K", "10-K/A"] and v.get("fp") in ["FY", "CY"]]
            if len(annual) >= 2:
                annual.sort(key=lambda x: x.get("end", ""))
                v_latest = annual[-1].get("val", 0.0)
                v_prev = annual[-2].get("val", 0.0)
                if v_prev > 0:
                    g = (v_latest - v_prev) / v_prev
                    return max(0.03, min(g, 0.18))
    return 0.10

def fetch_live_price(ticker_symbol):
    """Retrieves live stock price."""
    try:
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker_symbol}?interval=1d&range=1d"
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        res = requests.get(url, headers=headers, timeout=5)
        meta = res.json()["chart"]["result"][0]["meta"]
        return float(meta.get("regularMarketPrice", meta.get("chartPreviousClose", 100.0)))
    except Exception:
        return 100.0

def fetch_sec_financials(symbol):
    """Ingests comprehensive financial and leverage data directly from SEC EDGAR."""
    ticker_map = get_sec_ticker_mapping()
    if symbol not in ticker_map:
        raise ValueError(f"Ticker '{symbol}' not found in SEC EDGAR directory.")
    
    cik = ticker_map[symbol]
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
    res = requests.get(url, headers=SEC_HEADERS, timeout=10)
    if res.status_code != 200:
        raise ConnectionError(f"SEC EDGAR returned status {res.status_code}")
    
    facts = res.json()

    # 1. Income & SBC Items (Flow metrics: full year duration)
    N_raw = extract_latest_xbrl_facts(facts, ["NetIncomeLossAvailableToCommonStockholdersBasic", "NetIncomeLoss", "ProfitLoss"], is_duration=True)
    G_raw = extract_latest_xbrl_facts(facts, ["AllocatedShareBasedCompensationExpense", "ShareBasedCompensation", "ShareBasedCompensationArrangementByShareBasedPaymentAwardExpense"], is_duration=True)
    T_raw = extract_latest_xbrl_facts(facts, ["PaymentsForRepurchaseOfCommonStock", "PaymentsForRepurchaseOfEquity", "PaymentsRelatedToTaxWithholdingForShareBasedCompensation"], is_duration=True)

    # 2. Cash & Liquid Securities
    cash_raw = extract_latest_xbrl_facts(facts, ["CashAndCashEquivalentsAtCarryingValue", "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"])
    st_inv_raw = extract_latest_xbrl_facts(facts, ["MarketableSecuritiesCurrent", "AvailableForSaleSecuritiesCurrent", "OtherShortTermInvestments"])

    # 3. Comprehensive Total Debt (Short-Term, Long-Term, and Leases)
    lt_debt = extract_latest_xbrl_facts(facts, ["LongTermDebtNoncurrent", "LongTermDebtAndCapitalLeaseObligations"])
    st_debt = extract_latest_xbrl_facts(facts, ["DebtCurrent", "ShortTermBorrowings", "CommercialPaper"])
    leases_nc = extract_latest_xbrl_facts(facts, ["OperatingLeaseLiabilityNoncurrent", "FinanceLeaseLiabilityNoncurrent"])
    leases_cur = extract_latest_xbrl_facts(facts, ["OperatingLeaseLiabilityCurrent", "FinanceLeaseLiabilityCurrent"])
    
    total_debt_raw = lt_debt + st_debt + leases_nc + leases_cur
    if total_debt_raw == 0.0:
        total_debt_raw = extract_latest_xbrl_facts(facts, ["LongTermDebt", "DebtAndCapitalLeaseObligations", "LiabilitiesOtherThanLongTermDebtAndCapitalLeasesCurrent"])

    # 4. Diluted Share Count
    shares_raw = extract_latest_xbrl_facts(facts, ["WeightedAverageNumberOfDilutedSharesOutstanding"], is_duration=True)
    if shares_raw == 0.0:
        shares_raw = extract_latest_xbrl_facts(facts, ["EntityCommonStockSharesOutstanding"], taxonomy="dei")
    if shares_raw == 0.0:
        shares_raw = extract_latest_xbrl_facts(facts, ["CommonStockSharesOutstanding"])

    price = fetch_live_price(symbol)
    g1 = extract_annual_growth_rate(facts)

    return {
        "ticker": symbol,
        "price": price,
        "shares": (shares_raw / 1e6) if shares_raw > 0 else 100.0,
        "N": N_raw / 1e6,
        "G": G_raw / 1e6,
        "T": abs(T_raw) / 1e6,
        "total_cash": (cash_raw + st_inv_raw) / 1e6,
        "total_debt": total_debt_raw / 1e6,
        "g1": g1
    }

# User Controls
ticker = st.text_input("Enter Stock Ticker", value="ADBE").upper().strip()
tier_name = st.selectbox("Baseline AICT Moat Tier", list(AICT_TIERS.keys()), index=2)
tier = AICT_TIERS[tier_name]

if st.button("Evaluate Stock", type="primary"):
    with st.spinner(f"Loading audited SEC facts for {ticker}..."):
        try:
            st.session_state["calc_data"] = fetch_sec_financials(ticker)
        except Exception as e:
            st.error(f"Error loading SEC data: {e}")

if "calc_data" in st.session_state and st.session_state["calc_data"]["ticker"] == ticker:
    cd = st.session_state["calc_data"]

    # 1. Tragic Algebra SBC Dilution (Ω)
    C_val = cd["G"] * 0.20
    V_val = min(cd["T"], cd["G"] * 0.90 + cd["T"] * 0.20) if cd["T"] > 0 else cd["G"] * 1.10
    Omega = C_val + V_val
    auto_oe = max(1.0, cd["N"] + cd["G"] - Omega)

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

    # 3. 15-Year Hybrid Valuation Engine (r = 15%)
    g2 = g1_adj * tier["mult"]
    r = 0.15
    stream = []
    val = base_oe
    for yr in range(1, 16):
        val *= (1 + g1_adj) if yr <= tier["s1"] else (1 + g2)
        stream.append(val)

    pv_cf = sum([v / ((1 + r) ** y) for y, v in enumerate(stream, start=1)])
    tv_dcf = (stream[-1] * (1 + tier["gt"])) / (r - tier["gt"]) if (r > tier["gt"]) else 0.0
    pv_m1 = pv_cf + (tv_dcf / ((1 + r) ** 15))
    pv_m2 = pv_cf + ((stream[-1] * exit_m_adj) / ((1 + r) ** 15))
    hybrid_pv = (pv_m1 + pv_m2) / 2.0

    net_cash = cash_adj - debt_adj
    total_equity = hybrid_pv + net_cash
    iv15 = total_equity / shares_adj if shares_adj > 0 else 0.0

    p_iv15 = cd["price"] / iv15 if iv15 > 0 else 999.0
    iv12 = iv15 * ((1.15 / 1.12) ** 15)
    iv18 = iv15 * ((1.15 / 1.18) ** 15)

    # 4. Valuation Results Display
    st.markdown("---")
    st.subheader(f"📊 Valuation Verdict: {ticker}")

    res_c1, res_c2, res_c3 = st.columns(3)
    res_c1.metric("Market Price", f"${cd['price']:.2f}")
    res_c2.metric("Estimated IV15", f"${iv15:.2f}", f"P/IV15: {p_iv15:.2f}x")
    res_c3.metric("Owners' Earnings (OE)", f"${base_oe:,.0f} M", f"Growth: {g1_adj*100:.1f}%")

    if p_iv15 <= 1.0:
        st.success(f"🎯 **FAT PITCH (BUY)**: At **${cd['price']:.2f}**, {ticker} is priced below IV15 (**${iv15:.2f}**) and offers an estimated ≥15% annualized return.")
    elif p_iv15 <= 1.5:
        st.info(f"⚠️ **JUST OUTSIDE (WATCHLIST)**: At **${cd['price']:.2f}**, price approaches value territory. Expected return ~12%–14% (IV12: **${iv12:.2f}**).")
    else:
        st.error(f"⛔ **OUT FIELD (OVERVALUED)**: Price (**${cd['price']:.2f}**) is well above IV15 (**${iv15:.2f}**).")

    st.write("**Target Entry Bands:**")
    st.write(f"- **18% Annual Return (Deep Margin of Safety):** Buy under **${iv18:.2f}**")
    st.write(f"- **15% Annual Return (IV15 Baseline):** Buy under **${iv15:.2f}**")
    st.write(f"- **12% Annual Return (Fair / Moderate):** Buy under **${iv12:.2f}**")
