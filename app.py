
import math
import re
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests
import streamlit as st


# ============================================================
# Burry IV15 Screener — Simple UI / robust data layer
# ============================================================
#
# Important:
# - The Tragic Algebra equations below come from Burry's published papers.
# - The exact company-by-company IV15 spreadsheet is NOT publicly disclosed.
# - For companies covered in the attached Burry papers, we include a published
#   IV15 reference so the tool can reproduce the article value rather than
#   pretending an automatic approximation is exact.
# - For other tickers, the app calculates a transparent reconstruction.
#
# The tool intentionally keeps the UI simple:
#   ticker -> evaluate -> key data -> verdict -> targets
#
# Advanced assumptions are hidden in an expander.
# ============================================================

st.set_page_config(
    page_title="Burry IV15 Value Screener",
    page_icon="🎯",
    layout="centered",
)

AICT_TIERS = {
    "Fortress": {
        "s1": 8,
        "mult": 0.70,
        "gt": 0.07,
        "debt_cap": 3.0,
        "default_exit": 20.0,
        "description": "Little competitive weakness / very durable position",
    },
    "Castle": {
        "s1": 7,
        "mult": 0.55,
        "gt": 0.05,
        "debt_cap": 2.5,
        "default_exit": 16.0,
        "description": "Strong moat, owned AI / scale, generally secure",
    },
    "Chapel": {
        "s1": 5,
        "mult": 0.45,
        "gt": 0.04,
        "debt_cap": 2.0,
        "default_exit": 14.5,
        "description": "Real moat but meaningful AI / competitive uncertainty",
    },
    "Stone": {
        "s1": 4,
        "mult": 0.35,
        "gt": 0.03,
        "debt_cap": 0.0,
        "default_exit": 9.0,
        "description": "Vulnerable position / weak adaptation",
    },
    "Wood": {
        "s1": 2,
        "mult": 0.25,
        "gt": 0.00,
        "debt_cap": 0.0,
        "default_exit": 5.0,
        "description": "Fragile wrapper / little owned AI / high long-term risk",
    },
}

# Published examples from the attached Burry articles.
# These are used when available so the screener can reproduce the source value
# rather than inventing a "close enough" formula for an article name.
BURRY_REFERENCE = {
    # Productivity / Cybersecurity article
    "ADBE": {"iv15": 262.00, "tier": "Chapel", "source": "Software & Payments Part II"},
    "DOCU": {"iv15": 11.37, "tier": "Stone", "source": "Software & Payments Part II"},
    "INTU": {"iv15": 114.03, "tier": "Castle", "source": "Software & Payments Part II"},
    "ADSK": {"iv15": 66.22, "tier": "Chapel", "source": "Software & Payments Part II"},
    "U": {"iv15": 13.03, "tier": "Castle", "source": "Software & Payments Part II"},
    "ZS": {"iv15": 21.62, "tier": "Chapel", "source": "Software & Payments Part II"},
    "PANW": {"iv15": 31.00, "tier": "Castle", "source": "Software & Payments Part II"},
    "NOW": {"iv15": 43.56, "tier": "Chapel", "source": "D'AI of the Triffids"},
    "PAYC": {"iv15": 91.38, "tier": "Stone", "source": "D'AI of the Triffids"},
    "FRSH": {"iv15": 5.65, "tier": "Chapel", "source": "D'AI of the Triffids"},
    "HUBS": {"iv15": 39.94, "tier": "Stone", "source": "D'AI of the Triffids"},
    "MNDY": {"iv15": 55.21, "tier": "Chapel", "source": "D'AI of the Triffids"},
}

SEC_HEADERS = {
    "User-Agent": "BurryIV15ResearchTool/2.0 contact@example.com",
    "Accept-Encoding": "gzip, deflate",
}

SESSION = requests.Session()
SESSION.headers.update(SEC_HEADERS)


@st.cache_data(ttl=86400, show_spinner=False)
def get_sec_ticker_mapping():
    r = SESSION.get("https://www.sec.gov/files/company_tickers.json", timeout=20)
    r.raise_for_status()
    data = r.json()
    return {
        item["ticker"].upper(): {
            "cik": str(item["cik_str"]).zfill(10),
            "title": item.get("title", ""),
        }
        for item in data.values()
    }


@st.cache_data(ttl=21600, show_spinner=False)
def get_sec_facts(ticker: str):
    mapping = get_sec_ticker_mapping()
    if ticker.upper() not in mapping:
        raise ValueError(f"{ticker.upper()} was not found in SEC ticker mapping.")

    cik = mapping[ticker.upper()]["cik"]
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
    r = SESSION.get(url, timeout=30)
    r.raise_for_status()
    return r.json()


def _units(fact, preferred=("USD", "shares", "USD/shares")):
    units = fact.get("units", {})
    for u in preferred:
        if u in units:
            return units[u]
    return next(iter(units.values()), [])


def _annual_rows(fact):
    rows = []
    for r in _units(fact):
        if r.get("form") not in ("10-K", "10-K/A"):
            continue
        if not r.get("end"):
            continue

        start = r.get("start")
        end = r.get("end")

        # Prefer true FY rows. Some issuers have CY frames on 10-K facts;
        # those can still be useful, but FY is preferred.
        fp = r.get("fp")
        days = None
        if start and end:
            try:
                days = (
                    datetime.fromisoformat(end) - datetime.fromisoformat(start)
                ).days
            except Exception:
                pass

        if fp == "FY" or (days is not None and days >= 300):
            rr = dict(r)
            rr["_days"] = days
            rows.append(rr)

    return rows


def _latest_annual_fact(facts, concepts):
    us = facts.get("facts", {}).get("us-gaap", {})
    candidates = []

    for concept in concepts:
        if concept not in us:
            continue
        for r in _annual_rows(us[concept]):
            candidates.append(
                (
                    r.get("end", ""),
                    r.get("filed", ""),
                    concept,
                    float(r.get("val", 0.0)),
                    r.get("start"),
                    r.get("end"),
                    r.get("form"),
                    r.get("fp"),
                )
            )

    if not candidates:
        return None

    # Latest fiscal-end wins; within the same end date use latest filing.
    candidates.sort(key=lambda x: (x[0], x[1]))
    row = candidates[-1]

    return {
        "concept": row[2],
        "value": row[3],
        "start": row[4],
        "end": row[5],
        "form": row[6],
        "fp": row[7],
    }


def _annual_history(facts, concepts, n=6):
    us = facts.get("facts", {}).get("us-gaap", {})
    all_rows = []

    for concept in concepts:
        if concept not in us:
            continue
        for r in _annual_rows(us[concept]):
            all_rows.append(
                {
                    "concept": concept,
                    "start": r.get("start"),
                    "end": r.get("end"),
                    "filed": r.get("filed"),
                    "fp": r.get("fp"),
                    "form": r.get("form"),
                    "value": float(r.get("val", 0.0)),
                }
            )

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)

    # For annual history, deduplicate duplicate XBRL presentations of the same
    # fiscal period by taking the latest filed row.
    df = df.sort_values(["end", "filed"]).drop_duplicates(["end"], keep="last")
    df = df.sort_values("end").tail(n).reset_index(drop=True)
    return df


def _instant_history(facts, taxonomy, concepts, n=8):
    tree = facts.get("facts", {}).get(taxonomy, {})
    all_rows = []

    for concept in concepts:
        if concept not in tree:
            continue

        for r in _units(tree[concept], preferred=("USD", "shares")):
            if not r.get("end"):
                continue
            all_rows.append(
                {
                    "concept": concept,
                    "end": r.get("end"),
                    "filed": r.get("filed"),
                    "value": float(r.get("val", 0.0)),
                }
            )

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    df["end"] = pd.to_datetime(df["end"])
    df = df.sort_values(["end", "filed"]).drop_duplicates(["end"], keep="last")
    return df.tail(n).reset_index(drop=True)


def get_latest_instant(facts, taxonomy, concepts, end_on_or_before=None):
    df = _instant_history(facts, taxonomy, concepts, n=40)
    if df.empty:
        return 0.0

    if end_on_or_before is not None:
        dt = pd.Timestamp(end_on_or_before)
        df = df[df["end"] <= dt]
        if df.empty:
            return 0.0

    return float(df.iloc[-1]["value"])


@st.cache_data(ttl=1800, show_spinner=False)
def get_yahoo_history(ticker: str, period_days=2200):
    end = int(datetime.utcnow().timestamp())
    start = int((datetime.utcnow() - timedelta(days=period_days)).timestamp())
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
        f"?period1={start}&period2={end}&interval=1d&events=div%2Csplits"
    )
    r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=20)
    r.raise_for_status()
    payload = r.json()
    result = payload["chart"]["result"][0]

    ts = result.get("timestamp", [])
    quote = result.get("indicators", {}).get("quote", [{}])[0]
    close = quote.get("close", [])

    df = pd.DataFrame({"timestamp": pd.to_datetime(ts, unit="s"), "close": close})
    df = df.dropna(subset=["close"]).reset_index(drop=True)
    return df


@st.cache_data(ttl=900, show_spinner=False)
def get_yahoo_quote(ticker: str):
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
        f"?interval=1d&range=5d"
    )
    r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
    r.raise_for_status()
    result = r.json()["chart"]["result"][0]
    meta = result.get("meta", {})
    price = meta.get("regularMarketPrice")
    if price is None:
        price = meta.get("chartPreviousClose")
    return float(price)


def annual_share_count_history(facts, n=8):
    # DEI is generally the best source for end-of-period shares outstanding.
    df = _instant_history(
        facts,
        "dei",
        ["EntityCommonStockSharesOutstanding"],
        n=n,
    )
    return df


def _select_prior_share_value(shares_df, target_date):
    if shares_df.empty:
        return None
    td = pd.Timestamp(target_date)
    before = shares_df[shares_df["end"] <= td]
    if before.empty:
        return None
    return float(before.iloc[-1]["value"])


def annual_buyback_price(ticker, start_date, end_date, buyback_dollars):
    if buyback_dollars <= 0:
        return 0.0, 0.0, "No buybacks"

    try:
        px = get_yahoo_history(ticker, period_days=2600)
        s = pd.Timestamp(start_date)
        e = pd.Timestamp(end_date)
        subset = px[(px["timestamp"] >= s) & (px["timestamp"] <= e)]

        if subset.empty:
            current = get_yahoo_quote(ticker)
            return current, buyback_dollars / current, "Fallback: current price"

        avg_px = float(subset["close"].mean())
        repurchased = buyback_dollars / avg_px
        return avg_px, repurchased, "Fiscal-year average closing price proxy"
    except Exception:
        try:
            current = get_yahoo_quote(ticker)
            return current, buyback_dollars / current, "Fallback: current price"
        except Exception:
            return 0.0, 0.0, "No market-price proxy"


def tax_withholding_annual(facts):
    fact = _latest_annual_fact(
        facts,
        [
            "PaymentsRelatedToTaxWithholdingForShareBasedCompensation",
            "PaymentsForTaxWithholdingForShareBasedCompensation",
        ],
    )
    if not fact:
        return 0.0, None
    return abs(fact["value"]), fact["concept"]


def fetch_company_data(ticker):
    ticker = ticker.upper().strip()
    facts = get_sec_facts(ticker)
    sec_meta = get_sec_ticker_mapping()[ticker]

    net_income = _latest_annual_fact(
        facts,
        ["NetIncomeLoss", "ProfitLoss", "NetIncomeLossAvailableToCommonStockholdersBasic"],
    )
    sbc = _latest_annual_fact(
        facts,
        [
            "AllocatedShareBasedCompensationExpense",
            "ShareBasedCompensation",
            "ShareBasedCompensationArrangementByShareBasedPaymentAwardExpense",
            "ShareBasedCompensationArrangementsByShareBasedPaymentAwardCompensationExpense",
        ],
    )
    buybacks = _latest_annual_fact(
        facts,
        [
            "PaymentsForRepurchaseOfCommonStock",
            "PaymentsForRepurchaseOfEquity",
            "PaymentsForRepurchaseOfCommonAndPreferredStock",
        ],
    )
    revenue = _latest_annual_fact(
        facts,
        [
            "RevenueFromContractWithCustomerExcludingAssessedTax",
            "Revenues",
            "SalesRevenueNet",
            "RevenueFromContractWithCustomerIncludingAssessedTax",
        ],
    )

    if not net_income or not revenue:
        raise ValueError(
            "The SEC facts feed does not expose enough clean annual data for this ticker."
        )

    current_price = get_yahoo_quote(ticker)

    cash = get_latest_instant(
        facts,
        "us-gaap",
        [
            "CashAndCashEquivalentsAtCarryingValue",
            "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
            "CashAndDueFromBanks",
        ],
    )

    marketable = get_latest_instant(
        facts,
        "us-gaap",
        [
            "MarketableSecuritiesCurrent",
            "AvailableForSaleSecuritiesCurrent",
            "ShortTermInvestments",
            "OtherShortTermInvestments",
        ],
    )

    long_debt = get_latest_instant(
        facts,
        "us-gaap",
        ["LongTermDebtNoncurrent", "LongTermDebt", "LongTermDebtAndFinanceLeaseObligations"],
    )

    short_debt = get_latest_instant(
        facts,
        "us-gaap",
        ["DebtCurrent", "ShortTermBorrowings", "CommercialPaper"],
    )

    shares_df = annual_share_count_history(facts, n=10)
    current_shares = None
    if not shares_df.empty:
        current_shares = float(shares_df.iloc[-1]["value"])

    # Fallback: latest DEI shares if annual history didn't resolve.
    if not current_shares or current_shares <= 0:
        current_shares = get_latest_instant(
            facts,
            "dei",
            ["EntityCommonStockSharesOutstanding"],
        )

    # If the DEI figure is unavailable, use diluted weighted average shares.
    if not current_shares or current_shares <= 0:
        weighted = _latest_annual_fact(
            facts,
            [
                "WeightedAverageNumberOfDilutedSharesOutstanding",
                "WeightedAverageNumberOfSharesOutstandingDiluted",
            ],
        )
        current_shares = weighted["value"] if weighted else 0.0

    # SEC share units are raw shares.
    shares_m = current_shares / 1e6

    # Revenue history for a conservative automatic growth estimate.
    revenue_hist = _annual_history(
        facts,
        [
            "RevenueFromContractWithCustomerExcludingAssessedTax",
            "Revenues",
            "SalesRevenueNet",
            "RevenueFromContractWithCustomerIncludingAssessedTax",
        ],
        n=6,
    )

    if len(revenue_hist) >= 2:
        latest = float(revenue_hist.iloc[-1]["value"])
        prev = float(revenue_hist.iloc[-2]["value"])
        yoy = latest / prev - 1 if prev > 0 else 0.10
    else:
        yoy = 0.10

    if len(revenue_hist) >= 4:
        old = float(revenue_hist.iloc[-4]["value"])
        cagr_3y = (latest / old) ** (1 / 3) - 1 if old > 0 else yoy
    else:
        cagr_3y = yoy

    # We do not use a single-year revenue number blindly. The automatic rate
    # is a blend of recent growth and 3-year growth, bounded to avoid absurd
    # one-year values. User can override it in Advanced.
    auto_growth = 0.65 * yoy + 0.35 * cagr_3y
    auto_growth = float(np.clip(auto_growth, 0.00, 0.22))

    # Tragic Algebra history using fiscal-year average market price as a proxy
    # for average repurchase price.
    tax_w, tax_concept = tax_withholding_annual(facts)
    N = net_income["value"]
    G = abs(sbc["value"]) if sbc else 0.0
    T = abs(buybacks["value"]) if buybacks else 0.0

    annual = []
    if not shares_df.empty:
        sh = shares_df.copy()
        sh["end"] = pd.to_datetime(sh["end"])
        # Use up to the latest 6 share snapshots.
        sh = sh.sort_values("end").tail(8)
    else:
        sh = pd.DataFrame()

    # Build the latest fiscal year explicitly.
    fy_start = net_income.get("start")
    fy_end = net_income.get("end")

    start_shares = _select_prior_share_value(sh, fy_start) if fy_start else None
    end_shares = _select_prior_share_value(sh, fy_end) if fy_end else None

    if end_shares is None:
        end_shares = current_shares
    if start_shares is None:
        start_shares = end_shares

    buy_px, W, buy_px_method = annual_buyback_price(
        ticker,
        fy_start,
        fy_end,
        T,
    ) if fy_start and fy_end else (0.0, 0.0, "Not available")

    delta_s = end_shares - start_shares

    if W > 0:
        # Burry recurrence:
        # I = ΔS + W
        # V = T * (W + ΔS) / W
        I = W + delta_s
        V = T * (W + delta_s) / W
    else:
        I = delta_s
        V = 0.0

    omega = tax_w + V
    true_oe = N + G - omega

    # Guard against pathological XBRL cases:
    # If the inferred anti-dilution value is negative or wildly large relative
    # to the economics, fall back to the simpler SBC + withholding treatment
    # and mark confidence lower.
    data_quality = "High"
    fallback_reason = ""

    if V < 0:
        data_quality = "Medium"
        fallback_reason = "Inferred share issuance exceeded repurchases; anti-dilution term set to zero."
        V = 0.0
        I = max(0.0, delta_s)
        omega = tax_w + V
        true_oe = N + G - omega

    elif N > 0 and V > 3 * max(abs(N), 1.0):
        data_quality = "Medium"
        fallback_reason = "XBRL/share-count relationship produced an unusually large anti-dilution term; using SBC + tax withholding only."
        V = 0.0
        I = max(0.0, delta_s)
        omega = tax_w
        true_oe = N + G - omega

    # Useful diagnostics
    delta_e = true_oe / N if N else math.nan
    dilution_tax_rate = ((omega - G) / N) if N else math.nan
    buyback_offset_pct = (V / T) if T else 0.0

    return {
        "ticker": ticker,
        "company": sec_meta["title"],
        "price": current_price,
        "shares_m": shares_m,
        "cash_m": cash / 1e6,
        "marketable_m": marketable / 1e6,
        "debt_m": (long_debt + short_debt) / 1e6,
        "net_cash_m": (cash + marketable - long_debt - short_debt) / 1e6,
        "net_income_m": N / 1e6,
        "gaap_sbc_m": G / 1e6,
        "buybacks_m": T / 1e6,
        "tax_withholding_m": tax_w / 1e6,
        "buyback_avg_price": buy_px,
        "shares_repurched_m": W / 1e6,
        "delta_shares_m": delta_s / 1e6,
        "sbc_issued_m": I / 1e6,
        "anti_dilution_cost_m": V / 1e6,
        "omega_m": omega / 1e6,
        "owners_earnings_m": true_oe / 1e6,
        "delta_e": delta_e,
        "dilution_tax_rate": dilution_tax_rate,
        "buyback_offset_pct": buyback_offset_pct,
        "revenue_yoy": yoy,
        "revenue_3y_cagr": cagr_3y,
        "auto_growth": auto_growth,
        "latest_fy_end": fy_end,
        "latest_fy_start": fy_start,
        "data_quality": data_quality,
        "fallback_reason": fallback_reason,
        "buy_px_method": buy_px_method,
        "tax_concept": tax_concept,
    }


def owners_earnings_ps(data):
    if data["shares_m"] <= 0:
        return 0.0
    return data["owners_earnings_m"] / data["shares_m"]


def project_year15(oe_ps, stage1_growth, tier):
    s1_years = tier["s1"]
    g2 = stage1_growth * tier["mult"]
    value = oe_ps

    for year in range(1, 16):
        value *= 1 + (stage1_growth if year <= s1_years else g2)
    return value, g2


def calculate_iv15(oe_ps, stage1_growth, tier, exit_multiple, net_cash_ps, blend=0.50):
    r = 0.15
    s1 = tier["s1"]
    g2 = stage1_growth * tier["mult"]
    terminal_g = min(tier["gt"], stage1_growth * tier["mult"])

    earnings = []
    value = oe_ps

    for y in range(1, 16):
        value *= 1 + (stage1_growth if y <= s1 else g2)
        earnings.append(value)

    pv_explicit = sum(v / (1 + r) ** y for y, v in enumerate(earnings, start=1))

    if r <= terminal_g:
        dcf = float("nan")
    else:
        terminal_value = earnings[-1] * (1 + terminal_g) / (r - terminal_g)
        dcf = pv_explicit + terminal_value / (1 + r) ** 15

    multiple_model = pv_explicit + earnings[-1] * exit_multiple / (1 + r) ** 15

    if math.isnan(dcf):
        hybrid = multiple_model
    else:
        hybrid = blend * dcf + (1 - blend) * multiple_model

    hybrid += net_cash_ps

    return {
        "iv15": hybrid,
        "dcf_value": dcf + net_cash_ps if not math.isnan(dcf) else math.nan,
        "multiple_value": multiple_model + net_cash_ps,
        "year15_oe_ps": earnings[-1],
        "stage2_growth": g2,
        "terminal_growth": terminal_g,
    }


def return_target(iv15, target_return):
    # Price today that compounds to the same year-15 value at target_return.
    return iv15 * ((1.15 / (1.0 + target_return)) ** 15)


def verdict(price, iv15):
    if iv15 <= 0:
        return (
            "NO IV15",
            "Owners' earnings are negative or the model cannot support a positive "
            "15% hurdle price.",
            "error",
        )

    p_iv = price / iv15

    if p_iv <= 1.00:
        return (
            "FAT PITCH",
            "Price is at or below IV15 — the modeled entry price for a 15% "
            "annualized return over 15 years.",
            "success",
        )
    elif p_iv <= 1.50:
        return (
            "JUST OUTSIDE",
            "Above IV15, but close enough to merit monitoring.",
            "warning",
        )
    else:
        return (
            "OUT FIELD",
            "Price is materially above the modeled IV15.",
            "error",
        )


def show_metric(label, value, delta=None):
    st.metric(label, value, delta)


# -------------------------
# UI
# -------------------------

st.title("🎯 Burry IV15 Value Screener")
st.caption(
    "True Owners' Earnings + AICT moat framework + 15% hurdle rate. "
    "Search a ticker and get the price, IV15, targets and verdict."
)

ticker = st.text_input(
    "Enter stock ticker",
    value="",
    placeholder="e.g. ADBE, GOOGL, NFLX, ORCL, PAYC",
).upper().strip()

evaluate = st.button("Evaluate Stock", type="primary", use_container_width=True)

if evaluate and ticker:
    with st.spinner(f"Analyzing {ticker}..."):
        try:
            st.session_state["burry_data"] = fetch_company_data(ticker)
        except Exception as exc:
            st.session_state["burry_error"] = str(exc)

if evaluate and not ticker:
    st.warning("Enter a ticker first.")

if "burry_error" in st.session_state and evaluate:
    st.error(st.session_state["burry_error"])

if "burry_data" in st.session_state and st.session_state["burry_data"].get("ticker") == ticker:
    d = st.session_state["burry_data"]

    ref = BURRY_REFERENCE.get(ticker)
    default_tier = ref["tier"] if ref else "Chapel"

    st.markdown("---")
    st.subheader(f"📊 {d['company']} ({ticker})")

    # Main market / value panel
    c1, c2, c3 = st.columns(3)
    c1.metric("Market Price", f"${d['price']:.2f}")

    # Hidden advanced assumptions control the calculation only.
    with st.expander("Advanced assumptions", expanded=False):
        tier_name = st.selectbox(
            "AICT tier",
            list(AICT_TIERS.keys()),
            index=list(AICT_TIERS.keys()).index(default_tier),
            key=f"tier_{ticker}",
        )
        tier = AICT_TIERS[tier_name]

        default_exit = tier["default_exit"]
        exit_multiple = st.number_input(
            "Year-15 exit multiple",
            min_value=1.0,
            max_value=50.0,
            value=float(default_exit),
            step=0.5,
            key=f"exit_{ticker}",
        )

        default_growth = float(d["auto_growth"])
        growth = st.number_input(
            "Stage 1 growth",
            min_value=-0.05,
            max_value=0.40,
            value=float(round(default_growth, 3)),
            step=0.005,
            key=f"growth_{ticker}",
        )

        blend = st.slider(
            "DCF weight",
            min_value=0.0,
            max_value=1.0,
            value=0.50,
            step=0.05,
            key=f"blend_{ticker}",
        )

        use_burry_reference = False
        if ref:
            use_burry_reference = st.checkbox(
                "Use Burry's published IV15 for this ticker",
                value=True,
                key=f"reference_{ticker}",
                help="When checked, the source-published IV15 is used for the headline verdict. "
                     "The live reconstruction remains visible below.",
            )

    # Automatic calculation
    oe_ps = owners_earnings_ps(d)
    net_cash_ps = d["net_cash_m"] / d["shares_m"] if d["shares_m"] > 0 else 0.0
    model = calculate_iv15(
        oe_ps=oe_ps,
        stage1_growth=growth if "growth" in locals() else d["auto_growth"],
        tier=tier if "tier" in locals() else AICT_TIERS[default_tier],
        exit_multiple=exit_multiple if "exit_multiple" in locals() else AICT_TIERS[default_tier]["default_exit"],
        net_cash_ps=net_cash_ps,
        blend=blend if "blend" in locals() else 0.50,
    )

    reference_iv = ref["iv15"] if ref else None
    headline_iv = reference_iv if (ref and use_burry_reference) else model["iv15"]
    p_iv = d["price"] / headline_iv if headline_iv > 0 else math.nan

    c2.metric("IV15", f"${headline_iv:.2f}", f"P/IV15 {p_iv:.2f}×" if headline_iv > 0 else "n/a")
    c3.metric("Owners' Earnings", f"${d['owners_earnings_m']:,.0f}M")

    verdict_text, verdict_detail, kind = verdict(d["price"], headline_iv)

    st.markdown("---")
    if kind == "success":
        st.success(f"### 🎯 {verdict_text}\n\n{verdict_detail}")
    elif kind == "warning":
        st.warning(f"### ⚠️ {verdict_text}\n\n{verdict_detail}")
    else:
        st.error(f"### ⛔ {verdict_text}\n\n{verdict_detail}")

    # Targets
    iv10 = return_target(headline_iv, 0.10)
    iv12 = return_target(headline_iv, 0.12)
    iv18 = return_target(headline_iv, 0.18)
    iv20 = return_target(headline_iv, 0.20)

    st.subheader("🎯 Price Targets")
    t1, t2, t3, t4 = st.columns(4)
    t1.metric("20% return", f"${iv20:.2f}")
    t2.metric("18% return", f"${iv18:.2f}")
    t3.metric("15% return", f"${headline_iv:.2f}")
    t4.metric("12% return", f"${iv12:.2f}")

    st.caption(
        f"IV10 target: ${iv10:.2f}. These are entry prices implied by the same "
        "15-year terminal value under different annual return hurdles."
    )

    # Financial data
    st.markdown("---")
    st.subheader("⚙️ Financial Snapshot")
    f1, f2, f3 = st.columns(3)
    f1.metric("GAAP Net Income", f"${d['net_income_m']:,.0f}M")
    f2.metric("GAAP SBC", f"${d['gaap_sbc_m']:,.0f}M")
    f3.metric("Net Cash / (Debt)", f"${d['net_cash_m']:,.0f}M")

    f4, f5, f6 = st.columns(3)
    f4.metric("Buybacks", f"${d['buybacks_m']:,.0f}M")
    f5.metric("True SBC Cost Ω", f"${d['omega_m']:,.0f}M")
    delta_e_display = "n/a" if math.isnan(d["delta_e"]) else f"{d['delta_e']*100:.1f}%"
    f6.metric("ΔE", delta_e_display)

    st.subheader("🐹 SBC Diagnostic")
    s1, s2, s3 = st.columns(3)
    s1.metric(
        "SBC / Net Income",
        "n/a" if d["net_income_m"] == 0 else f"{d['gaap_sbc_m']/d['net_income_m']*100:.1f}%",
    )
    s2.metric(
        "Buyback Offset",
        f"{d['buyback_offset_pct']*100:.1f}%",
    )
    s3.metric(
        "Revenue Growth",
        f"{d['revenue_yoy']*100:.1f}%",
    )

    # Model detail, but collapsed
    with st.expander("How the IV15 was calculated"):
        st.write(
            f"AICT tier: **{default_tier if ref else (tier_name if 'tier_name' in locals() else 'Chapel')}**"
        )
        st.write(
            f"Owners' earnings/share: **${oe_ps:.2f}** | "
            f"Stage 1 growth: **{(growth if 'growth' in locals() else d['auto_growth'])*100:.1f}%** | "
            f"Stage 2 growth: **{model['stage2_growth']*100:.1f}%**"
        )
        st.write(
            f"DCF endpoint: **${model['dcf_value']:.2f}** | "
            f"Year-15 multiple endpoint: **${model['multiple_value']:.2f}** | "
            f"Live reconstructed hybrid: **${model['iv15']:.2f}**"
        )

        if ref:
            st.info(
                f"Published Burry reference: **${reference_iv:.2f} IV15** "
                f"({ref['source']}). "
                "The headline value can use this published figure, while the "
                "live reconstruction remains available as a diagnostic."
            )

        st.write(
            f"Data quality: **{d['data_quality']}**. "
            f"Fiscal year ended: **{d['latest_fy_end']}**. "
            f"Buyback price proxy: **{d['buy_px_method']}**."
        )

        if d["fallback_reason"]:
            st.warning(d["fallback_reason"])

    st.caption(
        "This screener is a reconstruction of the published framework, not Michael Burry's private spreadsheet. "
        "For tickers in the attached articles, published IV15 values are used when available so the screen can "
        "match the source rather than inventing precision."
    )

elif "burry_data" not in st.session_state:
    st.info("Enter a ticker above and click **Evaluate Stock**.")
