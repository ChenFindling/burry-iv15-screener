"""
Burry IV15 Screener
===================
Owners' earnings adjusted for the true cost of stock compensation, then the
intrinsic value ladder that follows from them.

THE KEY SIMPLIFICATION
----------------------
The published cost formula is  V = T x (W + dS) / W  , which needs W, the number
of shares repurchased. W is almost never tagged in XBRL — it lives in the share
repurchase footnote.

But P = T / W, so:

    V = T x (W + dS)/W  =  T + (T/W) x dS  =  T + P x dS

W cancels. Only the average share price is needed, and that is always
obtainable. Verified exact against all ten published Alphabet years.

So:
    V  = max(0, T + P x dS)      market value of shares handed to employees
    C  = Cw - Ce                 net cash award payments
    Om = C + V                   true SBC cost, replaces GAAP's estimate
    OE = N + G - Om              owners' earnings
    dE = OE / N                  fraction of reported profit that is really yours

Pooled over ~10 years as sum(OE)/sum(N) — never an average of annual ratios.

Run:  streamlit run app.py
"""

from __future__ import annotations

import datetime as dt
import statistics
from dataclasses import dataclass, field

import pandas as pd
import requests
import streamlit as st

# ══════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ══════════════════════════════════════════════════════════════════════

SEC_HEADERS = {
    # Put your own email here. The SEC blocks generic user agents.
    "User-Agent": "IV15 Research Tool contact@example.com",
    "Accept-Encoding": "gzip, deflate",
}


@dataclass(frozen=True)
class Tier:
    stage1_years: int
    stage2_years: int
    stage2_multiplier: float
    terminal_growth_cap: float
    debt_capacity_ebitda: float

    @property
    def horizon(self) -> int:
        return self.stage1_years + self.stage2_years

    @property
    def default_exit_multiple(self) -> float:
        """(1+g)/(r-g) at r=15%, i.e. the multiple the tier's own terminal
        growth cap already implies. Internally consistent rather than invented,
        and a far better starting point than one number for every tier — a
        Stone business in decline should never inherit a Fortress multiple."""
        return (1 + self.terminal_growth_cap) / (0.15 - self.terminal_growth_cap)


AICT: dict[str, Tier] = {
    "Fortress": Tier(8, 16, 0.70, 0.07, 3.0),
    "Castle":   Tier(7, 13, 0.55, 0.05, 2.5),
    "Chapel":   Tier(5, 10, 0.45, 0.04, 2.0),
    "Stone":    Tier(4,  7, 0.35, 0.03, 0.0),
    "Wood":     Tier(2,  4, 0.25, 0.00, 0.0),
}

TIER_BLURB = {
    "Fortress": "regulated or platform; owns its AI; no acute seat risk",
    "Castle":   "strong moat; owned AI at material scale; outcome fairly certain",
    "Chapel":   "acute AI threat but owned AI at decent scale plus switching costs",
    "Stone":    "meaningful threat without strong adaptability, or chronic pressure",
    "Wood":     "borrowed AI; no credible R&D; direct attack from foundation models",
}

VALUATION_BRACKETS = [(0.50, 35), (0.75, 32), (0.90, 28), (1.00, 24), (1.25, 20),
                      (1.50, 17), (2.00, 14), (3.00, 8), (5.00, 5), (10.0, 3)]

RUNG_MEANING = {8: "baseline intrinsic value, upper", 10: "baseline intrinsic value, lower",
                12: "a fair price", 15: "the benchmark buy target",
                18: "deep margin of safety", 20: "crisis pricing"}

# ══════════════════════════════════════════════════════════════════════
#  TRAGIC ALGEBRA
# ══════════════════════════════════════════════════════════════════════


@dataclass
class Year:
    """One fiscal year. Dollars in $M, shares in millions."""
    fy: int
    N: float                  # GAAP net income
    G: float = 0.0            # GAAP SBC expense
    T: float = 0.0            # buyback dollars
    dS: float = 0.0           # change in shares outstanding (+ = dilution)
    Cw: float = 0.0           # tax withheld on vesting
    Ce: float = 0.0           # option / ESPP proceeds
    price: float = 0.0        # average share price for the year
    cash_settled_sbc: bool = False   # MELI-style: no equity gap to close

    @property
    def C(self) -> float:
        return self.Cw - self.Ce

    @property
    def V(self) -> float:
        """Market value of shares delivered to employees.
        Floored at zero: you cannot deliver a negative number of shares."""
        if self.cash_settled_sbc:
            return 0.0
        return max(0.0, self.T + self.price * self.dS)

    @property
    def omega(self) -> float:
        return self.C + self.V

    @property
    def OE(self) -> float:
        return self.N + self.G - self.omega

    @property
    def dE(self) -> float | None:
        return self.OE / self.N if self.N else None


@dataclass
class Pooled:
    dE: float
    sum_N: float
    sum_OE: float
    sum_omega: float
    sum_G: float
    years: int

    @property
    def gaap_overstatement(self) -> float:
        return (self.sum_omega - self.sum_G) / self.sum_OE if self.sum_OE else float("nan")

    @property
    def street_overstatement(self) -> float:
        return self.sum_omega / self.sum_OE if self.sum_OE else float("nan")

    @property
    def tragic_tier(self) -> bool:
        return self.sum_OE < 0

    def retention(self, t: int) -> float:
        """Share of reported value growth that survives to year t. dE compounds."""
        return self.dE ** t

    def true_cagr(self, gaap_growth: float) -> float:
        """Break-even dE is 1/(1+g). Below it, reported growth never reaches you."""
        return self.dE * (1.0 + gaap_growth) - 1.0


def pool_recent(years: list[Year], n: int = 3) -> Pooled:
    """Pooled dE over just the last n years.

    The long window is the honest diagnostic, but where capital policy has
    changed the recent regime is what should feed a forward estimate.
    Salesforce is the clear case: 54.7% pooled over eleven years, 90.4% over
    the last three once buybacks overwhelmed issuance.
    """
    return pool(years[-n:])


def pool(years: list[Year]) -> Pooled:
    sN = sum(y.N for y in years)
    if not years or sN == 0:
        raise ValueError("Not enough data to pool.")
    return Pooled(
        dE=sum(y.OE for y in years) / sN,
        sum_N=sN, sum_OE=sum(y.OE for y in years),
        sum_omega=sum(y.omega for y in years), sum_G=sum(y.G for y in years),
        years=len(years),
    )


# ══════════════════════════════════════════════════════════════════════
#  INTRINSIC VALUE LADDER
# ══════════════════════════════════════════════════════════════════════


@dataclass
class IVParams:
    OE: float               # $M
    shares: float           # M
    tier: str
    growth: float           # decimal
    net_cash: float = 0.0   # $M
    exit_multiple: float = 20.0
    blend: float = 0.5      # weight on the perpetuity model
    stage0_years: int = 0
    stage0_growth: float = 0.0


def _stream(p: IVParams, n: int) -> list[float]:
    t = AICT[p.tier]
    g2 = p.growth * t.stage2_multiplier
    out, e = [], p.OE
    for y in range(1, n + 1):
        if y <= p.stage0_years:
            g = p.stage0_growth
        elif y <= p.stage0_years + t.stage1_years:
            g = p.growth
        else:
            g = g2
        e *= 1.0 + g
        out.append(e)
    return out


def intrinsic_value(p: IVParams, required_return_pct: float) -> float:
    """IV15 -> intrinsic_value(p, 15).

    Two models sharing one earnings stream, blended:
      model 1  stages 1 and 2, then a terminal perpetuity at the tier cap
      model 2  project to year 15, apply a market multiple

    Every rung is a full re-run at its own discount rate. Scaling one rung off
    another does not work — published IV12/IV15 ratios span 1.33 to 1.44.

    A negative result is meaningful: no share price delivers that return.
    """
    r = required_return_pct / 100.0
    t = AICT[p.tier]
    if r <= t.terminal_growth_cap or p.shares <= 0:
        return float("nan")

    n = t.horizon + p.stage0_years
    s = _stream(p, n)
    pv = sum(cf / (1 + r) ** y for y, cf in enumerate(s, 1))
    m1 = pv + s[-1] * (1 + t.terminal_growth_cap) / (r - t.terminal_growth_cap) / (1 + r) ** n

    s2 = _stream(p, 15)
    pv2 = sum(cf / (1 + r) ** y for y, cf in enumerate(s2, 1))
    m2 = pv2 + s2[-1] * p.exit_multiple / (1 + r) ** 15

    return (p.blend * m1 + (1 - p.blend) * m2 + p.net_cash) / p.shares


def ladder(p: IVParams) -> dict[int, float]:
    return {n: intrinsic_value(p, n) for n in (8, 10, 12, 15, 18, 20)}


def expected_return(price: float, p: IVParams) -> float:
    """IVB — the CAGR implied by today's price. Needs no required return chosen
    in advance, which arguably makes it the most useful single output."""
    lo, hi = AICT[p.tier].terminal_growth_cap + 1e-6, 3.0
    for _ in range(200):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if intrinsic_value(p, mid * 100) > price else (lo, mid)
    return (lo + hi) / 2


def valuation_points(ratio: float) -> int:
    if ratio < 0:
        return -2
    for ceiling, pts in VALUATION_BRACKETS:
        if ratio <= ceiling:
            return pts
    return -2


def zone(ratio: float) -> tuple[str, str]:
    if ratio < 0:
        return "Not investible", "error"
    if ratio <= 1.0:
        return "Fat Pitch", "success"
    if ratio <= 1.5:
        return "Just Outside", "info"
    return "Out Field", "error"


# ══════════════════════════════════════════════════════════════════════
#  DATA
# ══════════════════════════════════════════════════════════════════════

CONCEPTS = {
    "N":  (["NetIncomeLoss", "ProfitLoss"],
           ["ProfitLoss", "ProfitLossAttributableToOwnersOfParent"]),
    "G":  (["ShareBasedCompensation", "AllocatedShareBasedCompensationExpense"],
           ["ShareBasedPaymentsExpense"]),
    "T":  (["PaymentsForRepurchaseOfCommonStock", "PaymentsForRepurchaseOfEquity"],
           ["PaymentsToAcquireOrRedeemEntitysShares"]),
    "Cw": (["PaymentsRelatedToTaxWithholdingForShareBasedCompensation"], []),
    "Ce": (["ProceedsFromIssuanceOfSharesUnderIncentiveAndShareBasedCompensationPlans",
            "ProceedsFromStockOptionsExercised", "ProceedsFromIssuanceOfTreasuryStock",
            "ProceedsFromSaleOfTreasuryStock", "ProceedsFromStockPlans",
            "ProceedsFromEmployeeStockPurchasePlan", "ProceedsFromIssuanceOfCommonStock"], []),
    "REV": (["RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues",
             "RevenueFromContractWithCustomerIncludingAssessedTax"], ["Revenue"]),
    "SHD": (["WeightedAverageNumberOfDilutedSharesOutstanding",
             "WeightedAverageNumberOfSharesOutstandingDiluted"], []),
}

BALANCE = {
    "cash": ["CashAndCashEquivalentsAtCarryingValue",
             "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"],
    "sti":  ["ShortTermInvestments", "MarketableSecuritiesCurrent",
             "AvailableForSaleSecuritiesDebtSecuritiesCurrent"],
    "lti":  ["MarketableSecuritiesNoncurrent",
             "AvailableForSaleSecuritiesDebtSecuritiesNoncurrent"],
    "ltd":  ["LongTermDebtNoncurrent", "LongTermDebt"],
    "std":  ["LongTermDebtCurrent", "DebtCurrent", "ShortTermBorrowings", "CommercialPaper"],
}

ANNUAL_FORMS = ("10-K", "10-K/A", "20-F", "40-F")


@st.cache_data(ttl=86400, show_spinner=False)
def _ticker_map() -> dict[str, str]:
    r = requests.get("https://www.sec.gov/files/company_tickers.json",
                     headers=SEC_HEADERS, timeout=15)
    r.raise_for_status()
    return {e["ticker"].upper(): str(e["cik_str"]).zfill(10) for e in r.json().values()}


@st.cache_data(ttl=86400, show_spinner=False)
def _facts(cik: str) -> dict:
    r = requests.get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json",
                     headers=SEC_HEADERS, timeout=30)
    r.raise_for_status()
    return r.json()


def _annual(facts: dict, us: list[str], ifrs: list[str]) -> dict[int, tuple[str, str, float]]:
    """{fy: (start, end, value)} for full-year facts from annual reports only.

    Three filters that matter: the period must be roughly a year (so quarterly
    rows tagged fp='FY' cannot slip through); annual forms only; and where a
    year appears in several filings, keep the latest — a 10-K restates the
    prior year as a comparative.
    """
    out: dict[int, tuple[str, str, str, float]] = {}
    for taxonomy, concepts in (("us-gaap", us), ("ifrs-full", ifrs)):
        tax = facts.get("facts", {}).get(taxonomy, {})
        for concept in concepts:
            if concept not in tax:
                continue
            units = tax[concept].get("units", {})
            for row in units.get("USD", []) or units.get("shares", []):
                if row.get("form") not in ANNUAL_FORMS:
                    continue
                start, end = row.get("start"), row.get("end")
                if not (start and end):
                    continue
                if not 330 <= (dt.date.fromisoformat(end) - dt.date.fromisoformat(start)).days <= 400:
                    continue
                fy, filed = int(end[:4]), row.get("filed", "")
                if fy not in out or filed > out[fy][0]:
                    out[fy] = (filed, start, end, float(row.get("val", 0.0)))
            if out:
                return {k: (v[1], v[2], v[3]) for k, v in out.items()}
    return {}


def _instant(facts: dict, concepts: list[str], unit: str = "USD") -> dict[int, float]:
    out: dict[int, tuple[str, float]] = {}
    for taxonomy in ("us-gaap", "dei", "ifrs-full"):
        tax = facts.get("facts", {}).get(taxonomy, {})
        for concept in concepts:
            if concept not in tax:
                continue
            for row in tax[concept].get("units", {}).get(unit, []):
                if row.get("start") or not row.get("end"):
                    continue
                if row.get("form") not in ANNUAL_FORMS:
                    continue
                fy, filed = int(row["end"][:4]), row.get("filed", "")
                if fy not in out or filed > out[fy][0]:
                    out[fy] = (filed, float(row["val"]))
    return {k: v[1] for k, v in out.items()}


@st.cache_data(ttl=86400, show_spinner=False)
def _monthly_closes(ticker: str) -> dict[str, float]:
    """Monthly closes for ~11 years, keyed 'YYYY-MM'."""
    r = requests.get(
        f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
        "?interval=1mo&range=11y", headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
    res = r.json()["chart"]["result"][0]
    closes = res["indicators"]["quote"][0]["close"]
    out = {}
    for ts, c in zip(res["timestamp"], closes):
        if c:
            d = dt.datetime.utcfromtimestamp(ts)
            out[f"{d.year:04d}-{d.month:02d}"] = float(c)
    return out


def _avg_price(closes: dict[str, float], start: str, end: str) -> float | None:
    s, e = dt.date.fromisoformat(start), dt.date.fromisoformat(end)
    vals, d = [], s
    while d <= e:
        v = closes.get(f"{d.year:04d}-{d.month:02d}")
        if v:
            vals.append(v)
        d = (d.replace(day=1) + dt.timedelta(days=32)).replace(day=1)
    return statistics.fmean(vals) if vals else None


@st.cache_data(ttl=900, show_spinner=False)
def current_price(ticker: str) -> float | None:
    try:
        r = requests.get(
            f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?interval=1d&range=1d",
            headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        m = r.json()["chart"]["result"][0]["meta"]
        return float(m.get("regularMarketPrice") or m.get("chartPreviousClose"))
    except Exception:
        return None


def load(ticker: str, n_years: int = 10):
    cmap = _ticker_map()
    if ticker not in cmap:
        raise ValueError(f"'{ticker}' is not in the SEC company list.")
    facts = _facts(cmap[ticker])

    series = {k: _annual(facts, us, ifrs) for k, (us, ifrs) in CONCEPTS.items()}
    if not series["N"]:
        raise ValueError("No annual net income found — this filer uses a taxonomy "
                         "the app does not map. Try the manual tab.")

    shares_out = _instant(facts, ["CommonStockSharesOutstanding", "CommonStockSharesIssued",
                                  "EntityCommonStockSharesOutstanding"], unit="shares")
    try:
        closes = _monthly_closes(ticker)
    except Exception:
        closes = {}

    fys = sorted(series["N"])[-n_years:]
    notes: list[str] = []
    years: list[Year] = []

    for fy in fys:
        start, end, N = series["N"][fy]
        get = lambda k: abs(series[k][fy][2]) / 1e6 if fy in series[k] else 0.0

        dS = ((shares_out[fy] - shares_out[fy - 1]) / 1e6
              if fy in shares_out and fy - 1 in shares_out else 0.0)
        price = _avg_price(closes, start, end) or 0.0

        years.append(Year(fy=fy, N=N / 1e6, G=get("G"), T=get("T"), dS=dS,
                          Cw=get("Cw"), Ce=get("Ce"), price=price))

    if any(y.price == 0 for y in years):
        notes.append("No share price for some years — their SBC cost is understated.")
    if not any(y.Cw for y in years):
        notes.append("No tax-withholding line found. That understates the SBC cost, so "
                     "owners' earnings here are flattering rather than conservative.")

    g = lambda ks: (max(_instant(facts, ks).items(), default=(0, 0.0))[1]) / 1e6
    net_cash = g(BALANCE["cash"]) + g(BALANCE["sti"]) + g(BALANCE["lti"]) \
        - g(BALANCE["ltd"]) - g(BALANCE["std"])

    sh = series.get("SHD", {})
    diluted = (sh[fys[-1]][2] if fys[-1] in sh else shares_out.get(fys[-1], 0.0)) / 1e6

    rev = series.get("REV", {})
    ry = sorted(rev)
    growth = 0.08
    if len(ry) >= 4 and rev[ry[-4]][2] > 0 and rev[ry[-1]][2] > 0:
        growth = (rev[ry[-1]][2] / rev[ry[-4]][2]) ** (1 / 3) - 1

    return years, notes, {"net_cash": net_cash, "shares": diluted, "growth": growth}


# ══════════════════════════════════════════════════════════════════════
#  SELF-TEST
# ══════════════════════════════════════════════════════════════════════

def self_test() -> list[tuple[str, bool, str]]:
    out = []
    goog = [(2016, 19478, 6900, 3693, 3304, 97, 47), (2017, 12662, 7900, 4846, 4166, 78, 55),
            (2018, 30736, 10000, 9075, 4993, -2, 61), (2019, 34343, 11700, 18396, 4765, -158, 70),
            (2020, 40269, 12991, 31149, 5720, -263, 73), (2021, 76033, 15376, 50274, 10162, -264, 125),
            (2022, 59972, 19362, 59296, 9300, -412, 117), (2023, 73795, 22460, 61504, 9837, -374, 115),
            (2024, 100118, 22785, 62222, 12190, -243, 164), (2025, 132170, 24953, 45709, 14167, -93, 206)]
    ys = [Year(fy=f, N=n, G=g, T=t, Cw=c, dS=d, price=p) for f, n, g, t, c, d, p in goog]
    out.append(("Alphabet FY2016 V = $8,252M", abs(ys[0].V - 8252) < 1, f"${ys[0].V:,.0f}M"))
    out.append(("Alphabet FY2025 V = $26,551M", abs(ys[-1].V - 26551) < 1, f"${ys[-1].V:,.0f}M"))
    p = pool(ys)
    out.append(("Alphabet pooled ΔE = 88.7%", abs(p.dE - 0.887) < 0.002, f"{p.dE:.2%}"))

    m16 = Year(fy=2016, N=10217, G=3218, T=0, Cw=-10, dS=46, price=107)
    out.append(("Meta FY2016 ΔE = 83.4% (no buyback)", abs(m16.dE - 0.834) < 0.005, f"{m16.dE:.1%}"))

    N_, G_, OM_ = 4925.5, 919.0, 1732.2
    out.append(("NDX-97 GAAP overstatement = 19.78%",
                abs((OM_ - G_) / (N_ + G_ - OM_) - 0.1978) < 0.001,
                f"{(OM_-G_)/(N_+G_-OM_):.2%}"))
    out.append(("Break-even ΔE = 87%", abs(1 / 1.15 - 0.870) < 0.001, f"{1/1.15:.1%}"))

    crm = IVParams(OE=7300, shares=1073.3, tier="Chapel", growth=0.069,
                   exit_multiple=21.8, blend=1.0)
    out.append(("Salesforce IV15 ≈ $69.81", abs(intrinsic_value(crm, 15) - 69.81) < 1.0,
                f"${intrinsic_value(crm,15):.2f}"))
    out.append(("Salesforce IVB ≈ 8.6%", abs(expected_return(165.84, crm) - 0.086) < 0.005,
                f"{expected_return(165.84, crm):.1%}"))
    return out


# ══════════════════════════════════════════════════════════════════════
#  UI
# ══════════════════════════════════════════════════════════════════════

st.set_page_config(page_title="Burry IV15 Screener", layout="centered")

st.title("🎯 Burry IV15 Value Screener")
st.caption("Michael Burry's True Owners' Earnings (OE) & AICT Moat Valuation Engine")

with st.sidebar:
    st.subheader("Method self-test")
    st.caption("Checks the engine against Burry's own published figures.")
    if st.button("Run self-test"):
        for name, ok, got in self_test():
            st.write(("✅ " if ok else "❌ ") + f"{name} — {got}")
    st.divider()
    st.caption("IV15 is the price giving about 15% a year over 15+ years. It is a buy price "
               "target from a multi-stage cash flow model, not an earnings multiple. Real "
               "intrinsic value sits between IV8 and IV10 — below that, buybacks add value "
               "per share; above it they quietly destroy it.")

ticker = st.text_input("Enter Stock Ticker", value="",
                       placeholder="e.g. ADBE, NOW, CRM, GOOGL, PAYC").upper().strip()
tier_name = st.selectbox("Baseline AICT Moat Tier", list(AICT), index=2,
                         format_func=lambda t: f"{t} — {TIER_BLURB[t]}")

if st.button("Evaluate Stock", type="primary"):
    if not ticker:
        st.warning("Please enter a stock ticker symbol first.")
    else:
        try:
            with st.spinner(f"Pulling SEC EDGAR audited 10-K data for {ticker}..."):
                yrs, notes, pre = load(ticker, 10)
            st.session_state.update(years=yrs, notes=notes, pre=pre, tk=ticker)
        except Exception as e:
            st.error(f"Error fetching SEC data: {e}")

years = st.session_state.get("years", [])
if years and ticker and st.session_state.get("tk") == ticker:
    notes, pre, tk = st.session_state["notes"], st.session_state["pre"], st.session_state["tk"]
    pooled = pool(years)

    # ── 1. adjustments ──────────────────────────────────────────────
    st.markdown("---")
    st.subheader("⚙️ Audited SEC Financials & Adjustments")

    recent = pool_recent(years, 3) if len(years) >= 3 else pooled
    regime_gap = abs(recent.dE - pooled.dE)

    f1, f2, f3 = st.columns(3)
    fwd_N = f1.number_input("Forward GAAP Net Income ($M)",
                            value=float(round(years[-1].N, 1)), step=10.0,
                            help="Next year's expected GAAP net income — analyst consensus or "
                                 "your own estimate. Burry values on forward owners' earnings, "
                                 "not trailing.")
    which = f2.radio("Apply ΔE from",
                     [f"Last 3 years ({recent.dE:.1%})", f"Full period ({pooled.dE:.1%})"],
                     help="The long window is the diagnostic. Where capital policy has changed, "
                          "the recent regime is what will actually apply going forward.")
    use_dE = recent.dE if which.startswith("Last 3") else pooled.dE
    dE_usable = 0.0 < use_dE <= 1.25
    derived_OE = fwd_N * use_dE if dE_usable else float(round(years[-1].OE, 1))
    f3.metric("Implied Forward OE", f"${derived_OE:,.0f}M",
              f"{fwd_N:,.0f} x {use_dE:.1%}" if dE_usable else "ΔE unusable — set OE by hand")

    if not dE_usable:
        st.error(
            f"⛔ **ΔE of {use_dE:.1%} cannot be projected forward.** A negative or absurd ratio "
            "means stock compensation has swamped earnings over this window, so multiplying it "
            "by next year's profit is meaningless. Burry handles these by estimating recurring "
            "owners' earnings directly — DocuSign is his clearest example: ΔE of roughly −950%, "
            "yet he still assigns about $195M of forward owners' earnings on judgement. "
            "**Enter your own figure in Base Owners' Earnings below.** The other window may help."
        )

    if regime_gap > 0.15:
        st.warning(f"⚠️ **Regime change:** ΔE was {pooled.dE:.1%} over {pooled.years} years but "
                   f"{recent.dE:.1%} over the last three. Capital policy has shifted materially. "
                   "The recent figure is the fairer basis for a forward estimate — but satisfy "
                   "yourself the change is durable, not one good year.")

    a1, a2, a3 = st.columns(3)
    with a1:
        OE = st.number_input("Base Owners' Earnings ($M)",
                             value=float(round(derived_OE, 1)), step=1.0,
                             help="Seeded from forward net income x ΔE. Adjust further for "
                                  "maintenance capex, working capital and one-offs.")
        growth = st.number_input("Stage 1 Growth Rate (%)",
                                 value=round(pre["growth"] * 100, 1), step=0.5,
                                 help="Seeded from 3-year revenue growth. Return on capital "
                                      "is the ceiling — nothing outgrows it forever.") / 100
    with a2:
        shares = st.number_input("Diluted Shares (M)",
                                 value=float(round(pre["shares"], 1)), step=1.0)
        exit_m = st.number_input(
            "Exit Multiple (M15)", value=round(AICT[tier_name].default_exit_multiple, 2), step=0.5,
            help="Multiple applied to year-15 owners' earnings. Defaults to the multiple this "
                 "tier's own terminal growth cap implies at 15%. Burry has never published his, "
                 "so raise it for genuinely durable businesses and cut it for fading ones.")
    with a3:
        net_cash = st.number_input("Net Cash ($M)",
                                   value=float(round(pre["net_cash"], 1)), step=10.0)
        blend = st.slider("Model Blend (long-horizon weight)", 0.0, 1.0, 0.5, 0.05,
                          help="IV15 blends a long-horizon perpetuity model with an exit-multiple "
                               "model. The ratio depends on confidence in each and is not "
                               "published. This moves the answer a lot — on CRM, roughly $10.")

    for nte in notes:
        st.caption("ℹ️ " + nte)

    if net_cash and shares > 0:
        _p = current_price(tk) or 0.0
        if _p > 0 and abs(net_cash / (shares * _p)) > 0.08:
            st.info(f"💰 Net cash of ${net_cash:,.0f}M is {net_cash/(shares*_p):.0%} of market cap "
                    f"— about ${net_cash/shares:,.2f} per share of the IV15 below. Subtract only "
                    "what is freely deployable: restricted, regulated and operationally-tied "
                    "cash (customer float especially) funds the business and belongs in it.")

    # ── 2. tragic algebra ───────────────────────────────────────────
    st.markdown("---")
    st.subheader("🐹 Tragic Algebra Diagnostic")

    d1, d2, d3 = st.columns(3)
    _rec = pool_recent(years, 3) if len(years) >= 3 else pooled
    d1.metric("Owners' Earnings Kept (ΔE)", f"{pooled.dE:.1%}",
              f"{pooled.years}y pooled · last 3y: {_rec.dE:.1%}")
    d2.metric("True SBC Cost (Ω)", f"${pooled.sum_omega:,.0f}M",
              f"GAAP SBC: ${pooled.sum_G:,.0f}M")
    d3.metric("GAAP Overstates By", f"{pooled.gaap_overstatement:.1%}",
              f"Wall St: {pooled.street_overstatement:.1%}")

    if pooled.tragic_tier:
        st.error("⚠️ **Tragic Tier:** Over this period the cost of stock compensation exceeded "
                 "everything the business earned. Owners' earnings are negative, and not from one "
                 "bad year — shareholders were net funders of employee pay.")
    elif pooled.dE < 1 / 1.15:
        st.warning(f"⚠️ **Below the 87% break-even:** at ΔE of {pooled.dE:.1%}, even 15% reported "
                   f"growth compounds value per share at just {pooled.true_cagr(0.15):+.2%} a year. "
                   f"After ten years only {pooled.retention(10):.1%} of reported growth survives.")
    else:
        st.success(f"✅ **Above the 87% break-even** — reported growth actually reaches you. "
                   f"After ten years {pooled.retention(10):.1%} of it survives.")

    with st.expander("Year-by-year detail"):
        st.dataframe(pd.DataFrame([{
            "FY": y.fy, "Net Income": y.N, "GAAP SBC": y.G, "Buybacks": y.T,
            "Share Δ": y.dS, "Avg Price": y.price, "Ω": y.omega, "OE": y.OE, "ΔE": y.dE
        } for y in years]).style.format({
            "Net Income": "{:,.0f}", "GAAP SBC": "{:,.0f}", "Buybacks": "{:,.0f}",
            "Share Δ": "{:+,.1f}", "Avg Price": "${:,.2f}", "Ω": "{:,.0f}",
            "OE": "{:,.0f}", "ΔE": "{:.1%}"}, na_rep="—"),
            use_container_width=True, hide_index=True)

    # ── 3. valuation ────────────────────────────────────────────────
    if shares <= 0:
        st.error("Enter the diluted share count — everything below divides by it.")
        st.stop()

    par = IVParams(OE=OE, shares=shares, tier=tier_name, growth=growth,
                   net_cash=net_cash, exit_multiple=exit_m, blend=blend)
    lad = ladder(par)
    iv15 = lad[15]
    price = current_price(tk) or 0.0

    st.markdown("---")
    st.subheader("🧪 Scenario Stress-Testing Engine")
    s1, s2 = st.columns(2)
    keys = list(AICT)
    worse = s1.selectbox("Stress-Test Moat Downgrade", keys,
                         index=min(len(keys) - 1, keys.index(tier_name) + 1))
    cut = s2.slider("Growth Haircut (%)", 0, 80, 30, 5)
    spar = IVParams(OE=OE, shares=shares, tier=worse, growth=growth * (1 - cut / 100),
                    net_cash=net_cash, exit_multiple=exit_m, blend=blend)
    siv = intrinsic_value(spar, 15)

    st.markdown("---")
    st.subheader(f"📊 Valuation Verdict: {tk}")

    price = st.number_input("Market Price", value=float(price or 100.0), step=0.01)
    st.caption(f"Implied market cap ${shares * price / 1000:,.1f}B — if that is not roughly "
               "right, the share count is wrong and so is everything below.")

    r1, r2, r3 = st.columns(3)
    r1.metric("Market Price", f"${price:,.2f}")
    if iv15 == iv15 and iv15 > 0:
        r2.metric("Baseline IV15", f"${iv15:,.2f}", f"P/IV15: {price/iv15:.2f}x")
    else:
        r2.metric("Baseline IV15", "negative")
    if siv == siv and siv > 0:
        r3.metric("Stressed IV15", f"${siv:,.2f}", f"P/IV15: {price/siv:.2f}x")
    else:
        r3.metric("Stressed IV15", "negative")

    if iv15 != iv15:
        st.error("Required return must exceed the tier's terminal growth cap.")
    elif iv15 < 0:
        st.error(f"⛔ **NOT INVESTIBLE**: no share price — not even $0.01 — delivers 15% a year "
                 f"to a long-term shareholder in {tk} on these inputs.")
    else:
        ratio = price / iv15
        er = expected_return(price, par)
        if ratio <= 1.0:
            st.success(f"🎯 **FAT PITCH (BUY)**: at **${price:,.2f}**, {tk} trades below its IV15 "
                       f"of **${iv15:,.2f}**, implying about **{er:.1%}** a year held long term.")
        elif ratio <= 1.5:
            st.info(f"⚠️ **JUST OUTSIDE (WATCHLIST)**: at **${price:,.2f}**, {tk} is at "
                    f"{ratio:.2f}x its IV15 of **${iv15:,.2f}**. Expected return **{er:.1%}** "
                    f"a year.")
        else:
            st.error(f"⛔ **OUT FIELD (OVERVALUED)**: **${price:,.2f}** is well above IV15 of "
                     f"**${iv15:,.2f}**. Expected return just **{er:.1%}** a year.")

        st.write("**Target Entry Bands:**")
        for n in (20, 18, 15, 12, 10, 8):
            v = lad[n]
            if v == v and v > 0:
                st.write(f"- **{n}% Annual Return** ({RUNG_MEANING[n]}): buy under **${v:,.2f}**")
        st.caption("Each band is a separate run at its own required return — they cannot be "
                   "scaled off one another. Set price alerts at all of them.")

        with st.expander("Calibrate against a published IV15"):
            st.caption("If you have Burry's own IV15 for this stock, enter it and this solves "
                       "for the growth rate that reproduces it — given your current exit multiple "
                       "and blend. Useful for learning what assumptions his figures embed.")
            target = st.number_input("Published IV15 ($)", value=0.0, step=0.01)
            if target > 0:
                try:
                    from scipy.optimize import brentq
                    solved = brentq(
                        lambda g: intrinsic_value(
                            IVParams(OE=OE, shares=shares, tier=tier_name, growth=g,
                                     net_cash=net_cash, exit_multiple=exit_m,
                                     blend=blend), 15) - target, -0.30, 1.00)
                    st.success(f"A stage-1 growth rate of **{solved:.2%}** reproduces "
                               f"${target:,.2f} with your current settings "
                               f"(exit {exit_m:g}, blend {blend:g}).")
                except Exception:
                    st.error("No growth rate between -30% and +100% reaches that IV15. The exit "
                             "multiple, blend, owners' earnings or share count is likely off.")

        if siv == siv and siv > 0 and price <= siv:
            st.success("Still below IV15 after a moat downgrade and a growth haircut. That is "
                       "what a real margin of safety looks like.")
