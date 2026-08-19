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
    "User-Agent": "IV15 Research Tool chenfind@hotmail.com",
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

    traded_multiple: float = 14.5

    @property
    def perpetuity_equivalent(self) -> float:
        """(1+g)/(r-g) at r=15% — the multiple this tier's own terminal growth
        already implies. A useful floor, but too punitive as a default: Burry
        applies 'a multiple based on my experience with traded multiples' to
        year-15 earnings, and traded multiples sit well above perpetuity maths."""
        return (1 + self.terminal_growth_cap) / (0.15 - self.terminal_growth_cap)

    @property
    def default_exit_multiple(self) -> float:
        return self.traded_multiple


# Stage durations, multipliers, terminal caps and debt capacity are published.
# The traded exit multiple is NOT — these are calibrated so that the growth rate
# needed to reproduce a published IV15 matches the company's actual growth.
# Adobe is the anchor: at 14.5x, reaching his $262 needs 11.1% growth, and Adobe
# grew 11%. Treat them as reasonable starting points, not gospel.
AICT: dict[str, Tier] = {
    "Fortress": Tier(8, 16, 0.70, 0.07, 3.0, 20.0),
    "Castle":   Tier(7, 13, 0.55, 0.05, 2.5, 16.0),
    "Chapel":   Tier(5, 10, 0.45, 0.04, 2.0, 14.5),
    "Stone":    Tier(4,  7, 0.35, 0.03, 0.0,  9.0),
    "Wood":     Tier(2,  4, 0.25, 0.00, 0.0,  5.0),
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
    m2_style: str = "dcf"   # "dcf" = discount the stream, then a year-15 exit
                            # multiple. "hold" = buy, let it compound, sell in
                            # year 15 — no interim cash. See note below.


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

    # Two readings of the Buffett leg, and the published figures do not settle
    # which is right:
    #   "dcf"  — a normal DCF that finishes with a market multiple instead of a
    #            perpetuity. Fits Salesforce, Adobe, Paycom at blends of 0.5-1.
    #   "hold" — buy the business, let it reinvest, sell in year 15. No interim
    #            cash reaches you. Only this reading reaches Paylocity's
    #            published IV15, but it makes the blend a ~3x lever.
    s2 = _stream(p, 15)
    if p.m2_style == "hold":
        m2 = s2[-1] * p.exit_multiple / (1 + r) ** 15
    else:
        pv2 = sum(cf / (1 + r) ** y for y, cf in enumerate(s2, 1))
        m2 = pv2 + s2[-1] * p.exit_multiple / (1 + r) ** 15

    return (p.blend * m1 + (1 - p.blend) * m2 + p.net_cash) / p.shares


def model_legs(p: IVParams, required_return_pct: float = 15.0) -> tuple[float, float]:
    """Per-share value from each leg, so the blend's effect is visible rather
    than buried. A wide spread means the blend choice is doing a lot of work."""
    a = IVParams(**{**p.__dict__, "blend": 1.0})
    b = IVParams(**{**p.__dict__, "blend": 0.0})
    return (intrinsic_value(a, required_return_pct),
            intrinsic_value(b, required_return_pct))


def ladder(p: IVParams) -> dict[int, float]:
    return {n: intrinsic_value(p, n) for n in (8, 10, 12, 15, 18, 20)}


def expected_return(price: float, p: IVParams) -> float:
    """IVB — the CAGR implied by today's price. Needs no required return chosen
    in advance, which arguably makes it the most useful single output."""
    lo, hi = AICT[p.tier].terminal_growth_cap + 1e-6, 3.0
    for _ in range(200):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if intrinsic_value(p, mid * 100) > price else (lo, mid)
    out = (lo + hi) / 2
    # Saturating at the ceiling is not a 300% forecast, it means the inputs are
    # wrong — nearly always a bad share count.
    return float("inf") if out > 2.5 else out


def solve_growth(target_iv15: float, p: IVParams,
                 lo: float = -0.30, hi: float = 1.00) -> float | None:
    """Growth rate reproducing a given IV15, by bisection.

    Intrinsic value rises monotonically with growth, so bisection is exact
    enough and avoids a scipy dependency for one root-find.
    """
    f = lambda g: intrinsic_value(IVParams(**{**p.__dict__, "growth": g}), 15) - target_iv15
    flo, fhi = f(lo), f(hi)
    if flo != flo or fhi != fhi or flo * fhi > 0:
        return None
    for _ in range(200):
        mid = (lo + hi) / 2
        if f(lo) * f(mid) <= 0:
            hi = mid
        else:
            lo = mid
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
    # Shares issued for reasons that are NOT compensation. The extraction
    # protocol excludes these from dS explicitly: M&A issuance, public
    # offerings and debt-to-equity conversions are corporate transactions,
    # not pay. Salesforce issued heavily for Slack, Tableau, MuleSoft and
    # Informatica; charging those to employees makes dE far too negative.
    "MA":   (["StockIssuedDuringPeriodSharesAcquisitions"], []),
    "OFFER": (["StockIssuedDuringPeriodSharesNewIssues"], []),
    "CONV": (["StockIssuedDuringPeriodSharesConversionOfConvertibleSecurities",
              "StockIssuedDuringPeriodSharesConversionOfUnits"], []),
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
def _sic(cik: str) -> tuple[str, str]:
    """SIC code and description, for sector-specific guards."""
    try:
        r = requests.get(f"https://data.sec.gov/submissions/CIK{cik}.json",
                         headers=SEC_HEADERS, timeout=20)
        j = r.json()
        return str(j.get("sic", "")), str(j.get("sicDescription", ""))
    except Exception:
        return "", ""


def is_financial(sic: str) -> bool:
    """SIC 6000-6799: banks, insurers, brokers, REITs. For these, investments
    back policyholder or depositor liabilities and are not shareholder cash, so
    a balance-sheet 'net cash' figure is meaningless and hugely overstated."""
    return sic.isdigit() and 6000 <= int(sic) <= 6799


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


def split_adjust(shares: dict[int, float]) -> tuple[dict[int, float], list[str]]:
    """Restate historical share counts onto the current basis.

    Gate 3 of the published QA protocol: year-on-year share counts must stay
    within [0.35, 2.85]. Outside that band is a stock split, not real dilution.

    This matters enormously. XBRL reports shares as-filed, so pre-split years
    carry the old basis, while market prices are already split-adjusted. Mixing
    the two makes dS jump by the whole split in one year, and since
    V = T + P x dS, the SBC cost explodes. ServiceNow's 5-for-1 turned a
    pooled dE of about -79% into -2391%.
    """
    fys, notes = sorted(shares), []
    if len(fys) < 2:
        return dict(shares), notes
    adjusted, factor = {}, 1.0
    for i in range(len(fys) - 1, -1, -1):
        fy = fys[i]
        adjusted[fy] = shares[fy] * factor
        if i > 0 and shares[fys[i - 1]] > 0:
            # Raw-to-raw. Comparing the ADJUSTED current year against the raw
            # prior year re-detects the same split on every pass and compounds
            # the factor geometrically.
            ratio = shares[fy] / shares[fys[i - 1]]
            if ratio > 2.85 or ratio < 0.35:
                # Round to a plausible split ratio. Reverse splits must be
                # rounded on the reciprocal: round(0.1 * 2) / 2 is zero.
                if ratio >= 1:
                    clean = round(ratio * 2) / 2
                    label = f"{clean:g}:1"
                else:
                    inv = round((1 / ratio) * 2) / 2
                    clean = 1 / inv if inv > 0 else 0.0
                    label = f"1:{inv:g}"
                if clean > 0:
                    factor *= clean
                    notes.append(f"Stock split detected in FY{fy} (about {label}). Earlier "
                                 "share counts restated onto the current basis — without this "
                                 "the SBC cost would be wildly overstated.")
    return adjusted, notes


def load(ticker: str, n_years: int = 10):
    cmap = _ticker_map()
    if ticker not in cmap:
        raise ValueError(f"'{ticker}' is not in the SEC company list.")
    facts = _facts(cmap[ticker])
    sic, sic_desc = _sic(cmap[ticker])

    series = {k: _annual(facts, us, ifrs) for k, (us, ifrs) in CONCEPTS.items()}
    if not series["N"]:
        raise ValueError("No annual net income found — this filer uses a taxonomy "
                         "the app does not map. Try the manual tab.")

    shares_out = _instant(facts, ["CommonStockSharesOutstanding", "CommonStockSharesIssued",
                                  "EntityCommonStockSharesOutstanding"], unit="shares")
    shares_out, split_notes = split_adjust(shares_out)
    try:
        closes = _monthly_closes(ticker)
    except Exception:
        closes = {}

    fys = sorted(series["N"])[-n_years:]
    notes: list[str] = list(split_notes)
    non_sbc_total = 0.0
    years: list[Year] = []

    for fy in fys:
        start, end, N = series["N"][fy]
        get = lambda k: abs(series[k][fy][2]) / 1e6 if fy in series[k] else 0.0

        dS = ((shares_out[fy] - shares_out[fy - 1]) / 1e6
              if fy in shares_out and fy - 1 in shares_out else 0.0)
        non_sbc = sum(abs(series[k][fy][2]) / 1e6
                      for k in ("MA", "OFFER", "CONV") if fy in series.get(k, {}))
        if non_sbc:
            dS -= non_sbc
            non_sbc_total += non_sbc
        price = _avg_price(closes, start, end) or 0.0

        years.append(Year(fy=fy, N=N / 1e6, G=get("G"), T=get("T"), dS=dS,
                          Cw=get("Cw"), Ce=get("Ce"), price=price))

    if non_sbc_total:
        notes.append(f"Excluded {non_sbc_total:,.1f}M shares issued for acquisitions, offerings "
                     "or conversions — those are corporate transactions, not compensation. "
                     "Where a company issues stock for deals this matters a great deal.")
    if any(y.price == 0 for y in years):
        notes.append("No share price for some years — their SBC cost is understated.")
    if not any(y.Cw for y in years):
        notes.append("No tax-withholding line found. That understates the SBC cost, so "
                     "owners' earnings here are flattering rather than conservative.")

    g = lambda ks: (max(_instant(facts, ks).items(), default=(0, 0.0))[1]) / 1e6
    cash_total = g(BALANCE["cash"]) + g(BALANCE["sti"]) + g(BALANCE["lti"])
    debt_total = g(BALANCE["ltd"]) + g(BALANCE["std"])
    net_cash = cash_total - debt_total
    if is_financial(sic):
        notes.append(f"{sic_desc or 'Financial company'} (SIC {sic}). Investments here back "
                     "policyholder or depositor liabilities rather than belonging to "
                     "shareholders, so net cash has been set to zero. The Tragic Algebra still "
                     "works, but treat the valuation as indicative — this framework was built "
                     "for software, and insurers, banks and REITs need book-value and "
                     "combined-ratio thinking it does not contain.")
        cash_total = debt_total = net_cash = 0.0

    # Most recent shares OUTSTANDING beats trailing weighted-average diluted.
    # Under a heavy buyback the weighted average is stale and systematically
    # high, which depresses every per-share figure. Adobe: 427M weighted vs
    # ~408M actual, a 4.7% error straight through to IV15.
    # Shares outstanding is preferred (buybacks make the trailing weighted
    # average stale), BUT under a dual-class structure the outstanding count is
    # tagged per class and we may be seeing only one of them. Weighted-average
    # diluted is reported consolidated, so when the two diverge by more than a
    # buyback could explain, trust the diluted figure.
    sh = series.get("SHD", {})
    outstanding = shares_out[max(shares_out)] / 1e6 if shares_out else 0.0
    wavg = sh[max(sh)][2] / 1e6 if sh else 0.0
    diluted = outstanding or wavg
    if outstanding > 0 and wavg > 0:
        if outstanding / wavg < 0.65:
            diluted = wavg
            notes.append(f"Shares outstanding read as {outstanding:,.1f}M but weighted-average "
                         f"diluted is {wavg:,.1f}M — too big a gap for buybacks. This usually "
                         "means multiple share classes and only one was picked up. Using the "
                         "diluted figure; check it against the market cap below.")
        elif abs(outstanding / wavg - 1) > 0.03:
            notes.append(f"Shares outstanding {outstanding:,.1f}M vs weighted-average diluted "
                         f"{wavg:,.1f}M. Using the current count; buybacks make the average stale.")

    rev = series.get("REV", {})
    ry = sorted(rev)
    growth = 0.08
    if len(ry) >= 4 and rev[ry[-4]][2] > 0 and rev[ry[-1]][2] > 0:
        growth = (rev[ry[-1]][2] / rev[ry[-4]][2]) ** (1 / 3) - 1

    return years, notes, {"net_cash": net_cash, "cash": cash_total, "debt": debt_total,
                          "shares": diluted, "growth": growth, "sic": sic,
                          "sic_desc": sic_desc, "financial": is_financial(sic)}


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
#
# NOTE ON DOLLAR SIGNS: Streamlit markdown parses $...$ as LaTeX. Any literal
# dollar amount inside st.write/markdown/success/error/info/warning must be
# escaped as \$ or the text between two of them silently becomes an equation.
# st.metric, st.code and st.dataframe are unaffected.

st.set_page_config(page_title="Burry IV15 Screener", layout="centered")


def d(x, dp=2):
    """Escaped dollar amount, safe inside markdown."""
    return f"\\${x:,.{dp}f}"


st.title("🎯 Burry IV15 Value Screener")
st.caption("True owners' earnings after stock compensation, then the price ladder that follows")

with st.sidebar:
    st.subheader("Self-test")
    st.caption("Checks the engine against Burry's published figures.")
    if st.button("Run"):
        for name, ok, got in self_test():
            st.write(("✅ " if ok else "❌ ") + f"{name} — {got}")
    st.divider()
    with st.expander("What the numbers mean"):
        st.markdown(
            "**ΔE** — the share of each reported dollar of profit that actually reaches "
            "shareholders once the true cost of stock compensation is charged. Below about "
            "87%, a company needs 15% reported growth just to hold value per share steady.\n\n"
            "**IV15** — the price at which the stock would return roughly 15% a year over "
            "15+ years. A buy target from a cash flow model, not an earnings multiple.\n\n"
            "**IV8 to IV10** — closer to what the business is actually worth. Buybacks below "
            "that range add value per share; above it they destroy it.\n\n"
            "**Expected return** — what today's price implies you'd earn annually, held long "
            "term. The most useful single figure, since it needs no target return chosen "
            "in advance.\n\n"
            "**Moat tier** — sets how long growth lasts and how fast it fades, not the "
            "starting rate. Fortress holds growth 8 years; Wood gets 2."
        )
    st.divider()
    st.caption(
        "Research aid, not financial advice. Outputs depend on estimates you supply — change "
        "the growth rate and the answer changes a lot. Method follows Michael Burry's published "
        "writing; this tool is not affiliated with or endorsed by him."
    )

if "years" not in st.session_state:
    st.info(
        "**Reported profit is not what reaches you.** Shares handed to employees cost real "
        "money the income statement never shows. This works out what is left, then the price "
        "at which the stock would return about 15% a year over the long run.\n\n"
        "Enter a US-listed ticker to start. Built for software and other operating companies — "
        "banks, insurers and REITs need tools this one does not contain."
    )

# A form submits on Enter as well as on the button click.
with st.form("lookup"):
    ticker = st.text_input("Stock ticker",
                           placeholder="ADBE · CRM · NOW · GOOGL — press Enter").upper().strip()
    submitted = st.form_submit_button("Evaluate", type="primary")

tier_name = st.selectbox("Moat tier", list(AICT), index=2,
                         format_func=lambda t: f"{t} — {TIER_BLURB[t]}",
                         help="Sets stage lengths, how far growth fades in stage 2, the terminal "
                              "cap and the exit multiple. It does NOT set your stage 1 growth "
                              "rate — that is company-specific and yours to judge.")

if submitted:
    if not ticker:
        st.warning("Enter a ticker first.")
    else:
        try:
            with st.spinner(f"Reading {ticker} annual filings…"):
                yrs, notes, pre = load(ticker, 10)
            st.session_state.update(years=yrs, notes=notes, pre=pre, tk=ticker)
        except Exception as e:
            st.error(f"Could not load {ticker}: {e}")

years = st.session_state.get("years", [])
if years and ticker and st.session_state.get("tk") == ticker:
    notes, pre, tk = st.session_state["notes"], st.session_state["pre"], st.session_state["tk"]
    pooled = pool(years)
    recent = pool_recent(years, 3) if len(years) >= 3 else pooled
    alerts: list[tuple[str, str]] = [("info", n) for n in notes]

    # ══ inputs ═══════════════════════════════════════════════════════
    st.markdown("---")
    st.subheader("Inputs")

    use_recent = st.radio(
        "Apply ΔE from", ["Last 3 years", "Full period"], horizontal=True,
        captions=[f"{recent.dE:.1%}", f"{pooled.dE:.1%}"],
        help="ΔE is the share of reported profit that actually reaches shareholders. The long "
             "window is the diagnostic; where capital policy has changed, the recent one is "
             "what will apply going forward.") == "Last 3 years"
    use_dE = recent.dE if use_recent else pooled.dE
    dE_ok = 0.0 < use_dE <= 1.25

    hist = sorted(y.OE for y in years[-5:])
    median_OE = hist[len(hist) // 2] if hist else 0.0

    c1, c2, c3 = st.columns(3)
    fwd_N = c1.number_input("Forward net income ($M)", value=float(round(years[-1].N, 1)), step=10.0,
                            help="Next year's expected GAAP net income.")
    if dE_ok:
        derived = fwd_N * use_dE
    elif median_OE > 0:
        derived = median_OE
    else:
        # Every recent year is negative. Seeding zero makes IV15 collapse to
        # net cash per share, which looks like an answer but is not one.
        # Forward net income is at least a defensible ceiling to revise down from.
        derived = fwd_N
    OE = c1.number_input("Owners' earnings ($M)", value=float(round(derived, 1)), step=1.0,
                         help="Seeded from forward net income x ΔE. Adjust for maintenance "
                              "capex, working capital and anything non-recurring.")
    shares = c2.number_input("Diluted shares (M)", value=float(round(pre["shares"], 1)), step=1.0)
    growth = c2.number_input("Growth rate (%)", value=round(pre["growth"] * 100, 1), step=0.5,
                             help="Return on capital is the ceiling — nothing outgrows it "
                                  "forever.") / 100
    price = c3.number_input("Price", value=float(current_price(tk) or 100.0), step=0.01)
    cash = c3.number_input("Cash & investments ($M)", value=float(round(pre.get("cash", 0.0), 1)),
                           step=10.0, help="Only what is freely deployable. Restricted, regulated "
                                           "and operationally-tied cash funds the business.")
    debt = c2.number_input("Total debt ($M)", value=float(round(pre.get("debt", 0.0), 1)),
                           step=10.0, help="Short-term plus long-term borrowings. Subtracted from "
                                           "cash to give the net figure added to intrinsic value.")
    net_cash = cash - debt
    c1.caption(f"Net cash {d(net_cash,0)}M  ·  {d(cash,0)}M cash less {d(debt,0)}M debt")

    with st.expander("Model settings"):
        m1, m2 = st.columns(2)
        exit_m = m1.number_input(
            "Exit multiple", value=round(AICT[tier_name].default_exit_multiple, 2), step=0.5,
            help=f"Applied to year-15 owners' earnings. Burry never published his; this default "
                 f"is calibrated against Adobe. This tier's perpetuity floor is "
                 f"{AICT[tier_name].perpetuity_equivalent:.1f}x.")
        m2_style = m1.radio(
            "Exit-multiple leg", ["dcf", "hold"], horizontal=True,
            format_func=lambda v: "Cash flows + exit" if v == "dcf" else "Buy and hold to year 15",
            help="Two readings of Burry's Buffett leg. 'Cash flows + exit' discounts the earnings "
                 "stream then adds a year-15 multiple, and fits Salesforce, Adobe and Paycom. "
                 "'Buy and hold' counts only the year-15 sale — the only reading that reaches "
                 "Paylocity's published figure, but it makes the blend swing results about 3x.")
        blend = m2.slider("Long-horizon weight", 0.0, 1.0, 0.5, 0.05,
                          help="IV15 blends a perpetuity model with an exit-multiple model. "
                               "Moves the answer materially — about \\$10 on CRM.")
        t = AICT[tier_name]
        st.caption(f"{tier_name}: stage 1 {t.stage1_years}y, stage 2 {t.stage2_years}y at "
                   f"{t.stage2_multiplier:.2f}x, terminal cap {t.terminal_growth_cap:.0%}, "
                   f"total horizon {t.horizon} years.")
        _l1, _l2 = model_legs(IVParams(OE=OE, shares=shares, tier=tier_name, growth=growth,
                                       net_cash=net_cash, exit_multiple=exit_m, blend=blend,
                                       m2_style=m2_style))
        if _l1 == _l1 and _l2 == _l2:
            st.caption(f"Long-horizon leg ${_l1:,.2f} · exit-multiple leg ${_l2:,.2f}. "
                       + ("They agree closely, so the blend barely matters here."
                          if abs(_l1 - _l2) / max(_l1, 1) < 0.1 else
                          "They diverge, so the blend is doing real work — worth a look."))

    if not dE_ok and median_OE <= 0:
        st.error(
            f"**Set owners' earnings yourself.** ΔE of {use_dE:.1%} cannot be projected, and "
            "every recent year is negative too, so the field below is seeded with forward net "
            f"income of {d(fwd_N,0)}M as a ceiling — it is certainly too high. Burry does this "
            "by hand: ServiceNow gets about 620M against reported profit near 1,750M, adjusted "
            "upward from a negative ΔE for its dilution-neutral pledge and its sub-10%-of-revenue "
            "SBC target.")

    if not dE_ok:
        alerts.append(("error",
            f"ΔE of {use_dE:.1%} cannot be projected forward — stock compensation has swamped "
            "earnings over this window. Set owners' earnings by hand. Burry does exactly this "
            "for DocuSign: ΔE deeply negative, yet about 195M of forward owners' earnings on "
            "judgement, worked down from free cash flow."))
    elif median_OE > 0 and derived > 2 * median_OE:
        alerts.append(("warning",
            f"Derived owners' earnings of {d(derived,0)}M are {derived/median_OE:.1f}x the "
            f"{d(median_OE,0)}M median of the last five years. Forward profit may carry a "
            "one-off. Check the yearly table and override."))
    if abs(recent.dE - pooled.dE) > 0.15:
        alerts.append(("warning",
            f"Regime change: ΔE was {pooled.dE:.1%} over {pooled.years} years but "
            f"{recent.dE:.1%} over the last three. Satisfy yourself the shift is durable."))
    if shares > 0 and price > 0 and abs(net_cash / (shares * price)) > 0.08:
        alerts.append(("info",
            f"Net cash is {net_cash/(shares*price):.0%} of market cap — about "
            f"{d(net_cash/shares)} per share of the IV15 below."))

    if shares <= 0:
        st.error("Enter the diluted share count — everything divides by it.")
        st.stop()

    mcap = shares * price / 1000.0
    if net_cash > 0 and price > 0 and net_cash / (shares * price) > 0.60:
        st.error(
            f"**Check the share count before trusting anything below.** Net cash of "
            f"{d(net_cash,0)}M is {net_cash/(shares*price):.0%} of a {d(mcap,2)}B market cap, "
            "which almost never happens. The usual cause is a company with more than one share "
            "class where only one was picked up. Look up the real share count and type it in.")
    elif mcap < 0.05:
        st.warning(f"Implied market cap is only {d(mcap,2)}B. If that looks too small, the share "
                   "count is likely wrong — everything scales inversely with it.")

    par = IVParams(OE=OE, shares=shares, tier=tier_name, growth=growth,
                   net_cash=net_cash, exit_multiple=exit_m, blend=blend,
                   m2_style=m2_style)
    lad = ladder(par)
    iv15 = lad[15]

    # ══ verdict ══════════════════════════════════════════════════════
    st.markdown("---")
    st.subheader(f"Verdict · {tk}")

    if iv15 != iv15:
        st.error("Required return must exceed the tier's terminal growth cap.")
        st.stop()
    if iv15 < 0:
        st.error(f"**Not investible.** No share price — not even one cent — delivers 15% a year "
                 f"to a long-term shareholder in {tk} on these inputs.")
        st.stop()

    ratio = price / iv15
    er = expected_return(price, par)
    zn, kind = zone(ratio)
    er_txt = "implausible" if er == float("inf") else f"{er:.1%}"

    v1, v2, v3 = st.columns(3)
    v1.metric("IV15", f"${iv15:,.2f}", f"market ${price:,.2f}")
    v2.metric("Price / IV15", f"{ratio:.2f}x", zn)
    v3.metric("Expected return", er_txt, f"score {valuation_points(ratio)}/35")
    if er == float("inf"):
        st.error("Expected return came out beyond any believable range, which means an input is "
                 "wrong rather than that a bargain has been found. Check the share count first.")

    verdict = {
        "success": f"**Fat pitch.** {tk} trades below its IV15 of {d(iv15)}, implying about "
                   f"{er_txt} a year held long term.",
        "info":    f"**Just outside.** {tk} is at {ratio:.2f}x its IV15 of {d(iv15)} — a "
                   f"watchlist candidate at about {er_txt} a year.",
        "error":   f"**Out field.** At {ratio:.2f}x its IV15 of {d(iv15)}, {tk} offers only "
                   f"about {er_txt} a year.",
    }[kind]
    getattr(st, kind)(verdict)

    st.write("**Entry bands** — set alerts at each")
    st.dataframe(
        pd.DataFrame([{"Target return": f"{n}%", "Buy under": v, "Meaning": RUNG_MEANING[n]}
                      for n, v in lad.items() if v == v and v > 0][::-1])
        .style.format({"Buy under": "${:,.2f}"}),
        use_container_width=True, hide_index=True)

    # ══ quality ══════════════════════════════════════════════════════
    st.markdown("---")
    st.subheader("Shareholder quality")

    q1, q2, q3 = st.columns(3)
    q1.metric("Owners' earnings kept", f"{pooled.dE:.1%}" if abs(pooled.dE) < 10 else "deeply negative",
              f"last 3y: {recent.dE:.1%}")
    q2.metric("True SBC cost", f"${pooled.sum_omega:,.0f}M", f"GAAP says ${pooled.sum_G:,.0f}M")
    q3.metric("Value kept after 10y", f"{pooled.retention(10):.1%}" if 0 < pooled.dE <= 1.25 else "—",
              "of reported growth")

    if pooled.tragic_tier:
        st.error("**Tragic tier.** Stock compensation cost more than the business earned over "
                 "this period. Shareholders were net funders of employee pay.")
    elif pooled.dE < 1 / 1.15:
        st.warning(f"**Below the 87% break-even.** Even 15% reported growth compounds value per "
                   f"share at just {pooled.true_cagr(0.15):+.2%} a year.")
    else:
        st.success("**Above the 87% break-even** — reported growth actually reaches you.")

    # ══ stress ═══════════════════════════════════════════════════════
    st.markdown("---")
    st.subheader("Stress test")

    s1, s2 = st.columns(2)
    keys = list(AICT)
    worse = s1.selectbox("Downgrade tier to", keys,
                         index=min(len(keys) - 1, keys.index(tier_name) + 1))
    cut = s2.slider("Cut growth by (%)", 0, 80, 30, 5)
    siv = intrinsic_value(IVParams(OE=OE, shares=shares, tier=worse,
                                   growth=growth * (1 - cut / 100), net_cash=net_cash,
                                   exit_multiple=exit_m, blend=blend), 15)
    if siv == siv and siv > 0:
        t1, t2 = st.columns(2)
        t1.metric("Stressed IV15", f"${siv:,.2f}", f"{siv/iv15-1:+.1%}")
        t2.metric("Stressed P/IV15", f"{price/siv:.2f}x", zone(price / siv)[0])
        if price <= siv:
            st.success("Still below IV15 after a downgrade and a growth cut. That is a real "
                       "margin of safety.")
    else:
        st.error("Not investible under stressed assumptions.")

    # ══ detail ═══════════════════════════════════════════════════════
    st.markdown("---")
    label = f"Notes and detail" + (f" · {len(alerts)} to review" if alerts else "")
    with st.expander(label):
        for kind_, msg in alerts:
            getattr(st, kind_)(msg)

        st.write("**Year by year**")
        st.dataframe(pd.DataFrame([{
            "FY": y.fy, "Net income": y.N, "GAAP SBC": y.G, "Buybacks": y.T,
            "Share change": y.dS, "Avg price": y.price, "True SBC cost": y.omega,
            "Owners' earnings": y.OE, "ΔE": y.dE} for y in years]).style.format({
                "Net income": "{:,.0f}", "GAAP SBC": "{:,.0f}", "Buybacks": "{:,.0f}",
                "Share change": "{:+,.1f}", "Avg price": "${:,.2f}", "True SBC cost": "{:,.0f}",
                "Owners' earnings": "{:,.0f}", "ΔE": "{:.1%}"}, na_rep="—"),
            use_container_width=True, hide_index=True)

        st.write("**Assumptions used** — paste this if something looks wrong")
        st.code(
            f"{tk}   price {price:,.2f}   shares {shares:,.1f}M   "
            f"mkt cap ${shares*price/1000:,.2f}B\n"
            f"forward net income  {fwd_N:,.0f}\n"
            f"ΔE applied          {use_dE:.1%}   (full {pooled.dE:.1%} / 3y {recent.dE:.1%})\n"
            f"median OE, 5y       {median_OE:,.0f}\n"
            f"owners' earnings    {OE:,.0f}   ({OE/shares:,.2f}/share)\n"
            f"net cash            {net_cash:,.0f}   ({net_cash/shares:,.2f}/share)\n"
            f"tier                {tier_name}   growth {growth:.2%}\n"
            f"exit multiple       {exit_m:g}x   blend {blend:g}   leg {m2_style}\n"
            f"IV15                {iv15:,.2f}   P/IV15 {ratio:.2f}x", language="text")

        st.write("**Calibrate against a published IV15**")
        target = st.number_input("Published IV15", value=0.0, step=0.01,
                                 label_visibility="collapsed")
        if target > 0:
            solved = solve_growth(target, par)
            if solved is None:
                st.error("No growth rate between -30% and +100% reaches that. Owners' earnings, "
                         "share count, exit multiple or blend is likely off.")
            else:
                st.success(f"Growth of **{solved:.2%}** reproduces {d(target)} at your current "
                           f"exit multiple and blend.")
