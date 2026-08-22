"""
100-Bagger Checker
==================
Christopher Mayer's criteria, checked against the filings, with Michael Burry's
fully-adjusted return on invested capital as the centrepiece.

WHAT THIS IS FOR
----------------
Not a screener. It takes one ticker you already like and tells you which of
Mayer's conditions the filings support, which they contradict, and which cannot
be answered from EDGAR at all.

THE ARITHMETIC THAT DOES MOST OF THE WORK
-----------------------------------------
A 100-bagger is two engines multiplied: earnings growth and multiple change.

    100  =  (1+g)^N  x  (M_exit / M_now)  /  (1+dilution)^N

Solve for g and you get the growth rate the business must actually deliver.
Then compare it against what its own return on capital allows:

    g_max  =  ROIC  x  reinvestment rate

Nothing outgrows its return on capital for long, because growth has to be
funded and reinvested profit is the only self-funded source. When the required
rate sits above the ceiling, the case is not optimistic — it is arithmetically
closed, and no amount of narrative reopens it.

That comparison is the whole tool. Everything else is the work of computing the
two numbers honestly.

BURRY'S ROIC
------------
    ROIC = (Owners' earnings - interest income - capital lease payments
            - other expense)
           / (total capital - LT operating leases - net cash + other capital)

Owners' earnings come from the Tragic Algebra engine, ported unchanged from
tool 1 so both pages agree to the dollar. The rest is assembled from the
balance sheet, with two rules that matter:

  * Only genuinely deployable cash leaves the capital base. Restricted,
    regulated and operationally-tied cash funds the business and stays in.
  * Anything not obtainable from XBRL is exposed as an input seeded at zero
    and labelled as judgement, never guessed and quietly folded in.

Run:  streamlit run Home.py   (this file lives in pages/)
"""

from __future__ import annotations

import datetime as dt
import os
import statistics
import threading
import time
from dataclasses import dataclass

import pandas as pd
import requests
import streamlit as st

# ══════════════════════════════════════════════════════════════════════
#  SEC PLUMBING
#
#  Duplicated from tool 1 rather than imported. A Streamlit page module
#  cannot be imported without executing its UI, and a page filename
#  starting with a digit is not a legal module name either. The copy is
#  deliberate; the interval below is the one thing that had to change.
# ══════════════════════════════════════════════════════════════════════


def _sec_contact() -> str:
    try:
        v = st.secrets.get("sec_contact", "")
        if v:
            return str(v)
    except Exception:
        pass
    return os.environ.get("SEC_CONTACT", "")


SEC_HEADERS = {
    "User-Agent": f"Tragic Algebra Analyzer {_sec_contact() or 'contact-not-set'}",
    "Accept-Encoding": "gzip, deflate",
}

# Tool 1 spaces its requests at 0.15s. This module keeps its own counter — two
# copies of the same limiter in one process can interleave, and 0.15 each would
# put the app at ~13 req/s against a 10 req/s limit with two people on it. 0.30
# here holds the combined worst case under the ceiling. Nothing on this page is
# latency-sensitive: it is one filing fetch per ticker.
_SEC_MIN_INTERVAL = 0.30
_sec_lock = threading.Lock()
_sec_last = [0.0]


def _sec_get(url: str, timeout: int = 25) -> requests.Response:
    for attempt in range(4):
        with _sec_lock:
            wait = _SEC_MIN_INTERVAL - (time.monotonic() - _sec_last[0])
            if wait > 0:
                time.sleep(wait)
            _sec_last[0] = time.monotonic()
        try:
            r = requests.get(url, headers=SEC_HEADERS, timeout=timeout)
        except requests.RequestException:
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)
            continue
        if r.status_code == 200:
            return r
        if r.status_code in (403, 429, 502, 503):
            time.sleep(2 ** attempt)
            continue
        r.raise_for_status()
    raise RuntimeError(
        "SEC is throttling this app. Wait a minute and try again. If it keeps happening, "
        "check that a real contact address is set in Streamlit secrets — the SEC blocks "
        "generic user agents outright.")


ANNUAL_FORMS = ("10-K", "10-K/A", "20-F", "40-F")


@st.cache_data(ttl=86400, show_spinner=False)
def _ticker_map() -> dict[str, str]:
    r = _sec_get("https://www.sec.gov/files/company_tickers.json", timeout=15)
    return {e["ticker"].upper(): str(e["cik_str"]).zfill(10) for e in r.json().values()}


@st.cache_data(ttl=86400, show_spinner=False)
def _submissions(cik: str) -> dict:
    """Company metadata plus the recent filing index.

    Two things come from here that companyfacts cannot give: the SIC code, and
    a link to the latest proxy statement. Insider ownership is never tagged in
    XBRL — it lives in a beneficial ownership table inside the DEF 14A — so the
    most this tool can honestly do is take you straight to it.
    """
    try:
        return _sec_get(f"https://data.sec.gov/submissions/CIK{cik}.json", timeout=20).json()
    except Exception:
        return {}


def _latest_filing(subs: dict, forms: tuple[str, ...]) -> tuple[str, str] | None:
    """(url, filing date) of the most recent filing of one of these forms."""
    rec = subs.get("filings", {}).get("recent", {})
    cik_int = str(int(subs.get("cik", 0) or 0))
    for form, acc, doc, date in zip(rec.get("form", []), rec.get("accessionNumber", []),
                                    rec.get("primaryDocument", []), rec.get("filingDate", [])):
        if form in forms and acc:
            return (f"https://www.sec.gov/Archives/edgar/data/{cik_int}/"
                    f"{acc.replace('-', '')}/{doc}", date)
    return None


def _form4_count(subs: dict, days: int = 365) -> int:
    rec = subs.get("filings", {}).get("recent", {})
    cutoff = dt.date.today() - dt.timedelta(days=days)
    n = 0
    for form, date in zip(rec.get("form", []), rec.get("filingDate", [])):
        if form == "4" and date:
            try:
                if dt.date.fromisoformat(date) >= cutoff:
                    n += 1
            except ValueError:
                continue
    return n


def is_financial(sic: str) -> bool:
    """SIC 6000-6799: banks, insurers, brokers, REITs. Leverage is the product
    for these, not a financing choice, so an invested-capital denominator built
    from equity plus borrowings describes nothing real."""
    return sic.isdigit() and 6000 <= int(sic) <= 6799


@st.cache_data(ttl=86400, show_spinner=False)
def _facts(cik: str) -> dict:
    return _sec_get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json",
                    timeout=30).json()


def _annual(facts: dict, us: list[str], ifrs: list[str]) -> dict[int, tuple[str, str, float]]:
    """{fy: (start, end, value)} for full-year facts from annual reports only."""
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
                if not 330 <= (dt.date.fromisoformat(end)
                               - dt.date.fromisoformat(start)).days <= 400:
                    continue
                fy, filed = int(end[:4]), row.get("filed", "")
                if fy not in out or filed > out[fy][0]:
                    out[fy] = (filed, start, end, float(row.get("val", 0.0)))
            if out:
                return {k: (v[1], v[2], v[3]) for k, v in out.items()}
    return {}


def reporting_currency(facts: dict, concepts: list[str]) -> str | None:
    for taxonomy in ("us-gaap", "ifrs-full"):
        tax = facts.get("facts", {}).get(taxonomy, {})
        for concept in concepts:
            if concept not in tax:
                continue
            units = [u for u in tax[concept].get("units", {}) if u != "shares"]
            if units:
                return "USD" if "USD" in units else units[0]
    return None


def _instant(facts: dict, concepts: list[str], unit: str = "USD") -> dict[int, float]:
    """Latest balance-sheet value per fiscal year. First concept with data wins;
    merging them silently mixes incompatible definitions."""
    for taxonomy in ("us-gaap", "dei", "ifrs-full"):
        tax = facts.get("facts", {}).get(taxonomy, {})
        for concept in concepts:
            if concept not in tax:
                continue
            out: dict[int, tuple[str, float]] = {}
            for row in tax[concept].get("units", {}).get(unit, []):
                if row.get("start") or not row.get("end"):
                    continue
                if row.get("form") not in ANNUAL_FORMS:
                    continue
                fy, filed = int(row["end"][:4]), row.get("filed", "")
                if fy not in out or filed > out[fy][0]:
                    out[fy] = (filed, float(row["val"]))
            if out:
                return {k: v[1] for k, v in out.items()}
    return {}


def _instant_first(facts: dict, groups: list[list[str]],
                   unit: str = "USD") -> tuple[dict[int, float], int]:
    """Like _instant across several concept groups, returning which group won.

    Needed for cash: CashAndCashEquivalentsAtCarryingValue excludes restricted
    balances, the combined tag does not, and the difference is not shareholder
    money. Knowing which one answered is what lets the restricted amount be
    taken back out only when it was actually included.
    """
    for i, g in enumerate(groups):
        s = _instant(facts, g, unit)
        if s:
            return s, i
    return {}, -1


def _instant_sum(facts: dict, groups: list[list[str]]) -> dict[int, float]:
    """Sum of several independent balance-sheet lines, per year.

    A missing component is treated as zero, which is right far more often than
    not: a company with no commercial paper simply does not tag it. It is wrong
    when a filer uses a tag this reader does not know, which is why every
    capital figure is shown line by line rather than only as a total.
    """
    out: dict[int, float] = {}
    for g in groups:
        for fy, v in _instant(facts, g).items():
            out[fy] = out.get(fy, 0.0) + v
    return out


@st.cache_data(ttl=86400, show_spinner=False)
def _monthly_closes(ticker: str) -> dict[str, float]:
    r = requests.get(
        f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
        "?interval=1mo&range=11y", headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
    res = r.json()["chart"]["result"][0]
    closes = res["indicators"]["quote"][0]["close"]
    out = {}
    for ts, c in zip(res["timestamp"], closes):
        if c:
            d_ = dt.datetime.utcfromtimestamp(ts)
            out[f"{d_.year:04d}-{d_.month:02d}"] = float(c)
    return out


def _avg_price(closes: dict[str, float], start: str, end: str) -> float | None:
    s, e = dt.date.fromisoformat(start), dt.date.fromisoformat(end)
    vals, day = [], s
    while day <= e:
        v = closes.get(f"{day.year:04d}-{day.month:02d}")
        if v:
            vals.append(v)
        day = (day.replace(day=1) + dt.timedelta(days=32)).replace(day=1)
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
    """Restate historical share counts onto the current basis. Ported from
    tool 1: XBRL reports shares as filed, market prices arrive already
    split-adjusted, and mixing the two makes one year's dilution equal the
    whole split."""
    shares = {k: v for k, v in shares.items() if v and v > 0}
    fys, notes = sorted(shares), []
    if len(fys) < 2:
        return dict(shares), notes
    adjusted, factor = {}, 1.0
    for i in range(len(fys) - 1, -1, -1):
        fy = fys[i]
        adjusted[fy] = shares[fy] * factor
        if i > 0 and shares[fys[i - 1]] > 0:
            ratio = shares[fy] / shares[fys[i - 1]]
            if ratio > 2.85 and shares[fys[i - 1]] < 25e6:
                continue
            if ratio > 0 and (ratio > 2.85 or ratio < 0.35):
                if ratio >= 1:
                    clean = round(ratio * 2) / 2
                    label = f"{clean:g}:1"
                else:
                    inv = round((1 / ratio) * 2) / 2
                    clean = 1 / inv if inv > 0 else 0.0
                    label = f"1:{inv:g}"
                if clean > 0:
                    factor *= clean
                    notes.append(f"Stock split detected in FY{fy} (about {label}). Earlier share "
                                 "counts restated onto the current basis — without this both the "
                                 "SBC cost and the dilution rate would be wildly overstated.")
    return adjusted, notes


# ══════════════════════════════════════════════════════════════════════
#  TRAGIC ALGEBRA  — ported unchanged from tool 1
#
#  Owners' earnings are the numerator of Burry's ROIC, so this page needs
#  the same engine. The self-test at the foot re-runs tool 1's Alphabet
#  checks against this copy: if the two ever drift apart, that is where it
#  will show.
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
    excluded: str = ""        # non-empty means capital formation, not pay

    @property
    def C(self) -> float:
        return self.Cw - self.Ce

    @property
    def V(self) -> float:
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
    def dE_defined(self) -> bool:
        return self.sum_N > 0


def pool(years: list[Year]) -> Pooled:
    years = [y for y in years if not y.excluded]
    sN = sum(y.N for y in years)
    if not years or sN == 0:
        raise ValueError("Not enough data to pool.")
    return Pooled(dE=sum(y.OE for y in years) / sN, sum_N=sN,
                  sum_OE=sum(y.OE for y in years), sum_omega=sum(y.omega for y in years),
                  sum_G=sum(y.G for y in years), years=len(years))


# ══════════════════════════════════════════════════════════════════════
#  RETURN ON INVESTED CAPITAL
# ══════════════════════════════════════════════════════════════════════


@dataclass
class Capital:
    """One year's capital base, kept as separate lines so the total can be
    audited. Everything in $M."""
    fy: int
    equity: float = 0.0
    minority: float = 0.0
    debt: float = 0.0              # borrowings, short and long
    finance_leases: float = 0.0    # capitalised leases: debt in all but name
    operating_leases: float = 0.0  # shown, not applied — see note below
    cash: float = 0.0              # cash and investments, restricted removed
    restricted: float = 0.0
    goodwill: float = 0.0
    intangibles: float = 0.0
    revenue: float = 0.0
    op_cash_pct: float = 0.02
    other_capital: float = 0.0     # judgement, seeded at zero
    equity_found: bool = False

    @property
    def op_cash_need(self) -> float:
        """Cash the business cannot actually hand out. Burry's rule is that
        only genuinely deployable cash is subtracted from the capital base;
        working balances fund the business and belong in it. No published
        figure exists for the split, so this is a stated convention — a
        percentage of revenue, adjustable, and visible in the waterfall."""
        return max(0.0, self.revenue * self.op_cash_pct)

    @property
    def deployable_cash(self) -> float:
        return max(0.0, self.cash - self.op_cash_need)

    @property
    def total_capital(self) -> float:
        return self.equity + self.debt + self.finance_leases

    @property
    def invested(self) -> float:
        return self.total_capital - self.deployable_cash + self.other_capital

    @property
    def tangible_invested(self) -> float:
        return self.invested - self.goodwill - self.intangibles


@dataclass
class RoicYear:
    fy: int
    OE: float
    interest_income: float
    lease_payments: float
    other_expense: float
    cap: Capital
    excluded: str = ""

    @property
    def numerator(self) -> float:
        return self.OE - self.interest_income - self.lease_payments - self.other_expense

    @property
    def roic(self) -> float | None:
        c = self.cap.invested
        return self.numerator / c if c > 0 else None

    @property
    def tangible_roic(self) -> float | None:
        c = self.cap.tangible_invested
        return self.numerator / c if c > 0 else None

    @property
    def reason(self) -> str:
        """Empty when the year's ROIC can be trusted; otherwise why it cannot.

        Every one of these produced a confident wrong number before it produced
        a refusal. A negative capital base is the worst of them: buybacks that
        push equity below zero flip the sign, and a superb business prints as
        a catastrophic one.
        """
        if self.excluded:
            return f"{self.excluded} — owners' earnings distorted"
        if not self.cap.equity_found:
            return "no equity figure in this year's filing"
        if self.cap.invested <= 0:
            return "invested capital is zero or negative"
        if self.cap.revenue > 0 and self.cap.invested / self.cap.revenue < 0.05:
            return "capital base under 5% of revenue — ratio not informative"
        return ""


def median_roic(rows: list[RoicYear], n: int = 5) -> float | None:
    vals = [r.roic for r in rows[-n:] if not r.reason and r.roic is not None]
    return statistics.median(vals) if vals else None


# ══════════════════════════════════════════════════════════════════════
#  THE 100-BAGGER ARITHMETIC
# ══════════════════════════════════════════════════════════════════════


def required_growth(multiple_now: float, multiple_exit: float, years: int,
                    target: float = 100.0, dilution: float = 0.0) -> float | None:
    """Annual growth in owners' earnings needed for a target total return.

        target = (1+g)^N x (M_exit/M_now) / (1+dilution)^N

    Dilution enters as a straight drag on the per-share result, which is the
    only result that matters. A business can multiply its earnings a hundred
    times and still hand you far less if it pays for the growth in stock.
    """
    if multiple_now <= 0 or multiple_exit <= 0 or years <= 0 or target <= 0:
        return None
    return (target * multiple_now / multiple_exit) ** (1.0 / years) * (1.0 + dilution) - 1.0


def sustainable_growth(roic: float, payout_ratio: float) -> float:
    """Growth a business can fund from its own profits: ROIC x reinvestment.

    Above this it must raise capital — debt, which is finite, or stock, which
    is the dilution term above. This is the ceiling Burry means when he says
    ROIC bounds growth.
    """
    return roic * max(0.0, 1.0 - payout_ratio)


def per_share_ceiling(roic: float, payout_ratio: float, buyback_yield: float) -> float:
    """Total-earnings growth plus the lift from a shrinking share count.

    Buybacks do not grow the business, but a hundredfold on fewer shares is
    still a hundredfold to whoever stayed. Retiring 3% a year adds roughly
    3 points to per-share compounding.
    """
    g = sustainable_growth(roic, payout_ratio)
    b = min(max(buyback_yield, -0.20), 0.20)
    return (1.0 + g) / (1.0 - b) - 1.0


def cagr(first: float, last: float, years: int) -> float | None:
    if years <= 0 or first <= 0 or last <= 0:
        return None
    return (last / first) ** (1.0 / years) - 1.0


# Bands for the starting size, stated as this tool's own convention rather than
# attributed to a precise figure in Mayer. What is not a convention is the
# arithmetic in the second column: it is simply what a hundredfold means.
SIZE_BANDS = [
    (500, "genuinely small base — the size range most 100-baggers started from"),
    (2_000, "small, not micro — 100x still lands inside what has been done before"),
    (10_000, "mid cap — 100x means a business worth several hundred billion"),
    (100_000, "large cap — 100x lands beyond anything that has ever traded"),
]

WORLD_GDP_M = 110_000_000.0   # world GDP, roughly $110T, expressed in $M


def size_band(mcap_m: float) -> str:
    for ceiling, label in SIZE_BANDS:
        if mcap_m <= ceiling:
            return label
    return "the arithmetic refuses this one on size alone"


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
    "MA":   (["StockIssuedDuringPeriodSharesAcquisitions"], []),
    "OFFER": (["StockIssuedDuringPeriodSharesNewIssues"], []),
    "CONV": (["StockIssuedDuringPeriodSharesConversionOfConvertibleSecurities",
              "StockIssuedDuringPeriodSharesConversionOfUnits"], []),
    # Interest earned on the cash pile. It comes OUT of the numerator because
    # the cash came out of the denominator — leave it in and a company with a
    # large treasury books its deposit income as an operating return.
    "INT": (["InvestmentIncomeInterest", "InvestmentIncomeInterestAndDividend",
             "InterestIncomeOther"], []),
    # Finance lease principal. A financing outflow that never touches the
    # income statement, so earnings do not yet reflect it.
    "LEASEPAY": (["FinanceLeasePrincipalPayments",
                  "RepaymentsOfLongTermCapitalLeaseObligations"], []),
    "DIV": (["PaymentsOfDividendsCommonStock", "PaymentsOfDividends"], []),
    "CAPEX": (["PaymentsToAcquirePropertyPlantAndEquipment",
               "PaymentsToAcquireProductiveAssets"], []),
}

# Balance-sheet groups. Each inner list is an ordered fallback where the first
# tag with data wins; the outer list is summed.
EQUITY = [["StockholdersEquity"],
          ["StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"]]
MINORITY = [["MinorityInterest"]]
DEBT = [["LongTermDebtNoncurrent", "LongTermDebt"],
        ["LongTermDebtCurrent", "DebtCurrent", "ShortTermBorrowings"],
        ["CommercialPaper"]]
FIN_LEASE = [["FinanceLeaseLiabilityNoncurrent", "CapitalLeaseObligationsNoncurrent"],
             ["FinanceLeaseLiabilityCurrent", "CapitalLeaseObligationsCurrent"]]
OP_LEASE = [["OperatingLeaseLiabilityNoncurrent", "OperatingLeaseLiability"]]
CASH_PLAIN = ["CashAndCashEquivalentsAtCarryingValue"]
CASH_WITH_RESTRICTED = ["CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"]
INVESTMENTS = [["ShortTermInvestments", "MarketableSecuritiesCurrent",
                "AvailableForSaleSecuritiesDebtSecuritiesCurrent"],
               ["MarketableSecuritiesNoncurrent",
                "AvailableForSaleSecuritiesDebtSecuritiesNoncurrent"]]
RESTRICTED = [["RestrictedCashAndCashEquivalentsNoncurrent", "RestrictedCashNoncurrent"],
              ["RestrictedCashAndCashEquivalentsCurrent", "RestrictedCashCurrent"]]
GOODWILL = [["Goodwill"]]
INTANGIBLES = [["FiniteLivedIntangibleAssetsNet", "IntangibleAssetsNetExcludingGoodwill"]]


def load(ticker: str, n_years: int = 10):
    """Everything this page needs, in one pass over the filings."""
    cmap = _ticker_map()
    if ticker not in cmap:
        raise ValueError(f"'{ticker}' is not in the SEC company list.")
    cik = cmap[ticker]
    facts = _facts(cik)
    subs = _submissions(cik)
    sic, sic_desc = str(subs.get("sic", "")), str(subs.get("sicDescription", ""))

    series = {k: _annual(facts, us, ifrs) for k, (us, ifrs) in CONCEPTS.items()}
    if not series["N"]:
        ccy = reporting_currency(facts, CONCEPTS["N"][0] + CONCEPTS["N"][1])
        if ccy and ccy != "USD":
            raise ValueError(
                f"{ticker} reports in {ccy}, not US dollars. Every figure here assumes one "
                "currency throughout, and a euro capital base against a dollar share price "
                "would look fine and be wrong. Foreign private issuers filing in their home "
                "currency are not supported.")
        raise ValueError(
            f"No annual net income found for {ticker}. The filer uses tags this reader does not "
            "recognise. Owners' earnings are the numerator of everything here, so nothing can "
            "be computed without it.")

    shares_out = _instant(facts, ["CommonStockSharesOutstanding", "CommonStockSharesIssued",
                                  "EntityCommonStockSharesOutstanding"], unit="shares")
    shares_out = {k: v for k, v in shares_out.items() if v and v > 0}
    shares_out, notes = split_adjust(shares_out)
    try:
        closes = _monthly_closes(ticker)
    except Exception:
        closes = {}

    fys = sorted(series["N"])[-n_years:]
    years: list[Year] = []
    non_sbc_total = 0.0
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
        years.append(Year(fy=fy, N=N / 1e6, G=get("G"), T=get("T"), dS=dS,
                          Cw=get("Cw"), Ce=get("Ce"),
                          price=_avg_price(closes, start, end) or 0.0))

    # Capital events. A listing converts preferred to common and sells new
    # stock; an all-stock acquisition issues a year's payroll many times over.
    # Priced at market, either one charges the whole transaction to employees.
    priced = [i for i, y in enumerate(years) if y.price > 0]
    for i in priced:
        base = shares_out.get(fys[i] - 1, 0.0) / 1e6
        if base <= 0:
            continue
        jump = years[i].dS / base
        first_priced = (i == priced[0])
        if jump > (0.25 if first_priced else 0.15):
            years[i].excluded = "listing year" if first_priced else "share-funded acquisition"
            notes.append(
                f"FY{years[i].fy} excluded — the share count rose {jump:.0%} in one year, which no "
                "payroll produces. Owners' earnings and ROIC are both blank for that year.")
    if non_sbc_total:
        notes.append(f"Excluded {non_sbc_total:,.1f}M shares issued for acquisitions, offerings or "
                     "conversions. Those are corporate transactions, not compensation.")
    if any(y.price == 0 for y in years):
        notes.append("No share price for some years — their stock-comp cost is understated, so "
                     "owners' earnings and ROIC read high for those years.")
    if not any(y.Cw for y in years):
        notes.append("No tax-withholding line found. That understates the SBC cost, so owners' "
                     "earnings here are flattering rather than conservative.")

    # ── capital base, per year ───────────────────────────────────────
    # Parent-only equity is preferred because net income is parent-only too.
    # Mixing a consolidated capital base with a parent's earnings understates
    # the return by exactly the minority's share.
    eq = _instant(facts, EQUITY[0]) or _instant(facts, EQUITY[1])
    minority = _instant_sum(facts, MINORITY)
    debt = _instant_sum(facts, DEBT)
    fin_lease = _instant_sum(facts, FIN_LEASE)
    op_lease = _instant_sum(facts, OP_LEASE)
    restricted = _instant_sum(facts, RESTRICTED)
    cash_ser, which = _instant_first(facts, [CASH_PLAIN, CASH_WITH_RESTRICTED])
    invest = _instant_sum(facts, INVESTMENTS)
    goodwill = _instant_sum(facts, GOODWILL)
    intang = _instant_sum(facts, INTANGIBLES)
    rev = series.get("REV", {})

    if which == 1:
        notes.append("This filer tags only the combined cash-including-restricted line. The "
                     "restricted balance has been subtracted back out where it was tagged "
                     "separately; where it was not, deployable cash is overstated and ROIC "
                     "reads high.")
    if not eq:
        notes.append("No shareholders' equity figure found in the annual filings. Without it "
                     "there is no capital base and ROIC cannot be computed at all.")

    caps: dict[int, Capital] = {}
    for fy in fys:
        c_raw = cash_ser.get(fy, 0.0) / 1e6
        r_ = restricted.get(fy, 0.0) / 1e6
        caps[fy] = Capital(
            fy=fy,
            equity=eq.get(fy, 0.0) / 1e6,
            minority=minority.get(fy, 0.0) / 1e6,
            debt=debt.get(fy, 0.0) / 1e6,
            finance_leases=fin_lease.get(fy, 0.0) / 1e6,
            operating_leases=op_lease.get(fy, 0.0) / 1e6,
            cash=max(0.0, c_raw - (r_ if which == 1 else 0.0)) + invest.get(fy, 0.0) / 1e6,
            restricted=r_,
            goodwill=goodwill.get(fy, 0.0) / 1e6,
            intangibles=intang.get(fy, 0.0) / 1e6,
            revenue=rev[fy][2] / 1e6 if fy in rev else 0.0,
            equity_found=fy in eq,
        )

    latest_cap = caps.get(fys[-1], Capital(fy=fys[-1]))
    if latest_cap.minority > 0 and latest_cap.equity > 0 \
            and latest_cap.minority / latest_cap.equity > 0.05:
        notes.append(
            f"Non-controlling interests are {latest_cap.minority/latest_cap.equity:.0%} of "
            "shareholders' equity. Net income here is the parent's slice only, and so is the "
            "equity used in the capital base — consistent, but both understate the "
            "consolidated business. Read ROIC as the return on your slice.")
    if is_financial(sic):
        notes.append(
            f"{sic_desc or 'Financial company'} (SIC {sic}). Leverage is the product for banks, "
            "insurers and REITs rather than a financing decision, so equity plus borrowings "
            "does not describe capital at work and cash is not free. ROIC is not shown.")

    # Share count: outstanding beats trailing weighted average, but a dual-class
    # filer tags each class separately and only one may be picked up.
    sh_out = shares_out[max(shares_out)] / 1e6 if shares_out else 0.0
    wavg = _annual(facts, ["WeightedAverageNumberOfDilutedSharesOutstanding",
                           "WeightedAverageNumberOfSharesOutstandingDiluted"], [])
    wavg_v = wavg[max(wavg)][2] / 1e6 if wavg else 0.0
    diluted = sh_out or wavg_v
    if sh_out > 0 and wavg_v > 0 and sh_out / wavg_v < 0.65:
        diluted = wavg_v
        notes.append(f"Shares outstanding read as {sh_out:,.1f}M but weighted-average diluted is "
                     f"{wavg_v:,.1f}M — too big a gap for buybacks, and the usual cause is a "
                     "second share class that was missed. Using the diluted figure.")

    # Net annual change in the share count, over the window and excluding the
    # capital-event years. Positive is dilution, negative is retirement.
    clean = [fy for fy in fys if not any(y.fy == fy and y.excluded for y in years)]
    dil = None
    if len(clean) >= 3 and clean[0] in shares_out and clean[-1] in shares_out:
        dil = cagr(shares_out[clean[0]], shares_out[clean[-1]], clean[-1] - clean[0])

    proxy = _latest_filing(subs, ("DEF 14A", "DEFA14A", "DEF14A"))
    pre = {
        "sic": sic, "sic_desc": sic_desc, "financial": is_financial(sic),
        "shares": diluted, "dilution": dil, "caps": caps, "fys": fys,
        "interest": {fy: abs(series["INT"][fy][2]) / 1e6 for fy in series.get("INT", {})},
        "leasepay": {fy: abs(series["LEASEPAY"][fy][2]) / 1e6 for fy in series.get("LEASEPAY", {})},
        "dividends": {fy: abs(series["DIV"][fy][2]) / 1e6 for fy in series.get("DIV", {})},
        "capex": {fy: abs(series["CAPEX"][fy][2]) / 1e6 for fy in series.get("CAPEX", {})},
        "revenue": {fy: rev[fy][2] / 1e6 for fy in rev},
        "name": subs.get("name", ticker),
        "proxy": proxy,
        "form4": _form4_count(subs),
        "cik": str(int(cik)),
    }
    return years, notes, pre


def build_roic(years: list[Year], pre: dict, op_cash_pct: float,
               other_expense: float, other_capital: float) -> list[RoicYear]:
    """Assemble Burry's ROIC per year.

    The two judgement terms are applied to the latest year only. Spreading a
    single forensic estimate back over a decade would imply a precision the
    estimate does not have, and would move the historical trend — the one thing
    on this page that is pure arithmetic.
    """
    out = []
    last = years[-1].fy if years else None
    for y in years:
        cap = pre["caps"].get(y.fy, Capital(fy=y.fy))
        cap.op_cash_pct = op_cash_pct
        cap.other_capital = other_capital if y.fy == last else 0.0
        out.append(RoicYear(
            fy=y.fy, OE=y.OE,
            interest_income=pre["interest"].get(y.fy, 0.0),
            lease_payments=pre["leasepay"].get(y.fy, 0.0),
            other_expense=other_expense if y.fy == last else 0.0,
            cap=cap, excluded=y.excluded))
    return out


# ══════════════════════════════════════════════════════════════════════
#  SELF-TEST
# ══════════════════════════════════════════════════════════════════════

def self_test() -> list[tuple[str, bool, str]]:
    out = []

    # 1. The ported engine still agrees with tool 1, to the dollar.
    goog = [(2016, 19478, 6900, 3693, 3304, 97, 47), (2017, 12662, 7900, 4846, 4166, 78, 55),
            (2018, 30736, 10000, 9075, 4993, -2, 61), (2019, 34343, 11700, 18396, 4765, -158, 70),
            (2020, 40269, 12991, 31149, 5720, -263, 73), (2021, 76033, 15376, 50274, 10162, -264, 125),
            (2022, 59972, 19362, 59296, 9300, -412, 117), (2023, 73795, 22460, 61504, 9837, -374, 115),
            (2024, 100118, 22785, 62222, 12190, -243, 164), (2025, 132170, 24953, 45709, 14167, -93, 206)]
    ys = [Year(fy=f, N=n, G=g, T=t, Cw=c, dS=d, price=p) for f, n, g, t, c, d, p in goog]
    out.append(("Ported engine: Alphabet FY2016 V = $8,252M",
                abs(ys[0].V - 8252) < 1, f"${ys[0].V:,.0f}M"))
    out.append(("Ported engine: Alphabet pooled ΔE = 88.7%",
                abs(pool(ys).dE - 0.887) < 0.002, f"{pool(ys).dE:.2%}"))

    # 2. Mayer's arithmetic. 100x at a flat multiple over 25 years is the
    #    figure the book leads with.
    g25 = required_growth(20, 20, 25)
    out.append(("100x in 25 years, flat multiple → 20.2%/yr",
                abs(g25 - 0.2022) < 0.001, f"{g25:.2%}"))
    g15 = required_growth(20, 20, 15)
    out.append(("100x in 15 years, flat multiple → 35.9%/yr",
                abs(g15 - 0.3594) < 0.001, f"{g15:.2%}"))
    gm = required_growth(10, 20, 25)
    out.append(("Multiple 10x→20x halves the work → 16.9%/yr",
                abs(gm - 0.1694) < 0.001, f"{gm:.2%}"))
    gd = required_growth(20, 20, 25, dilution=0.02)
    out.append(("2%/yr dilution adds 2.4 points → 22.6%/yr",
                abs(gd - 0.2263) < 0.001, f"{gd:.2%}"))

    # 3. The ROIC ceiling.
    out.append(("ROIC 20%, half paid out → 10% ceiling",
                abs(sustainable_growth(0.20, 0.5) - 0.10) < 1e-9,
                f"{sustainable_growth(0.20, 0.5):.1%}"))
    out.append(("Same, plus 3% of stock retired → 13.4% per share",
                abs(per_share_ceiling(0.20, 0.5, 0.03) - 0.13402) < 0.0005,
                f"{per_share_ceiling(0.20, 0.5, 0.03):.2%}"))

    # 4. Burry's ROIC, wired to a hand-worked example. This checks the
    #    plumbing, not the framework — there is no published figure to
    #    validate against the way Alphabet validates the Tragic Algebra.
    c = Capital(fy=2025, equity=800, debt=200, cash=250, revenue=2500,
                op_cash_pct=0.02, other_capital=50)
    r = RoicYear(fy=2025, OE=100, interest_income=5, lease_payments=2, other_expense=3, cap=c)
    #  capital 1000 - deployable (250-50) + 50 = 850 ; numerator 90
    out.append(("ROIC formula wiring: 90 / 850 → 10.59%",
                r.roic is not None and abs(r.roic - 0.10588) < 0.0005,
                f"{r.roic:.2%}" if r.roic else "n/a"))
    out.append(("Operating cash stays in the capital base",
                abs(c.deployable_cash - 200) < 1e-9, f"${c.deployable_cash:,.0f}M deployable"))

    # 5. The refusals.
    neg = RoicYear(fy=2025, OE=500, interest_income=0, lease_payments=0, other_expense=0,
                   cap=Capital(fy=2025, equity=-300, debt=100, cash=50, revenue=5000,
                               equity_found=True))
    out.append(("Negative capital base refuses rather than flipping sign",
                neg.reason != "" and neg.roic is None, neg.reason or "printed a number"))
    return out


# ══════════════════════════════════════════════════════════════════════
#  UI
# ══════════════════════════════════════════════════════════════════════
#
# NOTE ON DOLLAR SIGNS: Streamlit markdown parses $...$ as LaTeX, so any
# literal dollar amount inside st.write/markdown/success/error/info/warning
# must be escaped. st.metric, st.code and st.dataframe are unaffected.


def d(x, dp=2):
    return f"\\${x:,.{dp}f}"


def money(x: float) -> str:
    """$M in, human-readable out."""
    if abs(x) >= 1_000_000:
        return f"${x/1_000_000:,.2f}T"
    if abs(x) >= 1_000:
        return f"${x/1_000:,.2f}B"
    return f"${x:,.0f}M"


st.set_page_config(
    page_title="100-Bagger Checker — Mayer's criteria and Burry's ROIC",
    page_icon="💯",
    layout="centered",
    initial_sidebar_state="collapsed",
)
st.title("💯 100-Bagger Checker")
st.caption("Christopher Mayer's criteria against the filings, with a fully-adjusted "
           "return on invested capital doing the arguing")

if not _sec_contact():
    st.warning(
        "**No SEC contact address set.** The SEC requires a real email in the request header "
        "and blocks generic user agents, so lookups will fail. Add `sec_contact = "
        "\"you@example.com\"` in Streamlit Settings → Secrets, or set a SEC_CONTACT "
        "environment variable locally.")

if "hb_years" not in st.session_state:
    st.info(
        "**A hundredfold is two engines multiplied: earnings growth and multiple change.** "
        "This works out the growth rate your holding period actually requires, then checks it "
        "against the ceiling the company's own return on capital sets. When the required rate "
        "is above the ceiling, the case is closed by arithmetic rather than by judgement.\n\n"
        "Enter a US-listed ticker you already like. This is a checker, not a screener.")

with st.form("hb_lookup"):
    ticker = st.text_input("Stock ticker",
                           placeholder="CPRX · MATX · CXDO · AGX — press Enter").upper().strip()
    submitted = st.form_submit_button("Check", type="primary")

if submitted:
    if not ticker:
        st.warning("Enter a ticker first.")
    else:
        try:
            with st.spinner(f"Reading {ticker} annual filings…"):
                yrs, nts, pre_ = load(ticker, 10)
            st.session_state.update(hb_years=yrs, hb_notes=nts, hb_pre=pre_, hb_tk=ticker)
        except ValueError as e:
            st.error(f"Could not load {ticker}: {e}")
        except Exception as e:
            st.error(
                f"Could not load {ticker} — {type(e).__name__}: {e}\n\n"
                "This is a gap in how the filings were read, not something you did. Recent "
                "listings, several share classes and foreign issuers are the usual causes.")

years = st.session_state.get("hb_years", [])
if years and ticker and st.session_state.get("hb_tk") == ticker:
    notes = list(st.session_state["hb_notes"])
    pre = st.session_state["hb_pre"]
    tk = st.session_state["hb_tk"]
    fys = pre["fys"]
    alerts: list[tuple[str, str]] = [("info", n) for n in notes]

    price = current_price(tk) or 0.0
    shares = pre["shares"]
    latest = years[-1]
    rev_now = pre["revenue"].get(fys[-1], 0.0)

    # ══ inputs ═══════════════════════════════════════════════════════
    st.markdown("---")
    st.subheader("Inputs")

    c1, c2, c3 = st.columns(3)
    price = c1.number_input("Price", value=float(price or 100.0), step=0.01)
    shares = c2.number_input("Diluted shares (M)", value=float(round(shares, 1)), step=1.0,
                             help="Everything per-share divides by this. Check it against the "
                                  "market cap shown below — a missed second share class is the "
                                  "most common reading error there is.")
    OE = c3.number_input(
        "Owners' earnings ($M)", value=float(round(latest.OE, 1)), step=1.0,
        help="From the Tragic Algebra engine: net income, plus the GAAP stock-comp charge, less "
             "what the stock actually cost. Override for anything non-recurring, or with the "
             "figure you think the business earns once profitable.")
    mcap = shares * price
    c1.caption(f"Market cap {money(mcap)}")

    with st.expander("Judgement inputs — the parts EDGAR cannot answer"):
        st.caption(
            "Burry's ROIC has two terms that are not tagged anywhere in XBRL, so they are seeded "
            "at zero and left to you. Zero is not neutral: it makes the return read high.\n\n"
            "**Other expense** covers forensic depreciation and amortisation, a normalised tax "
            "rate and cyclical adjustment. A company under-depreciating its asset base, or "
            "sitting on a tax rate that will not last, belongs here.\n\n"
            "**Other capital** adds back what funds the business without appearing as capital: "
            "purchase obligations, customer float, restricted cash and loans held for "
            "settlement. Payroll and payments processors are the obvious cases — client funds "
            "are somebody else's money, but they are working in the business.\n\n"
            "**Operating cash** is the split between cash the business needs and cash you could "
            "actually have. Burry's rule is that only the second comes out of the capital base. "
            "He publishes no percentage; 2% of revenue is this tool's stated convention, not "
            "his figure. Raising it lowers ROIC.")
        j1, j2, j3 = st.columns(3)
        other_expense = j1.number_input("Other expense ($M)", value=0.0, step=1.0)
        other_capital = j2.number_input("Other capital ($M)", value=0.0, step=1.0)
        op_cash_pct = j3.number_input("Operating cash (% of revenue)", value=2.0, step=0.5,
                                      min_value=0.0, max_value=25.0) / 100.0
        st.caption("Both dollar figures apply to the latest year only. Spreading one forensic "
                   "estimate back over a decade would move the historical trend, which is the "
                   "one thing on this page that is pure arithmetic.")

    rows = build_roic(years, pre, op_cash_pct, other_expense, other_capital)
    latest_r = rows[-1]
    roic_now = latest_r.roic
    roic_med = median_roic(rows, 5)
    financial = pre["financial"]

    # Cash returned to shareholders, as a share of owners' earnings. This is the
    # reinvestment rate's complement and it sets the growth ceiling.
    div_now = pre["dividends"].get(fys[-1], 0.0)
    buyback_now = latest.T
    payout = (div_now + buyback_now) / OE if OE > 0 else 0.0
    buyback_yield = buyback_now / mcap if mcap > 0 else 0.0

    e1, e2 = st.columns(2)
    horizon = e1.slider("Holding period (years)", 5, 30, 20, 1,
                        help="Mayer's own study runs on 25-year holding periods. Shorter needs a "
                             "faster rate, and the requirement is brutally non-linear.")
    exit_default = (price * shares / OE) if OE > 0 else 20.0
    exit_mult = e2.number_input(
        "Exit multiple on owners' earnings", value=float(round(min(max(exit_default, 3.0), 60.0), 1)),
        step=0.5, min_value=1.0,
        help="What the market pays for a dollar of owners' earnings at the end. Seeded flat at "
             "today's multiple, which is the honest default — assuming expansion is where most "
             "of the wishful thinking in this arithmetic hides.")
    dil_seed = pre.get("dilution")
    dilution = st.slider(
        "Net annual share issuance (%)", -6.0, 12.0,
        float(round((dil_seed or 0.0) * 100, 1)) if dil_seed is not None else 1.5, 0.1,
        help="Seeded from the actual share count over the window, capital events excluded. "
             "Negative means the count is shrinking. This is a straight drag on your per-share "
             "result and it compounds for the whole holding period.") / 100.0
    if dil_seed is None:
        st.caption("Share-count history was too short or too broken to measure dilution — the "
                   "1.5% above is a placeholder, not a measurement.")

    # ══ the arithmetic ═══════════════════════════════════════════════
    st.markdown("---")
    st.subheader(f"The arithmetic · {tk}")

    if OE <= 0:
        st.error(
            f"**No entry multiple exists.** Owners' earnings are {d(OE,0)}M, and Mayer's "
            "arithmetic is a ratio of what you pay to what the business earns. A hundredfold "
            "from a loss is not a calculation, it is a story about a recovery. Enter the "
            "owners' earnings you believe the business reaches in a normal year and the "
            "arithmetic below will mean something.")
    elif shares <= 0 or price <= 0:
        st.error("Enter a price and a share count — every figure below divides by them.")
    else:
        mult_now = mcap / OE
        req = required_growth(mult_now, exit_mult, horizon, 100.0, dilution)
        ceiling = (per_share_ceiling(roic_med, payout, buyback_yield)
                   if roic_med is not None and not financial else None)

        a1, a2, a3 = st.columns(3)
        a1.metric("Market cap", money(mcap), f"100x = {money(mcap*100)}")
        a2.metric("Growth needed", f"{req:.1%}" if req else "—",
                  f"in owners' earnings, for {horizon}y")
        a3.metric("ROIC ceiling", f"{ceiling:.1%}" if ceiling is not None else "n/a",
                  "what capital allows" if ceiling is not None else "see below")

        if mcap * 100 > WORLD_GDP_M * 0.05:
            st.error(
                f"**Size closes this before growth does.** A hundredfold on {money(mcap)} is "
                f"{money(mcap*100)} — against a world economy of roughly \\$110 trillion. "
                "Mayer's whole point is that the base has to be small enough for the arithmetic "
                "to have somewhere to go. Nothing below rescues this.")
        elif req is None:
            st.warning("Required growth could not be computed from these inputs.")
        elif ceiling is None:
            st.warning(
                f"**{req:.1%} a year for {horizon} years, and no ceiling to check it against.** "
                + ("Return on capital is not shown for financials — see the ROIC section. "
                   if financial else
                   "The capital base could not be read, so the one number that would tell you "
                   "whether that rate is fundable is missing. ")
                + "Judge the growth rate on its own and treat this page as incomplete.")
        elif req > ceiling:
            st.error(
                f"**The arithmetic does not close.** A hundredfold in {horizon} years needs "
                f"{req:.1%} a year in owners' earnings. Reinvesting at a {roic_med:.1%} return "
                f"on capital"
                + (f", after returning {payout:.0%} of earnings to shareholders," if payout > 0.02
                   else "")
                + f" funds about {ceiling:.1%}. The gap has to come from outside — debt, which "
                  "runs out, or stock, which is the dilution term you already set. This is not a "
                  "verdict on the business. It is a statement that this holding period and this "
                  "entry multiple cannot get there together.")
        else:
            st.success(
                f"**The arithmetic closes.** {req:.1%} a year is inside the {ceiling:.1%} that a "
                f"{roic_med:.1%} return on capital funds from its own profits. That makes the "
                "hundredfold possible, not likely — everything now turns on whether the return "
                "on capital and the runway both last, which no filing can tell you.")

        st.write("**What it takes, at each exit multiple** — required growth in owners' earnings")
        cols = [10.0, 15.0, 20.0, 25.0, 30.0]
        grid = []
        for h in (10, 15, 20, 25):
            row = {"Held for": f"{h} years"}
            for m in cols:
                g = required_growth(mult_now, m, h, 100.0, dilution)
                if g is None:
                    row[f"{m:g}x"] = "—"
                elif ceiling is not None and g > ceiling:
                    row[f"{m:g}x"] = f"{g:.0%} ✗"
                else:
                    row[f"{m:g}x"] = f"{g:.0%}"
            grid.append(row)
        st.dataframe(pd.DataFrame(grid), width="stretch", hide_index=True)
        st.caption(
            f"Today's multiple is {mult_now:,.1f}x owners' earnings, and dilution of "
            f"{dilution:.1%} a year is included. "
            + (f"A ✗ marks a rate above the {ceiling:.1%} the capital base funds. "
               if ceiling is not None else "")
            + "Notice how much work the exit multiple does: buying cheap and selling dear is "
              "half of Mayer's engine, and it is the half you control at purchase.")

    # ══ ROIC ═════════════════════════════════════════════════════════
    st.markdown("---")
    st.subheader("Return on invested capital")

    if financial:
        st.error(
            f"**Not shown for financials.** {pre['sic_desc'] or 'This company'} (SIC "
            f"{pre['sic']}) runs on leverage as its product rather than as a financing choice. "
            "Equity plus borrowings is not capital at work, and its cash is not free — it backs "
            "deposits or policyholder liabilities. Return on equity against a combined ratio or "
            "a net interest margin is the right frame, and this tool does not contain it.")
    elif roic_now is None or latest_r.reason:
        _w = latest_r.cap
        _why = ""
        if _w.invested <= 0 and _w.equity_found:
            if _w.equity < 0:
                _why = ("\n\nEquity is negative, which for a profitable company almost always "
                        "means buybacks have retired more capital than the balance sheet "
                        "carries. That is usually strength rather than distress — the business "
                        "funds itself and returns the rest — but a return on a negative "
                        "denominator flips sign, so nothing is printed. Judge reinvestment by "
                        "cash flow instead.")
            elif _w.deployable_cash > _w.total_capital:
                _why = (f"\n\nThe cause here is cash, not losses: {money(_w.deployable_cash)} of "
                        f"deployable cash against {money(_w.total_capital)} of total capital. "
                        "The operating business runs on less than nothing, which is a genuinely "
                        "excellent property and not a computable one. Two things to check before "
                        "accepting it — that the cash really is deployable rather than earmarked "
                        "for an acquisition or a trial, and that the share count is right, since "
                        "a missed share class inflates nothing but makes everything else look "
                        "strange. Raising the operating-cash percentage above will keep more of "
                        "it in the base if you think the business needs it.")
            else:
                _why = ("\n\nCapital at or below zero, with equity positive — the subtractions "
                        "have outrun the base. Check the waterfall against the balance sheet.")
        st.error(f"**ROIC is n/a for FY{latest_r.fy} — {latest_r.reason or 'capital base unread'}.**"
                 + _why)
    else:
        r1, r2, r3 = st.columns(3)
        r1.metric("ROIC", f"{roic_now:.1%}", f"FY{latest_r.fy}")
        _readable = len([r for r in rows[-5:] if not r.reason and r.roic is not None])
        r2.metric("Median, 5 years", f"{roic_med:.1%}" if roic_med is not None else "—",
                  f"{_readable} of {len(rows[-5:])} years readable")
        tang = latest_r.tangible_roic
        r3.metric("Ex-goodwill", f"{tang:.1%}" if tang is not None else "n/a",
                  "return on tangible capital")
        if roic_now > 1.0:
            st.warning(
                f"**{roic_now:.0%} is either a capital-light franchise or a reading error.** "
                "Both exist. Businesses that collect cash before they spend it genuinely earn "
                "these returns; so do companies whose capital base was mis-assembled. Check the "
                "waterfall below against the balance sheet before believing it.")
        if tang is None and latest_r.cap.invested > 0:
            st.info(
                f"Ex-goodwill is n/a because goodwill and acquired intangibles of "
                f"{money(latest_r.cap.goodwill + latest_r.cap.intangibles)} exceed the whole "
                f"{money(latest_r.cap.invested)} capital base. The tangible business is "
                "carrying less capital than the company paid for acquisitions — common in "
                "roll-ups, and the reason the all-in figure is the one to judge management by.")
        if tang is not None and roic_now is not None and tang > roic_now * 2:
            st.info(
                f"Return on tangible capital is {tang:.0%} against {roic_now:.0%} all-in, so "
                "goodwill and acquired intangibles are most of the capital base. The tangible "
                "figure is what tells you the return on the next dollar reinvested; the all-in "
                "figure is what tells you how well the acquisitions were priced.")

    if not financial:
        st.write("**Burry's formula, line by line** — latest year")
        w = latest_r.cap
        wf = [
            ("Owners' earnings", latest_r.OE, "from the Tragic Algebra engine"),
            ("less interest income", -latest_r.interest_income,
             "the cash left the denominator, so its income leaves the numerator"),
            ("less capital lease payments", -latest_r.lease_payments,
             "a financing outflow earnings never saw"),
            ("less other expense", -latest_r.other_expense,
             "forensic D&A, normalised tax, cyclical — yours to set"),
            ("= adjusted return", latest_r.numerator, ""),
            ("Shareholders' equity", w.equity, "parent's share" if w.minority else ""),
            ("plus borrowings", w.debt, "short and long term"),
            ("plus finance leases", w.finance_leases, "capitalised leases are debt in all but name"),
            ("= total capital", w.total_capital, ""),
            ("less deployable cash", -w.deployable_cash,
             f"of {money(w.cash)} held; {money(w.op_cash_need)} kept in as working cash"),
            ("plus other capital", w.other_capital, "float, obligations, restricted — yours to set"),
            ("= invested capital", w.invested, ""),
        ]
        st.dataframe(
            pd.DataFrame([{"Line": a, "$M": b, "Why": c} for a, b, c in wf])
            .style.format({"$M": "{:,.0f}"}), width="stretch", hide_index=True)
        st.caption(
            f"Long-term operating leases of {money(w.operating_leases)} are **not** subtracted "
            "here, and that is deliberate: Burry's formula takes them out of a total-capital "
            "figure that included them. This capital base is built from equity and borrowings, "
            "which never included them, so subtracting again would count them twice. Restricted "
            f"cash of {money(w.restricted)} stays in the base for the same reason it always "
            "should — it funds the business and you cannot have it.")

        hist = [{"FY": r.fy, "Owners' earnings": r.OE, "Invested capital": r.cap.invested,
                 "ROIC": r.roic if not r.reason else None,
                 "Ex-goodwill": r.tangible_roic if not r.reason else None,
                 "Revenue": r.cap.revenue,
                 "n/a because": r.reason} for r in rows]
        st.write("**Year by year** — the trend matters more than the level")
        st.dataframe(pd.DataFrame(hist).style.format(
            {"Owners' earnings": "{:,.0f}", "Invested capital": "{:,.0f}", "Revenue": "{:,.0f}",
             "ROIC": "{:.1%}", "Ex-goodwill": "{:.1%}"}, na_rep="n/a"),
            width="stretch", hide_index=True)
        good = [r.roic for r in rows if not r.reason and r.roic is not None]
        if good:
            st.caption(
                f"{sum(1 for v in good if v >= 0.15)} of {len(good)} readable years at or above "
                f"15%, {sum(1 for v in good if v >= 0.20)} at or above 20%. Capital is measured "
                "at the year end rather than averaged, which understates the return for anything "
                "growing its asset base quickly — the conservative direction, and the one "
                "Burry's formula reads literally.")

    # ══ Mayer's criteria ═════════════════════════════════════════════
    st.markdown("---")
    st.subheader("Mayer's criteria")

    rev_hist = [pre["revenue"].get(fy) for fy in fys if pre["revenue"].get(fy)]
    rev_cagr = (cagr(rev_hist[0], rev_hist[-1], len(rev_hist) - 1) if len(rev_hist) >= 3 else None)
    oe_years = [y for y in years if not y.excluded]
    oe_cagr = (cagr(oe_years[0].OE, oe_years[-1].OE, oe_years[-1].fy - oe_years[0].fy)
               if len(oe_years) >= 3 and oe_years[0].OE > 0 and oe_years[-1].OE > 0 else None)
    up_years = sum(1 for a, b in zip(rev_hist, rev_hist[1:]) if b > a)

    insider = st.number_input(
        "Insider ownership (%) — from the proxy", value=0.0, step=0.5,
        min_value=0.0, max_value=100.0,
        help="Never tagged in XBRL. It lives in the beneficial ownership table of the DEF 14A, "
             "linked below. Type what you find there and it joins the table.")

    facts_rows = [
        {"Criterion": "Small base",
         "What the filings say": f"{money(mcap)} market cap, {money(rev_now)} revenue"
                                 if mcap > 0 else "no price or share count",
         "Reading": size_band(mcap) if mcap > 0 else "n/a"},
        {"Criterion": "High return on capital",
         "What the filings say": (f"{roic_med:.1%} median over 5 years"
                                  if roic_med is not None else "n/a"),
         "Reading": ("n/a — financial company" if financial else
                     "n/a — " + (latest_r.reason or "capital base unread")
                     if roic_med is None else
                     "high — reinvestment compounds" if roic_med >= 0.20 else
                     "adequate" if roic_med >= 0.12 else
                     "too low to compound from")},
        {"Criterion": "Sustained, not a spike",
         "What the filings say": (f"{sum(1 for r in rows if not r.reason and r.roic and r.roic >= 0.15)}"
                                  f" of {len([r for r in rows if not r.reason])} years above 15%"
                                  if any(not r.reason for r in rows) and not financial else "n/a"),
         "Reading": "the year-by-year table above is the evidence"},
        {"Criterion": "Durable growth",
         "What the filings say": (f"revenue {rev_cagr:.1%}/yr over {len(rev_hist)-1} years, "
                                  f"up in {up_years} of {len(rev_hist)-1}"
                                  if rev_cagr is not None else "too few years of revenue"),
         "Reading": (f"owners' earnings {oe_cagr:.1%}/yr" if oe_cagr is not None
                     else "owners' earnings growth n/a — negative at one end")},
        {"Criterion": "Reasonable entry multiple",
         "What the filings say": (f"{mcap/OE:,.1f}x owners' earnings" if OE > 0 and mcap > 0
                                  else "n/a — no positive owners' earnings"),
         "Reading": ("half the engine, and the half you control at purchase"
                     if OE > 0 else "n/a")},
        {"Criterion": "Owner-operator",
         "What the filings say": "not in XBRL — read the proxy",
         "Reading": "n/a — no filing tags founder involvement"},
        {"Criterion": "Insider ownership",
         "What the filings say": (f"{insider:.1f}% — your figure" if insider > 0
                                  else "not in XBRL — read the proxy"),
         "Reading": ("aligned — Mayer's threshold territory" if insider >= 10 else
                     "some skin in the game" if insider >= 3 else
                     "low, if that is the whole picture" if insider > 0 else "n/a")},
        {"Criterion": "Dilution",
         "What the filings say": (f"{dil_seed:+.1%}/yr share count"
                                  if dil_seed is not None else "history too short"),
         "Reading": ("retiring stock — a tailwind" if dil_seed is not None and dil_seed < 0 else
                     "modest" if dil_seed is not None and dil_seed < 0.02 else
                     "heavy — it compounds against you" if dil_seed is not None else "n/a")},
    ]
    st.dataframe(pd.DataFrame(facts_rows), width="stretch", hide_index=True)

    p1, p2 = st.columns(2)
    if pre.get("proxy"):
        url, date = pre["proxy"]
        p1.markdown(f"[Latest proxy statement (DEF 14A), filed {date}]({url})")
        p1.caption("Beneficial ownership table — insider percentage, founder holdings, and who "
                   "actually controls the vote.")
    else:
        p1.caption("No DEF 14A found. Foreign private issuers do not file proxies; a recent "
                   "listing may not have filed its first one yet.")
    p2.markdown(f"[Insider transactions (Form 4) on EDGAR]"
                f"(https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={pre['cik']}"
                f"&type=4&dateb=&owner=include&count=40)")
    p2.caption(f"{pre['form4']} Form 4 filings in the last twelve months. A count is not a "
               "signal — open them and see whether anyone bought with their own money.")

    st.info(
        "**Two criteria stay blank on purpose.** Owner-operator and insider ownership are not "
        "tagged anywhere in XBRL, and every method for inferring them from structured data is a "
        "guess dressed as a measurement. They are also, in Mayer's account, among the ones that "
        "matter most. The links above go straight to where the answers actually are.")

    # ══ feed to tool 1 ═══════════════════════════════════════════════
    st.markdown("---")
    with st.expander("Feed this into the IV15 tool"):
        st.caption(
            "The Tragic Algebra Analyzer asks for a growth rate and has no way to sanity-check "
            "it — it seeds from revenue, which is why the note there says return on capital is "
            "the real ceiling. This is that ceiling, computed.")
        if roic_med is not None and not financial:
            g_ceiling = sustainable_growth(roic_med, payout)
            st.code(
                f"{tk}\n"
                f"owners' earnings     {OE:,.0f} M\n"
                f"shares               {shares:,.1f} M\n"
                f"ROIC, 5y median      {roic_med:.1%}\n"
                f"cash returned        {payout:.0%} of owners' earnings\n"
                f"growth ceiling       {g_ceiling:.1%}   <- do not exceed this in tool 1\n"
                f"per-share ceiling    {per_share_ceiling(roic_med, payout, buyback_yield):.1%}"
                f"   (includes buybacks)", language="text")
            st.caption(
                f"A growth rate above {g_ceiling:.1%} in tool 1 is a claim that this company will "
                "fund expansion from outside — a rights issue, more debt, or stock. Sometimes "
                "true, always worth stating out loud rather than assuming.")
        else:
            st.warning("No ROIC, so no ceiling. Tool 1's growth input stays unconstrained here.")

    # ══ detail ═══════════════════════════════════════════════════════
    st.markdown("---")
    label = "Notes and detail" + (f" · {len(alerts)} to review" if alerts else "")
    with st.expander(label):
        for kind_, msg in alerts:
            getattr(st, kind_)(msg)

        st.write("**Owners' earnings, year by year**")
        st.dataframe(pd.DataFrame([{
            "FY": f"{y.fy}*" if y.excluded else str(y.fy),
            "Net income": y.N, "GAAP SBC": y.G, "Buybacks": y.T, "Share change": y.dS,
            "Avg price": y.price, "True SBC cost": y.omega, "Owners' earnings": y.OE}
            for y in years]).style.format({
                "Net income": "{:,.0f}", "GAAP SBC": "{:,.0f}", "Buybacks": "{:,.0f}",
                "Share change": "{:+,.1f}", "Avg price": "${:,.2f}",
                "True SBC cost": "{:,.0f}", "Owners' earnings": "{:,.0f}"}, na_rep="—"),
            width="stretch", hide_index=True)
        st.caption("Identical to tool 1's table for the same ticker. If it is not, the two "
                   "engines have drifted apart and the self-test at the foot will say so.")

        st.write("**Assumptions used** — paste this if something looks wrong")
        st.code(
            f"{tk}   price {price:,.2f}   shares {shares:,.1f}M   mkt cap {money(mcap)}\n"
            f"owners' earnings    {OE:,.0f} M\n"
            f"ROIC latest         "
            + (f"{roic_now:.2%}" if roic_now is not None else f"n/a ({latest_r.reason})") + "\n"
            f"ROIC 5y median      "
            + (f"{roic_med:.2%}" if roic_med is not None else "n/a") + "\n"
            f"invested capital    {latest_r.cap.invested:,.0f} M\n"
            f"other expense       {other_expense:,.0f} M   other capital {other_capital:,.0f} M\n"
            f"operating cash      {op_cash_pct:.1%} of revenue\n"
            f"payout              {payout:.1%} of owners' earnings\n"
            f"dilution            {dilution:+.2%}/yr   horizon {horizon}y   "
            f"exit {exit_mult:g}x", language="text")

# ══════════════════════════════════════════════════════════════════════
#  REFERENCE
# ══════════════════════════════════════════════════════════════════════

st.divider()
_r1, _r2 = st.columns(2)
with _r1:
    with st.expander("What the numbers mean"):
        st.markdown(
            "**Required growth** — the annual rate in owners' earnings that turns today's price "
            "into a hundredfold over your holding period, after the multiple you assume at the "
            "end and the dilution along the way.\n\n"
            "**ROIC** — Burry's fully-adjusted return on invested capital. Owners' earnings "
            "stripped of interest income and lease payments, over the capital genuinely at work "
            "once deployable cash is removed and operational cash left in.\n\n"
            "**ROIC ceiling** — return on capital multiplied by the share of earnings retained, "
            "plus the lift from any stock retired. The fastest a business compounds per share "
            "without outside money.\n\n"
            "**Ex-goodwill ROIC** — the same return on tangible capital. It tells you the return "
            "on the next dollar reinvested; the all-in figure tells you how well past "
            "acquisitions were priced.\n\n"
            "**Dilution** — the net annual change in the share count. A hundredfold in the "
            "business is not a hundredfold for you if the count grew the whole way.")

with _r2:
    with st.expander("Verify the engine"):
        st.caption(
            "Two different kinds of check. The Alphabet lines re-run **Burry's published "
            "inputs** through this page's copy of the Tragic Algebra engine and confirm it "
            "still matches tool 1 to the dollar. The rest are arithmetic and wiring tests: "
            "Mayer's own 25-year figure, and Burry's ROIC formula against a hand-worked "
            "example.\n\n"
            "There is no published ROIC to validate against the way Alphabet validates owners' "
            "earnings, so the ROIC test proves the plumbing is right, not that the framework "
            "is. That distinction is worth keeping.")
        if st.button("Run checks"):
            for name, ok, got in self_test():
                st.write(("✅ " if ok else "❌ ") + f"{name} — {got}")

st.caption(
    "Research aid, not financial advice. Outputs depend on estimates you supply. Method follows "
    "Christopher Mayer's published framework and Michael Burry's published ROIC formula; this "
    "project is independent and is not affiliated with or endorsed by either of them.")
