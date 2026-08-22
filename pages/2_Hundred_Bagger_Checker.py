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


# Lines where a second tag means the same thing, so a gap in one can be filled
# from another. Deliberately not net income, stock comp, revenue or share
# counts: their alternates carry different definitions (parent-only versus
# consolidated, per-plan versus total, ASC 606 versus legacy), and stitching
# those across years would put a step change in a growth rate and call it
# history.
FILL_KEYS = {"T", "Cw", "Ce", "DIV", "INT", "LEASEPAY", "CAPEX", "MA", "OFFER", "CONV"}


def _annual(facts: dict, us: list[str], ifrs: list[str],
            sources: list[str] | None = None,
            fill: bool = False) -> dict[int, tuple[str, str, float]]:
    """{fy: (start, end, value)} for full-year facts from annual reports only.

    Three filters that matter: the period must be roughly a year (so quarterly
    rows tagged fp='FY' cannot slip through); annual forms only; and where a
    year appears in several filings, keep the latest — a 10-K restates the
    prior year as a comparative.

    On the choice between concepts: the first one with data used to win
    outright, and everything after it was ignored. That is right when the
    alternates mean different things, and quietly wrong when they do not.
    H&R Block retired shares rather than holding them as treasury, so
    PaymentsForRepurchaseOfCommonStock covered three of nineteen years and the
    remaining sixteen sat in StockRepurchasedAndRetiredDuringPeriodValue —
    unread, because three years was enough to stop the search. The buyback
    column printed zeros for a decade and every figure downstream inherited it.

    With fill=True the concepts are tried in order and later ones fill only the
    years earlier ones left empty. Priority is preserved; nothing is summed;
    and every concept that contributed is appended to `sources` so the panel in
    the UI can show which ones answered.
    """
    out: dict[int, tuple[str, str, str, float]] = {}
    for taxonomy, concepts in (("us-gaap", us), ("ifrs-full", ifrs)):
        tax = facts.get("facts", {}).get(taxonomy, {})
        for concept in concepts:
            if concept not in tax:
                continue
            units = tax[concept].get("units", {})
            got: dict[int, tuple[str, str, str, float]] = {}
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
                if fy not in got or filed > got[fy][0]:
                    got[fy] = (filed, start, end, float(row.get("val", 0.0)))
            fresh = {fy: v for fy, v in got.items() if fy not in out}
            if fresh:
                out.update(fresh)
                if sources is not None:
                    sources.append(concept)
            if out and not fill:
                return {k: (v[1], v[2], v[3]) for k, v in out.items()}
    return {k: (v[1], v[2], v[3]) for k, v in out.items()}


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


def pool_safe(years: list[Year], fallback: "Pooled") -> "Pooled":
    """pool() over a slice that may be empty or sum to zero net income."""
    try:
        return pool(years)
    except (ValueError, ZeroDivisionError):
        return fallback


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
    # Two switches, both off by default, which together are the difference
    # between this number and the one on a data provider's website. Neither
    # is a correction: they are different questions about the same filings.
    include_leases: bool = False   # capitalised office and store networks
    cash_in_base: bool = False     # leave cash in, as most providers do

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
        if self.cash_in_base:
            return 0.0
        return max(0.0, self.cash - self.op_cash_need)

    @property
    def total_capital(self) -> float:
        base = self.equity + self.debt + self.finance_leases
        # A leased office network is capital at work whatever the accounting
        # calls it. Burry's formula removes operating leases from a
        # total-capital figure that already contained them; this base never
        # did, so the switch adds them rather than subtracting. For a filer
        # with thousands of leased locations this is the difference between
        # a return on almost nothing and a return on the actual footprint.
        return base + (self.operating_leases if self.include_leases else 0.0)

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
    "T":  (["PaymentsForRepurchaseOfCommonStock", "PaymentsForRepurchaseOfEquity",
            "PaymentsForRepurchaseOfCommonStockAndRestrictedStockUnits",
            "StockRepurchasedAndRetiredDuringPeriodValue",
            "StockRepurchasedDuringPeriodValue"],
           ["PaymentsToAcquireOrRedeemEntitysShares"]),
    "Cw": (["PaymentsRelatedToTaxWithholdingForShareBasedCompensation",
            "TreasuryStockValueAcquiredCostMethod"], []),
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


TAG_LABELS = {
    "N": "Net income", "G": "GAAP stock comp", "T": "Buybacks",
    "Cw": "Tax withheld on vesting", "Ce": "Option / ESPP proceeds",
    "REV": "Revenue", "INT": "Interest income", "LEASEPAY": "Finance lease payments",
    "DIV": "Dividends paid", "CAPEX": "Capital expenditure",
    "MA": "Shares issued for acquisitions", "OFFER": "Shares issued in offerings",
    "CONV": "Shares from conversions",
}


def tag_report(facts: dict, series: dict, sources: dict[str, list[str]]) -> list[dict]:
    """Which tags answered for each line, and how many years they covered.

    Every silent zero in this app is a tag that did not match. Reading the
    panel is how the H&R Block buyback bug was found: the tag was there, it
    covered three years of nineteen, and nothing said so.
    """
    rows = []
    for key, (us, ifrs) in CONCEPTS.items():
        used = sources.get(key, [])
        n = len(series.get(key, {}))
        present = ""
        for taxonomy, concepts in (("us-gaap", us), ("ifrs-full", ifrs)):
            for c in concepts:
                if c in facts.get("facts", {}).get(taxonomy, {}):
                    present = c
                    break
            if present:
                break
        rows.append({
            "Line": TAG_LABELS.get(key, key),
            "Years read": n,
            "XBRL tag": " + ".join(used) if used else (present or "—"),
            "Status": ("read" if n and len(used) <= 1 else
                       f"read — gaps filled from {len(used)} tags" if n else
                       "tag present but no annual figures survived the filters" if present else
                       "none of the tags this reader knows are in the filing"),
        })
    return rows


def load(ticker: str, n_years: int = 10):
    """Everything this page needs, in one pass over the filings."""
    cmap = _ticker_map()
    if ticker not in cmap:
        raise ValueError(f"'{ticker}' is not in the SEC company list.")
    cik = cmap[ticker]
    facts = _facts(cik)
    subs = _submissions(cik)
    sic, sic_desc = str(subs.get("sic", "")), str(subs.get("sicDescription", ""))

    tag_sources: dict[str, list[str]] = {k: [] for k in CONCEPTS}
    series = {k: _annual(facts, us, ifrs, tag_sources[k], k in FILL_KEYS)
              for k, (us, ifrs) in CONCEPTS.items()}
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
    if "TreasuryStockValueAcquiredCostMethod" in tag_sources.get("Cw", []):
        capped = 0
        for y in years:
            if y.G > 0 and y.Cw > 3 * y.G:
                y.Cw, capped = 0.0, capped + 1
        if capped:
            notes.append(
                f"A treasury-stock line was read as tax withholding, but in {capped} year(s) it "
                "was more than three times the GAAP stock-comp charge — too large to be shares "
                "surrendered for employee tax, and almost certainly an ordinary repurchase. Those "
                "years were dropped rather than charged twice, once as withholding and again as "
                "the market value of shares delivered.")
        else:
            notes.append(
                "Tax withholding was read from a treasury-stock line rather than the usual "
                "withholding tag. Filers that retire shares on repurchase report it this way. "
                "The amounts are withholding-sized, so they were accepted.")

    # A blank year inside the window is invisible in a table full of numbers.
    # H&R Block read 12 of 19 years after the gap-filling fix, and the four
    # blanks that remained sat in the middle of the window while the share
    # count fell in every one of them. Nothing said so.
    _gap = [y.fy for y in years
            if y.fy not in series["T"] and y.dS < 0
            and shares_out.get(y.fy - 1, 0) > 0
            and abs(y.dS) / (shares_out[y.fy - 1] / 1e6) > 0.01]
    if _gap:
        notes.append(
            "No repurchase figure was found for FY"
            + ", FY".join(str(f) for f in _gap)
            + ", yet the share count fell by more than 1% in each. Those years are almost "
              "certainly buybacks tagged under an element this reader does not know. Two "
              "consequences: owners' earnings for those years are a ceiling, since the market "
              "value of shares delivered floors at zero without a repurchase figure; and cash "
              "returned to shareholders is understated, which flatters the growth a company "
              "looks able to fund. The tag panel shows which elements did answer.")

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
    # A fiscal year ending outside November-January means the balance sheet is
    # a snapshot taken at a busy point in the company's own cycle. H&R Block
    # closes on 30 April, days after tax season delivers its cash.
    fye_month = int(series["N"][fys[-1]][1][5:7]) if fys else 12
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
        "cik": str(int(cik)), "fye_month": fye_month,
        "tags": tag_report(facts, series, tag_sources),
    }
    return years, notes, pre


def build_roic(years: list[Year], pre: dict, op_cash_pct: float,
               other_expense: float, other_capital: float,
               include_leases: bool = False, cash_in_base: bool = False) -> list[RoicYear]:
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
        cap.include_leases, cap.cash_in_base = include_leases, cash_in_base
        out.append(RoicYear(
            fy=y.fy, OE=y.OE,
            interest_income=pre["interest"].get(y.fy, 0.0),
            lease_payments=pre["leasepay"].get(y.fy, 0.0),
            other_expense=other_expense if y.fy == last else 0.0,
            cap=cap, excluded=y.excluded))
    return out


# ══════════════════════════════════════════════════════════════════════
#  THE VERDICT
#
#  One question, asked in one place: is the growth this price requires
#  inside what the business can fund and has ever managed?
#
#  Three rates, and the answer is which of them is smallest:
#    NEEDS      required_growth, from the price you pay and the multiple
#               you assume at the end
#    CAN FUND   ROIC x retention, plus whatever a shrinking share count
#               adds. Capital's ceiling.
#    HAS DONE   what it has actually delivered. History's ceiling.
#
#  Neither ceiling is a law. Capital's can be raised by borrowing;
#  history's can be broken by a genuinely new business. But when the
#  required rate sits above both, the case is closed by arithmetic
#  rather than by opinion, and that is worth saying plainly.
# ══════════════════════════════════════════════════════════════════════


@dataclass
class Verdict:
    label: str      # short headline
    kind: str       # streamlit method: success / warning / error / info
    why: str        # which constraint decided it


def assess(required: float | None, fundable: float | None,
           delivered: float | None, mcap_m: float) -> Verdict:
    """The whole tool, in one function, so it can be tested without a browser."""
    if mcap_m > 0 and mcap_m * 100 > WORLD_GDP_M * 0.05:
        return Verdict("Closed on size", "error", "size")
    if required is None:
        return Verdict("Cannot be computed", "error", "no earnings base")
    if fundable is None and delivered is None:
        return Verdict("Cannot be computed", "error", "no ceiling of either kind")
    if fundable is not None and required > fundable:
        return Verdict("The arithmetic does not close", "error", "capital")
    # Fundable, but the company has never gone at anything like this rate. Both
    # ceilings are generous here — delivered takes the better of revenue and
    # owners' earnings — so failing this one is a real finding.
    if delivered is not None and required > max(delivered * 1.5, delivered + 0.03):
        return Verdict("Fundable, but unprecedented", "warning", "history")
    if fundable is None:
        return Verdict("Open, on history alone", "info", "no capital base")
    if delivered is None:
        return Verdict("Open, on capital alone", "info", "no growth history")
    return Verdict("The arithmetic is open", "success", "both")


@dataclass
class Payout:
    """What a company hands back, kept in pieces.

    A single ratio was the wrong output here. H&R Block came back as "retains
    70%" — which was dividends alone, with several hundred million a year of
    buybacks missing, because one XBRL tag did not match. A ratio cannot show
    you that; its components can.
    """
    dividends: float = 0.0        # $M a year, averaged
    buybacks: float = 0.0         # as filed
    implied: float = 0.0          # from shares retired x price: a floor
    oe: float = 0.0
    used_implied: bool = False

    @property
    def returned(self) -> float:
        return self.dividends + (self.implied if self.used_implied else self.buybacks)

    @property
    def ratio(self) -> float | None:
        return self.returned / self.oe if self.oe > 0 else None

    @property
    def warning(self) -> str:
        if not self.used_implied:
            return ""
        return (f"No repurchase line was found in the filings, but the share count fell by about "
                f"\\${self.implied:,.0f}M a year at market prices. The payout below uses that "
                "implied figure instead of zero — it is a floor, since shares issued to "
                "employees offset some of what was bought.")


def pooled_payout(years: list[Year], dividends: dict[int, float], n: int = 5) -> Payout:
    """Cash returned to shareholders, pooled over several years.

    Pooled rather than taken from the latest year for the same reason ΔE is
    pooled in tool 1: one big repurchase is a decision, not a policy.

    Buybacks appear on both sides of the ceiling and that is not double
    counting. They leave the numerator because a dollar returned is a dollar
    not reinvested; they come back as a share-count effect because a
    hundredfold on fewer shares is still a hundredfold to whoever stayed.
    """
    clean = [y for y in years if not y.excluded][-n:]
    if not clean:
        return Payout()
    k = len(clean)
    p = Payout(
        dividends=sum(dividends.get(y.fy, 0.0) for y in clean) / k,
        buybacks=sum(y.T for y in clean) / k,
        # Shares retired, valued at that year's average price. dS is already net
        # of issuance, so this understates gross repurchases — a floor, which is
        # the right direction for a fallback.
        implied=sum(max(0.0, -y.dS) * y.price for y in clean) / k,
        oe=sum(y.OE for y in clean) / k,
    )
    # Only step in when the filed figure is not merely lower but absent, and the
    # share count says real money was spent. A company that genuinely does not
    # repurchase has an implied figure of zero and is untouched.
    if p.implied > 0 and p.buybacks < 0.25 * p.implied:
        p.used_implied = True
    return p


def roic_caveat(r: RoicYear, fye_month: int) -> str:
    """One line on how solid a ROIC reading is, instead of a warning box.

    Everything here was, at some point, a place the number went wrong. None of
    them makes it worthless; all of them change how hard you should lean on it.
    """
    c, bits = r.cap, []
    if r.roic is not None and r.roic > 0.50:
        bits.append("at this level capital is not what limits growth — the payout is, and the "
                    "ceiling already reflects that")
    if c.total_capital > 0 and c.deployable_cash > 0.5 * c.total_capital:
        bits.append("over half the gross capital base is cash being subtracted, so the answer "
                    "moves a long way with the operating-cash setting")
    if c.invested > 0 and (c.goodwill + c.intangibles) > 0.5 * c.invested:
        bits.append("acquisitions are most of the base, so the tangible figure is the one that "
                    "describes reinvestment")
    if fye_month not in (11, 12, 1):
        bits.append(f"the fiscal year ends in month {fye_month}, so the balance sheet is a "
                    "snapshot taken mid-cycle rather than at a quiet point")
    return "; ".join(bits)


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
    # The seed both pages show: latest net income x 3-year pooled ΔE.
    seed = ys[-1].N * pool_safe(ys[-3:], pool(ys)).dE
    out.append(("Owners' earnings seed matches tool 1's box: $117,743M",
                abs(seed - 117_743) < 500, f"${seed:,.0f}M"))

    # 2. Mayer's arithmetic.
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

    # 3. The ceiling. The second of these is the H&R Block shape: a superb
    #    return on capital that funds no growth at all, because essentially
    #    every dollar is paid out.
    out.append(("ROIC 20%, half paid out → 10% ceiling",
                abs(sustainable_growth(0.20, 0.5) - 0.10) < 1e-9,
                f"{sustainable_growth(0.20, 0.5):.1%}"))
    hrb = per_share_ceiling(1.05, 1.0, 0.07)
    out.append(("ROIC 105% but everything paid out → 7.5%, not 105%",
                abs(hrb - 0.0753) < 0.001, f"{hrb:.2%}"))

    # 4. Burry's ROIC, wired to a hand-worked example.
    c = Capital(fy=2025, equity=800, debt=200, cash=250, revenue=2500,
                op_cash_pct=0.02, other_capital=50)
    r = RoicYear(fy=2025, OE=100, interest_income=5, lease_payments=2, other_expense=3, cap=c)
    out.append(("ROIC formula wiring: 90 / 850 → 10.59%",
                r.roic is not None and abs(r.roic - 0.10588) < 0.0005,
                f"{r.roic:.2%}" if r.roic else "n/a"))
    out.append(("Operating cash stays in the capital base",
                abs(c.deployable_cash - 200) < 1e-9, f"${c.deployable_cash:,.0f}M deployable"))
    neg = RoicYear(fy=2025, OE=500, interest_income=0, lease_payments=0, other_expense=0,
                   cap=Capital(fy=2025, equity=-300, debt=100, cash=50, revenue=5000,
                               equity_found=True))
    out.append(("Negative capital base refuses rather than flipping sign",
                neg.reason != "" and neg.roic is None, neg.reason or "printed a number"))

    # 4b. The payout cross-check. HRB shaped: dividends tagged, buybacks not,
    #     share count visibly shrinking.
    hrb = [Year(fy=f, N=600, G=70, T=0.0, dS=-8.0, price=55.0) for f in range(2022, 2027)]
    pz = pooled_payout(hrb, {f: 190.0 for f in range(2022, 2027)}, 5)
    out.append(("Missing buyback line caught by the share count",
                pz.used_implied and abs(pz.implied - 440) < 1,
                f"implied ${pz.implied:,.0f}M/yr vs ${pz.buybacks:,.0f}M filed"))
    out.append(("...and payout rises from 28% to 94%",
                pz.ratio is not None and abs(pz.ratio - 0.9403) < 0.005,
                f"{pz.ratio:.1%}, filed-only would be {190/670:.0%}"))
    real = pooled_payout([Year(fy=f, N=600, G=70, T=400.0, dS=-8.0, price=55.0)
                          for f in range(2022, 2027)], {}, 5)
    out.append(("A filed buyback line is left alone",
                not real.used_implied, f"used filed ${real.buybacks:,.0f}M"))

    # 4c. The capital-base switches, on the H&R Block shape: they should carry
    #     a triple-digit Burry ROIC down toward what a data provider publishes.
    base = Capital(fy=2026, equity=100, debt=1500, cash=900, operating_leases=250,
                   revenue=3600, equity_found=True)
    r_burry = RoicYear(fy=2026, OE=770, interest_income=0, lease_payments=0,
                       other_expense=0, cap=base)
    conv = Capital(**{**base.__dict__, "include_leases": True, "cash_in_base": True})
    r_conv = RoicYear(fy=2026, OE=770, interest_income=0, lease_payments=0,
                      other_expense=0, cap=conv)
    out.append(("Switches reconcile a Burry ROIC toward a published one",
                r_burry.roic > 0.90 and r_conv.roic < 0.50,
                f"{r_burry.roic:.0%} -> {r_conv.roic:.0%}"))

    # 4d. The bug the tag panel found: a tag that answers for a few years and
    #     stops the search. Built to the exact shape of H&R Block's buybacks.
    def _rows(concept, yrs, val):
        return {concept: {"units": {"USD": [
            {"form": "10-K", "start": f"{y}-07-01", "end": f"{y+1}-06-30",
             "filed": f"{y+1}-08-01", "val": val} for y in yrs]}}}
    facts = {"facts": {"us-gaap": {
        **_rows("PaymentsForRepurchaseOfCommonStock", range(2008, 2011), 100e6),
        **_rows("StockRepurchasedAndRetiredDuringPeriodValue", range(2008, 2026), 500e6)}}}
    first_wins = _annual(facts, CONCEPTS["T"][0], [], None, False)
    filled = _annual(facts, CONCEPTS["T"][0], [], None, True)
    out.append(("First-tag-wins read 3 years and stopped",
                len(first_wins) == 3, f"{len(first_wins)} years"))
    out.append(("Gap-filling reads all 18, priority preserved",
                len(filled) == 18 and abs(filled[2009][2] - 100e6) < 1
                and abs(filled[2020][2] - 500e6) < 1,
                f"{len(filled)} years; 2009 from the cash-flow tag, 2020 from the retirement tag"))
    src = []
    _annual(facts, CONCEPTS["T"][0], [], src, True)
    out.append(("...and both tags are named in the panel", len(src) == 2, " + ".join(src)))

    # 5. The verdict itself.
    v = assess(0.272, 0.075, 0.04, 5_000)
    out.append(("Needs 27%, funds 7.5% → closed by capital",
                v.kind == "error" and v.why == "capital", f"{v.label} ({v.why})"))
    v = assess(0.27, 0.35, 0.10, 5_000)
    out.append(("Fundable at 35% but has only done 10% → unprecedented",
                v.kind == "warning" and v.why == "history", f"{v.label} ({v.why})"))
    v = assess(0.22, 0.30, 0.20, 5_000)
    out.append(("Needs 22%, funds 30%, has done 20% → open",
                v.kind == "success", f"{v.label} ({v.why})"))
    v = assess(0.15, 0.90, 0.60, 4_000_000)
    out.append(("Size closes it even when everything else passes",
                v.why == "size", f"{v.label} ({v.why})"))
    return out


# ══════════════════════════════════════════════════════════════════════
#  UI
# ══════════════════════════════════════════════════════════════════════
#
# NOTE ON DOLLAR SIGNS: Streamlit markdown parses $...$ as LaTeX, so any
# literal dollar amount inside st.write/markdown/success/error/info/warning
# must be escaped. st.metric, st.code and st.dataframe are unaffected.


def money(x: float) -> str:
    """$M in, human-readable out. Escaped, so it is safe inside markdown."""
    if abs(x) >= 1_000_000:
        return f"\\${x/1_000_000:,.2f}T"
    if abs(x) >= 1_000:
        return f"\\${x/1_000:,.2f}B"
    return f"\\${x:,.0f}M"


def plain(x: float) -> str:
    """Same, unescaped, for st.metric and st.code."""
    return money(x).replace("\\", "")


st.set_page_config(
    page_title="100-Bagger Checker — Mayer's criteria and Burry's ROIC",
    page_icon="💯",
    layout="centered",
    initial_sidebar_state="collapsed",
)
st.title("💯 100-Bagger Checker")
st.caption("What a hundredfold requires, against what this business can fund and has ever done")

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
        "against two ceilings — what the company's return on capital can fund, and what it has "
        "ever delivered. Below both, the case is open. Above either, it is closed by arithmetic "
        "rather than by opinion.\n\n"
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
    fys, financial = pre["fys"], pre["financial"]
    alerts: list[tuple[str, str]] = [("info", n) for n in notes]
    latest = years[-1]

    # ── owners' earnings, seeded exactly as tool 1 seeds its box ─────
    # Both pages run the same engine on the same filings. What differed was
    # which figure got shown: tool 1 puts forward net income x pooled ΔE in
    # its box, this page was showing the latest year as filed. Same formula,
    # different year, and the two disagreed by whatever that year was unusual
    # by. They now seed identically.
    pooled = pool(years)
    recent = pool_safe(years[-3:], pooled)
    use_dE = recent.dE if 0 < recent.dE <= 1.25 else pooled.dE
    hist_oe = sorted(y.OE for y in years[-5:] if not y.excluded)
    median_OE = hist_oe[len(hist_oe) // 2] if hist_oe else 0.0
    if 0 < use_dE <= 1.25:
        seed_OE = latest.N * use_dE
    elif median_OE > 0:
        seed_OE = median_OE
    else:
        seed_OE = latest.N

    # ══ inputs ═══════════════════════════════════════════════════════
    st.markdown("---")
    st.subheader("Inputs")

    c1, c2, c3 = st.columns(3)
    price = c1.number_input("Price", value=float(current_price(tk) or 100.0), step=0.01)
    shares = c2.number_input("Diluted shares (M)", value=float(round(pre["shares"], 1)), step=1.0,
                             help="Everything per-share divides by this. Check it against the "
                                  "market cap below — a missed second share class is the most "
                                  "common reading error there is.")
    OE = c3.number_input(
        "Owners' earnings ($M)", value=float(round(seed_OE, 1)), step=1.0,
        help="Seeded exactly as the IV15 tool seeds its box: latest net income times pooled ΔE. "
             "Both pages should show the same figure for the same ticker.")
    mcap = shares * price

    c4, c5, c6 = st.columns(3)
    horizon = c4.slider("Holding period (years)", 5, 30, 20, 1,
                        help="Mayer's study runs on 25-year holds. The requirement is brutally "
                             "non-linear in this: five years less can add ten points a year.")
    exit_default = (mcap / OE) if OE > 0 else 20.0
    exit_mult = c5.number_input(
        "Exit multiple", value=float(round(min(max(exit_default, 3.0), 60.0), 1)), step=0.5,
        min_value=1.0,
        help="What the market pays for a dollar of owners' earnings at the end. Seeded flat at "
             "today's multiple, which is the honest default — assuming expansion is where most "
             "of the wishful thinking in this arithmetic hides.")
    dil_seed = pre.get("dilution")
    dilution = c6.number_input(
        "Share issuance (%/yr)",
        value=float(round((dil_seed if dil_seed is not None else 0.015) * 100, 1)), step=0.1,
        help="Measured from the actual share count over the window, capital events excluded. "
             "Negative means the count is shrinking. A straight drag on your per-share "
             "result, compounding for the whole holding period.") / 100.0
    st.caption(
        f"Market cap {money(mcap)} · a hundredfold is {money(mcap*100)}"
        + (f" · trading at {mcap/OE:,.1f}x owners' earnings" if OE > 0 else "")
        + ("" if dil_seed is not None else
           " · share-count history too short to measure dilution, so 1.5% is a placeholder"))

    with st.expander("Judgement inputs — the parts EDGAR cannot answer"):
        st.caption(
            "Burry's ROIC has two terms that are not tagged anywhere in XBRL, so they are seeded "
            "at zero and left to you. Zero is not neutral: it makes the return read high.\n\n"
            "**Other expense** — forensic depreciation and amortisation, a normalised tax rate, "
            "cyclical adjustment.\n\n"
            "**Other capital** — what funds the business without appearing as capital: purchase "
            "obligations, customer float, restricted cash, loans held for settlement.\n\n"
            "**Operating cash** — the split between cash the business needs and cash you could "
            "actually have. Only the second comes out of the capital base. Burry publishes no "
            "percentage; 2% of revenue is this tool's convention, not his figure.")
        j1, j2, j3 = st.columns(3)
        other_expense = j1.number_input("Other expense ($M)", value=0.0, step=1.0)
        other_capital = j2.number_input("Other capital ($M)", value=0.0, step=1.0)
        op_cash_pct = j3.number_input("Operating cash (% of revenue)", value=2.0, step=0.5,
                                      min_value=0.0, max_value=25.0) / 100.0

        st.caption(
            "**Reconciling with the websites.** A ROIC here will usually read higher than "
            "GuruFocus, Stockanalysis or Morningstar, and two choices account for most of the "
            "gap. Burry takes deployable cash out of the capital base; almost no provider does. "
            "And this base is equity plus borrowings, which leaves a leased store or office "
            "network out of capital altogether. Turn both on to see roughly the number they "
            "publish, then decide which question you wanted answered.")
        k1, k2 = st.columns(2)
        include_leases = k1.checkbox(
            "Count long-term operating leases as capital", value=False,
            help="For a company with thousands of leased locations, the lease obligation is the "
                 "asset base. Leaving it out flatters the return enormously.")
        cash_in_base = k2.checkbox(
            "Leave cash in the capital base", value=False,
            help="What most data providers do. Turn it on to compare like with like; turn it off "
                 "to follow Burry.")

    # ── the three rates ──────────────────────────────────────────────
    rows = build_roic(years, pre, op_cash_pct, other_expense, other_capital,
                      include_leases, cash_in_base)
    rows[-1].OE = OE                      # latest year follows the box above
    rows[-1].other_expense = other_expense
    latest_r = rows[-1]
    roic_med = None if financial else median_roic(rows, 5)

    pay = pooled_payout(years, pre["dividends"], 5)
    payout = pay.ratio
    buyback_yield = (pay.implied if pay.used_implied else pay.buybacks) / mcap if mcap > 0 else 0.0
    fundable = (per_share_ceiling(roic_med, payout, buyback_yield)
                if roic_med is not None and payout is not None else None)

    rev_hist = [pre["revenue"][fy] for fy in fys if pre["revenue"].get(fy)]
    rev_cagr = cagr(rev_hist[0], rev_hist[-1], len(rev_hist) - 1) if len(rev_hist) >= 3 else None
    oe_clean = [y for y in years if not y.excluded]
    oe_cagr = (cagr(oe_clean[0].OE, oe_clean[-1].OE, oe_clean[-1].fy - oe_clean[0].fy)
               if len(oe_clean) >= 3 else None)
    # Deliberately generous: the better of the two. A refusal that survives the
    # kindest reading of the history is a refusal worth trusting.
    delivered = max([g for g in (rev_cagr, oe_cagr) if g is not None], default=None)

    required = (required_growth(mcap / OE, exit_mult, horizon, 100.0, dilution)
                if OE > 0 and mcap > 0 else None)
    v = assess(required, fundable, delivered, mcap)

    # ══ verdict ══════════════════════════════════════════════════════
    st.markdown("---")
    st.subheader(f"Verdict · {tk}")

    m1, m2, m3 = st.columns(3)
    m1.metric("Needs", f"{required:.1%}" if required is not None else "—",
              f"100x in {horizon}y at {exit_mult:g}x")
    m2.metric("Has delivered", f"{delivered:.1%}" if delivered is not None else "n/a",
              (f"revenue {rev_cagr:.0%}" if rev_cagr is not None else "revenue n/a")
              + (f" · OE {oe_cagr:.0%}" if oe_cagr is not None else ""))
    m3.metric("Can fund", f"{fundable:.1%}" if fundable is not None else "n/a",
              (f"ROIC {roic_med:.0%} · retains {max(0.0,1-payout):.0%}"
               if fundable is not None else
               "financial company" if financial else "capital base unread"))

    if pay.warning:
        st.warning("**Check this before trusting the ceiling.** " + pay.warning)
    if payout is not None:
        st.caption(
            f"**Can fund** is {money(pay.oe)} of owners' earnings a year, less the "
            f"{money(pay.returned)} handed back — {money(pay.dividends)} of dividends and "
            f"{money(pay.implied if pay.used_implied else pay.buybacks)} of buybacks — leaving "
            f"{max(0.0, 1-payout):.0%} retained, reinvested at "
            + (f"{roic_med:.0%}" if roic_med is not None else "an unreadable return")
            + ". It is an upper bound, not a forecast: it assumes every retained dollar finds a "
              "project as good as the business already is. A very high return on capital usually "
              "means the business needs little capital, which is also a reason there may be "
              "nowhere to put more of it.")

    if v.why == "size":
        st.error(
            f"**{v.label}.** A hundredfold on {money(mcap)} is {money(mcap*100)}, against a world "
            "economy of roughly \\$110 trillion. Mayer's whole point is that the base has to be "
            "small enough for the arithmetic to have somewhere to go. Nothing below rescues this.")
    elif v.why == "no earnings base":
        st.error(
            f"**{v.label}.** Owners' earnings are {money(OE)}, and Mayer's arithmetic is a ratio "
            "of what you pay to what the business earns. A hundredfold from a loss is not a "
            "calculation, it is a story about a recovery. Enter the owners' earnings you believe "
            "the business reaches in a normal year and this page will mean something.")
    elif v.why == "capital":
        st.error(
            f"**{v.label}.** A hundredfold in {horizon} years needs {required:.1%} a year in "
            f"owners' earnings. This business funds about {fundable:.1%}"
            + (f" — it earns {roic_med:.0%} on capital but returns {payout:.0%} of its earnings "
               "to shareholders, so the compounding happens in your dividend and your share of "
               "the company, not in the earnings themselves."
               if payout is not None and payout > 0.6 else
               f" from a {roic_med:.0%} return on capital with {max(0.0,1-payout):.0%} retained.")
            + " The gap would have to come from outside — debt, which runs out, or stock, which "
              "is the dilution you already set. Raising the exit multiple is the only other "
              "lever, and that is a bet on the market, not on the business.")
    elif v.why == "history":
        st.warning(
            f"**{v.label}.** {required:.1%} a year is inside the {fundable:.1%} its capital could "
            f"fund, but the company has delivered {delivered:.1%} — and that is the kinder of its "
            "revenue and owners'-earnings rates. It has never grown at the rate this price "
            "requires. Not impossible; it is a bet on an inflection, and you should be able to "
            "say what causes it.")
    elif v.kind == "success":
        st.success(
            f"**{v.label}.** {required:.1%} a year sits inside both ceilings — the {fundable:.1%} "
            f"its capital funds and the {delivered:.1%} it has delivered. That makes a "
            "hundredfold possible, not likely. Everything now turns on how long the return on "
            "capital and the runway last, which no filing can tell you.")
    else:
        st.info(
            f"**{v.label}.** {required:.1%} a year, checked against only one ceiling — "
            + ("no growth history long enough to measure." if v.why == "no growth history" else
               "the capital base could not be read, so there is no funding ceiling to test.")
            + " Treat this page as incomplete for this company.")

    # ══ criteria ═════════════════════════════════════════════════════
    st.markdown("---")
    st.subheader("Mayer's criteria")

    insider = st.number_input(
        "Insider ownership (%) — from the proxy", value=0.0, step=0.5,
        min_value=0.0, max_value=100.0,
        help="Never tagged in XBRL. It lives in the beneficial ownership table of the DEF 14A, "
             "linked below. Type what you find there and it joins the table.")

    up_years = sum(1 for a, b in zip(rev_hist, rev_hist[1:]) if b > a)
    readable = [r for r in rows if not r.reason]
    facts_rows = [
        {"Criterion": "Small base",
         "Filings": f"{plain(mcap)} cap, {plain(pre['revenue'].get(fys[-1], 0.0))} revenue"
                    if mcap > 0 else "no price or share count",
         "Reading": size_band(mcap) if mcap > 0 else "n/a"},
        {"Criterion": "Return on capital",
         "Filings": f"{roic_med:.1%} median, 5 years" if roic_med is not None else "n/a",
         "Reading": ("n/a — financial company" if financial else
                     "n/a — " + (latest_r.reason or "capital base unread") if roic_med is None
                     else "high — reinvestment compounds" if roic_med >= 0.20
                     else "adequate" if roic_med >= 0.12 else "too low to compound from")},
        {"Criterion": "Reinvests it",
         "Filings": f"returns {payout:.0%} of owners' earnings" if payout is not None else "n/a",
         "Reading": ("n/a — owners' earnings negative over the window" if payout is None
                     else "distributes nearly everything — a payer, not a compounder"
                     if payout > 0.8 else "retains most of it" if payout < 0.35 else "mixed")},
        {"Criterion": "Sustained, not a spike",
         "Filings": (f"{sum(1 for r in readable if r.roic and r.roic >= 0.15)} of {len(readable)}"
                     " years above 15%" if readable and not financial else "n/a"),
         "Reading": "the year-by-year table is the evidence"},
        {"Criterion": "Durable growth",
         "Filings": (f"revenue {rev_cagr:.1%}/yr, up in {up_years} of {len(rev_hist)-1}"
                     if rev_cagr is not None else "too few years of revenue"),
         "Reading": (f"owners' earnings {oe_cagr:.1%}/yr" if oe_cagr is not None
                     else "owners' earnings growth n/a — negative at one end")},
        {"Criterion": "Entry multiple",
         "Filings": f"{mcap/OE:,.1f}x owners' earnings" if OE > 0 and mcap > 0 else "n/a",
         "Reading": "half the engine, and the half you control at purchase"},
        {"Criterion": "Dilution",
         "Filings": f"{dil_seed:+.1%}/yr share count" if dil_seed is not None
                    else "history too short",
         "Reading": ("retiring stock — a tailwind" if dil_seed is not None and dil_seed < 0 else
                     "modest" if dil_seed is not None and dil_seed < 0.02 else
                     "heavy — it compounds against you" if dil_seed is not None else "n/a")},
        {"Criterion": "Owner-operator",
         "Filings": "not in XBRL — read the proxy",
         "Reading": "n/a — no filing tags founder involvement"},
        {"Criterion": "Insider ownership",
         "Filings": f"{insider:.1f}% — your figure" if insider > 0 else "not in XBRL",
         "Reading": ("aligned" if insider >= 10 else "some skin in the game" if insider >= 3
                     else "low, if that is the whole picture" if insider > 0 else "n/a")},
    ]
    st.dataframe(pd.DataFrame(facts_rows), width="stretch", hide_index=True)

    p1, p2 = st.columns(2)
    if pre.get("proxy"):
        url, date = pre["proxy"]
        p1.markdown(f"[Proxy statement (DEF 14A), filed {date}]({url})")
        p1.caption("Beneficial ownership table — insider percentage, founder holdings, who "
                   "controls the vote.")
    else:
        p1.caption("No DEF 14A found. Foreign private issuers do not file proxies; a recent "
                   "listing may not have filed its first one yet.")
    p2.markdown(f"[Insider transactions (Form 4)]"
                f"(https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={pre['cik']}"
                f"&type=4&dateb=&owner=include&count=40)")
    p2.caption(f"{pre['form4']} filings in the last twelve months. A count is not a signal — "
               "open them and see whether anyone bought with their own money.")

    # ══ detail, all folded away ══════════════════════════════════════
    st.markdown("---")

    with st.expander("What it would take at other exit multiples"):
        if required is not None:
            mult_now = mcap / OE
            grid = []
            for h in (10, 15, 20, 25):
                row = {"Held for": f"{h} years"}
                for m in (10.0, 15.0, 20.0, 25.0, 30.0):
                    g = required_growth(mult_now, m, h, 100.0, dilution)
                    row[f"{m:g}x"] = ("—" if g is None else
                                      f"{g:.0%} ✗" if fundable is not None and g > fundable
                                      else f"{g:.0%}")
                grid.append(row)
            st.dataframe(pd.DataFrame(grid), width="stretch", hide_index=True)
            st.caption(
                f"Required growth in owners' earnings. Today's multiple is {mult_now:,.1f}x and "
                f"{dilution:.1%} annual issuance is included. "
                + (f"A ✗ marks a rate above the {fundable:.1%} this business can fund. "
                   if fundable is not None else "")
                + "Notice how much work the exit multiple does — buying cheap and selling dear "
                  "is half of Mayer's engine, and it is the half you fix at purchase.")
        else:
            st.caption("Needs a positive owners' earnings figure.")

    with st.expander("Return on invested capital — Burry's formula, line by line"):
        if financial:
            st.error(
                f"**Not shown for financials.** {pre['sic_desc'] or 'This company'} (SIC "
                f"{pre['sic']}) runs on leverage as its product rather than as a financing "
                "choice. Equity plus borrowings is not capital at work, and its cash is not "
                "free — it backs deposits or policyholder liabilities. Return on equity against "
                "a combined ratio or a net interest margin is the right frame, and this tool "
                "does not contain it.")
        elif latest_r.reason:
            w = latest_r.cap
            st.error(f"**n/a for FY{latest_r.fy} — {latest_r.reason}.**"
                     + ("\n\nEquity is negative, which for a profitable company almost always "
                        "means buybacks have retired more capital than the balance sheet "
                        "carries. Usually strength rather than distress, but a return on a "
                        "negative denominator flips sign, so nothing is printed."
                        if w.equity < 0 and w.equity_found else
                        f"\n\nThe cause is cash, not losses: {money(w.deployable_cash)} "
                        f"deployable against {money(w.total_capital)} of total capital. The "
                        "operating business runs on less than nothing — a genuinely excellent "
                        "property and not a computable one. Check that the cash really is "
                        "deployable rather than earmarked, and raise the operating-cash "
                        "percentage if the business needs more of it."
                        if w.deployable_cash > w.total_capital and w.equity_found else ""))
        else:
            k1, k2, k3 = st.columns(3)
            k1.metric("ROIC", f"{latest_r.roic:.1%}", f"FY{latest_r.fy}")
            k2.metric("Median, 5 years", f"{roic_med:.1%}" if roic_med is not None else "—",
                      f"{len([r for r in rows[-5:] if not r.reason])} of {len(rows[-5:])} readable")
            tang = latest_r.tangible_roic
            k3.metric("Ex-goodwill", f"{tang:.1%}" if tang is not None else "n/a",
                      "return on tangible capital")
            cav = roic_caveat(latest_r, pre.get("fye_month", 12))
            if cav:
                st.caption("How hard to lean on this: " + cav + ".")

            w = latest_r.cap
            wf = [
                ("Owners' earnings", latest_r.OE, "the figure in the box above"),
                ("less interest income", -latest_r.interest_income,
                 "the cash left the denominator, so its income leaves the numerator"),
                ("less capital lease payments", -latest_r.lease_payments,
                 "a financing outflow earnings never saw"),
                ("less other expense", -latest_r.other_expense,
                 "forensic D&A, normalised tax, cyclical — yours to set"),
                ("= adjusted return", latest_r.numerator, ""),
                ("Shareholders' equity", w.equity, "parent's share" if w.minority else ""),
                ("plus borrowings", w.debt, "short and long term"),
                ("plus finance leases", w.finance_leases, "capitalised leases are debt in all "
                                                          "but name"),
                ("= total capital", w.total_capital, ""),
                ("less deployable cash", -w.deployable_cash,
                 f"of {plain(w.cash)} held; {plain(w.op_cash_need)} kept in as working cash"),
                ("plus other capital", w.other_capital, "float, obligations — yours to set"),
                ("= invested capital", w.invested, ""),
            ]
            st.dataframe(
                pd.DataFrame([{"Line": a, "$M": b, "Why": c} for a, b, c in wf])
                .style.format({"$M": "{:,.0f}"}), width="stretch", hide_index=True)
            st.caption(
                f"Long-term operating leases of {money(w.operating_leases)} are **not** "
                "subtracted, and that is deliberate: Burry's formula removes them from a "
                "total-capital figure that included them. This base is equity plus borrowings, "
                "which never did, so subtracting again would count them twice. Restricted cash "
                f"of {money(w.restricted)} stays in for the reason it always should — it funds "
                "the business and you cannot have it.")

            st.write("**Year by year** — the trend matters more than the level")
            st.dataframe(pd.DataFrame([{
                "FY": r.fy, "Owners' earnings": r.OE, "Invested capital": r.cap.invested,
                "ROIC": r.roic if not r.reason else None,
                "Ex-goodwill": r.tangible_roic if not r.reason else None,
                "n/a because": r.reason} for r in rows]).style.format(
                {"Owners' earnings": "{:,.0f}", "Invested capital": "{:,.0f}",
                 "ROIC": "{:.1%}", "Ex-goodwill": "{:.1%}"}, na_rep="n/a"),
                width="stretch", hide_index=True)
            st.caption(
                f"FY{latest_r.fy} uses the owners' earnings figure from the input box; earlier "
                "years are as filed. Capital is measured at each year end rather than averaged, "
                "which understates the return for anything growing its asset base quickly — the "
                "conservative direction, and the one Burry's formula reads literally.")

    with st.expander("Feed this into the IV15 tool"):
        st.caption(
            "The Tragic Algebra Analyzer asks for a growth rate and has no way to sanity-check "
            "it. This is that ceiling, computed.")
        if roic_med is not None and payout is not None:
            g_ceiling = sustainable_growth(roic_med, payout)
            st.code(
                f"{tk}\n"
                f"owners' earnings     {OE:,.0f} M      <- same figure tool 1 seeds\n"
                f"shares               {shares:,.1f} M\n"
                f"ROIC, 5y median      {roic_med:.1%}\n"
                f"cash returned        {payout:.0%} of owners' earnings\n"
                f"growth ceiling       {g_ceiling:.1%}   <- do not exceed this in tool 1\n"
                f"per-share ceiling    {fundable:.1%}   (adds the buyback effect)",
                language="text")
            st.caption(
                f"A growth rate above {g_ceiling:.1%} in tool 1 is a claim that this company "
                "funds expansion from outside — more debt, or stock. Sometimes true, always "
                "worth stating out loud rather than assuming.")
        else:
            st.warning("No ROIC or no positive earnings base, so no ceiling. Tool 1's growth "
                       "input stays unconstrained here.")

    label = "Notes and detail" + (f" · {len(alerts)} to review" if alerts else "")
    with st.expander(label):
        for kind_, msg in alerts:
            getattr(st, kind_)(msg)

        st.write("**Owners' earnings, year by year** — identical to tool 1 for the same ticker")
        st.dataframe(pd.DataFrame([{
            "FY": f"{y.fy}*" if y.excluded else str(y.fy),
            "Net income": y.N, "GAAP SBC": y.G, "Buybacks": y.T, "Share change": y.dS,
            "Avg price": y.price, "True SBC cost": y.omega, "Owners' earnings": y.OE}
            for y in years]).style.format({
                "Net income": "{:,.0f}", "GAAP SBC": "{:,.0f}", "Buybacks": "{:,.0f}",
                "Share change": "{:+,.1f}", "Avg price": "${:,.2f}",
                "True SBC cost": "{:,.0f}", "Owners' earnings": "{:,.0f}"}, na_rep="—"),
            width="stretch", hide_index=True)
        st.caption(
            f"ΔE pooled: {pooled.dE:.1%} over {pooled.years} years, {recent.dE:.1%} over the "
            f"last three. The box above shows {plain(latest.N)} of net income times "
            f"{use_dE:.1%}, which is how tool 1 seeds it too — the latest year as filed came in "
            f"at {plain(latest.OE)}.")

        # Two tells that owners' earnings here are flattering. Both are visible
        # in the table above, and neither announced itself before.
        _neg = [y for y in years if y.omega < 0 and not y.excluded]
        if pay.used_implied:
            st.error(
                "**Owners' earnings are overstated in this table, and so are tool 1's.** No "
                "buyback line was read, so the market value of shares delivered to employees "
                "floors at zero: V = max(0, buybacks + price x share change), and with buybacks "
                "missing and the share count shrinking, that maximum is always zero. Real "
                "repurchases are gross of the stock issued to employees, so the true cost is "
                "positive. Both tools read this filer the same way and are wrong the same way. "
                "The tag panel below is what to send me to fix it.")
        elif len(_neg) >= max(2, len(years) // 3):
            st.warning(
                f"**The true stock-comp cost reads negative in {len(_neg)} years**, which adds to "
                "owners' earnings rather than subtracting. That happens when option and ESPP "
                "proceeds exceed the tax withheld on vesting — real, but it usually also means "
                "the share-delivery term is being floored at zero because a buyback or issuance "
                "line was not read. Treat these owners' earnings as a ceiling.")

        st.write("**Assumptions used** — paste this if something looks wrong")
        st.code(
            f"{tk}   price {price:,.2f}   shares {shares:,.1f}M   mkt cap {plain(mcap)}\n"
            f"owners' earnings    {OE:,.0f} M   ({mcap/OE:,.1f}x)\n"
            f"needs               "
            + (f"{required:.2%}/yr" if required is not None else "n/a") + "\n"
            f"can fund            "
            + (f"{fundable:.2%}/yr" if fundable is not None else "n/a") + "\n"
            f"has delivered       "
            + (f"{delivered:.2%}/yr" if delivered is not None else "n/a") + "\n"
            f"ROIC 5y median      "
            + (f"{roic_med:.2%}" if roic_med is not None else "n/a") + "\n"
            f"invested capital    {latest_r.cap.invested:,.0f} M\n"
            f"payout              "
            + (f"{payout:.1%}" if payout is not None else "n/a") + "\n"
            f"other expense       {other_expense:,.0f} M   other capital {other_capital:,.0f} M\n"
            f"operating cash      {op_cash_pct:.1%} of revenue\n"
            f"dilution            {dilution:+.2%}/yr   horizon {horizon}y   exit {exit_mult:g}x\n"
            f"verdict             {v.label} ({v.why})", language="text")

    with st.expander("What was read from the filings — every tag, found or missing"):
        st.caption(
            "A zero in this app is either something the company did not do or a tag this reader "
            "does not know. Only the filing settles which, and this is where to look. If a line "
            "you know exists reads zero years, that is a bug worth reporting — the tag name is "
            "the whole fix.")
        st.dataframe(pd.DataFrame(pre.get("tags", [])), width="stretch", hide_index=True)

# ══════════════════════════════════════════════════════════════════════
#  REFERENCE
# ══════════════════════════════════════════════════════════════════════

st.divider()
_r1, _r2 = st.columns(2)
with _r1:
    with st.expander("What the numbers mean"):
        st.markdown(
            "**Needs** — the annual growth in owners' earnings that turns today's price into a "
            "hundredfold over your holding period, after the exit multiple you assume and the "
            "dilution along the way.\n\n"
            "**Can fund** — return on capital multiplied by the share of earnings retained, plus "
            "the lift from any stock retired. The fastest a business compounds per share without "
            "outside money. A company earning 100% on capital and paying all of it out funds "
            "almost no growth, which is why this is the number to read rather than ROIC "
            "itself.\n\n"
            "**Has delivered** — the better of its revenue and owners'-earnings growth over the "
            "window. History is not a ceiling, but a price that requires a rate the company has "
            "never reached is a bet on an inflection.\n\n"
            "**ROIC** — Burry's fully-adjusted return on invested capital, computed on owners' "
            "earnings over capital genuinely at work: deployable cash removed, operational cash "
            "left in.")

with _r2:
    with st.expander("Verify the engine"):
        st.caption(
            "Three kinds of check. The Alphabet lines re-run **Burry's published inputs** through "
            "this page's copy of the Tragic Algebra engine and confirm it still matches tool 1 to "
            "the dollar. The Mayer lines check the 100x arithmetic against the figures his book "
            "leads with. The rest are wiring and verdict tests.\n\n"
            "There is no published ROIC to validate against the way Alphabet validates owners' "
            "earnings, so that test proves the plumbing is right, not that the framework is.")
        if st.button("Run checks"):
            for name, ok, got in self_test():
                st.write(("✅ " if ok else "❌ ") + f"{name} — {got}")

st.caption(
    "Research aid, not financial advice. Outputs depend on estimates you supply. Method follows "
    "Christopher Mayer's published framework and Michael Burry's published ROIC formula; this "
    "project is independent and is not affiliated with or endorsed by either of them.")
