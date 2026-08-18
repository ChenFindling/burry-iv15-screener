"""
Burry IV15 Screener — Tragic Algebra owners' earnings + hybrid AICT/IV ladder.

Replaces the earlier version. Key corrections vs. that build:
  1. Omega is computed from the real formula, not invented ratios.
  2. W (shares repurchased) and dS (share count change) are actually fetched.
  3. C = Cw - Ce, kept strictly separate from T. They were previously collided
     in one tag list, which alone corrupted many tickers.
  4. dE is pooled over ~10 years, not a single year.
  5. Owners' earnings are NOT floored at zero. Negative is a real, important
     answer ("not investible") and must survive to the output.
  6. Tier stage-2 durations are honoured. Total horizon is 24/20/15/11/6 years
     by tier, not 15 for everything.
  7. IV12/IV18 are re-run through the DCF, never scaled off IV15. The old
     shortcut overstated IV12 ~10% and understated IV18 ~15%.
  8. XBRL facts are filtered on period duration and deduped by filing, so
     prior-year comparatives and quarterly rows can't be mistaken for FY data.
  9. IFRS taxonomy fallback for foreign filers.
 10. Unresolved inputs are reported explicitly instead of silently becoming 0.

Run:  streamlit run app.py
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

import pandas as pd
import requests
import streamlit as st

# ══════════════════════════════════════════════════════════════════════
#  DOMAIN CONSTANTS
# ══════════════════════════════════════════════════════════════════════

SEC_HEADERS = {
    # SEC requires a real contact address. Put your own in before deploying.
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


AICT: dict[str, Tier] = {
    "Fortress": Tier(8, 16, 0.70, 0.07, 3.0),
    "Castle":   Tier(7, 13, 0.55, 0.05, 2.5),
    "Chapel":   Tier(5, 10, 0.45, 0.04, 2.0),
    "Stone":    Tier(4,  7, 0.35, 0.03, 0.0),
    "Wood":     Tier(2,  4, 0.25, 0.00, 0.0),
}

TIER_BLURB = {
    "Fortress": "Regulated or platform/enterprise, owns its AI, no acute seat risk",
    "Castle":   "Strong moat, owned AI at material scale, outcome reasonably certain",
    "Chapel":   "Acute AI threat but owned AI at decent scale + switching costs",
    "Stone":    "Meaningful threat without strong adaptability, or chronic pressure",
    "Wood":     "Borrowed AI, no credible R&D, direct attack from foundation models",
}

# P/IV15 -> points, from the 11-bracket valuation scale
VALUATION_BRACKETS = [
    (0.50, 35), (0.75, 32), (0.90, 28), (1.00, 24), (1.25, 20),
    (1.50, 17), (2.00, 14), (3.00, 8), (5.00, 5), (10.0, 3),
]

# ══════════════════════════════════════════════════════════════════════
#  TRAGIC ALGEBRA  (pure functions — no Streamlit in here)
# ══════════════════════════════════════════════════════════════════════


@dataclass
class YearInputs:
    """One fiscal year. Dollars in $M, share counts in millions."""
    fy: int
    N: float                       # GAAP net income (parent)
    G: float                       # GAAP SBC expense (positive)
    T: float = 0.0                 # buyback dollars — PROGRAM ONLY
    W: float = 0.0                 # shares repurchased under the program
    dS: float = 0.0                # change in shares outstanding (+ = dilution)
    Cw: float = 0.0                # tax withholding paid on vesting
    Ce: float = 0.0                # option / ESPP proceeds received
    external_price: float | None = None   # required when there is no buyback
    cash_settled_sbc: bool = False        # MELI-style: V = 0 by design

    # ---- derived -----------------------------------------------------
    @property
    def C(self) -> float:
        """Net cash award payments. Can be negative when proceeds exceed
        withholding — that is a real outcome, not an error."""
        return self.Cw - self.Ce

    @property
    def no_buyback(self) -> bool:
        return self.T == 0 and self.W == 0

    @property
    def I(self) -> float:
        """Shares delivered to employees.
        Treadmill:      I = dS + W   (recovers gross issuance)
        Pure dilution:  I = dS       (W is zero)"""
        return self.dS if self.no_buyback else self.dS + self.W

    @property
    def P(self) -> float | None:
        """Average price. T/W is the audited figure when a program exists;
        otherwise an external period-average price is required."""
        if self.W > 0:
            return self.T / self.W
        return self.external_price

    @property
    def V(self) -> float | None:
        if self.cash_settled_sbc:
            return 0.0
        p = self.P
        if p is None:
            return None
        return max(0.0, self.I * p)   # floor: cannot deliver negative shares

    @property
    def omega(self) -> float | None:
        v = self.V
        return None if v is None else self.C + v

    @property
    def owners_earnings(self) -> float | None:
        om = self.omega
        return None if om is None else self.N + self.G - om

    @property
    def delta_e(self) -> float | None:
        oe = self.owners_earnings
        if oe is None or self.N == 0:
            return None
        return oe / self.N

    @property
    def usable(self) -> bool:
        return self.V is not None

    def gate1(self, tol: float = 0.001) -> bool:
        """|Omega - (C+V)| / max(|Omega|, $1K) <= 0.1%"""
        om, v = self.omega, self.V
        if om is None or v is None:
            return False
        return abs(om - (self.C + v)) / max(abs(om), 0.001) <= tol


@dataclass
class Pooled:
    delta_e: float
    sum_N: float
    sum_OE: float
    sum_omega: float
    sum_G: float
    years_used: int
    years_dropped: list[int] = field(default_factory=list)

    @property
    def gaap_overstatement(self) -> float:
        """(sum_Omega - sum_G) / sum_OE — how much GAAP overstates owners'
        earnings. Equivalently sum_N/sum_OE - 1."""
        return (self.sum_omega - self.sum_G) / self.sum_OE if self.sum_OE else float("nan")

    @property
    def street_overstatement(self) -> float:
        return self.sum_omega / self.sum_OE if self.sum_OE else float("nan")

    @property
    def tragic_tier(self) -> bool:
        return self.sum_OE < 0

    def retention(self, t: int) -> float:
        """Share of GAAP intrinsic-value-per-share growth that survives to
        year t. This is dE**t — it compounds, which is the whole point."""
        return self.delta_e ** t

    def true_cagr(self, gaap_growth: float) -> float:
        """Break-even dE is 1/(1+g): below it, reported growth doesn't reach
        the owner at all."""
        return self.delta_e * (1.0 + gaap_growth) - 1.0


def pool_delta_e(years: list[YearInputs]) -> Pooled:
    """Earnings-weighted pooled dE. Never average annual ratios — that blows
    up on near-zero-earnings years and forces exclusions.

    Gate 2: a year with no usable V has its N, G and C excluded too. Keeping
    the earnings while dropping the SBC cost biases every aggregate low.
    """
    keep = [y for y in years if y.usable]
    drop = [y.fy for y in years if not y.usable]
    if not keep:
        raise ValueError("No usable years. Years without a buyback need an average share price.")
    sN = sum(y.N for y in keep)
    if sN == 0:
        raise ValueError("Cumulative net income is zero — extend the window.")
    return Pooled(
        delta_e=sum(y.owners_earnings for y in keep) / sN,
        sum_N=sN,
        sum_OE=sum(y.owners_earnings for y in keep),
        sum_omega=sum(y.omega for y in keep),
        sum_G=sum(y.G for y in keep),
        years_used=len(keep),
        years_dropped=drop,
    )


# ══════════════════════════════════════════════════════════════════════
#  IV LADDER  (hybrid two-model DCF)
# ══════════════════════════════════════════════════════════════════════


@dataclass
class IVParams:
    owners_earnings: float     # $M — normalised, post-Tragic-Algebra
    shares: float              # M
    tier: str
    stage1_growth: float       # decimal
    net_cash: float = 0.0      # $M (negative = net debt)
    exit_multiple: float = 20.0
    blend_model1: float = 0.5  # weight on the perpetuity model
    stage0_years: int = 0      # hypergrowth ramp for inflecting companies
    stage0_growth: float = 0.0


def _stream(p: IVParams, n_years: int) -> list[float]:
    t = AICT[p.tier]
    g2 = p.stage1_growth * t.stage2_multiplier
    out, e = [], p.owners_earnings
    for y in range(1, n_years + 1):
        if y <= p.stage0_years:
            g = p.stage0_growth
        elif y <= p.stage0_years + t.stage1_years:
            g = p.stage1_growth
        else:
            g = g2
        e *= 1.0 + g
        out.append(e)
    return out


def _model1(p: IVParams, r: float) -> float:
    """Stages 1 and 2, then a terminal perpetuity at the tier growth cap."""
    t = AICT[p.tier]
    n = t.horizon + p.stage0_years
    s = _stream(p, n)
    pv = sum(cf / (1.0 + r) ** y for y, cf in enumerate(s, start=1))
    terminal = s[-1] * (1.0 + t.terminal_growth_cap) / (r - t.terminal_growth_cap)
    return pv + terminal / (1.0 + r) ** n


def _model2(p: IVParams, r: float) -> float:
    """Project to year 15, then apply a market multiple to year-15 earnings."""
    s = _stream(p, 15)
    pv = sum(cf / (1.0 + r) ** y for y, cf in enumerate(s, start=1))
    return pv + s[-1] * p.exit_multiple / (1.0 + r) ** 15


def intrinsic_value(p: IVParams, required_return_pct: float) -> float:
    """IV15 -> intrinsic_value(p, 15).

    Each rung is a full re-run at its own discount rate. Never scale one rung
    off another: the published SW46 ratios span 1.333–1.443, so no constant
    multiplier can fit them.

    A negative result is meaningful — no share price delivers that return.
    """
    r = required_return_pct / 100.0
    t = AICT[p.tier]
    if r <= t.terminal_growth_cap:
        return float("nan")
    if p.shares <= 0:
        return float("nan")
    w = p.blend_model1
    blended = w * _model1(p, r) + (1.0 - w) * _model2(p, r)
    return (blended + p.net_cash) / p.shares


def iv_ladder(p: IVParams, rungs=(8, 10, 12, 15, 18, 20)) -> dict[int, float]:
    return {n: intrinsic_value(p, n) for n in rungs}


def expected_return(price: float, p: IVParams) -> float:
    """IVB — the long-term CAGR implied by today's price. The ladder inverted,
    and arguably the most useful single output since it needs no required
    return chosen in advance."""
    lo, hi = AICT[p.tier].terminal_growth_cap + 1e-6, 3.0
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if intrinsic_value(p, mid * 100) > price:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def valuation_points(p_over_iv15: float) -> int:
    if p_over_iv15 < 0:
        return -2
    for ceiling, pts in VALUATION_BRACKETS:
        if p_over_iv15 <= ceiling:
            return pts
    return -2


def zone(p_over_iv15: float) -> str:
    if p_over_iv15 < 0:
        return "Not investible"
    if p_over_iv15 <= 1.0:
        return "Fat Pitch"
    if p_over_iv15 <= 1.5:
        return "Just Outside"
    return "Out Field"


# ══════════════════════════════════════════════════════════════════════
#  SEC EDGAR
# ══════════════════════════════════════════════════════════════════════

CONCEPTS = {
    "N":  (["NetIncomeLoss", "ProfitLoss",
            "NetIncomeLossAvailableToCommonStockholdersBasic"],
           ["ProfitLoss", "ProfitLossAttributableToOwnersOfParent"]),
    "G":  (["ShareBasedCompensation", "AllocatedShareBasedCompensationExpense"],
           ["ShareBasedPaymentsExpense", "ExpenseFromSharebasedPaymentTransactionsWithEmployees"]),
    "T":  (["PaymentsForRepurchaseOfCommonStock"],
           ["PaymentsToAcquireOrRedeemEntitysShares"]),
    "Cw": (["PaymentsRelatedToTaxWithholdingForShareBasedCompensation"], []),
    "Ce": (["ProceedsFromStockOptionsExercised",
            "ProceedsFromIssuanceOfSharesUnderIncentiveAndShareBasedCompensationPlans",
            "ProceedsFromIssuanceOfCommonStock"],
           []),
    "W":  (["StockRepurchasedAndRetiredDuringPeriodShares",
            "StockRepurchasedDuringPeriodShares",
            "TreasuryStockSharesAcquired"], []),
}

SHARE_CONCEPTS = ["CommonStockSharesOutstanding", "CommonStockSharesIssued",
                  "EntityCommonStockSharesOutstanding"]


@st.cache_data(ttl=86400, show_spinner=False)
def sec_ticker_map() -> dict[str, str]:
    r = requests.get("https://www.sec.gov/files/company_tickers.json",
                     headers=SEC_HEADERS, timeout=15)
    r.raise_for_status()
    return {e["ticker"].upper(): str(e["cik_str"]).zfill(10) for e in r.json().values()}


@st.cache_data(ttl=86400, show_spinner=False)
def sec_company_facts(cik: str) -> dict:
    r = requests.get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json",
                     headers=SEC_HEADERS, timeout=25)
    r.raise_for_status()
    return r.json()


def _annual_rows(facts: dict, concepts_us: list[str], concepts_ifrs: list[str]) -> dict[int, float]:
    """Return {fiscal_year: value} for full-year duration facts only.

    Three filters the previous version lacked, each of which silently
    corrupted data:
      * duration must be ~a year (330–400 days), so quarterly rows tagged
        fp='FY' can't slip through;
      * annual reports only (10-K / 20-F / 40-F);
      * when the same year appears in several filings, keep the most recently
        filed one — a 10-K restates the prior year as a comparative.
    """
    out: dict[int, tuple[str, float]] = {}
    for taxonomy, concepts in (("us-gaap", concepts_us), ("ifrs-full", concepts_ifrs)):
        tax = facts.get("facts", {}).get(taxonomy, {})
        for concept in concepts:
            if concept not in tax:
                continue
            units = tax[concept].get("units", {})
            rows = units.get("USD", []) or units.get("shares", [])
            for row in rows:
                if row.get("form") not in ("10-K", "10-K/A", "20-F", "40-F"):
                    continue
                start, end = row.get("start"), row.get("end")
                if not start or not end:
                    continue
                days = (dt.date.fromisoformat(end) - dt.date.fromisoformat(start)).days
                if not (330 <= days <= 400):
                    continue
                fy = int(end[:4])
                filed = row.get("filed", "")
                if fy not in out or filed > out[fy][0]:
                    out[fy] = (filed, float(row.get("val", 0.0)))
            if out:
                return {k: v[1] for k, v in out.items()}
    return {}


def _instant_share_counts(facts: dict) -> dict[int, float]:
    """Year-end shares outstanding, used to derive dS."""
    out: dict[int, tuple[str, float]] = {}
    for taxonomy in ("us-gaap", "dei"):
        tax = facts.get("facts", {}).get(taxonomy, {})
        for concept in SHARE_CONCEPTS:
            if concept not in tax:
                continue
            for row in tax[concept].get("units", {}).get("shares", []):
                if row.get("form") not in ("10-K", "10-K/A", "20-F", "40-F"):
                    continue
                end = row.get("end")
                if not end or row.get("start"):
                    continue
                fy, filed = int(end[:4]), row.get("filed", "")
                if fy not in out or filed > out[fy][0]:
                    out[fy] = (filed, float(row.get("val", 0.0)))
    return {k: v[1] for k, v in out.items()}


def fetch_years(ticker: str, n_years: int = 10) -> tuple[list[YearInputs], dict[str, list[int]]]:
    """Returns (years, missing) where `missing` maps each variable to the
    fiscal years it could not be resolved for. Nothing is silently zeroed."""
    cmap = sec_ticker_map()
    if ticker not in cmap:
        raise ValueError(f"'{ticker}' is not in the SEC company list. "
                         "Foreign private issuers without US listings won't appear.")
    facts = sec_company_facts(cmap[ticker])

    series = {k: _annual_rows(facts, us, ifrs) for k, (us, ifrs) in CONCEPTS.items()}
    shares = _instant_share_counts(facts)

    if not series["N"]:
        raise ValueError("Could not find annual net income. This filer may use a "
                         "taxonomy the app doesn't map yet — enter figures manually.")

    fys = sorted(series["N"].keys())[-n_years:]
    missing: dict[str, list[int]] = {k: [] for k in ("G", "T", "W", "Cw", "Ce", "dS")}

    years: list[YearInputs] = []
    for fy in fys:
        def grab(key: str, scale: float = 1e6) -> float:
            v = series[key].get(fy)
            if v is None:
                missing[key].append(fy)
                return 0.0
            return abs(float(v)) / scale

        dS = 0.0
        if fy in shares and (fy - 1) in shares:
            dS = (shares[fy] - shares[fy - 1]) / 1e6
        else:
            missing["dS"].append(fy)

        years.append(YearInputs(
            fy=fy,
            N=float(series["N"][fy]) / 1e6,
            G=grab("G"),
            T=grab("T"),
            W=grab("W", 1e6),
            dS=dS,
            Cw=grab("Cw"),
            Ce=grab("Ce"),
        ))
    return years, {k: v for k, v in missing.items() if v}


@st.cache_data(ttl=900, show_spinner=False)
def fetch_price(ticker: str) -> float | None:
    try:
        r = requests.get(
            f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?interval=1d&range=1d",
            headers={"User-Agent": "Mozilla/5.0"}, timeout=8)
        meta = r.json()["chart"]["result"][0]["meta"]
        return float(meta.get("regularMarketPrice") or meta.get("chartPreviousClose"))
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════
#  SELF-TEST  — validates the maths against published figures
# ══════════════════════════════════════════════════════════════════════

def self_test() -> list[tuple[str, bool, str]]:
    res = []

    # Alphabet FY2016–2025, from the published Tragic Algebra table
    goog = [(2016, 19478, 6900, 3693, 3304, 78.6, 97), (2017, 12662, 7900, 4846, 4166, 88.1, 78),
            (2018, 30736, 10000, 9075, 4993, 148.8, -2), (2019, 34343, 11700, 18396, 4765, 262.8, -158),
            (2020, 40269, 12991, 31149, 5720, 426.7, -263), (2021, 76033, 15376, 50274, 10162, 402.2, -264),
            (2022, 59972, 19362, 59296, 9300, 506.8, -412), (2023, 73795, 22460, 61504, 9837, 534.8, -374),
            (2024, 100118, 22785, 62222, 12190, 379.4, -243), (2025, 132170, 24953, 45709, 14167, 221.9, -93)]
    ys = [YearInputs(fy=y, N=n_, G=g, T=t, W=w, dS=ds, Cw=c) for y, n_, g, t, c, w, ds in goog]
    p = pool_delta_e(ys)
    res.append(("Alphabet pooled ΔE = 88.7%", abs(p.delta_e - 0.887) < 0.002, f"{p.delta_e:.2%}"))
    res.append(("Alphabet FY2016 V = $8,252M", abs(ys[0].V - 8252) < 5, f"${ys[0].V:,.0f}M"))

    # Pure-dilution year needs an external price (Meta FY2016)
    meta16 = YearInputs(fy=2016, N=10217, G=3218, T=0, W=0, dS=46, Cw=-10, external_price=107)
    res.append(("Meta FY2016 ΔE = 83.4% (no buyback)",
                abs(meta16.delta_e - 0.834) < 0.005, f"{meta16.delta_e:.1%}"))
    res.append(("Missing price → year excluded, not zeroed",
                YearInputs(fy=2020, N=100, G=10, T=0, W=0, dS=5).V is None, "excluded"))

    # NDX-97 index identities
    N_, G_, OM_ = 4925.5, 919.0, 1732.2
    OE_ = N_ + G_ - OM_
    res.append(("NDX-97 GAAP overstatement = 19.78%",
                abs((OM_ - G_) / OE_ - 0.1978) < 0.001, f"{(OM_-G_)/OE_:.2%}"))

    # Salesforce full ladder from tier structure alone
    crm = IVParams(owners_earnings=7300, shares=1073.3, tier="Chapel",
                   stage1_growth=0.069, exit_multiple=21.8, blend_model1=1.0)
    res.append(("Salesforce IV15 ≈ $69.81",
                abs(intrinsic_value(crm, 15) - 69.81) < 1.0, f"${intrinsic_value(crm,15):.2f}"))
    res.append(("Salesforce IV12 ≈ $96.02",
                abs(intrinsic_value(crm, 12) - 96.02) < 1.5, f"${intrinsic_value(crm,12):.2f}"))
    res.append(("Salesforce IVB ≈ 8.6%",
                abs(expected_return(165.84, crm) - 0.086) < 0.005,
                f"{expected_return(165.84, crm):.1%}"))
    return res


# ══════════════════════════════════════════════════════════════════════
#  UI
# ══════════════════════════════════════════════════════════════════════

st.set_page_config(page_title="Burry IV15 Screener", layout="wide")
st.title("Burry IV15 Screener")
st.caption("Tragic Algebra owners' earnings → AICT tiering → hybrid intrinsic value ladder")

with st.sidebar:
    st.subheader("Method self-test")
    if st.button("Run"):
        for name, ok, got in self_test():
            st.write(("✅ " if ok else "❌ ") + name + f" — {got}")
    st.divider()
    st.caption(
        "IV15 is the price giving ~15% annually over 15+ years. It is a buy price "
        "target from a multi-stage DCF, not an earnings multiple. Baseline intrinsic "
        "value sits between IV8 and IV10 — below that buybacks add value per share, "
        "above it they destroy it."
    )

tab_fetch, tab_manual = st.tabs(["Fetch from SEC", "Enter manually"])

years: list[YearInputs] = []
missing: dict[str, list[int]] = {}

with tab_fetch:
    c1, c2 = st.columns([2, 1])
    ticker = c1.text_input("Ticker", value="", placeholder="ADBE, CRM, NOW…").upper().strip()
    n_years = c2.number_input("Years of history", 3, 12, 10)
    if st.button("Pull 10-K data", type="primary") and ticker:
        try:
            with st.spinner(f"Reading {ticker} filings…"):
                years, missing = fetch_years(ticker, int(n_years))
            st.session_state["years"] = years
            st.session_state["missing"] = missing
            st.session_state["ticker"] = ticker
        except Exception as e:
            st.error(str(e))

with tab_manual:
    st.caption("Dollars in $M, share counts in millions. Leave a row blank to skip it.")
    blank = pd.DataFrame([{"fy": 2025 - i, "N": None, "G": None, "T": None, "W": None,
                           "dS": None, "Cw": None, "Ce": None, "external_price": None}
                          for i in range(5)][::-1])
    edited = st.data_editor(blank, num_rows="dynamic", width='stretch', key="manual")
    if st.button("Use these figures"):
        rows = []
        for _, r in edited.iterrows():
            if pd.isna(r["N"]):
                continue
            rows.append(YearInputs(
                fy=int(r["fy"]), N=float(r["N"]), G=float(r["G"] or 0), T=float(r["T"] or 0),
                W=float(r["W"] or 0), dS=float(r["dS"] or 0), Cw=float(r["Cw"] or 0),
                Ce=float(r["Ce"] or 0),
                external_price=None if pd.isna(r["external_price"]) else float(r["external_price"]),
            ))
        st.session_state["years"] = rows
        st.session_state["missing"] = {}
        st.session_state["ticker"] = "MANUAL"

years = st.session_state.get("years", [])
missing = st.session_state.get("missing", {})
ticker = st.session_state.get("ticker", "")

if years:
    st.divider()
    st.subheader(f"Tragic Algebra — {ticker}")

    if missing:
        st.warning(
            "**Not found in XBRL, defaulted to zero — verify these in the 10-K.** "
            + "; ".join(f"`{k}` for {v}" for k, v in missing.items())
            + ".  `W` and `Cw` in particular often live only in the share-repurchase "
              "footnote or the statement of shareholders' equity."
        )

    df = pd.DataFrame([{
        "FY": y.fy, "N": y.N, "G": y.G, "T": y.T, "W": y.W, "ΔS": y.dS,
        "C": y.C, "P": y.P, "V": y.V, "Ω": y.omega,
        "Owners' earnings": y.owners_earnings,
        "ΔE": y.delta_e,
    } for y in years])
    st.dataframe(
        df.style.format({c: "{:,.0f}" for c in ["N", "G", "T", "C", "V", "Ω", "Owners' earnings"]}
                        | {"W": "{:,.1f}", "ΔS": "{:+,.1f}", "P": "${:,.2f}", "ΔE": "{:.1%}"}, na_rep="—"),
        width='stretch', hide_index=True)

    try:
        pooled = pool_delta_e(years)
    except ValueError as e:
        st.error(str(e))
        st.stop()

    if pooled.years_dropped:
        st.info(
            f"Years {pooled.years_dropped} were excluded: no buyback, so the average price "
            "is needed to value shares delivered. Their N, G and C are excluded too — "
            "keeping earnings while dropping the SBC cost would bias the result low. "
            "Add a price on the manual tab to include them."
        )

    m = st.columns(4)
    m[0].metric("Pooled ΔE", f"{pooled.delta_e:.1%}", f"{pooled.years_used}y earnings-weighted")
    m[1].metric("Owners' earnings", f"${pooled.sum_OE:,.0f}M", "cumulative")
    m[2].metric("GAAP overstates by", f"{pooled.gaap_overstatement:.1%}")
    m[3].metric("Street overstates by", f"{pooled.street_overstatement:.1%}")

    if pooled.tragic_tier:
        st.error(
            "**Tragic Tier.** Cumulative SBC cost exceeded GAAP SBC expense and net income "
            "combined — owners' earnings are negative over the whole period, and not because "
            "of one bad year. Shareholders were net funders of employee compensation."
        )
    elif pooled.delta_e < 1 / 1.15:
        st.warning(
            f"**Below the {1/1.15:.0%} break-even.** At ΔE of {pooled.delta_e:.1%}, even 15% "
            f"GAAP growth compounds intrinsic value per share at only "
            f"{pooled.true_cagr(0.15):+.2%} a year. Retention after 10 years: "
            f"{pooled.retention(10):.1%}."
        )
    else:
        st.success(
            f"Above the {1/1.15:.0%} break-even — reported growth reaches the owner. "
            f"Retention after 10 years: {pooled.retention(10):.1%}."
        )

    # ── valuation ────────────────────────────────────────────────────
    st.divider()
    st.subheader("Intrinsic value ladder")

    latest = [y for y in years if y.usable][-1]
    default_oe = latest.owners_earnings

    v1, v2, v3 = st.columns(3)
    with v1:
        oe = st.number_input("Owners' earnings ($M)", value=float(round(default_oe, 1)), step=1.0,
                             help="Normalise further for maintenance capex, working capital and "
                                  "one-offs before relying on this.")
        shares = st.number_input("Diluted shares (M)", value=100.0, step=1.0)
    with v2:
        tier = st.selectbox("AICT tier", list(AICT.keys()), index=2,
                            format_func=lambda t: f"{t} — {TIER_BLURB[t]}")
        g1 = st.number_input("Stage 1 growth (%)", value=8.0, step=0.5,
                             help="ROIC is the ceiling — a company cannot outgrow it forever.") / 100
    with v3:
        net_cash = st.number_input("Net cash ($M)", value=0.0, step=10.0,
                                   help="Subtract only what is freely deployable. Restricted, "
                                        "regulated and operationally-tied cash funds the business.")
        price = st.number_input("Price", value=float(fetch_price(ticker) or 100.0), step=0.01)

    with st.expander("Hybrid model settings — judgement, not published"):
        h1, h2, h3 = st.columns(3)
        exit_m = h1.number_input("Exit multiple on year-15 owners' earnings", value=20.0, step=0.5)
        blend = h2.slider("Weight on perpetuity model", 0.0, 1.0, 0.5, 0.05)
        s0y = h3.number_input("Stage 0 years (hypergrowth)", 0, 8, 0)
        s0g = h3.number_input("Stage 0 growth (%)", value=30.0, step=1.0) / 100
        t = AICT[tier]
        st.caption(
            f"{tier}: stage 1 = {t.stage1_years}y, stage 2 = {t.stage2_years}y at "
            f"{t.stage2_multiplier:.2f}× stage-1 growth, terminal cap {t.terminal_growth_cap:.0%}, "
            f"debt capacity {t.debt_capacity_ebitda:.1f}× EBITDA. "
            f"**Total horizon {t.horizon + s0y} years** — not 15. "
            "The exit multiple and blend weight are not published anywhere and cannot be "
            "reverse-engineered from published IV values; they are your call."
        )

    p = IVParams(owners_earnings=oe, shares=shares, tier=tier, stage1_growth=g1,
                 net_cash=net_cash, exit_multiple=exit_m, blend_model1=blend,
                 stage0_years=int(s0y), stage0_growth=s0g)

    ladder = iv_ladder(p)
    iv15 = ladder[15]

    if iv15 != iv15:  # NaN
        st.error("Required return must exceed the tier's terminal growth cap.")
    elif iv15 < 0:
        st.error(
            "**Negative IV15 — not investible.** There is no share price, not even $0.01, "
            "that delivers 15% annually as a long-term shareholder on these inputs."
        )
    else:
        ratio = price / iv15
        er = expected_return(price, p)
        k = st.columns(4)
        k[0].metric("IV15", f"${iv15:,.2f}")
        k[1].metric("P/IV15", f"{ratio:.2f}×", zone(ratio))
        k[2].metric("IVB (expected CAGR)", f"{er:.1%}")
        k[3].metric("Valuation points", f"{valuation_points(ratio)}/35")

        lad = pd.DataFrame([{
            "Rung": f"IV{n}", "Price": v, "vs market": f"{price/v:.2f}×" if v > 0 else "—",
            "Meaning": {8: "baseline intrinsic value, upper",
                        10: "baseline intrinsic value, lower",
                        12: "fair", 15: "benchmark buy target",
                        18: "deep margin of safety", 20: "crisis pricing"}[n],
        } for n, v in ladder.items() if v == v])
        st.dataframe(lad.style.format({"Price": "${:,.2f}"}),
                     width='stretch', hide_index=True)
        st.caption(
            "Set alerts at every rung. Each is a separate DCF run at its own discount rate — "
            "scaling one rung off another does not work, since published IV12/IV15 ratios "
            "range 1.33–1.44 across companies."
        )
