# 🎯 Cassandra IV15 Screener & Tragic Algebra Engine

A fundamental valuation engine implementing **Michael Burry's Tragic Algebra and AICT
moat framework**, computed directly from audited SEC EDGAR XBRL filings.

🔗 **Live app:** [burry-iv15-screener.streamlit.app](https://burry-iv15-screener.streamlit.app/)

---

## 📌 The problem: free cash flow lies in software

Standard tools add stock-based compensation back as a "non-cash expense." It is not free.
Granting equity either **dilutes owners** by expanding the share count, or **drains cash**
through buybacks that exist only to neutralise employee grants.

Across the NASDAQ-100 over ten years, that cost totals **$1.73 trillion**. Shareholders keep
about **83 cents** of every reported GAAP dollar — and Wall Street's "adjusted" earnings,
which add SBC back with no offset, overstate the real figure by **42%**.

---

## 🧮 Tragic Algebra

| Symbol | Meaning | Source |
| :--- | :--- | :--- |
| $N$ | GAAP net income | Income statement |
| $G$ | GAAP SBC expense | Cash flow, operating |
| $C_w$ | Tax withheld on vesting | Cash flow, financing |
| $C_e$ | Option and ESPP proceeds | Cash flow, financing |
| $T$ | Buyback dollars | Cash flow, financing |
| $W$ | Shares repurchased | Repurchase footnote |
| $\Delta S$ | Change in shares outstanding | Balance sheet |

$$I = \Delta S + W \qquad P = T / W \qquad V = I \times P$$

$$C = C_w - C_e \qquad \Omega = C + V \qquad OE = N + G - \Omega \qquad \Delta E = OE / N$$

$\Omega$ **replaces** $G$ rather than supplementing it — leaving the GAAP charge in would
double-count. Pooling over ~10 years uses $\sum OE / \sum N$, never an average of annual
ratios, which blows up on near-zero-earnings years.

### The simplification that makes this automatable

$W$ is almost never tagged in XBRL — it lives in the share repurchase footnote. But since
$P = T/W$:

$$V = T \cdot \frac{W + \Delta S}{W} = T + \frac{T}{W}\Delta S = T + P \cdot \Delta S$$

$W$ cancels. Only the average share price is needed, and that is always obtainable. This is
exact, not an approximation — verified against all ten published Alphabet years.

### Why ΔE compounds

$\Delta E$ is not a one-off haircut. It applies every year, so intrinsic value per share
retains $\Delta E^{t}$ after $t$ years.

**Break-even is $1/1.15 \approx 87\%$.** Below that, a company needs 15% reported growth just
to hold value per share steady. At the NASDAQ-100's 83.5%, 15% growth still compounds at
**−3.99% a year**.

---

## ✅ Validation

The engine reproduces Burry's published figures. Run the self-test in the sidebar.

| Check | Published | Engine |
| :--- | :---: | :---: |
| Alphabet FY2016 $V$ | $8,252M | $8,252M |
| Alphabet FY2025 $V$ | $26,551M | $26,551M |
| Alphabet pooled ΔE | 88.7% | 88.68% |
| Meta pooled ΔE | 83.35% | 83.35% |
| Meta FY2016 ΔE (no buyback) | 83.4% | 83.4% |
| NDX-97 GAAP overstatement | 19.78% | 19.77% |
| Salesforce IV15 | $69.81 | $69.63 |
| Salesforce IVB | 8.6% | 8.6% |

---

## 🏰 AICT moat tiers

| Tier | Stage 1 | Stage 2 | Stage 2 growth | Terminal cap | Debt capacity | Exit multiple |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Fortress** — regulated or platform, owns its AI | 8y | 16y | 70% | 7.0% | 3.0× EBITDA | 20.0× |
| **Castle** — strong moat, owned AI at scale | 7y | 13y | 55% | 5.0% | 2.5× EBITDA | 16.0× |
| **Chapel** — acute AI threat, real defences | 5y | 10y | 45% | 4.0% | 2.0× EBITDA | 14.5× |
| **Stone** — threatened, limited adaptability | 4y | 7y | 35% | 3.0% | 0 | 9.0× |
| **Wood** — borrowed AI, no credible R&D | 2y | 4y | 25% | 0.0% | 0 | 5.0× |

Total horizon is **24 / 20 / 15 / 11 / 6 years** — not 15 for everything. Tier sets how long
growth lasts and how fast it fades, never the starting rate.

Stage durations, multipliers, terminal caps and debt capacity are published. **Exit multiples
are not** — those are calibrated so the growth needed to reproduce a published IV15 matches
the company's actual growth. Adobe anchors them: at 14.5×, reaching his $262 needs 11.1%
growth, and Adobe grew 11%.

---

## 📐 The IV ladder

$IV_n$ is the price returning $n\%$ annually over the long run. Every rung is **one earnings
stream discounted at its own rate** — never scaled off another. Published IV12/IV15 ratios
span 1.33–1.44 across companies, so no constant multiplier fits.

Two models share the stream and are blended:

1. **Long-horizon** — stages 1 and 2, then a terminal perpetuity at the tier cap.
2. **Exit multiple** — project to year 15, apply a market multiple.

**IVB** inverts the ladder: the CAGR today's price implies. It needs no target return chosen
in advance, which arguably makes it the most useful single output.

A **negative IV15 is meaningful** — no share price delivers that return, not even $0.01. The
engine never floors it.

---

## 🚦 What is calculated vs. what is judgement

| Calculated — trust it | Judgement — yours to set |
| :--- | :--- |
| Every Tragic Algebra term | Normalised recurring owner earnings |
| Pooled ΔE and retention | Stage 1 growth rate |
| The full IV ladder and IVB | Moat tier |
| Split, listing and M&A adjustments | Exit multiple and model blend |

Burry writes thousands of words per company largely to justify the right-hand column. The app
seeds sensible defaults and flags when they cannot be trusted; it does not pretend to replace
the judgement.

---

## 🚀 Features

* **SEC EDGAR ingestion** — annual facts only, filtered on period duration and deduped by
  filing, with an IFRS fallback for foreign issuers. Rate-limited to ~6.7 req/s with backoff.
* **Watchlist screening** — up to 25 tickers ranked by ΔE, with CSV export. IV15 appears only
  where inputs pass every sanity check.
* **Stress testing** — downgrade the tier and cut growth, then re-value.
* **Calibration** — enter a published IV15 and solve for the growth it implies.
* **Structural adjustments** — stock splits restated onto a current basis (Gate 3
  continuity), listing years and share-funded acquisitions excluded, non-compensation
  issuance deducted from ΔS.
* **Guards that refuse to guess** — dual-class share counts, implausible P/E ratios,
  ΔE outside a meaningful range, financial-sector structures, and unbounded growth seeds all
  produce a warning rather than a confident wrong number.

---

## ⚠️ Known limitations

* **Financials.** Banks, insurers, brokers and REITs get net cash zeroed and a warning. The
  framework was built for software; these need book-value and combined-ratio thinking it does
  not contain.
* **Complex structures.** Up-C partnerships with large non-controlling interests report only
  the parent's slice of income against a full share count.
* **M&A share issuance.** Deducted where XBRL tags it, and whole years are excluded when the
  share count jumps more than 15%. Smaller untagged issuance still inflates ΔS.
* **Owner earnings normalisation.** Where ΔE is negative or absurd, the figure must be set by
  hand. Burry does the same — DocuSign's ΔE is deeply negative, yet he assigns ~$195M of
  forward owner earnings on judgement.
* **Paylocity** remains unreconciled against its published IV15, one outlier in eight.

---

## 🛠 Running locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Put a real email address in `SEC_HEADERS` at the top of `app.py`. The SEC blocks generic user
agents outright.

Dependencies: `streamlit`, `pandas`, `requests`. Nothing else.

---

## ⚖️ Disclaimer

Educational and analytical software. Not financial, tax, or investment advice. Outputs depend
on estimates you supply — change the growth rate and the answer changes a great deal. Method
follows Michael Burry's published writing; this project is independent and is not affiliated
with or endorsed by him or Scion Asset Management.
